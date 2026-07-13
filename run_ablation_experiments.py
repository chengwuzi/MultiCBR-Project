#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import gc
import json
import os
import re
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import torch
import yaml

from train import resolve_train_style, run_cbr_training, run_dwt_training, set_seed
from utility import Datasets


DEFAULT_TOPKS = (10, 20, 40, 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Run configured ablation experiments.")
    parser.add_argument("--config", default="config.yaml", help="Base tuned config file.")
    parser.add_argument("--experiments", default="ablation_experiments.yaml", help="Ablation experiment list.")
    parser.add_argument("--output-dir", default="ablation_runs", help="Directory for run logs and summaries.")
    parser.add_argument("--gpu", default="0", help="GPU id passed into the training config.")
    parser.add_argument("--model", default="AnchorViewBundleNet", help="Model name expected by the current project.")
    parser.add_argument("--max-retries", default=5, type=int, help="Maximum attempts for each experiment.")
    parser.add_argument("--heartbeat-seconds", default=300, type=int, help="Progress heartbeat interval.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print plans without training.")
    parser.add_argument("--only", default=None, help="Run only an experiment name or group.")
    parser.add_argument("--include-disabled", action="store_true", help="Include experiments with enabled: false.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip experiments already marked success in results.csv.")
    return parser.parse_args()


def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_elapsed(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def dump_yaml(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def set_by_path(config, dotted_path, value):
    if not dotted_path or not isinstance(dotted_path, str):
        raise ValueError(f"Invalid override path: {dotted_path!r}")
    parts = dotted_path.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or target[part] is None:
            target[part] = {}
        if not isinstance(target[part], dict):
            raise TypeError(f"Cannot set {dotted_path!r}: {part!r} is not a mapping")
        target = target[part]
    target[parts[-1]] = value


def apply_overrides(config, overrides):
    for key, value in (overrides or {}).items():
        if "." in key:
            set_by_path(config, key, value)
        else:
            config[key] = value


def normalize_experiment(raw):
    if "name" not in raw:
        raise ValueError(f"Experiment is missing name: {raw!r}")
    if "dataset" not in raw:
        raise ValueError(f"Experiment {raw['name']!r} is missing dataset")
    return {
        "name": str(raw["name"]),
        "dataset": str(raw["dataset"]),
        "group": str(raw.get("group", "default")),
        "enabled": bool(raw.get("enabled", True)),
        "overrides": dict(raw.get("overrides", {}) or {}),
    }


def resolve_experiment(base_configs, experiment, gpu, model):
    dataset = experiment["dataset"]
    base_key = dataset.split("_")[0] if "_" in dataset else dataset
    if base_key not in base_configs:
        raise KeyError(f"Dataset {dataset!r} resolves to {base_key!r}, which is not in config.yaml")

    conf = copy.deepcopy(base_configs[base_key])
    apply_overrides(conf, experiment["overrides"])
    conf["dataset"] = dataset
    conf["model"] = model
    conf["gpu"] = gpu
    conf["info"] = experiment["name"]
    return conf


def flatten_overrides(overrides):
    if not overrides:
        return ""
    return json.dumps(overrides, ensure_ascii=False, sort_keys=True)


def print_plan(experiments):
    print(f"Resolved {len(experiments)} enabled experiment(s).")
    for idx, exp in enumerate(experiments, start=1):
        print(f"[{idx}/{len(experiments)}] {exp['name']} | dataset={exp['dataset']} | group={exp['group']}")
        if exp["overrides"]:
            for key, value in exp["overrides"].items():
                print(f"  - {key} = {value}")
        else:
            print("  - overrides: <none>")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class Heartbeat:
    def __init__(self, message_func, seconds):
        self.message_func = message_func
        self.seconds = max(int(seconds), 1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=1)

    def _run(self):
        while not self.stop_event.wait(self.seconds):
            print(self.message_func(), flush=True)


def write_status(status_path, event):
    payload = dict(event)
    payload.setdefault("time", now_string())
    with open(status_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def prepare_training_config(conf):
    train_style = resolve_train_style(conf)
    if train_style == "dwt":
        set_seed(conf.get("seed", None))

    dataset = Datasets(conf)
    if train_style != "dwt":
        set_seed(conf.get("seed", None))

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf["gpu"])
    conf["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return train_style, dataset


def run_training(conf):
    train_style, dataset = prepare_training_config(conf)
    print(conf)
    print(f"train_style: {train_style}")
    if train_style == "cbr":
        run_cbr_training(conf, dataset, conf["device"])
    elif train_style == "dwt":
        run_dwt_training(conf, dataset, conf["device"])
    else:
        raise ValueError(f"Unsupported train_style: {train_style}")


def parse_best_metrics(log_path, topks=DEFAULT_TOPKS):
    pattern = re.compile(
        r"Best in epoch\s+(\d+),\s+TOP\s+(\d+):\s+REC_([VT])=([0-9.]+),\s+NDCG_\3=([0-9.]+)"
    )
    metrics = {"val": {}, "test": {}}
    best_epoch = ""
    if not os.path.exists(log_path):
        return best_epoch, metrics

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            epoch, topk, split_code, recall, ndcg = match.groups()
            split = "val" if split_code == "V" else "test"
            topk = int(topk)
            metrics[split][topk] = {
                "recall": float(recall),
                "ndcg": float(ndcg),
            }
            best_epoch = int(epoch)
    return best_epoch, metrics


def result_fieldnames(topks=DEFAULT_TOPKS):
    fields = [
        "name",
        "dataset",
        "group",
        "status",
        "attempts",
        "start_time",
        "end_time",
        "elapsed",
        "best_epoch",
    ]
    for split in ("val", "test"):
        for topk in topks:
            fields.append(f"{split}_recall@{topk}")
            fields.append(f"{split}_ndcg@{topk}")
    fields.extend(["log_path", "error", "overrides"])
    return fields


def append_result(results_path, row):
    fieldnames = result_fieldnames()
    exists = os.path.exists(results_path)
    with open(results_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def completed_names(results_path):
    if not os.path.exists(results_path):
        return set()
    names = set()
    with open(results_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "success":
                names.add(row.get("name"))
    return names


def build_result_row(experiment, status, attempts, start_time, end_time, log_path, error=""):
    best_epoch, metrics = parse_best_metrics(log_path)
    row = {
        "name": experiment["name"],
        "dataset": experiment["dataset"],
        "group": experiment["group"],
        "status": status,
        "attempts": attempts,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed": format_elapsed(time.time() - experiment["_started_at_seconds"]),
        "best_epoch": best_epoch,
        "log_path": log_path,
        "error": error,
        "overrides": flatten_overrides(experiment["overrides"]),
    }
    for split in ("val", "test"):
        for topk in DEFAULT_TOPKS:
            values = metrics.get(split, {}).get(topk, {})
            row[f"{split}_recall@{topk}"] = values.get("recall", "")
            row[f"{split}_ndcg@{topk}"] = values.get("ndcg", "")
    return row


def run_one_experiment(index, total, experiment, resolved_conf, paths, args):
    max_retries = max(int(args.max_retries), 1)
    final_error = ""
    last_log_path = ""
    experiment["_started_at_seconds"] = time.time()
    start_time = now_string()

    print("=" * 90)
    print(f"[{index}/{total}] START {experiment['name']}")
    print(f"Dataset: {experiment['dataset']} | Group: {experiment['group']}")
    print(f"Overrides: {flatten_overrides(experiment['overrides']) or '<none>'}")
    print("=" * 90)

    for attempt in range(1, max_retries + 1):
        log_path = os.path.join(paths["logs"], f"{experiment['name']}_attempt{attempt}.log")
        last_log_path = log_path
        write_status(
            paths["status"],
            {
                "event": "attempt_start",
                "name": experiment["name"],
                "dataset": experiment["dataset"],
                "group": experiment["group"],
                "attempt": attempt,
                "max_retries": max_retries,
            },
        )
        print(f"[{index}/{total}] Attempt {attempt}/{max_retries}: {experiment['name']}")

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                tee_out = Tee(sys.__stdout__, log_file)
                tee_err = Tee(sys.__stderr__, log_file)
                with redirect_stdout(tee_out), redirect_stderr(tee_err):
                    print(f"Experiment: {experiment['name']}")
                    print(f"Dataset: {experiment['dataset']}")
                    print(f"Group: {experiment['group']}")
                    print(f"Attempt: {attempt}/{max_retries}")
                    print(f"Start time: {now_string()}")
                    print("Overrides:")
                    print(yaml.safe_dump(experiment["overrides"], allow_unicode=True, sort_keys=False).strip() or "<none>")
                    print("Resolved config:")
                    print(yaml.safe_dump(resolved_conf, allow_unicode=True, sort_keys=False).strip())

                    attempt_start = time.time()
                    heartbeat = Heartbeat(
                        lambda: (
                            f"[Heartbeat] [{index}/{total}] {experiment['name']} "
                            f"attempt {attempt}/{max_retries} elapsed {format_elapsed(time.time() - attempt_start)}"
                        ),
                        args.heartbeat_seconds,
                    )
                    with heartbeat:
                        run_training(copy.deepcopy(resolved_conf))

                    print(f"End time: {now_string()}")

            end_time = now_string()
            write_status(
                paths["status"],
                {
                    "event": "success",
                    "name": experiment["name"],
                    "attempt": attempt,
                    "log_path": log_path,
                },
            )
            row = build_result_row(experiment, "success", attempt, start_time, end_time, log_path)
            append_result(paths["results"], row)
            print(f"[{index}/{total}] SUCCESS {experiment['name']} attempts={attempt} elapsed={row['elapsed']}")
            if row.get("test_recall@20") != "":
                print(f"  Test R@20={row['test_recall@20']} N@20={row['test_ndcg@20']}")
            return True
        except Exception as exc:
            final_error = f"{type(exc).__name__}: {exc}"
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("\nException traceback:\n")
                log_file.write(traceback.format_exc())
            write_status(
                paths["status"],
                {
                    "event": "attempt_failed",
                    "name": experiment["name"],
                    "attempt": attempt,
                    "error": final_error,
                    "log_path": log_path,
                },
            )
            print(f"[{index}/{total}] FAILED attempt {attempt}/{max_retries}: {experiment['name']} -> {final_error}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    end_time = now_string()
    row = build_result_row(experiment, "failed", max_retries, start_time, end_time, last_log_path, final_error)
    append_result(paths["results"], row)
    write_status(
        paths["status"],
        {
            "event": "failed",
            "name": experiment["name"],
            "attempts": max_retries,
            "error": final_error,
            "log_path": last_log_path,
        },
    )
    print(f"[{index}/{total}] FAILED {experiment['name']} after {max_retries} attempt(s)")
    return False


def main():
    args = parse_args()
    base_configs = load_yaml(args.config)
    experiment_file = load_yaml(args.experiments)
    raw_experiments = experiment_file.get("experiments", [])
    experiments = [normalize_experiment(item) for item in raw_experiments]

    if not args.include_disabled:
        experiments = [item for item in experiments if item["enabled"]]
    if args.only:
        experiments = [
            item for item in experiments
            if item["name"] == args.only or item["group"] == args.only
        ]

    run_dir = os.path.join(args.output_dir, stamp_string())
    paths = {
        "run_dir": run_dir,
        "logs": os.path.join(run_dir, "logs"),
        "resolved": os.path.join(run_dir, "resolved_configs"),
        "status": os.path.join(run_dir, "status.jsonl"),
        "results": os.path.join(run_dir, "results.csv"),
        "plan": os.path.join(run_dir, "run_plan.yaml"),
    }
    for key in ("run_dir", "logs", "resolved"):
        ensure_dir(paths[key])

    resolved_items = []
    for experiment in experiments:
        resolved_conf = resolve_experiment(base_configs, experiment, args.gpu, args.model)
        resolved_items.append((experiment, resolved_conf))

    print_plan([item[0] for item in resolved_items])
    dump_yaml(
        paths["plan"],
        {
            "created_at": now_string(),
            "config": args.config,
            "experiments": args.experiments,
            "dry_run": args.dry_run,
            "items": [
                {
                    "name": experiment["name"],
                    "dataset": experiment["dataset"],
                    "group": experiment["group"],
                    "enabled": experiment["enabled"],
                    "overrides": experiment["overrides"],
                }
                for experiment, _ in resolved_items
            ],
        },
    )

    for experiment, resolved_conf in resolved_items:
        dump_yaml(os.path.join(paths["resolved"], f"{experiment['name']}.yaml"), resolved_conf)

    if args.dry_run:
        print(f"Dry run complete. Plan written to {paths['plan']}")
        print(f"Resolved configs written to {paths['resolved']}")
        return

    skip_names = completed_names(paths["results"]) if args.skip_completed else set()
    total = len(resolved_items)
    successes = 0
    failures = 0

    for idx, (experiment, resolved_conf) in enumerate(resolved_items, start=1):
        if experiment["name"] in skip_names:
            print(f"[{idx}/{total}] SKIP completed {experiment['name']}")
            write_status(paths["status"], {"event": "skipped", "name": experiment["name"], "reason": "completed"})
            continue

        ok = run_one_experiment(idx, total, experiment, resolved_conf, paths, args)
        if ok:
            successes += 1
        else:
            failures += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 90)
    print(f"Ablation run finished. success={successes}, failed={failures}, total={total}")
    print(f"Run directory: {run_dir}")
    print(f"Results: {paths['results']}")
    print("=" * 90)


if __name__ == "__main__":
    main()
