#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the ReCoBR sensitivity campaign as isolated, recoverable processes.

This runner deliberately does not reuse historical results or merge repeated
configurations: every logical row is a newly launched physical training task.
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import yaml


REQUIRED_METRICS = (
    ("val", "recall", "20"),
    ("val", "ndcg", "20"),
    ("val", "recall", "40"),
    ("val", "ndcg", "40"),
    ("test", "recall", "20"),
    ("test", "ndcg", "20"),
    ("test", "recall", "40"),
    ("test", "ndcg", "40"),
)

RESULT_COLUMNS = (
    "dataset",
    "sweep_param",
    "sweep_value",
    "lambda_set",
    "lambda_a",
    "lambda_c",
    "seed",
    "best_epoch",
    "best_epoch_index",
    "best_val_score",
    "val_R20",
    "val_N20",
    "val_R40",
    "val_N40",
    "test_R20",
    "test_N20",
    "test_R40",
    "test_N40",
    "status",
    "run_id",
    "config_path",
    "log_path",
    "checkpoint_path",
    "result_path",
    "error",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated ReCoBR sensitivity tasks.")
    parser.add_argument("--plan", default="recobr_sensitivity_plan.yaml", help="Sensitivity campaign definition.")
    parser.add_argument("--base-config", default="config.yaml", help="Base project config used for deep copies.")
    parser.add_argument("--run-dir", default=None, help="Campaign directory; reuse it to resume.")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU order, e.g. 0,1. Defaults to plan.gpu_order.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for isolated training processes.")
    parser.add_argument("--dry-run", action="store_true", help="Write and preview all task configs without training.")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun failed tasks; successful valid tasks are always skipped.")
    parser.add_argument("--only", default=None, help="Only run task ids/datasets containing this case-insensitive text.")
    return parser.parse_args()


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def dump_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def read_json(path, default):
    if not path.is_file():
        return copy.deepcopy(default)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def set_by_path(mapping, dotted_path, value):
    parts = dotted_path.split(".")
    target = mapping
    for part in parts[:-1]:
        if part not in target or target[part] is None:
            target[part] = {}
        if not isinstance(target[part], dict):
            raise TypeError(f"Cannot set {dotted_path}: {part} is not a mapping")
        target = target[part]
    target[parts[-1]] = value


def get_by_path(mapping, dotted_path):
    target = mapping
    for part in dotted_path.split("."):
        if not isinstance(target, dict) or part not in target:
            raise KeyError(f"Missing configuration field {dotted_path}")
        target = target[part]
    return target


def apply_overrides(mapping, overrides):
    for path, value in (overrides or {}).items():
        set_by_path(mapping, path, copy.deepcopy(value))


def value_token(value):
    decimal = Decimal(str(value)).normalize()
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def as_float(value, name):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


def require_eight_values(name, values):
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError(f"{name} must contain exactly 8 values; got {values!r}")
    parsed = [as_float(value, name) for value in values]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} contains duplicate values: {values!r}")
    return parsed


def get_lambda_set(config):
    if config.get("train_style", "cbr").lower() == "dwt":
        return as_float(get_by_path(config, "dwt.latent_diffusion_set_loss_weight"), "lambda_set")
    return as_float(get_by_path(config, "latent_rebuild.latent_diffusion_set_loss_weight"), "lambda_set")


def get_lambda_c(config):
    values = config.get("c_lambdas")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"c_lambdas must be a singleton list for one isolated task, got {values!r}")
    return as_float(values[0], "lambda_c")


def get_lambda_a(config):
    return as_float(get_by_path(config, "anchor_cl.anchor_cl_lambda"), "lambda_a")


def assert_close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label} expected {expected}, got {actual}")


def build_task(base_config, dataset_name, dataset_spec, sweep_param, sweep_value, sequence):
    config = copy.deepcopy(base_config)
    apply_overrides(config, dataset_spec.get("baseline_overrides", {}))
    train_style = config.get("train_style", "cbr").strip().lower()

    if sweep_param == "lambda_set":
        if train_style == "dwt":
            set_by_path(config, "dwt.use_latent_diffusion_rebuild", True)
            set_by_path(config, "dwt.latent_diffusion_set_loss_weight", sweep_value)
        else:
            set_by_path(config, "latent_rebuild.enabled", True)
            set_by_path(config, "latent_rebuild.latent_diffusion_set_loss_weight", sweep_value)
    elif sweep_param == "lambda_a":
        set_by_path(config, "anchor_cl.enabled", True)
        set_by_path(config, "anchor_cl.anchor_cl_lambda", sweep_value)
        config["c_lambdas"] = [0.0]
    elif sweep_param == "lambda_c":
        config["c_lambdas"] = [sweep_value]
    else:
        raise ValueError(f"Unsupported sweep parameter {sweep_param!r}")

    lambda_set = get_lambda_set(config)
    lambda_a = get_lambda_a(config)
    lambda_c = get_lambda_c(config)
    run_id = f"{dataset_name.lower()}_{sweep_param}_{value_token(sweep_value)}"
    config["dataset"] = dataset_name
    config["model"] = "AnchorViewBundleNet"
    config["info"] = run_id
    config["sensitivity_task"] = {
        "campaign": "recobr_sensitivity",
        "run_id": run_id,
        "sweep_param": sweep_param,
        "sweep_value": float(sweep_value),
        "lambda_set": lambda_set,
        "lambda_a": lambda_a,
        "lambda_c": lambda_c,
    }

    task = {
        "sequence": sequence,
        "run_id": run_id,
        "dataset": dataset_name,
        "train_style": train_style,
        "sweep_param": sweep_param,
        "sweep_value": float(sweep_value),
        "lambda_set": lambda_set,
        "lambda_a": lambda_a,
        "lambda_c": lambda_c,
        "seed": config.get("seed"),
        "epochs": config.get("epochs"),
        "test_interval": config.get("test_interval"),
        "config": config,
    }
    validate_task_config(task)
    return task


def validate_task_config(task):
    config = task["config"]
    if task["seed"] is None:
        raise ValueError(f"{task['run_id']} has no fixed seed")
    assert_close(get_lambda_set(config), task["lambda_set"], f"{task['run_id']} lambda_set")
    assert_close(get_lambda_a(config), task["lambda_a"], f"{task['run_id']} lambda_a")
    assert_close(get_lambda_c(config), task["lambda_c"], f"{task['run_id']} lambda_c")

    if task["train_style"] == "cbr":
        if not get_by_path(config, "latent_rebuild.enabled"):
            raise ValueError(f"{task['run_id']} must enable latent_rebuild")
    elif task["train_style"] == "dwt":
        if not get_by_path(config, "dwt.use_latent_diffusion_rebuild"):
            raise ValueError(f"{task['run_id']} must enable DWT latent rebuild")
        if not get_by_path(config, "dwt.enable_recobr_auxiliary_losses"):
            raise ValueError(
                f"{task['run_id']} must enable DWT ReCoBR auxiliary losses "
                "so fixed lambda_a and lambda_c are effective during every sweep"
            )
    else:
        raise ValueError(f"{task['run_id']} uses unsupported train_style {task['train_style']!r}")

    if task["sweep_param"] == "lambda_a":
        assert_close(task["lambda_c"], 0.0, f"{task['run_id']} lambda_c while scanning lambda_a")


def build_tasks(plan, base_configs):
    dataset_order = plan.get("dataset_order", ["Youshu", "iFashion", "NetEase"])
    dataset_specs = plan.get("datasets", {})
    sweep_values = plan.get("sweep_values", {})
    tasks = []
    seen_ids = set()

    for dataset_name in dataset_order:
        if dataset_name not in base_configs:
            raise KeyError(f"Dataset {dataset_name!r} is absent from base config")
        if dataset_name not in dataset_specs:
            raise KeyError(f"Dataset {dataset_name!r} is absent from plan.datasets")
        dataset_spec = dataset_specs[dataset_name]
        for sweep_param in dataset_spec.get("sweeps", []):
            values = require_eight_values(
                f"sweep_values.{sweep_param}",
                sweep_values.get(sweep_param),
            )
            for sweep_value in values:
                task = build_task(
                    base_configs[dataset_name],
                    dataset_name,
                    dataset_spec,
                    sweep_param,
                    sweep_value,
                    len(tasks) + 1,
                )
                if task["run_id"] in seen_ids:
                    raise ValueError(f"Duplicate task id {task['run_id']}")
                seen_ids.add(task["run_id"])
                tasks.append(task)

    expected_count = int(plan.get("expected_task_count", 48))
    if len(tasks) != expected_count:
        raise ValueError(f"Expected {expected_count} fresh physical tasks, generated {len(tasks)}")
    return tasks


def compact_task(task, run_dir):
    task_dir = run_dir / "tasks" / task["run_id"]
    compact = {
        key: task[key]
        for key in (
            "sequence",
            "run_id",
            "dataset",
            "train_style",
            "sweep_param",
            "sweep_value",
            "lambda_set",
            "lambda_a",
            "lambda_c",
            "seed",
            "epochs",
            "test_interval",
        )
    }
    compact["task_dir"] = str(task_dir.relative_to(run_dir))
    return compact


def task_paths(run_dir, task):
    task_dir = run_dir / "tasks" / task["run_id"]
    return {
        "task_dir": task_dir,
        "preview_config": task_dir / "prepared_config.yaml",
        "metadata": task_dir / "task.json",
    }


def attempt_paths(run_dir, task, attempt):
    task_dir = run_dir / "tasks" / task["run_id"]
    attempt_dir = task_dir / "attempts" / f"attempt_{attempt:03d}"
    return {
        "task_dir": task_dir,
        "attempt_dir": attempt_dir,
        "config": attempt_dir / "config.yaml",
        "result": attempt_dir / "train_result.json",
        "stdout": attempt_dir / "stdout.log",
        "artifact_dir": attempt_dir / "artifacts",
    }


def resolve_record_path(run_dir, path_value):
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else run_dir / path


def validate_result(task, result_path):
    if not result_path.is_file():
        return False, "missing train_result.json", None
    try:
        result = read_json(result_path, None)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid train_result.json: {exc}", None
    if not isinstance(result, dict) or result.get("status") != "success":
        return False, "result JSON does not report success", None
    if result.get("dataset") != task["dataset"]:
        return False, "result dataset does not match task", None
    results = result.get("results")
    if not isinstance(results, list) or len(results) != 1:
        return False, "result must contain exactly one training result", None
    run_result = results[0]
    if not run_result.get("valid", False):
        return False, "training result has no valid best checkpoint", None
    checkpoint_path = Path(run_result.get("checkpoint_path", ""))
    if not checkpoint_path.is_file():
        return False, "best checkpoint is missing", None

    config = result.get("effective_config")
    try:
        assert_close(get_lambda_set(config), task["lambda_set"], "effective lambda_set")
        assert_close(get_lambda_a(config), task["lambda_a"], "effective lambda_a")
        assert_close(get_lambda_c(config), task["lambda_c"], "effective lambda_c")
        if config.get("seed") != task["seed"]:
            raise ValueError("effective seed differs")
        metrics = run_result["metrics"]
        for split, metric_name, topk in REQUIRED_METRICS:
            metric = float(metrics[split][metric_name][topk])
            if not math.isfinite(metric):
                raise ValueError(f"non-finite {split}.{metric_name}@{topk}")
        if not math.isfinite(float(run_result["best_val_score"])):
            raise ValueError("non-finite best validation score")
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid effective result: {exc}", None
    return True, "", result


class Campaign:
    def __init__(self, run_dir, tasks):
        self.run_dir = run_dir
        self.tasks = tasks
        self.task_by_id = {task["run_id"]: task for task in tasks}
        self.state_path = run_dir / "state.json"
        self.results_path = run_dir / "results.csv"
        self.lock = threading.Lock()
        self.state = read_json(self.state_path, {"schema_version": 1, "created_at": now(), "tasks": {}})
        self.state.setdefault("tasks", {})

    def initialise(self):
        with self.lock:
            for task in self.tasks:
                record = self.state["tasks"].setdefault(
                    task["run_id"],
                    {"status": "pending", "attempts": 0, "created_at": now()},
                )
                if record.get("status") == "running":
                    record.update({"status": "pending", "error": "previous launcher ended while task was running"})
                if record.get("status") == "success":
                    result_path = resolve_record_path(self.run_dir, record.get("result_path"))
                    if result_path is None:
                        valid, error = False, "successful task has no recorded result path"
                    else:
                        valid, error, _ = validate_result(task, result_path)
                    if not valid:
                        record.update({"status": "pending", "error": f"recovery validation failed: {error}"})
            self._persist_locked()

    def should_run(self, task, rerun_failed):
        record = self.state["tasks"][task["run_id"]]
        return record.get("status") == "pending" or (rerun_failed and record.get("status") == "failed")

    def update(self, run_id, **updates):
        with self.lock:
            record = self.state["tasks"][run_id]
            record.update(updates)
            self._persist_locked()

    def record(self, run_id):
        with self.lock:
            return copy.deepcopy(self.state["tasks"][run_id])

    def _persist_locked(self):
        self.state["updated_at"] = now()
        write_json_atomic(self.state_path, self.state)
        self._write_results_locked()

    def _write_results_locked(self):
        temporary_path = self.results_path.with_name(self.results_path.name + ".tmp")
        with open(temporary_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for task in self.tasks:
                writer.writerow(self._result_row(task))
        os.replace(temporary_path, self.results_path)

    def _result_row(self, task):
        paths = task_paths(self.run_dir, task)
        record = self.state["tasks"][task["run_id"]]
        config_path = record.get("config_path", str(paths["preview_config"].relative_to(self.run_dir)))
        log_path = record.get("log_path", "")
        result_path = record.get("result_path", "")
        row = {
            "dataset": task["dataset"],
            "sweep_param": task["sweep_param"],
            "sweep_value": task["sweep_value"],
            "lambda_set": task["lambda_set"],
            "lambda_a": task["lambda_a"],
            "lambda_c": task["lambda_c"],
            "seed": task["seed"],
            "status": record.get("status", "pending"),
            "run_id": task["run_id"],
            "config_path": config_path,
            "log_path": log_path,
            "result_path": result_path,
            "checkpoint_path": record.get("checkpoint_path", ""),
            "error": record.get("error", ""),
        }
        if record.get("status") != "success":
            return row
        completed_result_path = resolve_record_path(self.run_dir, result_path)
        if completed_result_path is None:
            row["status"] = "failed"
            row["error"] = "successful task has no recorded result path"
            return row
        valid, error, result = validate_result(task, completed_result_path)
        if not valid:
            row["status"] = "failed"
            row["error"] = f"post-success validation failed: {error}"
            return row
        run_result = result["results"][0]
        metrics = run_result["metrics"]
        row.update(
            {
                "best_epoch": run_result["best_epoch"],
                "best_epoch_index": run_result["best_epoch_index"],
                "best_val_score": run_result["best_val_score"],
                "val_R20": metrics["val"]["recall"]["20"],
                "val_N20": metrics["val"]["ndcg"]["20"],
                "val_R40": metrics["val"]["recall"]["40"],
                "val_N40": metrics["val"]["ndcg"]["40"],
                "test_R20": metrics["test"]["recall"]["20"],
                "test_N20": metrics["test"]["ndcg"]["20"],
                "test_R40": metrics["test"]["recall"]["40"],
                "test_N40": metrics["test"]["ndcg"]["40"],
                "checkpoint_path": run_result["checkpoint_path"],
            }
        )
        return row


def write_task_files(run_dir, task, gpu, attempt):
    paths = attempt_paths(run_dir, task, attempt)
    paths["attempt_dir"].mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(task["config"])
    config["gpu"] = str(gpu)
    config["artifact_dir"] = str(paths["artifact_dir"].resolve())
    dump_yaml(paths["config"], config)

    task_root_paths = task_paths(run_dir, task)
    task_root_paths["task_dir"].mkdir(parents=True, exist_ok=True)
    metadata = compact_task(task, run_dir)
    metadata.update(
        {
            "latest_attempt": attempt,
            "gpu": str(gpu),
            "config_path": str(paths["config"].relative_to(run_dir)),
            "result_path": str(paths["result"].relative_to(run_dir)),
            "log_path": str(paths["stdout"].relative_to(run_dir)),
        }
    )
    write_json_atomic(task_root_paths["metadata"], metadata)
    return paths


def write_preview_task_files(run_dir, task, gpu):
    paths = task_paths(run_dir, task)
    paths["task_dir"].mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(task["config"])
    config["gpu"] = str(gpu)
    dump_yaml(paths["preview_config"], config)
    metadata = compact_task(task, run_dir)
    metadata.update(
        {
            "planned_gpu": str(gpu),
            "prepared_config_path": str(paths["preview_config"].relative_to(run_dir)),
        }
    )
    write_json_atomic(paths["metadata"], metadata)
    return paths


def run_task(campaign, task, gpu, python_executable, project_root):
    old_record = campaign.record(task["run_id"])
    attempt = int(old_record.get("attempts", 0)) + 1
    paths = write_task_files(campaign.run_dir, task, gpu, attempt)
    command = [
        python_executable,
        "-u",
        "train.py",
        "--config",
        str(paths["config"].resolve()),
        "--artifact-dir",
        str(paths["task_dir"].resolve()),
        "--result-json",
        str(paths["result"].resolve()),
        "--gpu",
        str(gpu),
        "--dataset",
        task["dataset"],
        "--model",
        "AnchorViewBundleNet",
        "--info",
        task["run_id"],
    ]
    campaign.update(
        task["run_id"],
        status="running",
        attempts=attempt,
        gpu=str(gpu),
        started_at=now(),
        error="",
        command=command,
        config_path=str(paths["config"].relative_to(campaign.run_dir)),
        log_path=str(paths["stdout"].relative_to(campaign.run_dir)),
        result_path=str(paths["result"].relative_to(campaign.run_dir)),
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    with open(paths["stdout"], "w", encoding="utf-8") as log_handle:
        log_handle.write("Command: " + " ".join(command) + "\n\n")
        completed = subprocess.run(
            command,
            cwd=project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    valid, error, result = validate_result(task, paths["result"])
    if completed.returncode == 0 and valid:
        run_result = result["results"][0]
        campaign.update(
            task["run_id"],
            status="success",
            ended_at=now(),
            return_code=completed.returncode,
            checkpoint_path=run_result["checkpoint_path"],
            error="",
        )
        print(f"SUCCESS {task['run_id']} on GPU {gpu}", flush=True)
        return True

    reason = error if completed.returncode == 0 else f"training process exited {completed.returncode}; {error}"
    campaign.update(
        task["run_id"],
        status="failed",
        ended_at=now(),
        return_code=completed.returncode,
        error=reason,
    )
    print(f"FAILED {task['run_id']} on GPU {gpu}: {reason}", flush=True)
    return False


def parse_gpus(raw_value, fallback):
    values = raw_value if raw_value is not None else fallback
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    if not isinstance(values, list) or not values:
        raise ValueError("At least one GPU id is required")
    return [str(value) for value in values]


def print_plan(tasks):
    print(f"Prepared {len(tasks)} fresh physical training task(s); no result reuse is configured.")
    for task in tasks:
        print(
            f"[{task['sequence']:02d}] {task['run_id']} | {task['dataset']} | "
            f"{task['sweep_param']}={task['sweep_value']} | "
            f"set={task['lambda_set']} a={task['lambda_a']} c={task['lambda_c']} | "
            f"seed={task['seed']} epochs={task['epochs']} eval={task['test_interval']}"
        )


def selected_tasks(tasks, selector):
    if not selector:
        return tasks
    normalized = selector.lower()
    selected = [
        task for task in tasks
        if normalized in task["run_id"].lower() or normalized in task["dataset"].lower()
    ]
    if not selected:
        raise ValueError(f"--only={selector!r} did not match any task")
    return selected


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    plan_path = (project_root / args.plan).resolve()
    base_config_path = (project_root / args.base_config).resolve()
    plan = load_yaml(plan_path)
    base_configs = load_yaml(base_config_path)
    tasks = build_tasks(plan, base_configs)
    gpus = parse_gpus(args.gpus, plan.get("gpu_order", ["0"]))
    run_dir = Path(args.run_dir).resolve() if args.run_dir else project_root / "ablation_runs" / f"recobr_sensitivity_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        run_dir / "run_plan.json",
        {
            "created_at": now(),
            "plan_path": str(plan_path),
            "base_config_path": str(base_config_path),
            "gpu_order": gpus,
            "task_count": len(tasks),
            "reuse_results": False,
            "tasks": [compact_task(task, run_dir) for task in tasks],
        },
    )
    print_plan(tasks)

    campaign = Campaign(run_dir, tasks)
    campaign.initialise()
    if args.dry_run:
        for index, task in enumerate(tasks):
            write_preview_task_files(run_dir, task, gpus[index % len(gpus)])
        print(f"Dry run complete: {run_dir}")
        return

    execution_tasks = [task for task in selected_tasks(tasks, args.only) if campaign.should_run(task, args.rerun_failed)]
    if not execution_tasks:
        print("No runnable tasks. Successful valid tasks are skipped; use --rerun-failed for failures.")
        return

    queues = {gpu: [] for gpu in gpus}
    for task in execution_tasks:
        queues[gpus[(task["sequence"] - 1) % len(gpus)]].append(task)

    def worker(gpu, queue):
        outcomes = []
        for task in queue:
            outcomes.append(run_task(campaign, task, gpu, args.python, str(project_root)))
        return outcomes

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, queue) for gpu, queue in queues.items() if queue]
        outcomes = [outcome for future in as_completed(futures) for outcome in future.result()]
    print(
        f"Campaign finished: success={sum(outcomes)}, failed={len(outcomes) - sum(outcomes)}, "
        f"executed={len(outcomes)}, directory={run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
