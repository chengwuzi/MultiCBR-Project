#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml


# -----------------------------------------------------------------------------
# Sweep configuration
# Edit only this block before running the script.
# -----------------------------------------------------------------------------
SWEEP_CONFIG = {
    "dataset": "NetEase",
    "gpu": "0",
    "model": "MultiCBR",
    "info_prefix": "sweep",
    "max_retries_per_run": 3,
    "python_executable": sys.executable,
    "config_path": "config.yaml",
    "output_root": "sweep_outputs",
    "param_groups": [
        {
            "name": "latent_rebuild.rebuild_k",
            "values": [6, 9, 12],
        },
        {
            "name": "latent_rebuild.latent_diffusion_set_loss_weight",
            "values": [0.03, 0.04, 0.06, 0.07],
        },
        {
            "name": "latent_rebuild.latent_diffusion_num_steps",
            "values": [8, 10, 14, 16],
        },
    ],
}


BEST_LINE_RE = re.compile(
    r"Best in epoch (?P<epoch>\d+), TOP (?P<topk>\d+): "
    r"REC_(?P<split>[VT])=(?P<recall>[0-9.]+), "
    r"NDCG_(?P=split)=(?P<ndcg>[0-9.]+)"
)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ts_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def get_dataset_conf(config, dataset_name):
    if dataset_name not in config:
        raise KeyError(f"Dataset {dataset_name!r} not found in config.yaml")
    return config[dataset_name]


def set_nested_value(data, dotted_key, value):
    keys = dotted_key.split(".")
    cursor = data
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise KeyError(f"Config path {dotted_key!r} is invalid at {key!r}")
        cursor = cursor[key]
    leaf = keys[-1]
    if leaf not in cursor:
        raise KeyError(f"Config path {dotted_key!r} does not exist")
    cursor[leaf] = value


def format_run_config(dataset_name, model, gpu, info_value, dataset_conf):
    payload = {
        "dataset": dataset_name,
        "model": model,
        "gpu": gpu,
        "info": info_value,
        "config": dataset_conf,
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).rstrip()


def sanitize_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    return cleaned.strip("_") or "value"


def build_run_info(prefix, param_name, param_value):
    param_token = sanitize_name(param_name.replace(".", "_"))
    value_token = sanitize_name(param_value)
    return f"{prefix}_{param_token}_{value_token}"


def run_train(command, workdir, stdout_path):
    with open(stdout_path, "w", encoding="utf-8", newline="") as log_file:
        process = subprocess.run(
            command,
            cwd=workdir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def parse_best_results(text):
    best = {
        "V": {},
        "T": {},
    }
    for match in BEST_LINE_RE.finditer(text):
        split = match.group("split")
        topk = int(match.group("topk"))
        best[split][topk] = {
            "best_epoch": int(match.group("epoch")),
            "recall": float(match.group("recall")),
            "ndcg": float(match.group("ndcg")),
        }
    return best


def summarize_best_results(best_results):
    lines = []
    epoch_candidates = []
    for split in ("V", "T"):
        for topk in sorted(best_results[split]):
            epoch_candidates.append(best_results[split][topk]["best_epoch"])

    if epoch_candidates:
        lines.append(f"best_epoch_reference: {max(epoch_candidates)}")
    else:
        lines.append("best_epoch_reference: N/A")

    for topk in (10, 20, 40, 80):
        if topk in best_results["V"]:
            item = best_results["V"][topk]
            lines.append(
                f"TOP{topk}_V: epoch={item['best_epoch']}, "
                f"recall={item['recall']:.5f}, ndcg={item['ndcg']:.5f}"
            )
        if topk in best_results["T"]:
            item = best_results["T"][topk]
            lines.append(
                f"TOP{topk}_T: epoch={item['best_epoch']}, "
                f"recall={item['recall']:.5f}, ndcg={item['ndcg']:.5f}"
            )
    return "\n".join(lines)


def has_any_best_results(best_results):
    return bool(best_results["V"] or best_results["T"])


def append_detail_section(path, title, body):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{'=' * 24} {title} {'=' * 24}\n")
        f.write(body.rstrip() + "\n\n")


def append_summary_section(path, title, body):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{title}\n")
        f.write(body.rstrip() + "\n\n")


def main():
    project_root = Path(__file__).resolve().parent
    config_path = project_root / SWEEP_CONFIG["config_path"]
    output_root = project_root / SWEEP_CONFIG["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = SWEEP_CONFIG["dataset"]
    model = SWEEP_CONFIG["model"]
    gpu = str(SWEEP_CONFIG["gpu"])
    info_prefix = SWEEP_CONFIG["info_prefix"]
    python_executable = SWEEP_CONFIG["python_executable"]
    max_retries_per_run = int(SWEEP_CONFIG.get("max_retries_per_run", 3))
    if max_retries_per_run < 0:
        raise ValueError("SWEEP_CONFIG['max_retries_per_run'] must be >= 0")
    run_stamp = ts_str()
    run_dir = output_root / f"{dataset}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not SWEEP_CONFIG["param_groups"]:
        raise ValueError("SWEEP_CONFIG['param_groups'] cannot be empty")

    detail_path = run_dir / "detail.txt"
    summary_path = run_dir / "summary.txt"

    original_config_text = read_text(config_path)
    original_config_data = yaml.safe_load(original_config_text)

    sweep_header = {
        "dataset": dataset,
        "model": model,
        "gpu": gpu,
        "info_prefix": info_prefix,
        "max_retries_per_run": max_retries_per_run,
        "run_dir": str(run_dir),
        "started_at": now_str(),
        "param_groups": SWEEP_CONFIG["param_groups"],
    }

    append_detail_section(
        detail_path,
        "Sweep Start",
        json.dumps(sweep_header, ensure_ascii=False, indent=2),
    )
    append_summary_section(
        summary_path,
        "Sweep Start",
        json.dumps(sweep_header, ensure_ascii=False, indent=2),
    )

    try:
        for group_index, group in enumerate(SWEEP_CONFIG["param_groups"], start=1):
            param_name = group["name"]
            values = group["values"]
            if not values:
                raise ValueError(f"Parameter group {param_name!r} has an empty values list")

            group_title = f"Group {group_index}: {param_name}"
            append_detail_section(
                detail_path,
                group_title,
                f"values: {json.dumps(values, ensure_ascii=False)}",
            )
            append_summary_section(
                summary_path,
                group_title,
                f"values: {json.dumps(values, ensure_ascii=False)}",
            )

            for value_index, value in enumerate(values, start=1):
                run_id = f"{group_index:02d}_{value_index:02d}_{sanitize_name(param_name)}_{sanitize_name(value)}"

                config_data = copy.deepcopy(original_config_data)
                dataset_conf = get_dataset_conf(config_data, dataset)
                set_nested_value(dataset_conf, param_name, value)

                info_value = build_run_info(info_prefix, param_name, value)
                current_run_config_text = format_run_config(
                    dataset,
                    model,
                    gpu,
                    info_value,
                    dataset_conf,
                )
                command = [
                    python_executable,
                    "-u",
                    "train.py",
                    "-d",
                    dataset,
                    "-m",
                    model,
                    "-g",
                    gpu,
                    "-i",
                    info_value,
                ]

                started_at = now_str()
                max_attempts = 1 + max_retries_per_run
                attempts = []
                return_code = None
                best_results = {"V": {}, "T": {}}
                best_summary = "best_epoch_reference: N/A"
                final_stdout_path = None
                success = False

                for attempt_idx in range(1, max_attempts + 1):
                    stdout_path = run_dir / f"{run_id}_attempt{attempt_idx}_stdout.txt"
                    dump_yaml(config_path, config_data)

                    attempt_started_at = now_str()
                    current_return_code = run_train(command, str(project_root), stdout_path)
                    attempt_ended_at = now_str()
                    stdout_text = read_text(stdout_path)
                    current_best_results = parse_best_results(stdout_text)
                    parsed_best_results = has_any_best_results(current_best_results)
                    attempt_success = current_return_code == 0 and parsed_best_results

                    attempts.append(
                        {
                            "attempt": attempt_idx,
                            "started_at": attempt_started_at,
                            "ended_at": attempt_ended_at,
                            "return_code": current_return_code,
                            "stdout_file": stdout_path.name,
                            "parsed_best_results": parsed_best_results,
                            "status": "SUCCESS" if attempt_success else "FAILED",
                        }
                    )

                    return_code = current_return_code
                    final_stdout_path = stdout_path
                    best_results = current_best_results
                    best_summary = summarize_best_results(best_results)

                    if attempt_success:
                        success = True
                        break

                ended_at = now_str()

                detail_body = [
                    f"run_id: {run_id}",
                    f"started_at: {started_at}",
                    f"ended_at: {ended_at}",
                    f"parameter_name: {param_name}",
                    f"parameter_value: {json.dumps(value, ensure_ascii=False)}",
                    f"return_code: {return_code}",
                    f"final_stdout_file: {final_stdout_path.name if final_stdout_path else 'N/A'}",
                    f"max_retries_per_run: {max_retries_per_run}",
                    f"total_attempts_used: {len(attempts)}",
                    f"command: {' '.join(command)}",
                    "",
                    "[actual_run_config]",
                    current_run_config_text,
                    "",
                    "[attempts]",
                    json.dumps(attempts, ensure_ascii=False, indent=2),
                    "",
                    "[best_result_summary]",
                    best_summary,
                ]

                if not success:
                    detail_body.extend(
                        [
                            "",
                            "[status]",
                            "FAILED_AFTER_RETRIES",
                        ]
                    )

                append_detail_section(
                    detail_path,
                    f"Run {run_id}",
                    "\n".join(detail_body),
                )

                summary_body = [
                    f"run_id: {run_id}",
                    f"parameter_name: {param_name}",
                    f"parameter_value: {json.dumps(value, ensure_ascii=False)}",
                    f"return_code: {return_code}",
                    f"status: {'SUCCESS' if success else 'FAILED_AFTER_RETRIES'}",
                    f"max_retries_per_run: {max_retries_per_run}",
                    f"total_attempts_used: {len(attempts)}",
                    f"final_stdout_file: {final_stdout_path.name if final_stdout_path else 'N/A'}",
                    "",
                    "[actual_run_config]",
                    current_run_config_text,
                    "",
                    "[attempts]",
                    json.dumps(attempts, ensure_ascii=False, indent=2),
                    "",
                    "[best_result_summary]",
                    best_summary,
                ]
                append_summary_section(
                    summary_path,
                    f"Run {run_id}",
                    "\n".join(summary_body),
                )

    except Exception:
        append_detail_section(
            detail_path,
            "Unhandled Exception",
            traceback.format_exc(),
        )
        raise
    finally:
        write_text(config_path, original_config_text)
        append_detail_section(
            detail_path,
            "Sweep End",
            f"ended_at: {now_str()}\nconfig_restored: true",
        )
        append_summary_section(
            summary_path,
            "Sweep End",
            f"ended_at: {now_str()}\nconfig_restored: true",
        )


if __name__ == "__main__":
    main()
