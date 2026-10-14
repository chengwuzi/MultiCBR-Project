#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReCoBR noise-robustness benchmark orchestrator.

The script creates immutable noisy copies of only ``user_bundle_train.txt`` and
then executes CrossCBR, MultiCBR, and ReCoBR against the same noisy training
file. It is designed to be started from the repository root on the lab server:

    python noise_benchmark.py run --gpu 0

The default command prepares and sequentially runs 54 experiments:
3 models x 3 datasets x 6 noise ratios. State, logs, metrics, and generated
noise metadata are kept below ``noise_benchmark_artifacts/``.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from decimal import Decimal, InvalidOperation


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ("Youshu", "iFashion", "NetEase")
DEFAULT_MODELS = ("crosscbr", "multicbr", "recobr")
DEFAULT_NOISE_RATIOS = ("0", "0.05", "0.10", "0.15", "0.20", "0.25")
DEFAULT_NOISE_SEED = 20261014
DEFAULT_TRAIN_SEED = 20261014
DEFAULT_NOISE_MODE = "replacement"
NOISE_MODES = ("replacement", "injection")
MANIFEST_VERSION = 1
COMPATIBILITY_MANIFEST_VERSION = 1
SMOKE_DATASET = "Youshu"
SMOKE_CHECKPOINT_MARKER = "SMOKE_CHECKPOINT_SAVED:"


MODEL_SPECS = {
    "crosscbr": {
        "display_name": "CrossCBR",
        "workdir": "CrossCBR",
        "model_arg": "CrossCBR",
        "data_root": "CrossCBR/datasets",
    },
    "multicbr": {
        "display_name": "MultiCBR",
        "workdir": "MultiCBR",
        "model_arg": "MultiCBR",
        "data_root": "MultiCBR/datasets",
    },
    "recobr": {
        "display_name": "ReCoBR",
        "workdir": ".",
        "model_arg": "AnchorViewBundleNet",
        "data_root": "datasets",
    },
}


BEST_METRIC_PATTERN = re.compile(
    r"Best in epoch\s+(?P<epoch>\d+),\s+TOP\s+(?P<topk>\d+):\s+"
    r"REC_T=(?P<recall>[-+0-9.eE]+),\s+NDCG_T=(?P<ndcg>[-+0-9.eE]+)"
)


def utc_now():
    """Return an ISO-like timestamp for manifests and logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_path, str(path))
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path, default):
    path = Path(path)
    if not path.is_file():
        return default
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_noise_ratios(raw_ratios):
    ratios = []
    seen = set()
    for raw_ratio in raw_ratios:
        try:
            value = Decimal(str(raw_ratio))
        except InvalidOperation:
            raise ValueError("Invalid noise ratio: %s" % raw_ratio)
        if value < 0 or value >= 1:
            raise ValueError("Noise ratio must be in [0, 1): %s" % raw_ratio)
        if value in seen:
            raise ValueError("Duplicate noise ratio: %s" % raw_ratio)
        seen.add(value)
        ratios.append((str(raw_ratio), value))
    return ratios


def safe_ratio_tag(ratio_text):
    return str(ratio_text).replace("/", "_").replace("\\", "_").replace(" ", "")


def is_zero_noise_ratio(ratio_text):
    """Accept equivalent CLI spellings such as 0, 0.0, and 0.00."""
    try:
        return Decimal(str(ratio_text)) == Decimal("0")
    except InvalidOperation:
        return False


def validate_noise_mode(noise_mode):
    if noise_mode not in NOISE_MODES:
        raise ValueError("Unsupported noise mode: %s" % noise_mode)
    return noise_mode


def default_artifact_root(noise_mode):
    """Keep independent noise mechanisms in independent resumable workspaces."""
    validate_noise_mode(noise_mode)
    if noise_mode == "injection":
        return PROJECT_ROOT / "noise_benchmark_artifacts_injection"
    return PROJECT_ROOT / "noise_benchmark_artifacts"


def resolved_artifact_root(args):
    explicit_root = getattr(args, "artifact_root", None)
    if explicit_root:
        return Path(explicit_root).resolve()
    noise_mode = getattr(args, "noise_mode", DEFAULT_NOISE_MODE)
    return default_artifact_root(noise_mode).resolve()


def mode_specific_noise_data_root(artifact_root, noise_mode):
    """Avoid data-file collisions even when a user explicitly shares a root."""
    validate_noise_mode(noise_mode)
    root = Path(artifact_root) / "noisy_data"
    return root if noise_mode == DEFAULT_NOISE_MODE else root / noise_mode


def derive_seed(master_seed, dataset_name, ratio_text, noise_mode=DEFAULT_NOISE_MODE):
    """Derive deterministic data noise, preserving legacy replacement seeds."""
    validate_noise_mode(noise_mode)
    material = "%s|%s|%s" % (master_seed, dataset_name, ratio_text)
    if noise_mode != DEFAULT_NOISE_MODE:
        material += "|%s" % noise_mode
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def read_train_pairs(train_path):
    """Load pair lines while preserving the original line order."""
    pairs = []
    user_positions = {}
    user_bundles = {}
    all_train_bundles = set()

    with open(str(train_path), "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError(
                    "Expected two tab-separated IDs in %s at line %d" % (train_path, line_number)
                )
            try:
                user_id = int(fields[0])
                bundle_id = int(fields[1])
            except ValueError:
                raise ValueError(
                    "Non-integer ID in %s at line %d" % (train_path, line_number)
                )

            position = len(pairs)
            pairs.append([user_id, bundle_id])
            user_positions.setdefault(user_id, []).append(position)
            user_bundles.setdefault(user_id, set())
            if bundle_id in user_bundles[user_id]:
                raise ValueError(
                    "Duplicate original training UB edge (%d, %d) in %s" %
                    (user_id, bundle_id, train_path)
                )
            user_bundles[user_id].add(bundle_id)
            all_train_bundles.add(bundle_id)

    if not pairs:
        raise ValueError("Training UB file is empty: %s" % train_path)
    return pairs, user_positions, user_bundles, sorted(all_train_bundles)


def sample_unobserved_bundles(rng, all_train_bundles, observed_bundles, sample_size):
    """Uniformly sample from B_train minus a user's original train bundles."""
    available = len(all_train_bundles) - len(observed_bundles)
    if sample_size > available:
        raise ValueError(
            "Cannot draw %d replacement bundles from only %d candidates" % (sample_size, available)
        )

    selected = set()
    # With the project datasets, user degree is far below the global training
    # bundle count. Rejection sampling avoids materializing a large complement
    # for every user while retaining uniform sampling without replacement.
    maximum_attempts = max(100, sample_size * 100)
    attempts = 0
    while len(selected) < sample_size and attempts < maximum_attempts:
        candidate = rng.choice(all_train_bundles)
        attempts += 1
        if candidate not in observed_bundles and candidate not in selected:
            selected.add(candidate)

    if len(selected) < sample_size:
        remaining = [
            bundle_id for bundle_id in all_train_bundles
            if bundle_id not in observed_bundles and bundle_id not in selected
        ]
        selected.update(rng.sample(remaining, sample_size - len(selected)))

    return list(selected)


def noise_metadata_matches(
    metadata, source_hash, dataset_name, ratio_text, noise_seed, noise_mode=DEFAULT_NOISE_MODE
):
    validate_noise_mode(noise_mode)
    return (
        metadata.get("manifest_version") == MANIFEST_VERSION
        and metadata.get("dataset") == dataset_name
        and metadata.get("requested_noise_ratio") == ratio_text
        and metadata.get("master_noise_seed") == noise_seed
        and metadata.get("source_train_sha256") == source_hash
        # Older replacement metadata deliberately remains reusable.
        and metadata.get("noise_mode", DEFAULT_NOISE_MODE) == noise_mode
        and metadata.get("candidate_definition") == "B_train_minus_T_u"
        and metadata.get("rounding_rule") == "floor(x_times_n_u)"
    )


def normalize_noise_metadata(metadata):
    """Expose new bookkeeping fields when reusing older replacement metadata."""
    normalized = dict(metadata)
    noise_mode = normalized.setdefault("noise_mode", DEFAULT_NOISE_MODE)
    validate_noise_mode(noise_mode)
    original_edges = int(normalized.get("original_train_edge_count", normalized["total_train_edges"]))
    replacement_count = int(normalized.get("replacement_count", 0))
    injected_count = int(normalized.get("injected_count", 0))
    noise_edge_count = int(normalized.get("noise_edge_count", replacement_count + injected_count))
    output_edges = int(normalized.get(
        "output_train_edge_count",
        original_edges + injected_count,
    ))
    normalized["original_train_edge_count"] = original_edges
    normalized["output_train_edge_count"] = output_edges
    normalized["replacement_count"] = replacement_count
    normalized["injected_count"] = injected_count
    normalized["noise_edge_count"] = noise_edge_count
    normalized.setdefault("actual_noise_ratio", float(noise_edge_count) / original_edges)
    normalized.setdefault("noise_fraction_in_output", float(noise_edge_count) / output_edges)
    return normalized


def prepare_noisy_training_file(
    source_data_root,
    output_root,
    dataset_name,
    ratio_text,
    ratio_value,
    noise_seed,
    noise_mode=DEFAULT_NOISE_MODE,
    force=False,
):
    """Create or reuse one immutable noisy training UB file for one noise mode."""
    validate_noise_mode(noise_mode)
    source_path = Path(source_data_root) / dataset_name / "user_bundle_train.txt"
    if not source_path.is_file():
        raise FileNotFoundError("Missing source training UB file: %s" % source_path)

    ratio_dir = Path(output_root) / dataset_name / ("noise_" + safe_ratio_tag(ratio_text))
    output_path = ratio_dir / "user_bundle_train.txt"
    metadata_path = ratio_dir / "metadata.json"
    source_hash = sha256_file(source_path)

    if not force and output_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path, {})
        if noise_metadata_matches(
            metadata, source_hash, dataset_name, ratio_text, noise_seed, noise_mode
        ):
            return output_path, normalize_noise_metadata(metadata), False

    ratio_dir.mkdir(parents=True, exist_ok=True)
    pairs, user_positions, user_bundles, all_train_bundles = read_train_pairs(source_path)
    original_total_edges = len(pairs)
    replacement_count = 0
    injected_count = 0
    affected_user_count = 0

    if ratio_value == Decimal("0"):
        temp_output_path = ratio_dir / (output_path.name + ".tmp")
        shutil.copyfile(str(source_path), str(temp_output_path))
        os.replace(str(temp_output_path), str(output_path))
    else:
        rng = random.Random(derive_seed(noise_seed, dataset_name, ratio_text, noise_mode))
        injected_pairs = []

        for user_id in sorted(user_positions):
            positions = user_positions[user_id]
            noise_count = int(
                (ratio_value * Decimal(len(positions))).to_integral_value(rounding="ROUND_FLOOR")
            )
            if noise_count == 0:
                continue

            observed_bundles = user_bundles[user_id]
            if noise_mode == "replacement":
                replace_positions = rng.sample(positions, noise_count)
                noisy_bundles = sample_unobserved_bundles(
                    rng,
                    all_train_bundles,
                    observed_bundles,
                    noise_count,
                )
                for position, replacement_bundle in zip(replace_positions, noisy_bundles):
                    pairs[position][1] = replacement_bundle
                replacement_count += noise_count
            else:
                noisy_bundles = sample_unobserved_bundles(
                    rng,
                    all_train_bundles,
                    observed_bundles,
                    noise_count,
                )
                # Preserve every source interaction and append only false ones.
                # Sorting makes the generated file's line order stable as well.
                injected_pairs.extend((user_id, bundle_id) for bundle_id in sorted(noisy_bundles))
                injected_count += noise_count
            affected_user_count += 1

        if noise_mode == "injection":
            pairs.extend(injected_pairs)
        text = "".join("%d\t%d\n" % (user_id, bundle_id) for user_id, bundle_id in pairs)
        atomic_write_text(output_path, text)

    output_total_edges = len(pairs)
    user_noise_counts = []
    for user_id in user_positions:
        original_degree = len(user_positions[user_id])
        count = int((ratio_value * Decimal(original_degree)).to_integral_value(rounding="ROUND_FLOOR"))
        user_noise_counts.append(count)
    noise_edge_count = replacement_count + injected_count
    mode_detail = {
        "replacement": (
            "For each affected user, replace selected original training interactions with "
            "equally many false bundles."
        ),
        "injection": (
            "For each affected user, retain every original training interaction and append "
            "false bundle interactions."
        ),
    }[noise_mode]

    metadata = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "dataset": dataset_name,
        "requested_noise_ratio": ratio_text,
        "noise_mode": noise_mode,
        "noise_mode_detail": mode_detail,
        "master_noise_seed": noise_seed,
        "derived_noise_seed": derive_seed(noise_seed, dataset_name, ratio_text, noise_mode),
        "source_train_path": str(source_path.resolve()),
        "source_train_sha256": source_hash,
        "output_train_path": str(output_path.resolve()),
        "candidate_definition": "B_train_minus_T_u",
        "candidate_definition_detail": (
            "For user u, draw false bundles from the original global training bundle set "
            "B_train excluding u's original training set T_u."
        ),
        "rounding_rule": "floor(x_times_n_u)",
        "rounding_rule_detail": "m_u = floor(noise_ratio * original_user_train_degree)",
        # Keep the legacy field's original-denominator meaning while making both
        # edge totals explicit for injection experiments.
        "total_train_edges": original_total_edges,
        "original_train_edge_count": original_total_edges,
        "output_train_edge_count": output_total_edges,
        "active_train_users": len(user_positions),
        "global_train_bundle_count": len(all_train_bundles),
        "replacement_count": replacement_count,
        "injected_count": injected_count,
        "noise_edge_count": noise_edge_count,
        "affected_user_count": affected_user_count,
        "users_with_zero_replacements": sum(1 for count in user_noise_counts if count == 0),
        "users_with_zero_noise_edges": sum(1 for count in user_noise_counts if count == 0),
        # In both modes this is the realized false-edge count divided by the
        # clean input edge count, i.e. the requested ratio's denominator.
        "actual_noise_ratio": float(noise_edge_count) / original_total_edges,
        "noise_fraction_in_output": float(noise_edge_count) / output_total_edges,
        "output_train_sha256": sha256_file(output_path),
    }
    atomic_write_json(metadata_path, metadata)
    return output_path, metadata, True


def required_dataset_files(dataset_root, dataset_name):
    return [
        Path(dataset_root) / dataset_name / (dataset_name + "_data_size.txt"),
        Path(dataset_root) / dataset_name / "bundle_item.txt",
        Path(dataset_root) / dataset_name / "user_item.txt",
        Path(dataset_root) / dataset_name / "user_bundle_tune.txt",
        Path(dataset_root) / dataset_name / "user_bundle_test.txt",
    ]


def validate_project_layout(models, datasets, source_data_root):
    missing = []
    for dataset_name in datasets:
        source_train_path = Path(source_data_root) / dataset_name / "user_bundle_train.txt"
        if not source_train_path.is_file():
            missing.append(str(source_train_path))

    for model_key in models:
        spec = MODEL_SPECS[model_key]
        workdir = PROJECT_ROOT / spec["workdir"]
        if not (workdir / "train.py").is_file():
            missing.append(str(workdir / "train.py"))
        if not (workdir / "config.yaml").is_file():
            missing.append(str(workdir / "config.yaml"))
        data_root = PROJECT_ROOT / spec["data_root"]
        for dataset_name in datasets:
            missing.extend(str(path) for path in required_dataset_files(data_root, dataset_name) if not path.is_file())

    if missing:
        raise FileNotFoundError("Benchmark preflight failed; missing:\n- " + "\n- ".join(missing))


def experiment_id(
    model_key,
    dataset_name,
    ratio_text,
    noise_seed,
    train_seed,
    noise_mode=DEFAULT_NOISE_MODE,
):
    validate_noise_mode(noise_mode)
    identifier = "%s__%s__noise_%s__nseed_%s__tseed_%s" % (
        model_key,
        dataset_name,
        safe_ratio_tag(ratio_text),
        noise_seed,
        train_seed,
    )
    # Preserve legacy replacement IDs so an interrupted old benchmark resumes
    # exactly as before, even if a caller deliberately shares an artifact root.
    if noise_mode != DEFAULT_NOISE_MODE:
        identifier += "__mode_%s" % noise_mode
    return identifier


def load_manifest(manifest_path):
    default = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "experiments": {},
    }
    payload = read_json(manifest_path, default)
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported benchmark manifest version: %s" % payload.get("manifest_version"))
    payload.setdefault("experiments", {})
    return payload


def save_manifest(manifest_path, manifest):
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def build_command(
    model_key,
    dataset_name,
    ratio_text,
    train_ub_path,
    gpu,
    noise_seed,
    train_seed,
    python_executable,
    noise_mode=DEFAULT_NOISE_MODE,
):
    validate_noise_mode(noise_mode)
    spec = MODEL_SPECS[model_key]
    data_root = (PROJECT_ROOT / spec["data_root"]).resolve()
    info = "noise_%s_nseed_%s_tseed_%s" % (
        safe_ratio_tag(ratio_text),
        noise_seed,
        train_seed,
    )
    if noise_mode != DEFAULT_NOISE_MODE:
        info += "_mode_%s" % noise_mode
    return [
        python_executable,
        "train.py",
        "-g", str(gpu),
        "-d", dataset_name,
        "-m", spec["model_arg"],
        "-i", info,
        "--data_path", str(data_root),
        "--train_ub_path", str(Path(train_ub_path).resolve()),
        "--seed", str(train_seed),
    ]


def parse_best_metrics(log_path):
    metrics_by_topk = {}
    log_path = Path(log_path)
    if not log_path.is_file():
        return metrics_by_topk
    with open(str(log_path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = BEST_METRIC_PATTERN.search(line)
            if not match:
                continue
            topk = int(match.group("topk"))
            metrics_by_topk[topk] = {
                "best_epoch": int(match.group("epoch")),
                "recall": float(match.group("recall")),
                "ndcg": float(match.group("ndcg")),
            }
    return metrics_by_topk


def execute_experiment(experiment, attempt, dry_run, on_process_started=None):
    """Run one attempt and always return a recordable outcome.

    A process-launch error is treated exactly like a non-zero process exit so
    the caller can use the same retry policy for both cases.
    """
    attempt_dir = Path(attempt["attempt_dir"])
    attempt_dir.mkdir(parents=True, exist_ok=True)
    command_path = Path(attempt["command_path"])
    atomic_write_json(command_path, {
        "experiment_id": experiment["id"],
        "attempt": attempt["number"],
        "cwd": experiment["cwd"],
        "command": experiment["command"],
        "created_at": utc_now(),
    })

    if dry_run:
        print("[DRY RUN] %s (attempt %d)" % (experiment["id"], attempt["number"]))
        print("  cwd: %s" % experiment["cwd"])
        print("  command: %s" % " ".join(experiment["command"]))
        return 0, {}, ""

    stdout_path = Path(attempt["stdout_log"])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("\n[%s] START %s attempt=%d" % (utc_now(), experiment["id"], attempt["number"]))
    print("[%s] COMMAND %s" % (experiment["id"], " ".join(experiment["command"])))

    try:
        with open(str(stdout_path), "w", encoding="utf-8", newline="\n") as log_handle:
            try:
                process = subprocess.Popen(
                    experiment["command"],
                    cwd=experiment["cwd"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            except OSError as error:
                message = "Process launch failed: %s" % error
                log_handle.write(message + "\n")
                print("[%s] %s" % (experiment["id"], message))
                return 127, {}, message

            attempt["child_pid"] = process.pid
            attempt["child_started_at"] = utc_now()
            if on_process_started is not None:
                try:
                    on_process_started()
                except Exception as error:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    message = "Could not persist child PID; cancelled process: %s" % error
                    log_handle.write(message + "\n")
                    return 125, {}, message

            try:
                for line in iter(process.stdout.readline, ""):
                    log_handle.write(line)
                    log_handle.flush()
                    sys.stdout.write("[%s] %s" % (experiment["id"], line))
                    sys.stdout.flush()
                process.stdout.close()
                return_code = process.wait()
            except Exception as error:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                message = "Process I/O failed: %s" % error
                log_handle.write(message + "\n")
                return 70, {}, message
    except OSError as error:
        return 126, {}, "Unable to write attempt log: %s" % error

    metrics = parse_best_metrics(stdout_path)
    failure_reason = "" if return_code == 0 else "Training process exited with return code %d" % return_code
    print("[%s] END attempt=%d return_code=%d" % (experiment["id"], attempt["number"], return_code))
    return return_code, metrics, failure_reason


def normalized_attempt_history(experiment):
    """Keep older manifests usable after the retry-history schema upgrade."""
    attempts = list(experiment.get("attempts", []))
    legacy_count = int(experiment.get("attempt", 0))
    while len(attempts) < legacy_count:
        number = len(attempts) + 1
        attempts.append({
            "number": number,
            "status": "legacy_or_interrupted",
            "failure_reason": "Recorded by a pre-retry-history version of the controller.",
        })
    return attempts


def is_process_running(pid):
    """Best-effort same-host PID liveness check for interrupted controllers."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            return str(pid) in result.stdout
        os.kill(pid, 0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def active_running_attempt(attempts):
    """Return a still-live interrupted attempt, if one can be identified."""
    for attempt in reversed(attempts):
        if attempt.get("status") == "running" and is_process_running(attempt.get("child_pid")):
            return attempt
    return None


def summary_rows(manifest):
    """Build one compact CSV row per experiment, with metric columns per Top-K."""
    experiments = manifest.get("experiments", {})
    topks = sorted({
        int(topk_text)
        for experiment in experiments.values()
        for topk_text in experiment.get("metrics", {})
    })
    baseline_lookup = {}
    for experiment in experiments.values():
        if experiment.get("status") != "succeeded" or not is_zero_noise_ratio(
            experiment.get("noise_ratio", "")
        ):
            continue
        noise_mode = experiment.get("noise_mode", DEFAULT_NOISE_MODE)
        for topk_text, metric in experiment.get("metrics", {}).items():
            baseline_lookup[(noise_mode, experiment["model"], experiment["dataset"], int(topk_text))] = metric

    rows = []
    for experiment in experiments.values():
        attempts = normalized_attempt_history(experiment)
        row = {
            "experiment_id": experiment["id"],
            "model": experiment["model"],
            "dataset": experiment["dataset"],
            "noise_mode": experiment.get("noise_mode", DEFAULT_NOISE_MODE),
            "noise_ratio": experiment["noise_ratio"],
            "actual_noise_ratio": experiment.get("actual_noise_ratio", ""),
            "noise_fraction_in_output": experiment.get("noise_fraction_in_output", ""),
            "status": experiment.get("status", "unknown"),
            "attempts_used": len(attempts),
            "max_attempts": experiment.get("max_attempts", ""),
            "failed_attempts": sum(1 for item in attempts if item.get("status") == "failed"),
            "latest_return_code": experiment.get("return_code", ""),
            "latest_failure_reason": experiment.get("failure_reason", ""),
            "replacement_count": experiment.get("replacement_count", ""),
            "injected_count": experiment.get("injected_count", ""),
            "noise_edge_count": experiment.get("noise_edge_count", ""),
            "noise_seed": experiment.get("noise_seed", ""),
            "train_seed": experiment.get("train_seed", ""),
            "stdout_log": experiment.get("stdout_log", ""),
        }
        metrics = experiment.get("metrics", {})
        for topk in topks:
            metric = metrics.get(str(topk))
            baseline = baseline_lookup.get((
                experiment.get("noise_mode", DEFAULT_NOISE_MODE),
                experiment["model"],
                experiment["dataset"],
                topk,
            ))
            row["best_epoch@%d" % topk] = metric["best_epoch"] if metric else ""
            row["recall@%d" % topk] = metric["recall"] if metric else ""
            row["ndcg@%d" % topk] = metric["ndcg"] if metric else ""
            if metric and baseline and baseline["recall"] != 0:
                row["recall_change_vs_zero@%d" % topk] = (
                    (metric["recall"] - baseline["recall"]) / baseline["recall"]
                )
            else:
                row["recall_change_vs_zero@%d" % topk] = ""
            if metric and baseline and baseline["ndcg"] != 0:
                row["ndcg_change_vs_zero@%d" % topk] = (
                    (metric["ndcg"] - baseline["ndcg"]) / baseline["ndcg"]
                )
            else:
                row["ndcg_change_vs_zero@%d" % topk] = ""
        rows.append(row)
    return rows, topks


def write_summary(artifact_root, manifest):
    artifact_root = Path(artifact_root)
    rows, topks = summary_rows(manifest)
    csv_path = artifact_root / "results_summary.csv"
    # The manifest retains retry state; results themselves intentionally have
    # one compact, spreadsheet-friendly home.
    stale_json_path = artifact_root / "results_summary.json"
    if stale_json_path.is_file():
        stale_json_path.unlink()

    fieldnames = [
        "experiment_id", "model", "dataset", "noise_mode", "noise_ratio", "actual_noise_ratio",
        "noise_fraction_in_output", "status",
        "attempts_used", "max_attempts", "failed_attempts", "latest_return_code",
        "latest_failure_reason", "replacement_count", "injected_count", "noise_edge_count",
        "noise_seed", "train_seed", "stdout_log",
    ]
    for topk in topks:
        fieldnames.extend([
            "best_epoch@%d" % topk,
            "recall@%d" % topk,
            "ndcg@%d" % topk,
            "recall_change_vs_zero@%d" % topk,
            "ndcg_change_vs_zero@%d" % topk,
        ])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=csv_path.name + ".", suffix=".tmp", dir=str(csv_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, str(csv_path))
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def prepare_selected_data(args, datasets, ratios):
    source_data_root = Path(args.source_data_root).resolve()
    noise_data_root = mode_specific_noise_data_root(resolved_artifact_root(args), args.noise_mode)
    prepared = {}
    for dataset_name in datasets:
        for ratio_text, ratio_value in ratios:
            output_path, metadata, created = prepare_noisy_training_file(
                source_data_root,
                noise_data_root,
                dataset_name,
                ratio_text,
                ratio_value,
                args.noise_seed,
                noise_mode=args.noise_mode,
                force=args.force_prepare,
            )
            prepared[(dataset_name, ratio_text)] = (output_path, metadata)
            action = "created" if created else "reused"
            print(
                "[%s] %s %s-noise UB data: requested=%s actual=%.8f false_edges=%d path=%s" % (
                    dataset_name,
                    action,
                    args.noise_mode,
                    ratio_text,
                    metadata["actual_noise_ratio"],
                    metadata["noise_edge_count"],
                    output_path,
                )
            )
    return prepared


def build_smoke_command(model_key, train_ub_path, args):
    """Build a real, deliberately short Youshu compatibility command."""
    command = build_command(
        model_key,
        SMOKE_DATASET,
        "compatibility_smoke",
        train_ub_path,
        args.gpu,
        args.noise_seed,
        args.train_seed,
        args.python_executable,
        args.noise_mode,
    )
    command.append("--smoke-test")
    return command


def model_code_fingerprint(models):
    """Hash model source/config files so cached smoke passes cannot outlive edits."""
    digest = hashlib.sha256()
    included_files = []
    ignored_directories = {
        ".git", ".idea", ".pytest_cache", "__pycache__", "datasets",
        "noise_benchmark_artifacts", "noise_benchmark_artifacts_injection",
        "checkpoints", "runs", "log",
    }
    for model_key in sorted(models):
        workdir = (PROJECT_ROOT / MODEL_SPECS[model_key]["workdir"]).resolve()
        for directory, subdirectories, filenames in os.walk(str(workdir)):
            subdirectories[:] = sorted(
                name for name in subdirectories if name not in ignored_directories
            )
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix.lower() not in (".py", ".yaml", ".yml"):
                    continue
                # The ReCoBR root contains baseline subprojects too. Including them
                # is conservative: any relevant project code/config edit revalidates.
                relative_path = path.relative_to(PROJECT_ROOT).as_posix()
                digest.update(relative_path.encode("utf-8"))
                digest.update(b"\0")
                with open(str(path), "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                included_files.append(relative_path)
    return digest.hexdigest(), included_files


def compatibility_signature(args, smoke_train_path, models):
    code_hash, code_files = model_code_fingerprint(models)
    return {
        "dataset": SMOKE_DATASET,
        "train_ub_sha256": sha256_file(smoke_train_path),
        "noise_mode": args.noise_mode,
        "noise_seed": args.noise_seed,
        "train_seed": args.train_seed,
        "gpu": str(args.gpu),
        "python_executable": str(args.python_executable),
        "models": list(sorted(models)),
        "model_code_sha256": code_hash,
        "model_code_files": code_files,
    }


def load_compatibility_manifest(manifest_path, signature):
    default = {
        "manifest_version": COMPATIBILITY_MANIFEST_VERSION,
        "created_at": utc_now(),
        "signature": signature,
        "checks": {},
    }
    manifest = read_json(manifest_path, default)
    if (
        manifest.get("manifest_version") != COMPATIBILITY_MANIFEST_VERSION
        or manifest.get("signature") != signature
    ):
        return default
    manifest.setdefault("checks", {})
    return manifest


def save_compatibility_manifest(manifest_path, manifest):
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)


def saved_smoke_checkpoint(stdout_path, cwd):
    """Return the checkpoint path only when the model confirmed a real save."""
    stdout_path = Path(stdout_path)
    if not stdout_path.is_file():
        return None
    with open(str(stdout_path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(SMOKE_CHECKPOINT_MARKER):
                continue
            raw_path = line.split(":", 1)[1].strip()
            checkpoint_path = Path(raw_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(cwd) / checkpoint_path
            checkpoint_path = checkpoint_path.resolve()
            if checkpoint_path.is_file():
                return str(checkpoint_path)
    return None


def smoke_output_has_metrics(stdout_path):
    stdout_path = Path(stdout_path)
    if not stdout_path.is_file():
        return False
    with open(str(stdout_path), "r", encoding="utf-8", errors="replace") as handle:
        return any(BEST_METRIC_PATTERN.search(line) for line in handle)


def run_compatibility_smoke(args, models):
    """Gate formal runs on an actual Youshu train/evaluate/save cycle per model."""
    models = tuple(models)
    source_data_root = Path(args.source_data_root).resolve()
    validate_project_layout(models, (SMOKE_DATASET,), source_data_root)

    artifact_root = resolved_artifact_root(args)
    smoke_root = artifact_root / "compatibility_smoke"
    smoke_data_root = mode_specific_noise_data_root(smoke_root, args.noise_mode)
    smoke_train_path, smoke_metadata, created = prepare_noisy_training_file(
        source_data_root,
        smoke_data_root,
        SMOKE_DATASET,
        "0",
        Decimal("0"),
        args.noise_seed,
        noise_mode=args.noise_mode,
        force=args.force_prepare,
    )
    print(
        "[COMPATIBILITY] %s Youshu smoke input: %s" % (
            "created" if created else "reused",
            smoke_train_path,
        )
    )

    signature = compatibility_signature(args, smoke_train_path, models)
    manifest_path = smoke_root / "compatibility_manifest.json"
    manifest = load_compatibility_manifest(manifest_path, signature)
    failures = []

    for model_key in models:
        spec = MODEL_SPECS[model_key]
        prior = manifest["checks"].get(model_key, {})
        if prior.get("status") == "succeeded" and not args.rerun_compatibility_check:
            print("[COMPATIBILITY SKIP] %s already passed" % spec["display_name"])
            continue
        if prior.get("status") == "running" and is_process_running(prior.get("child_pid")):
            raise RuntimeError(
                "Compatibility smoke test for %s is still running (PID %s). "
                "Wait for it to finish before resuming." % (spec["display_name"], prior.get("child_pid"))
            )

        attempt_number = int(prior.get("attempt", 0)) + 1
        smoke_runs_root = smoke_root / "runs"
        if args.noise_mode != DEFAULT_NOISE_MODE:
            smoke_runs_root = smoke_runs_root / args.noise_mode
        attempt_dir = smoke_runs_root / model_key / ("attempt_%02d" % attempt_number)
        command = build_smoke_command(model_key, smoke_train_path, args)
        check = {
            "model": spec["display_name"],
            "model_key": model_key,
            "dataset": SMOKE_DATASET,
            "status": "running",
            "attempt": attempt_number,
            "started_at": utc_now(),
            "cwd": str((PROJECT_ROOT / spec["workdir"]).resolve()),
            "command": command,
            "smoke_train_ub_path": str(smoke_train_path.resolve()),
            "smoke_metadata_path": str((smoke_train_path.parent / "metadata.json").resolve()),
            "attempt_dir": str(attempt_dir.resolve()),
            "command_path": str((attempt_dir / "command.json").resolve()),
            "stdout_log": str((attempt_dir / "stdout.log").resolve()),
        }
        manifest["checks"][model_key] = check
        save_compatibility_manifest(manifest_path, manifest)

        attempt = {
            "number": attempt_number,
            "attempt_dir": check["attempt_dir"],
            "command_path": check["command_path"],
            "stdout_log": check["stdout_log"],
        }
        smoke_id = "compatibility__%s__%s" % (model_key, SMOKE_DATASET)
        if args.noise_mode != DEFAULT_NOISE_MODE:
            smoke_id += "__mode_%s" % args.noise_mode
        experiment = {
            "id": smoke_id,
            "cwd": check["cwd"],
            "command": command,
        }

        def persist_smoke_pid():
            check["child_pid"] = attempt["child_pid"]
            check["child_started_at"] = attempt["child_started_at"]
            manifest["checks"][model_key] = check
            save_compatibility_manifest(manifest_path, manifest)

        return_code, metrics, failure_reason = execute_experiment(
            experiment,
            attempt,
            False,
            on_process_started=persist_smoke_pid,
        )
        checkpoint_path = saved_smoke_checkpoint(check["stdout_log"], check["cwd"])
        metrics_seen = smoke_output_has_metrics(check["stdout_log"])
        check["finished_at"] = utc_now()
        check["return_code"] = return_code
        check["metrics"] = {str(topk): values for topk, values in metrics.items()}
        check["checkpoint_path"] = checkpoint_path or ""
        check["metrics_seen"] = metrics_seen
        if return_code == 0 and checkpoint_path and metrics_seen:
            check["status"] = "succeeded"
            check["failure_reason"] = ""
            print("[COMPATIBILITY PASS] %s" % spec["display_name"])
        else:
            check["status"] = "failed"
            checks_missing = []
            if return_code != 0:
                checks_missing.append(failure_reason or "non-zero return code")
            if not metrics_seen:
                checks_missing.append("evaluation metric marker was not found")
            if not checkpoint_path:
                checks_missing.append("checkpoint save marker/file was not found")
            check["failure_reason"] = "; ".join(checks_missing)
            failures.append(spec["display_name"])
            print("[COMPATIBILITY FAIL] %s: %s" % (spec["display_name"], check["failure_reason"]))
        manifest["checks"][model_key] = check
        save_compatibility_manifest(manifest_path, manifest)

    # A cached result still needs to be considered when another model was run.
    failures.extend(
        MODEL_SPECS[model_key]["display_name"]
        for model_key in models
        if manifest["checks"].get(model_key, {}).get("status") != "succeeded"
        and MODEL_SPECS[model_key]["display_name"] not in failures
    )
    if failures:
        raise RuntimeError(
            "Compatibility smoke test failed for: %s. Formal benchmark was not started. "
            "Inspect %s" % (", ".join(failures), manifest_path)
        )
    print("[COMPATIBILITY PASS] All selected models completed Youshu load/train/evaluate/save.")
    return manifest_path


def compatibility_only(args):
    run_compatibility_smoke(args, tuple(args.models))


def run_benchmark(args):
    models = tuple(args.models)
    datasets = tuple(args.datasets)
    ratios = parse_noise_ratios(args.noise_ratios)
    artifact_root = resolved_artifact_root(args)
    source_data_root = Path(args.source_data_root).resolve()
    validate_project_layout(models, datasets, source_data_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN] Compatibility smoke test is not launched in dry-run mode.")
    elif args.skip_compatibility_check:
        print("[WARNING] Compatibility smoke test was explicitly skipped.")
    else:
        run_compatibility_smoke(args, models)

    prepared = prepare_selected_data(args, datasets, ratios)
    manifest_path = artifact_root / "benchmark_manifest.json"
    manifest = load_manifest(manifest_path)

    # Dataset is deliberately the outermost loop: complete every model/noise
    # task for Youshu before iFashion, then complete iFashion before NetEase.
    for dataset_name in datasets:
        for model_key in models:
            spec = MODEL_SPECS[model_key]
            cwd = (PROJECT_ROOT / spec["workdir"]).resolve()
            for ratio_text, _ in ratios:
                train_ub_path, noise_metadata = prepared[(dataset_name, ratio_text)]
                task_id = experiment_id(
                    model_key,
                    dataset_name,
                    ratio_text,
                    args.noise_seed,
                    args.train_seed,
                    args.noise_mode,
                )
                prior = manifest["experiments"].get(task_id, {})
                if prior.get("status") == "succeeded" and not args.rerun_succeeded:
                    print("[SKIP] %s already succeeded" % task_id)
                    continue
                if prior.get("status") == "exhausted" and not args.rerun_exhausted:
                    print("[SKIP] %s exhausted its retry budget; recorded as failed" % task_id)
                    continue

                run_dir = artifact_root / "runs" / task_id
                command = build_command(
                    model_key,
                    dataset_name,
                    ratio_text,
                    train_ub_path,
                    args.gpu,
                    args.noise_seed,
                    args.train_seed,
                    args.python_executable,
                    args.noise_mode,
                )
                attempts = normalized_attempt_history(prior)
                is_manual_new_cycle = (
                    (prior.get("status") == "succeeded" and args.rerun_succeeded) or
                    (prior.get("status") == "exhausted" and args.rerun_exhausted)
                )
                max_attempts = args.max_retries + 1
                if is_manual_new_cycle:
                    # Keep earlier logs/history while granting an intentional
                    # manually requested re-run its own retry budget.
                    max_attempts += len(attempts)

                experiment = {
                    "id": task_id,
                    "model": spec["display_name"],
                    "model_key": model_key,
                    "dataset": dataset_name,
                    "noise_mode": args.noise_mode,
                    "noise_ratio": ratio_text,
                    "noise_seed": args.noise_seed,
                    "train_seed": args.train_seed,
                    "actual_noise_ratio": noise_metadata["actual_noise_ratio"],
                    "noise_fraction_in_output": noise_metadata["noise_fraction_in_output"],
                    "replacement_count": noise_metadata["replacement_count"],
                    "injected_count": noise_metadata["injected_count"],
                    "noise_edge_count": noise_metadata["noise_edge_count"],
                    "train_ub_path": str(train_ub_path.resolve()),
                    "noise_metadata_path": str((train_ub_path.parent / "metadata.json").resolve()),
                    "run_dir": str(run_dir.resolve()),
                    "cwd": str(cwd),
                    "command": command,
                    "status": "planned" if args.dry_run else prior.get("status", "pending"),
                    "attempt": len(attempts),
                    "attempts": attempts,
                    "max_attempts": max_attempts,
                }

                live_attempt = active_running_attempt(attempts)
                if live_attempt is not None:
                    experiment["status"] = "running"
                    experiment["stdout_log"] = live_attempt.get("stdout_log", "")
                    experiment["failure_reason"] = ""
                    manifest["experiments"][task_id] = experiment
                    save_manifest(manifest_path, manifest)
                    write_summary(artifact_root, manifest)
                    print(
                        "[WAIT] %s attempt %d (PID %s) is still running; "
                        "stopping controller to avoid a duplicate GPU job." % (
                            task_id,
                            live_attempt.get("number", 0),
                            live_attempt.get("child_pid"),
                        )
                    )
                    return

                if args.dry_run:
                    attempt_number = len(attempts) + 1
                    attempt_dir = run_dir / ("attempt_%02d" % attempt_number)
                    dry_attempt = {
                        "number": attempt_number,
                        "attempt_dir": str(attempt_dir.resolve()),
                        "command_path": str((attempt_dir / "command.json").resolve()),
                        "stdout_log": str((attempt_dir / "stdout.log").resolve()),
                    }
                    manifest["experiments"][task_id] = experiment
                    save_manifest(manifest_path, manifest)
                    execute_experiment(experiment, dry_attempt, True)
                    continue

                if len(attempts) >= max_attempts:
                    experiment["status"] = "exhausted"
                    experiment["failure_reason"] = "Retry budget was already exhausted before this resume."
                    manifest["experiments"][task_id] = experiment
                    save_manifest(manifest_path, manifest)
                    write_summary(artifact_root, manifest)
                    print("[SKIP] %s has no retry attempts remaining" % task_id)
                    continue

                while len(attempts) < max_attempts:
                    attempt_number = len(attempts) + 1
                    attempt_dir = run_dir / ("attempt_%02d" % attempt_number)
                    attempt = {
                        "number": attempt_number,
                        "status": "running",
                        "started_at": utc_now(),
                        "attempt_dir": str(attempt_dir.resolve()),
                        "command_path": str((attempt_dir / "command.json").resolve()),
                        "stdout_log": str((attempt_dir / "stdout.log").resolve()),
                    }
                    attempts.append(attempt)
                    experiment["attempt"] = len(attempts)
                    experiment["attempts"] = attempts
                    experiment["max_attempts"] = max_attempts
                    experiment["status"] = "running"
                    experiment["started_at"] = attempt["started_at"]
                    experiment["stdout_log"] = attempt["stdout_log"]
                    manifest["experiments"][task_id] = experiment
                    # Persist 'running' before launching. If the controller is
                    # killed, the next invocation will see this consumed attempt
                    # and continue from the remaining retry budget.
                    save_manifest(manifest_path, manifest)

                    def persist_child_pid():
                        # A controller interruption after Popen must not make a
                        # later resume accidentally launch a duplicate process.
                        manifest["experiments"][task_id] = experiment
                        save_manifest(manifest_path, manifest)

                    return_code, metrics, failure_reason = execute_experiment(
                        experiment,
                        attempt,
                        False,
                        on_process_started=persist_child_pid,
                    )
                    attempt["finished_at"] = utc_now()
                    attempt["return_code"] = return_code
                    attempt["status"] = "succeeded" if return_code == 0 else "failed"
                    attempt["failure_reason"] = failure_reason
                    attempt["metrics"] = {str(topk): values for topk, values in metrics.items()}

                    experiment["finished_at"] = attempt["finished_at"]
                    experiment["return_code"] = return_code
                    experiment["failure_reason"] = failure_reason
                    experiment["metrics"] = attempt["metrics"] if return_code == 0 else {}
                    if return_code == 0:
                        experiment["status"] = "succeeded"
                    elif len(attempts) >= max_attempts:
                        experiment["status"] = "exhausted"
                    else:
                        experiment["status"] = "failed"
                    manifest["experiments"][task_id] = experiment
                    save_manifest(manifest_path, manifest)
                    write_summary(artifact_root, manifest)

                    if return_code == 0:
                        break
                    if experiment["status"] == "exhausted":
                        print("[EXHAUSTED] %s failed %d attempts; skipping it" % (task_id, len(attempts)))
                        if args.fail_fast:
                            raise RuntimeError("Experiment exhausted all retries with --fail-fast: %s" % task_id)
                        break
                    print("[RETRY] %s failed attempt %d/%d; retrying" % (
                        task_id, len(attempts), max_attempts
                    ))

    write_summary(artifact_root, manifest)
    selected_tasks = [
        task
        for task in manifest["experiments"].values()
        if task.get("noise_mode", DEFAULT_NOISE_MODE) == args.noise_mode
    ]
    succeeded = sum(1 for task in selected_tasks if task.get("status") == "succeeded")
    exhausted = sum(1 for task in selected_tasks if task.get("status") == "exhausted")
    pending = sum(1 for task in selected_tasks if task.get("status") not in ("succeeded", "exhausted"))
    planned = len(models) * len(datasets) * len(ratios)
    print("\nBenchmark state: succeeded=%d exhausted=%d unfinished=%d planned=%d" % (
        succeeded, exhausted, pending, planned
    ))
    print("Manifest: %s" % manifest_path)
    print("Summary: %s" % (artifact_root / "results_summary.csv"))


def prepare_only(args):
    datasets = tuple(args.datasets)
    ratios = parse_noise_ratios(args.noise_ratios)
    validate_project_layout(tuple(args.models), datasets, Path(args.source_data_root).resolve())
    prepare_selected_data(args, datasets, ratios)


def show_status(args):
    artifact_root = resolved_artifact_root(args)
    manifest_path = artifact_root / "benchmark_manifest.json"
    manifest = load_manifest(manifest_path)
    experiments = {
        task_id: experiment
        for task_id, experiment in manifest.get("experiments", {}).items()
        if experiment.get("noise_mode", DEFAULT_NOISE_MODE) == args.noise_mode
    }
    if not experiments:
        print(
            "No %s benchmark experiments have been registered: %s" %
            (args.noise_mode, manifest_path)
        )
        return
    counts = {}
    for experiment in experiments.values():
        status = experiment.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    print("Benchmark manifest: %s" % manifest_path)
    for status in sorted(counts):
        print("%s: %d" % (status, counts[status]))
    for task_id in sorted(experiments):
        experiment = experiments[task_id]
        attempts_used = len(normalized_attempt_history(experiment))
        print(
            "%s | %s | %s | mode=%s | noise=%s | attempts=%s/%s" % (
                experiment.get("status", "unknown"),
                experiment["model"],
                experiment["dataset"],
                experiment.get("noise_mode", DEFAULT_NOISE_MODE),
                experiment["noise_ratio"],
                attempts_used,
                experiment.get("max_attempts", "?"),
            )
        )


def add_shared_arguments(parser):
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="models to include (default: all three)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
        help="datasets to include (default: all three)",
    )
    parser.add_argument(
        "--noise-ratios",
        nargs="+",
        default=list(DEFAULT_NOISE_RATIOS),
        help="noise ratios, e.g. 0 0.05 0.10",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help="master seed used only for noisy-data generation",
    )
    parser.add_argument(
        "--noise-mode",
        choices=NOISE_MODES,
        default=DEFAULT_NOISE_MODE,
        help=(
            "replacement: replace original UB edges (default); injection: preserve "
            "original UB edges and append false ones"
        ),
    )
    parser.add_argument(
        "--source-data-root",
        default=str(PROJECT_ROOT / "datasets"),
        help="canonical clean dataset root",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help=(
            "optional root for generated data, task state, logs, and summaries; defaults to "
            "noise_benchmark_artifacts/ for replacement or "
            "noise_benchmark_artifacts_injection/ for injection"
        ),
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="regenerate noisy data even if matching metadata already exists",
    )


def add_compatibility_arguments(parser):
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="models to smoke-test (default: all three)",
    )
    parser.add_argument("--gpu", default="0", help="GPU identifier passed to each smoke-test model")
    parser.add_argument(
        "--train-seed",
        type=int,
        default=DEFAULT_TRAIN_SEED,
        help="fixed model-training seed used for the smoke test",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for smoke-test child processes",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help="seed for the shared zero-noise Youshu smoke input",
    )
    parser.add_argument(
        "--noise-mode",
        choices=NOISE_MODES,
        default=DEFAULT_NOISE_MODE,
        help="artifact namespace for the smoke check (default: replacement)",
    )
    parser.add_argument(
        "--source-data-root",
        default=str(PROJECT_ROOT / "datasets"),
        help="canonical clean dataset root",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help=(
            "optional root for smoke-test data, logs, and status; defaults by noise mode"
        ),
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="regenerate the zero-noise Youshu smoke input",
    )
    parser.add_argument(
        "--rerun-compatibility-check",
        action="store_true",
        help="rerun smoke checks that already passed",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate UB noise and run CrossCBR, MultiCBR, and ReCoBR reproducibly."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="generate/reuse noisy training UB files only")
    add_shared_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=prepare_only)

    compatibility_parser = subparsers.add_parser(
        "compatibility",
        help="run the short Youshu load/train/evaluate/save smoke test only",
    )
    add_compatibility_arguments(compatibility_parser)
    compatibility_parser.set_defaults(handler=compatibility_only)

    run_parser = subparsers.add_parser("run", help="prepare data and run the selected benchmark tasks")
    add_shared_arguments(run_parser)
    run_parser.add_argument("--gpu", default="0", help="GPU identifier passed to each model's -g option")
    run_parser.add_argument(
        "--train-seed",
        type=int,
        default=DEFAULT_TRAIN_SEED,
        help="fixed model-training seed passed to every experiment",
    )
    run_parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for child training processes",
    )
    run_parser.add_argument(
        "--skip-compatibility-check",
        action="store_true",
        help="explicitly bypass the required Youshu compatibility smoke test",
    )
    run_parser.add_argument(
        "--rerun-compatibility-check",
        action="store_true",
        help="rerun compatibility checks even when their cached status passed",
    )
    run_parser.add_argument(
        "--rerun-succeeded",
        action="store_true",
        help="rerun experiments already marked succeeded",
    )
    run_parser.add_argument(
        "--rerun-exhausted",
        action="store_true",
        help="give an exhausted experiment a new retry budget",
    )
    run_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="retries after the initial attempt for each experiment (default: 3)",
    )
    run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop only after a task exhausts all of its retries",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show generated commands without launching any training",
    )
    run_parser.set_defaults(handler=run_benchmark)

    status_parser = subparsers.add_parser("status", help="show the saved benchmark task states")
    status_parser.add_argument(
        "--noise-mode",
        choices=NOISE_MODES,
        default=DEFAULT_NOISE_MODE,
        help="show replacement (default) or injection benchmark state",
    )
    status_parser.add_argument(
        "--artifact-root",
        default=None,
        help="optional benchmark artifact root (defaults by --noise-mode)",
    )
    status_parser.set_defaults(handler=show_status)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "run" and args.max_retries < 0:
            raise ValueError("--max-retries must be non-negative")
        args.handler(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
