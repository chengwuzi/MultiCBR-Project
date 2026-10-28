#!/usr/bin/env python3
"""Repeat MultiCBR with predeclared random seeds and report every run's mean/std."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid

import yaml

if __package__:
    from .configuration import (BASE_KEYS, IMPLEMENTATION, MODEL_DIR, PROJECT_ROOT, SELECTION_RULE,
                                SINGLETONS, digest, file_hash, load_config, project_path,
                                read_yaml, source_fingerprints)
    from .train import atomic_json, timestamp
else:
    from configuration import (BASE_KEYS, IMPLEMENTATION, MODEL_DIR, PROJECT_ROOT, SELECTION_RULE,
                               SINGLETONS, digest, file_hash, load_config, project_path,
                               read_yaml, source_fingerprints)
    from train import atomic_json, timestamp


def input_hashes(conf, dataset):
    directory = Path(conf["data_path"]) / dataset
    names = [dataset + "_data_size.txt"] + [name + ".txt" for name in
            ("user_bundle_train", "user_bundle_tune", "user_bundle_test", "user_item", "bundle_item")]
    return {name: file_hash(directory / name) for name in names}


@contextmanager
def batch_lock(directory):
    with (Path(directory) / "controller.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another controller is running this batch") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return False
            raise PermissionError(f"Cannot inspect previous PID {pid}")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError("Cannot read previous worker status")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_worker(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def verify_inputs(batch):
    if (digest(plan_identity(batch)) != batch["plan_hash"]
            or load_config(batch["config_snapshot"], batch["dataset"]) != batch["config"]):
        raise ValueError("Declared configuration, seeds or execution settings changed")
    if (batch["sources"] != source_fingerprints()
            or batch["data_files"] != input_hashes(batch["config"], batch["dataset"])):
        raise ValueError("Training source or data changed; create a new batch with the reviewed inputs")
    if file_hash(batch["config_snapshot"]) != batch["config_snapshot_sha256"]:
        raise ValueError("Batch configuration snapshot changed")


def plan_identity(batch):
    return {key: batch[key] for key in ("dataset", "config", "sources", "data_files", "gpu", "device")} | {
        "seeds": [run["seed"] for run in batch["runs"]]}


def validate_run(batch, run):
    attempt = run["attempts"][-1]
    record = json.loads(Path(attempt["result_json"]).read_text(encoding="utf-8"))
    expected = dict(batch["config"], seed=run["seed"])
    actual = record["config"]
    if (record.get("status") != "SUCCESS" or record.get("completed") is not True
            or record.get("completed_epochs") != expected["epochs"]
            or record.get("dataset") != batch["dataset"] or record.get("model") != "MultiCBR"
            or record.get("experiment_case") != "first" or record.get("implementation") != IMPLEMENTATION
            or record.get("selection_rule") != SELECTION_RULE
            or record.get("sources") != batch["sources"] or record.get("data_files") != batch["data_files"]):
        raise ValueError("Run is incomplete or its implementation/data/protocol differs")
    if any(actual.get(key) != expected[key] for key in BASE_KEYS):
        raise ValueError("Run's actual configuration/seed differs from the declared repeat")
    if (any(actual.get(single) != actual[plural][0] for plural, single in SINGLETONS.items())
            or record.get("config_hash") != digest(actual)
            or actual.get("selection_rule") != SELECTION_RULE or actual.get("dataset") != batch["dataset"]
            or actual.get("model") != "MultiCBR" or actual.get("implementation") != IMPLEMENTATION
            or actual.get("gpu") != batch["gpu"] or record.get("gpu") != batch["gpu"]
            or actual.get("device") != record.get("device")
            or (batch["device"] != "auto" and not record.get("device", "").startswith(batch["device"]))):
        raise ValueError("Actual configuration checksum, scalars or execution device differs")
    best = record["result"]
    epoch = best["best_epoch"]
    if (type(epoch) is not int or not 0 <= epoch < expected["epochs"]
            or best["best_epoch_1based"] != epoch + 1 or (epoch + 1) % expected["test_interval"]
            or best.get("selection_rule") != SELECTION_RULE):
        raise ValueError("Invalid selected checkpoint epoch/protocol")
    for split in ("val", "test"):
        for name in ("recall", "ndcg"):
            for k in expected["topk"]:
                value = best["metrics"][split][name][str(k)]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Missing/nonfinite/out-of-range metric")
    if any(best["selection_metrics"][name + "@20"] != best["metrics"]["val"][name]["20"]
           or best["selection_metrics"][name + "@20"] <= 0 for name in ("recall", "ndcg")):
        raise ValueError("Selection values differ from the same checkpoint's validation metrics")
    for key in ("checkpoint_model", "checkpoint_config", "config"):
        path = Path(record["artifacts"][key])
        if not path.is_file() or not path.stat().st_size or file_hash(path) != record["artifact_hashes"][key]:
            raise ValueError(f"Missing or changed artifact: {key}")
    if (json.loads(Path(record["artifacts"]["checkpoint_config"]).read_text(encoding="utf-8")) != actual
            or read_yaml(record["artifacts"]["config"]) != {batch["dataset"]: actual}
            or json.loads(Path(record["artifacts"]["result_json"]).read_text(encoding="utf-8")) != record):
        raise ValueError("Configuration snapshot or internal/external result differs")
    if run.get("execution_id") and run["execution_id"] != record["execution_id"]:
        raise ValueError("Successful execution identity changed")
    return record


def summarize(batch, directory):
    records = []
    for run in batch["runs"]:
        if run["status"] == "SUCCESS":
            records.append((run, validate_run(batch, run)))
    identities = [record["execution_id"] for _, record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated execution IDs cannot count as independent runs")
    complete = len(records) == len(batch["runs"])
    summary = dict(batch_id=batch["batch_id"], dataset=batch["dataset"], complete=complete,
                   planned=len(batch["runs"]), successful=len(records),
                   seeds=[run["seed"] for run in batch["runs"]], seed_origin=batch["seed_origin"],
                   selection_rule=SELECTION_RULE, config=batch["config"], sources=batch["sources"],
                   data_files=batch["data_files"], mean=None, std=None,
                   std_definition="sample standard deviation, ddof=1; null for n=1",
                   runs=[dict(index=run["index"], seed=run["seed"], execution_id=record["execution_id"],
                              best_epoch=record["result"]["best_epoch"],
                              best_epoch_1based=record["result"]["best_epoch_1based"],
                              metrics=record["result"]["metrics"], artifacts=record["artifacts"])
                         for run, record in records], updated_at=timestamp())
    summary["config_seed_note"] = "Base seed is overridden by each predeclared run seed; all other settings are fixed"
    if complete:
        summary["mean"], summary["std"] = {}, {}
        for split in ("val", "test"):
            summary["mean"][split], summary["std"][split] = {}, {}
            for metric in ("recall", "ndcg"):
                summary["mean"][split][metric], summary["std"][split][metric] = {}, {}
                for k in batch["config"]["topk"]:
                    values = [r["result"]["metrics"][split][metric][str(k)] for _, r in records]
                    summary["mean"][split][metric][str(k)] = statistics.mean(values)
                    summary["std"][split][metric][str(k)] = statistics.stdev(values) if len(values) > 1 else None
    atomic_json(Path(directory) / "summary.json", summary)
    lines = [f"MultiCBR {batch['dataset']} | {len(records)}/{len(batch['runs'])} 次有效成功",
             "随机种子: " + ", ".join(str(run["seed"]) for run in batch["runs"]),
             "每次均由验证双指标严格改善选取同一检查点；所有计划运行等权平均，不按测试成绩筛选。",
             "标准差：样本标准差(ddof=1)，n=1时不计算。"]
    if not complete:
        lines.append("本批次未完成，不输出部分运行的正式平均值。使用同一manifest恢复。")
    else:
        for split in ("val", "test"):
            for k in batch["config"]["topk"]:
                values = []
                for metric in ("recall", "ndcg"):
                    avg = summary["mean"][split][metric][str(k)]
                    std = summary["std"][split][metric][str(k)]
                    values.append(f"{metric}={avg:.8f} ± {std:.8f}" if std is not None else f"{metric}={avg:.8f} (n=1)")
                lines.append(f"{split} TOP {k}: " + ", ".join(values))
    (Path(directory) / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def execute_run(batch, run, directory, manifest):
    attempt_dir = Path(directory) / f"run_{run['index']:03d}" / f"attempt_{len(run['attempts'])+1}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    attempt = dict(status="RUNNING", started_at=timestamp(), host=socket.gethostname(), pid=None,
                   stdout=str(attempt_dir / "stdout.txt"), result_json=str(attempt_dir / "result.json"))
    command = [sys.executable, "-B", "-u", str(MODEL_DIR / "train.py"), "--config", batch["config_snapshot"],
               "-d", batch["dataset"], "--seed", str(run["seed"]), "-g", batch["gpu"], "--device", batch["device"],
               "--output-dir", str(attempt_dir / "artifacts"), "--record-file", batch["record_file"],
               "--result-json", attempt["result_json"], "--quiet", "--note",
               f"Repeated batch {batch['batch_id']} run {run['index']}/{len(batch['runs'])}; {batch['note']}"]
    attempt["command"] = command
    run["attempts"].append(attempt)
    run["status"] = "RUNNING"
    batch.update(status="RUNNING", updated_at=timestamp())
    atomic_json(manifest, batch)
    print(f"[{timestamp()}] {batch['dataset']} {run['index']}/{len(batch['runs'])} seed={run['seed']} | {attempt['stdout']}", flush=True)
    environment = dict(os.environ, PYTHONHASHSEED=str(run["seed"]), CUDA_VISIBLE_DEVICES=batch["gpu"],
                       PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = None
    started = time.monotonic()
    try:
        with Path(attempt["stdout"]).open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT, **options)
            attempt["pid"] = process.pid
            atomic_json(manifest, batch)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"[{timestamp()}] run {run['index']}/{len(batch['runs'])} seed={run['seed']} "
                          f"已运行 {time.monotonic()-started:.0f}s | 完整进度见 {attempt['stdout']}", flush=True)
        attempt["return_code"] = code
        if code in (130, -signal.SIGINT, -signal.SIGTERM, 0xC000013A, -1073741510):
            raise KeyboardInterrupt("Training worker interrupted")
        if code:
            raise RuntimeError(f"Training exited with code {code}; see {attempt['stdout']}")
        verify_inputs(batch)
        record = validate_run(batch, run)
        run.update(status="SUCCESS", execution_id=record["execution_id"])
        attempt["status"] = "SUCCESS"
        print(f"[{timestamp()}] run {run['index']} SUCCESS | seed={run['seed']} | "
              f"best epoch={record['result']['best_epoch_1based']}", flush=True)
    except BaseException as error:
        run["status"] = attempt["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        attempt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if process is not None:
            stop_worker(process)
        attempt.update(ended_at=timestamp(), duration_seconds=round(time.monotonic() - started, 3))
        batch["status"] = run["status"] if run["status"] != "SUCCESS" else "RUNNING"
        atomic_json(manifest, batch)


def run_batch(directory):
    directory = Path(directory).resolve()
    manifest = directory / "manifest.json"
    with batch_lock(directory):
        batch = json.loads(manifest.read_text(encoding="utf-8"))
        if batch.get("version") != 1 or not batch.get("runs"):
            raise ValueError("Invalid batch manifest")
        seeds = [run["seed"] for run in batch["runs"]]
        if len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
            raise ValueError("Manifest must contain distinct valid seeds")
        if [run["index"] for run in batch["runs"]] != list(range(1, len(seeds) + 1)):
            raise ValueError("Invalid run indices")
        verify_inputs(batch)
        # Never replace an unavailable successful run with an unrelated execution.
        summarize(batch, directory)
        try:
            for run in batch["runs"]:
                if run["status"] == "SUCCESS":
                    print(f"Skip verified run {run['index']} seed={run['seed']}", flush=True)
                    continue
                if run["status"] == "RUNNING":
                    attempt = run["attempts"][-1]
                    if (attempt["host"] != socket.gethostname() or not attempt.get("pid")
                            or process_alive(attempt["pid"])):
                        raise RuntimeError("Previous worker may still be running; inspect its host/PID before resuming")
                    attempt.update(status="INTERRUPTED", recovery_detected_at=timestamp(),
                                   error="Previous controller stopped; actual stop time/duration unknown")
                    run["status"] = "INTERRUPTED"
                verify_inputs(batch)
                execute_run(batch, run, directory, manifest)
            batch.update(status="SUCCESS", ended_at=timestamp())
            atomic_json(manifest, batch)
        finally:
            summary = summarize(batch, directory)
        print(f"Summary: {directory / 'summary.txt'} | complete={summary['complete']}", flush=True)
    return summary


def create_batch(args):
    if args.dataset is None:
        raise ValueError("Specify --dataset/-d")
    repeats = args.repeats if args.repeats is not None else len(args.seeds) if args.seeds else None
    if repeats is None or not 1 <= repeats <= 2**32:
        raise ValueError("Specify a positive --repeats/-n")
    if args.seeds is None:
        seeds = secrets.SystemRandom().sample(range(2**32), repeats)
    else:
        seeds = args.seeds
        if len(seeds) != repeats or len(set(seeds)) != repeats or any(not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("--seeds must contain exactly N distinct integers in [0,2**32-1]")
    overrides = {key: getattr(args, key) for key in ("epochs", "data_path", "num_workers", "batch_size_test")}
    conf = load_config(args.config or MODEL_DIR / "config.yaml", args.dataset, overrides)
    hashes = input_hashes(conf, args.dataset)
    print(f"Checking {args.dataset} configuration and data before starting {repeats} runs...", flush=True)
    # The native CLI performs full input checks once before any planned training.
    with __import__("tempfile").TemporaryDirectory(prefix="multicbr-repeat-preflight-") as temp:
        path = Path(temp) / "config.yaml"
        path.write_text(yaml.safe_dump({args.dataset: conf}), encoding="utf-8")
        checked = subprocess.run([sys.executable, "-B", str(MODEL_DIR / "train.py"), "--config", str(path),
                                  "-d", args.dataset, "--check-config"], cwd=PROJECT_ROOT,
                                 capture_output=True, text=True, encoding="utf-8")
        if checked.returncode:
            raise ValueError("Configuration/data preflight failed:\n" + checked.stdout + checked.stderr)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = project_path(args.output_dir) / args.dataset / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "config.yaml"
    snapshot.write_text(yaml.safe_dump({args.dataset: conf}, sort_keys=False), encoding="utf-8")
    batch = dict(version=1, batch_id=batch_id, dataset=args.dataset, created_at=timestamp(), status="PENDING",
                 seed_origin="random_without_replacement" if args.seeds is None else "explicit_replay",
                 config=conf, config_snapshot=str(snapshot), config_snapshot_sha256=file_hash(snapshot),
                 sources=source_fingerprints(), data_files=hashes, controller_sha256=file_hash(__file__),
                 gpu=args.gpu or "0", device=args.device or "cuda", note=args.note,
                 record_file=str(project_path(args.record_file or MODEL_DIR / "实验记录.txt")),
                 runs=[dict(index=i+1, seed=seed, status="PENDING", attempts=[]) for i, seed in enumerate(seeds)])
    batch["plan_hash"] = digest(plan_identity(batch))
    atomic_json(directory / "manifest.json", batch)
    print(f"Batch: {directory}\nDataset: {args.dataset} | repeats: {repeats} | seeds: {seeds}", flush=True)
    print(f"Resume: python -u MultiCBR/run_repeated.py --resume \"{directory}\"", flush=True)
    if args.dry_run:
        summarize(batch, directory)
        print("Plan saved; no training started. Run --resume with this directory to execute these exact seeds.", flush=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", choices=["Youshu", "NetEase", "iFashion"])
    parser.add_argument("-n", "--repeats", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional predeclared seeds for replay; default is random")
    parser.add_argument("--config", type=Path)
    parser.add_argument("-g", "--gpu")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-path")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size-test", type=int)
    parser.add_argument("--note", default="")
    parser.add_argument("--record-file")
    parser.add_argument("--output-dir", default=str(MODEL_DIR / "log/repeated"))
    parser.add_argument("--resume", type=Path, help="Existing batch directory or manifest.json; reuses the saved seed list")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and save a random-seed plan without training")
    args = parser.parse_args(argv)
    if args.gpu is not None and (not args.gpu.isascii() or not args.gpu.isdigit()):
        raise ValueError("Specify one nonnegative physical GPU ID")
    if args.resume:
        if any(getattr(args, key) is not None for key in
               ("dataset", "repeats", "seeds", "config", "epochs", "data_path", "num_workers", "batch_size_test", "record_file")):
            raise ValueError("--resume uses the existing manifest; do not override its configuration or repeats")
        if args.dry_run:
            raise ValueError("To inspect a saved plan, read manifest.json; --dry-run creates a new plan")
        directory = project_path(args.resume)
        if directory.name == "manifest.json":
            directory = directory.parent
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if any(value is not None and value != saved[key] for key, value in (("gpu", args.gpu), ("device", args.device))):
            raise ValueError("Resume preserves the batch GPU and device")
    else:
        directory = create_batch(args)
        if args.dry_run:
            return 0
    run_batch(directory)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Resume the saved batch; unfinished runs restart with their original seeds.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as error:
        print(f"Repeated batch stopped: {error}", file=sys.stderr)
        sys.exit(1)
