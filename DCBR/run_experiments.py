#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute reviewed DCBR search rounds serially, without importing torch/model code."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import yaml

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
TERMINAL = {"SUCCESS", "FAILED", "INTERRUPTED"}
RULE = "validation Recall@20 and NDCG@20 both strictly improve over the saved checkpoint"
CROSS_RULE = "validation_recall20_then_ndcg20"
DATA_NAMES = ("user_bundle_train", "user_bundle_tune", "user_bundle_test", "user_item", "bundle_item")
MANAGED = {"status", "success", "fingerprint", "execution_ids", "resolved_config",
           "dependency_selection", "blocked_by"}
# Protocol and budget are common to each round. Changes are explicit at the
# round level rather than hidden among individual experiment overrides.
ROUND_ONLY = {"seed", "epochs", "data_path", "topk", "test_interval", "batch_size_train",
              "batch_size_test", "neg_num", "embedding_size", "num_layers", "norm"}


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_yaml(path):
    # Reject duplicate keys: silently losing an experiment or an override makes
    # a centralized plan particularly difficult to audit.
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"Non-string or duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    with Path(path).open(encoding="utf-8-sig") as stream:
        return yaml.load(stream, Loader=UniqueLoader)


def require_fields(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f"Invalid {label} fields; allowed={sorted(allowed)}, required={sorted(required)}")


def identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError(f"{label} needs 1-64 ASCII letters, numbers, '_' or '-'")
    return value


def merge_config(base, overrides):
    if not isinstance(overrides, dict) or set(overrides) - set(base):
        raise ValueError("Overrides must be a mapping of existing model configuration keys")
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = copy.deepcopy(value)
    return result


def data_fingerprints(conf, dataset):
    directory = resolve_path(conf["data_path"]) / dataset
    names = [dataset + "_data_size.txt"] + [name + ".txt" for name in DATA_NAMES]
    return {name: file_hash(directory / name) for name in names}


def source_fingerprints():
    return {"train.py": file_hash(MODEL_DIR / "train.py"),
            "run_experiments.py": file_hash(Path(__file__))}


def experiment_fingerprint(conf, dataset, sources, data, settings, selection=None):
    signature = {"config": conf, "dataset": dataset, "sources": sources, "data": data,
                 "gpu": settings["gpu"], "device": settings["device"], "selection_rule": RULE}
    if selection is not None:
        signature["dependency_selection"] = selection
    return digest(signature)


def load_plan(path, gpu=None, device=None):
    plan = read_yaml(path)
    require_fields(plan, {"version", "base_config", "work_dir", "result_file", "defaults", "rounds"},
                   {"version", "base_config", "work_dir", "result_file", "defaults", "rounds"}, "plan")
    if type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("Unsupported plan version")
    defaults = plan["defaults"]
    require_fields(defaults, {"gpu", "device", "max_attempts", "retry_delay_seconds",
                             "console_interval_seconds", "stream_output"},
                   {"gpu", "device", "max_attempts", "retry_delay_seconds",
                    "console_interval_seconds", "stream_output"}, "defaults")
    if type(defaults["max_attempts"]) is not int or defaults["max_attempts"] < 1:
        raise ValueError("max_attempts must be a positive integer")
    for key in ("retry_delay_seconds", "console_interval_seconds"):
        if type(defaults[key]) not in (int, float) or not math.isfinite(defaults[key]) or defaults[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if defaults["console_interval_seconds"] < 1 or type(defaults["stream_output"]) is not bool:
        raise ValueError("console_interval_seconds must be >= 1 and stream_output must be boolean")
    settings = dict(defaults)
    settings["gpu"] = str(gpu if gpu is not None else settings["gpu"])
    settings["device"] = device if device is not None else settings["device"]
    if not re.fullmatch(r"[0-9]+", settings["gpu"]) or settings["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("Specify one numeric GPU ID and device cpu/cuda/auto")
    if not isinstance(plan["rounds"], list):
        raise ValueError("rounds must be a list")
    base_configs = read_yaml(resolve_path(plan["base_config"]))
    rounds = []
    seen_rounds = set()
    sources = source_fingerprints()
    data_cache = {}
    for raw_round in plan["rounds"]:
        require_fields(raw_round, {"id", "dataset", "note", "overrides", "experiments"},
                       {"id", "dataset", "note", "experiments"}, "round")
        rid = identifier(raw_round["id"], "round id")
        if rid in seen_rounds:
            raise ValueError(f"Duplicate round id: {rid}")
        seen_rounds.add(rid)
        dataset = identifier(raw_round["dataset"], "dataset")
        if dataset not in base_configs:
            raise ValueError(f"No base configuration for {dataset}")
        if not isinstance(raw_round["note"], str) or not raw_round["note"].strip():
            raise ValueError("Round note is required")
        base = merge_config(base_configs[dataset], raw_round.get("overrides", {}))
        base["data_path"] = str(resolve_path(base["data_path"]))
        # The shared protocol is not a search dimension.
        if any(base[key] != base_configs[dataset][key] for key in ("topk", "test_interval", "neg_num")):
            raise ValueError("topk, test_interval and neg_num must retain the shared base protocol")
        data_key = (base["data_path"], dataset)
        if data_key not in data_cache:
            data_cache[data_key] = data_fingerprints(base, dataset)
        data = data_cache[data_key]
        if not isinstance(raw_round["experiments"], list) or not raw_round["experiments"]:
            raise ValueError("Each scheduled round needs at least one experiment")
        experiments = []
        seen_experiments = set()
        for item in raw_round["experiments"]:
            require_fields(item, {"id", "note", "overrides", "inherit_best"} | MANAGED,
                           {"id", "note"}, "experiment")
            eid = identifier(item["id"], "experiment id")
            if eid in seen_experiments:
                raise ValueError(f"Duplicate experiment id in {rid}: {eid}")
            if not isinstance(item["note"], str) or not item["note"].strip():
                raise ValueError(f"Experiment {eid} needs a note")
            overrides = item.get("overrides", {})
            if not isinstance(overrides, dict) or set(overrides) & ROUND_ONLY:
                raise ValueError(f"Experiment {eid}: protocol/seed/budget settings must be common to the round")
            conf = merge_config(base, overrides)
            inherit = item.get("inherit_best")
            if inherit is not None:
                require_fields(inherit, {"candidates", "rule"}, {"candidates", "rule"}, "inherit_best")
                candidates = inherit["candidates"]
                if (inherit["rule"] != CROSS_RULE or not isinstance(candidates, list) or not candidates
                        or any(not isinstance(value, str) for value in candidates)
                        or len(set(candidates)) != len(candidates) or set(candidates) - seen_experiments):
                    raise ValueError("inherit_best needs unique earlier candidate IDs and validation_recall20_then_ndcg20")
            seen_experiments.add(eid)
            experiments.append({"id": eid, "note": item["note"], "config": conf,
                                "template_config": copy.deepcopy(conf),
                                "overrides": copy.deepcopy(overrides), "inherit_best": copy.deepcopy(inherit),
                                "dependency_selection": None, "blocked_by": [],
                                "fingerprint": experiment_fingerprint(conf, dataset, sources, data, settings)
                                if inherit is None else None})
        rounds.append({"id": rid, "dataset": dataset, "note": raw_round["note"],
                       "experiments": experiments, "data_files": data})
    return {"document": plan, "path": Path(path).resolve(), "settings": settings, "rounds": rounds,
            "work_dir": resolve_path(plan["work_dir"]), "result_file": resolve_path(plan["result_file"]),
            "sources": sources}


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
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
            raise RuntimeError(f"Another controller/report writer holds the lock: {path}") from error
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
            raise PermissionError(f"Cannot inspect worker PID {pid}")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError("Cannot read worker status")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_process(process):
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
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)


def empty_state():
    return {"version": 1, "executions": []}


def load_state(path):
    if not Path(path).exists():
        return empty_state()
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("version") != 1 or not isinstance(state.get("executions"), list):
        raise ValueError("Invalid search history")
    return state


def latest_execution(state, round_id, experiment):
    if experiment["fingerprint"] is None:
        return None
    return next((record for record in reversed(state["executions"])
                 if record["round_id"] == round_id and record["experiment_id"] == experiment["id"]
                 and record["fingerprint"] == experiment["fingerprint"]), None)


def validate_result(path, expected, dataset, sources, data, info, gpu, device, check_hashes=True):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "SUCCESS" or record.get("completed") is not True:
        raise ValueError("Training result is not complete SUCCESS")
    if record.get("experiment_case") != "second" or record.get("dataset") != dataset:
        raise ValueError("Wrong dataset or experiment category")
    if record.get("source_hash") != sources["train.py"] or record.get("data_files") != data:
        raise ValueError("Training source or data fingerprint mismatch")
    actual = record["config"]
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Actual configuration mismatch: {key}")
    if actual.get("info") != info or record.get("gpu") != gpu or actual.get("gpu") != gpu:
        raise ValueError("Execution identity or GPU mismatch")
    if device != "auto" and not str(record.get("device", "")).startswith(device):
        raise ValueError("Execution device mismatch")
    if record.get("completed_epochs") != expected["epochs"] or actual.get("selection_rule") != RULE:
        raise ValueError("Training budget or selection protocol mismatch")
    if digest(actual) != record.get("config_hash"):
        # Training uses json.dumps defaults (including ensure_ascii=True).
        legacy_hash = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        if legacy_hash != record.get("config_hash"):
            raise ValueError("Actual configuration checksum mismatch")
    best = record["result"]
    epoch = best["best_epoch"]
    if (type(epoch) is not int or not 0 <= epoch < expected["epochs"]
            or best["best_epoch_1based"] != epoch + 1 or (epoch + 1) % expected["test_interval"] != 0
            or best.get("selection_rule") != RULE):
        raise ValueError("Invalid best epoch or selection rule")
    for split in ("val", "test"):
        for name in ("recall", "ndcg"):
            for k in expected["topk"]:
                value = best["metrics"][split][name][str(k)]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid/missing metric")
    for name in ("recall", "ndcg"):
        value = best["metrics"]["val"][name]["20"]
        if value <= 0 or best["selection_metrics"][name + "@20"] != value:
            raise ValueError("Selection metrics do not match the selected checkpoint")
    for key in ("checkpoint_model", "checkpoint_config", "config"):
        artifact = Path(record["artifacts"][key])
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f"Missing/empty artifact: {key}")
        if check_hashes and file_hash(artifact) != record["artifact_hashes"][key]:
            raise ValueError(f"Artifact checksum mismatch: {key}")
    if json.loads(Path(record["artifacts"]["checkpoint_config"]).read_text(encoding="utf-8")) != actual:
        raise ValueError("Checkpoint configuration differs from actual configuration")
    if read_yaml(record["artifacts"]["config"]) != {dataset: actual}:
        raise ValueError("Configuration snapshot mismatch")
    if json.loads(Path(record["artifacts"]["result_json"]).read_text(encoding="utf-8")) != record:
        raise ValueError("Internal and external training results differ")
    return record


def check_success(record, experiment, round_, plan):
    if record is None or record["status"] != "SUCCESS":
        return False
    try:
        attempt = record["attempts"][-1]
        training = record["training_record"]
        paths = [attempt["result_path"]] + [training["artifacts"][key] for key in
                                          ("checkpoint_model", "checkpoint_config", "config", "result_json")]
        stamp = [(str(path), Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in paths]
        # Rehash once per controller invocation, and again whenever an artifact
        # changes. Avoid re-reading every historical checkpoint on each status
        # update when a plan contains hundreds of experiments.
        cache = plan.setdefault("_verified_success", {})
        key = (record["execution_id"], experiment["fingerprint"])
        if key in cache and cache[key][0] == stamp:
            return cache[key][1]
        verified = validate_result(attempt["result_path"], experiment["config"], round_["dataset"],
                                   plan["sources"], round_["data_files"], attempt["info"],
                                   record["gpu"], record["device"])
        valid = verified == training
        cache[key] = (stamp, valid)
        return valid
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def validation_key(training_record):
    """Cross-configuration ordering; test metrics are deliberately never read."""
    metrics = training_record["result"]["metrics"]["val"]
    values = tuple(metrics[name]["20"] for name in ("recall", "ndcg"))
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid cross-configuration validation metrics")
    return values


def resolve_dependencies(plan, round_, state):
    earlier = {}
    for item in round_["experiments"]:
        inherit = item.get("inherit_best")
        if inherit is not None:
            candidates = []
            missing = []
            for eid in inherit["candidates"]:
                parent = earlier[eid]
                record = latest_execution(state, round_["id"], parent)
                if not check_success(record, parent, round_, plan):
                    missing.append(eid)
                else:
                    candidates.append((parent, record))
            item.update(config=None, fingerprint=None, dependency_selection=None, blocked_by=missing)
            if not missing:
                # max retains the first candidate on an exact two-metric tie.
                # Require candidates to follow declaration order below.
                candidates.sort(key=lambda pair: list(earlier).index(pair[0]["id"]))
                winner, winner_record = max(candidates, key=lambda pair: validation_key(pair[1]["training_record"]))
                selection = {"rule": CROSS_RULE, "tie_break": "first experiment in declared order",
                             "winner_experiment_id": winner["id"],
                             "winner_execution_id": winner_record["execution_id"],
                             "winner_validation": dict(zip(("recall@20", "ndcg@20"),
                                                           validation_key(winner_record["training_record"]))),
                             "candidates": [{"experiment_id": parent["id"], "execution_id": record["execution_id"],
                                             "fingerprint": parent["fingerprint"],
                                             "validation": dict(zip(("recall@20", "ndcg@20"),
                                                                    validation_key(record["training_record"])))}
                                            for parent, record in candidates]}
                conf = merge_config(winner["config"], item["overrides"])
                item.update(config=conf, dependency_selection=selection,
                            fingerprint=experiment_fingerprint(conf, round_["dataset"], plan["sources"],
                                                               round_["data_files"], plan["settings"], selection))
        earlier[item["id"]] = item


def select_experiments(plan, round_, state, mode, only):
    resolve_dependencies(plan, round_, state)
    ids = {item["id"] for item in round_["experiments"]}
    if (mode == "selected" and (not only or set(only) - ids)) or (mode != "selected" and only):
        raise ValueError("--mode selected requires valid experiment IDs in --only; other modes reject --only")
    selected = []
    for item in round_["experiments"]:
        previous = latest_execution(state, round_["id"], item)
        valid = check_success(previous, item, round_, plan)
        if (mode == "all" or mode == "selected" and item["id"] in only
                or mode == "resume" and not valid
                or mode == "failed" and previous is not None and previous["status"] == "FAILED"):
            selected.append(item)
    return selected


def sync_plan(plan, state):
    # Re-read to preserve unrelated rounds and detect operator edits while the
    # controller runs, instead of silently overwriting a revised search plan.
    current = read_yaml(plan["path"])
    def declarative(document):
        result = copy.deepcopy(document)
        for round_ in result["rounds"]:
            for item in round_["experiments"]:
                for key in MANAGED:
                    item.pop(key, None)
        return result
    if declarative(current) != declarative(plan["document"]):
        raise RuntimeError("Search plan was edited during execution; history is saved. Restore/review it before resuming")
    for raw_round, round_ in zip(current["rounds"], plan["rounds"]):
        resolve_dependencies(plan, round_, state)
        for item, experiment in zip(raw_round["experiments"], round_["experiments"]):
            previous = latest_execution(state, round_["id"], experiment)
            valid = check_success(previous, experiment, round_, plan)
            status = previous["status"] if previous else "PENDING"
            if experiment.get("blocked_by"):
                status = "BLOCKED"
            if status == "SUCCESS" and not valid:
                status = "INVALIDATED"
            item.update(status=status, success=valid, fingerprint=experiment["fingerprint"],
                        resolved_config=copy.deepcopy(experiment["config"]),
                        dependency_selection=copy.deepcopy(experiment.get("dependency_selection")),
                        blocked_by=experiment.get("blocked_by", []),
                        execution_ids=[r["execution_id"] for r in state["executions"]
                                       if r["round_id"] == round_["id"] and r["experiment_id"] == item["id"]])
    atomic_text(plan["path"], yaml.safe_dump(current, allow_unicode=True, sort_keys=False))


def append_report(path, record):
    end = "END_EXECUTION: " + record["execution_id"]
    with file_lock(str(path) + ".lock"):
        old = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
        if end in old.splitlines():
            return
        lines = ["", "=" * 72, "EXECUTION_ID: " + record["execution_id"],
                 "实验类别: 第二种情况（新数据集参数搜索）", "模型: DCBR", "数据集: " + record["dataset"],
                 "轮次: " + record["round_id"], "轮次备注: " + record["round_note"],
                 "实验: " + record["experiment_id"], "实验备注: " + record["note"],
                 "状态: " + record["status"], "GPU: " + record["gpu"], "设备: " + record["device"],
                 "开始时间(UTC+08:00): " + record["started_at"], "结束时间(UTC+08:00): " + record["ended_at"],
                 "实验总耗时（秒，含重试）: " + str(record["duration_seconds"]),
                 f"尝试次数: {len(record['attempts'])}/{record['max_attempts']}",
                 "代码版本: " + record["git_commit"], "源码摘要: " + json.dumps(record["sources"], sort_keys=True),
                 "配置及数据指纹: " + record["fingerprint"],
                 "数据文件摘要: " + json.dumps(record["data_files"], sort_keys=True),
                 "选模规则: " + RULE, "完整配置:",
                 yaml.safe_dump(record.get("training_record", {}).get("config", record["config"]),
                                allow_unicode=True, sort_keys=False).rstrip()]
        if record.get("timing_note"):
            lines.append("计时说明: " + record["timing_note"])
        if record.get("dependency_selection"):
            lines.append("前置实验选择依据（仅验证集）: " + json.dumps(record["dependency_selection"], ensure_ascii=False))
        for attempt in record["attempts"]:
            lines.extend([f"尝试 {attempt['number']}: {attempt['status']}",
                          "  开始时间: " + attempt["started_at"], "  结束时间: " + attempt.get("ended_at", "未知"),
                          "  耗时（秒）: " + str(attempt.get("duration_seconds")),
                          "  日志: " + attempt["stdout"], "  配置快照: " + attempt["config_snapshot"],
                          "  结果: " + attempt["result_path"]])
            if attempt.get("error"):
                lines.append("  原因: " + attempt["error"])
        if record.get("error"):
            lines.append("原因: " + record["error"])
        training = record.get("training_record")
        if training:
            best = training["result"]
            lines.extend([f"best_epoch: {best['best_epoch']}（从0计数）",
                          f"最佳训练轮次: {best['best_epoch_1based']}",
                          "验证选模指标: " + json.dumps(best["selection_metrics"], sort_keys=True)])
            for split in ("val", "test"):
                for k in sorted(map(int, training["config"]["topk"])):
                    metric = best["metrics"][split]
                    lines.append(f"{split} TOP {k}: Recall={metric['recall'][str(k)]:.8f}, NDCG={metric['ndcg'][str(k)]:.8f}")
            lines.append("训练产物: " + json.dumps(training["artifacts"], ensure_ascii=False))
        lines.extend([end, ""])
        atomic_text(path, old + "\n".join(lines))


def save_history(path, state):
    atomic_json(path, state)
    for record in state["executions"]:
        atomic_json(Path(record["directory"]) / "execution.json", record)


def finish(record, started):
    record.update(ended_at=timestamp(), duration_seconds=round(max(0, time.monotonic() - started), 3))


def recover(state, history_path):
    for record in state["executions"]:
        if record["status"] != "RUNNING":
            continue
        attempt = record["attempts"][-1] if record["attempts"] else None
        pid = attempt.get("pid") if attempt else None
        if pid and (record["host"] != socket.gethostname() or process_alive(pid)):
            raise RuntimeError(f"Previous worker PID {pid} may still be alive on {record['host']}; inspect it before resuming")
        # Do not guess success from a file left behind after a hard controller
        # crash. Preserve the result, mark interruption, and re-run from epoch 0.
        record.update(status="INTERRUPTED", ended_at=timestamp(), duration_seconds=None,
                      error="Previous controller stopped before finalizing; next execution starts from epoch 0",
                      timing_note="结束时间为恢复检测时间，实际停止时间和耗时无法确定。")
        if attempt and attempt["status"] == "RUNNING":
            attempt.update(status="INTERRUPTED", ended_at=timestamp(), duration_seconds=None,
                           error="Controller stopped; actual stop time unknown")
    save_history(history_path, state)
    for record in state["executions"]:
        if record["status"] in TERMINAL:
            append_report(record["report_file"], record)


def show_progress(round_, plan, rows, current=""):
    # Dependency resolution can make the remaining children ready between two
    # snapshots. Keep their display in step with the actual scheduling state.
    for item in round_["experiments"]:
        if rows.get(item["id"]) in {"WAITING", "BLOCKED"}:
            rows[item["id"]] = "BLOCKED" if item.get("blocked_by") else "WAITING"
    counts = {status: sum(value == status for value in rows.values())
              for status in ("WAITING", "RUNNING", "RETRY", "SUCCESS", "FAILED", "INTERRUPTED", "SKIPPED", "BLOCKED")}
    print(f"\n[{timestamp()}] {round_['dataset']} | 轮次 {round_['id']} | GPU {plan['settings']['gpu']} | {current}", flush=True)
    print(f"轮次备注: {round_['note']}", flush=True)
    print("计划 {} | 本次 {} | 跳过 {} | 成功 {} | 失败 {} | 中断 {} | 运行 {} | 等待 {} | 等待前置结果 {}".format(
        len(rows), len(rows) - counts["SKIPPED"], counts["SKIPPED"], counts["SUCCESS"], counts["FAILED"],
        counts["INTERRUPTED"], counts["RUNNING"], counts["WAITING"] + counts["RETRY"], counts["BLOCKED"]), flush=True)
    for item in round_["experiments"]:
        print(f"  {item['id']}: {rows[item['id']]} | {item['note']}", flush=True)


def worker_main(gate, arguments):
    # The parent persists PID before opening the gate. If it crashes between
    # Popen and the durable save, this bootstrap expires without touching CUDA.
    deadline = time.monotonic() + 30
    while not Path(gate).exists():
        if time.monotonic() >= deadline:
            return 125
        time.sleep(.1)
    import runpy
    sys.argv = [str(MODEL_DIR / "train.py")] + arguments
    runpy.run_path(str(MODEL_DIR / "train.py"), run_name="__main__")
    return 0


def run_worker(command, attempt, record, state, history_path, settings, progress):
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                       PYTHONHASHSEED=str(record["config"]["seed"]), CUDA_VISIBLE_DEVICES=record["gpu"])
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = None
    reader = None
    last = [""]
    stream_error = []
    started = time.monotonic()
    with Path(attempt["stdout"]).open("w", encoding="utf-8") as output:
        def drain():
            try:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    last[0] = line.rstrip()[-300:]
                    if settings["stream_output"]:
                        print(line, end="", flush=True)
            except BaseException as error:
                stream_error.append(str(error))
        try:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **options)
            attempt["pid"] = process.pid
            save_history(history_path, state)
            atomic_text(attempt["gate"], "start\n")
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            last_shown = time.monotonic()
            while process.poll() is None:
                time.sleep(.2)
                if time.monotonic() - last_shown >= settings["console_interval_seconds"]:
                    progress(f"已运行 {time.monotonic() - started:.0f} 秒 | 最近输出: {last[0]}")
                    last_shown = time.monotonic()
            reader.join(timeout=10)
            if reader.is_alive() or stream_error:
                raise RuntimeError("Cannot capture training output completely: " + str(stream_error))
            return process.returncode
        finally:
            if process is not None:
                stop_process(process)
                if reader is not None:
                    reader.join(timeout=10)
                if process.stdout is not None:
                    process.stdout.close()


def execute_experiment(plan, round_, item, state, history_path, rows):
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = plan["work_dir"] / round_["dataset"] / round_["id"] / item["id"] / execution_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    started = time.monotonic()
    record = {"execution_id": execution_id, "round_id": round_["id"], "round_note": round_["note"],
              "experiment_id": item["id"], "note": item["note"], "dataset": round_["dataset"],
              "fingerprint": item["fingerprint"], "sources": plan["sources"], "data_files": round_["data_files"],
              "git_commit": commit, "config": item["config"], "gpu": plan["settings"]["gpu"],
              "device": plan["settings"]["device"], "host": socket.gethostname(), "status": "RUNNING",
              "runner_settings": copy.deepcopy(plan["settings"]),
              "dependency_selection": copy.deepcopy(item.get("dependency_selection")),
              "started_at": timestamp(), "max_attempts": plan["settings"]["max_attempts"], "attempts": [],
              "directory": str(directory), "report_file": str(plan["result_file"])}
    for name in plan["sources"]:
        destination = directory / ("source_" + name)
        shutil.copyfile(MODEL_DIR / name, destination)
        if file_hash(destination) != plan["sources"][name]:
            raise RuntimeError("Source changed before creating its snapshot")
    state["executions"].append(record)
    save_history(history_path, state)
    sync_plan(plan, state)
    attempt_started = started
    try:
        for number in range(1, record["max_attempts"] + 1):
            if source_fingerprints() != plan["sources"] or data_fingerprints(item["config"], round_["dataset"]) != round_["data_files"]:
                raise RuntimeError("Source/data changed during this round; stop and review before running more experiments")
            stem = directory / f"attempt_{number}"
            info = f"tune_{execution_id}_{item['id']}_a{number}"
            attempt = {"number": number, "status": "RUNNING", "started_at": timestamp(), "info": info,
                       "config_snapshot": str(stem) + "_config.yaml", "result_path": str(stem) + "_result.json",
                       "stdout": str(stem) + "_stdout.txt", "gate": str(stem) + "_start.gate"}
            record["attempts"].append(attempt)
            atomic_text(attempt["config_snapshot"], yaml.safe_dump({round_["dataset"]: item["config"]}, sort_keys=False))
            save_history(history_path, state)
            rows[item["id"]] = "RUNNING"
            show_progress(round_, plan, rows, f"实验 {item['id']} | 尝试 {number}/{record['max_attempts']} | 日志 {attempt['stdout']}")
            args = ["--config", attempt["config_snapshot"], "-d", round_["dataset"], "-g", record["gpu"],
                    "--device", record["device"], "--experiment-case", "second", "--quiet", "--info", info,
                    "--note", item["note"], "--output-dir", str(directory / f"attempt_{number}_artifacts"),
                    "--record-file", str(stem) + "_training.txt", "--result-json", attempt["result_path"]]
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", attempt["gate"]] + args
            attempt["command"] = command
            attempt_started = time.monotonic()
            try:
                code = run_worker(command, attempt, record, state, history_path, plan["settings"],
                                  lambda detail: show_progress(round_, plan, rows, f"{item['id']} | 尝试 {number} | {detail}"))
                attempt["return_code"] = code
                if code in (130, -signal.SIGINT, -signal.SIGTERM, 0xC000013A, -1073741510):
                    raise KeyboardInterrupt(f"Training worker was interrupted (exit code {code})")
                if code != 0:
                    raise RuntimeError(f"Training exited with code {code}")
                if (source_fingerprints() != plan["sources"]
                        or data_fingerprints(item["config"], round_["dataset"]) != round_["data_files"]):
                    raise RuntimeError("Source/data changed while training")
                record["training_record"] = validate_result(attempt["result_path"], item["config"], round_["dataset"],
                                                           plan["sources"], round_["data_files"], info,
                                                           record["gpu"], record["device"])
            except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
                attempt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                log = Path(attempt["stdout"])
                if log.exists():
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 4000))
                        attempt["error"] += "\n日志末尾:\n" + stream.read().decode("utf-8", errors="replace")
                record["error"] = attempt["error"]
                print(f"[{timestamp()}] {item['id']} 尝试 {number} FAILED: {error}", flush=True)
            else:
                attempt["status"] = record["status"] = "SUCCESS"
                record.pop("error", None)
                finish(record, started)  # A crash after the next save still has a complete terminal record.
            finish(attempt, attempt_started)
            save_history(history_path, state)
            if record["status"] == "SUCCESS":
                break
            if number < record["max_attempts"]:
                rows[item["id"]] = "RETRY"
                show_progress(round_, plan, rows, f"{item['id']} 尝试失败；{plan['settings']['retry_delay_seconds']} 秒后从第 1 轮重试")
                deadline = time.monotonic() + plan["settings"]["retry_delay_seconds"]
                while time.monotonic() < deadline:
                    time.sleep(min(.2, max(0, deadline - time.monotonic())))
        else:
            record["status"] = "FAILED"
    except BaseException as error:
        record.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                      error=f"{type(error).__name__}: {error}")
        if record["attempts"] and record["attempts"][-1]["status"] == "RUNNING":
            record["attempts"][-1].update(status=record["status"], error=record["error"])
            finish(record["attempts"][-1], attempt_started)
        raise
    finally:
        finish(record, started)
        save_history(history_path, state)
        if record["status"] in TERMINAL:
            append_report(plan["result_file"], record)
        sync_plan(plan, state)
        rows[item["id"]] = record["status"]
        show_progress(round_, plan, rows, f"{item['id']} 结束 | 总耗时 {record['duration_seconds']} 秒")
    return record["status"]


def preflight(plan, round_, experiments):
    # Validate using the actual training entry point, in a separate process, so
    # the controller never imports CUDA/torch or duplicates model validation.
    import tempfile
    for item in experiments:
        # A template checks already-known overrides before any GPU work. It is
        # never published/executed as a resolved inherited configuration. The
        # actual winner's configuration is checked again before child training.
        conf = item["config"] if item["config"] is not None else item["template_config"]
        with tempfile.TemporaryDirectory(prefix="dcbr-preflight-") as temp:
            path = Path(temp) / "config.yaml"
            path.write_text(yaml.safe_dump({round_["dataset"]: conf}), encoding="utf-8")
            result = subprocess.run([sys.executable, str(MODEL_DIR / "train.py"), "--config", str(path),
                                     "-d", round_["dataset"], "--check-config"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, encoding="utf-8", timeout=60)
            if result.returncode:
                raise ValueError(f"Invalid experiment {item['id']}:\n{result.stdout}\n{result.stderr}")


def write_round_summary(plan, round_, state):
    resolve_dependencies(plan, round_, state)
    verified = []
    for item in round_["experiments"]:
        record = latest_execution(state, round_["id"], item)
        if check_success(record, item, round_, plan):
            verified.append((item, record))
    complete = len(verified) == len(round_["experiments"])
    summary = {"round_id": round_["id"], "dataset": round_["dataset"], "complete": complete,
               "planned": len(round_["experiments"]), "valid_successes": len(verified),
               "configuration_selection_rule": CROSS_RULE, "checkpoint_selection_rule": RULE,
               "candidates": [{"experiment_id": item["id"], "execution_id": record["execution_id"],
                               "validation": dict(zip(("recall@20", "ndcg@20"),
                                                      validation_key(record["training_record"])))}
                              for item, record in verified], "best": None}
    if complete:
        winner, record = max(verified, key=lambda pair: validation_key(pair[1]["training_record"]))
        summary["best"] = {"experiment_id": winner["id"], "execution_id": record["execution_id"],
                           "config": winner["config"],
                           "validation": dict(zip(("recall@20", "ndcg@20"),
                                                  validation_key(record["training_record"]))),
                           "checkpoint": record["training_record"]["artifacts"]["checkpoint_model"]}
    summary_id = digest(summary)
    summary["updated_at"] = timestamp()
    path = plan["work_dir"] / round_["dataset"] / round_["id"] / "round_summary.json"
    atomic_json(path, summary)
    end = "END_ROUND_SUMMARY: " + summary_id
    with file_lock(str(plan["result_file"]) + ".lock"):
        old = plan["result_file"].read_text(encoding="utf-8") if plan["result_file"].exists() else ""
        if end not in old.splitlines():
            text = "\n" + "=" * 72 + "\n轮次汇总: " + round_["id"] + "\n"
            text += "跨配置规则: 验证 Recall@20 优先，完全相同再比较验证 NDCG@20；完全平局保留声明顺序在前者。\n"
            text += json.dumps(summary, ensure_ascii=False, indent=2) + "\n" + end + "\n"
            atomic_text(plan["result_file"], old + text)
    print(f"Round summary: {path} | complete={complete} | best={summary['best']}", flush=True)
    return summary


def run(plan_path, round_id=None, mode="resume", only=None, dry_run=False, gpu=None, device=None, list_only=False):
    plan_path = Path(plan_path).resolve()
    plan = load_plan(plan_path, gpu, device)
    if list_only:
        for round_ in plan["rounds"]:
            print(f"{round_['id']} | {round_['dataset']} | {len(round_['experiments'])} experiments | {round_['note']}")
        if not plan["rounds"]:
            print("No search rounds scheduled. Add a reviewed round to the central plan first.")
        return 0
    round_ = next((r for r in plan["rounds"] if r["id"] == round_id), None)
    if round_ is None:
        raise ValueError("Specify an existing --round ID; use --list to inspect scheduled rounds")
    history_path = plan["work_dir"] / "history.json"
    if dry_run:
        state = load_state(history_path)
        selected = select_experiments(plan, round_, state, mode, only or [])
        preflight(plan, round_, selected)
        print(f"DRY RUN | {round_['dataset']} | {round_['id']} | GPU {plan['settings']['gpu']} | device {plan['settings']['device']}")
        rows = {item["id"]: ("BLOCKED" if item.get("blocked_by") else "WAITING") if item in selected
                else "SKIPPED" for item in round_["experiments"]}
        show_progress(round_, plan, rows)
        for item in selected:
            if item["config"] is None:
                print(f"{item['id']}: 等待前置实验 {item['blocked_by']}；按 {CROSS_RULE} 继承获胜配置，"
                      f"再覆盖 {item['overrides']}。实际配置与指纹在前置实验成功后生成。")
            else:
                print(f"{item['id']} fingerprint={item['fingerprint']}\n" + yaml.safe_dump(item["config"], sort_keys=False))
        return 0
    # Both locks: protect the central YAML (even with a changed work_dir) and
    # protect a shared execution history (even with a different plan filename).
    with file_lock(str(plan_path) + ".lock"):
        plan = load_plan(plan_path, gpu, device)
        with file_lock(plan["work_dir"] / "controller.lock"):
            history_path = plan["work_dir"] / "history.json"
            round_ = next(r for r in plan["rounds"] if r["id"] == round_id)
            state = load_state(history_path)
            recover(state, history_path)
            selected = select_experiments(plan, round_, state, mode, only or [])
            preflight(plan, round_, selected)
            sync_plan(plan, state)
            rows = {item["id"]: ("BLOCKED" if item.get("blocked_by") else "WAITING") if item in selected
                    else "SKIPPED" for item in round_["experiments"]}
            show_progress(round_, plan, rows, "开始串行执行；所有重试均从第 1 轮训练开始")
            statuses = []
            for item in selected:
                resolve_dependencies(plan, round_, state)
                if item.get("blocked_by"):
                    rows[item["id"]] = "BLOCKED"
                    statuses.append("BLOCKED")
                    sync_plan(plan, state)
                    show_progress(round_, plan, rows, f"{item['id']} 等待前置实验成功: {item['blocked_by']}")
                    continue
                previous = latest_execution(state, round_["id"], item)
                if mode == "resume" and check_success(previous, item, round_, plan):
                    rows[item["id"]] = "SKIPPED"
                    continue
                if item.get("inherit_best"):
                    selection = item["dependency_selection"]
                    print("前置配置已选定（仅验证集）: " + json.dumps(selection, ensure_ascii=False), flush=True)
                    preflight(plan, round_, [item])
                statuses.append(execute_experiment(plan, round_, item, state, history_path, rows))
            summary = write_round_summary(plan, round_, state) if any(
                item.get("inherit_best") for item in round_["experiments"]) else None
            print(f"Round finished: SUCCESS={statuses.count('SUCCESS')} FAILED={statuses.count('FAILED')} "
                  f"BLOCKED={statuses.count('BLOCKED')}", flush=True)
            return 1 if "FAILED" in statuses or "BLOCKED" in statuses or (
                summary is not None and not summary["complete"] and mode in {"resume", "all"}) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=MODEL_DIR / "搜参配置.yaml")
    parser.add_argument("--round", dest="round_id")
    parser.add_argument("--mode", choices=["resume", "failed", "selected", "all"], default="resume")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--gpu", "-g")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    args = parser.parse_args(argv)
    try:
        return run(args.plan, args.round_id, args.mode, args.only, args.dry_run,
                   args.gpu, args.device, args.list_only)
    except KeyboardInterrupt:
        print("Interrupted. History is saved; resume restarts unfinished experiments from epoch 0.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration,
            subprocess.TimeoutExpired, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        sys.exit(worker_main(sys.argv[2], sys.argv[3:]))
    sys.exit(main())
