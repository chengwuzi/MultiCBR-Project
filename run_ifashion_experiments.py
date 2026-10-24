"""Run a serial iFashion experiment plan without importing the training model."""

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import yaml

from experiment_io import atomic_write_json, validation_score


PROJECT_ROOT = Path(__file__).resolve().parent
GRID_KEYS = (
    "lrs", "l2_regs", "UB_ratios", "UI_ratios", "BI_ratios",
    "embedding_sizes", "num_layerss", "c_lambdas", "c_temps",
)
TERMINAL = {"SUCCESS", "FAILED", "INTERRUPTED"}


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def finish_timing(record, started):
    record["ended_at"] = timestamp()
    record["duration_seconds"] = round(max(0, time.monotonic() - started), 3)


def elapsed_text(record):
    seconds = record.get("duration_seconds")
    suffix = ""
    if seconds is None and not record.get("timing_note"):
        # Older progress files contain timestamps but no monotonic duration.
        try:
            seconds = max(0, (datetime.fromisoformat(record["ended_at"])
                              - datetime.fromisoformat(record["started_at"])).total_seconds())
        except (KeyError, TypeError, ValueError):
            return "未知"
        suffix = "（根据开始和结束时间估算）"
    if seconds is None:
        return "未知"
    hours, remainder = divmod(round(seconds * 1000), 3600000)
    minutes, milliseconds = divmod(remainder, 60000)
    return f"{hours:02d}:{minutes:02d}:{milliseconds / 1000:06.3f}（共 {seconds:.3f} 秒）" + suffix


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError(f"{label} must contain 1-64 ASCII letters, numbers, '_' or '-'")
    return value


def deep_merge(base, overrides, prefix=""):
    if not isinstance(overrides, dict):
        raise ValueError(f"{prefix or 'overrides'} must be a mapping")
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown configuration key: {name}")
        old = base[key]
        if isinstance(old, dict):
            merged[key] = deep_merge(old, value, name)
        elif isinstance(old, list) != isinstance(value, list) or isinstance(old, bool) != isinstance(value, bool):
            raise ValueError(f"Wrong configuration type: {name}")
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(conf, seed):
    if conf.get("seed") != seed:
        raise ValueError("Experiment overrides must not change the round's fixed seed")
    for key in GRID_KEYS:
        if not isinstance(conf[key], list) or len(conf[key]) != 1:
            raise ValueError(f"{key} must contain exactly one value; each process runs one experiment")
        value = conf[key][0]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} requires a finite number")
    for key in ("lrs", "embedding_sizes", "num_layerss", "c_temps"):
        if conf[key][0] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("embedding_sizes", "num_layerss"):
        if type(conf[key][0]) is not int:
            raise ValueError(f"{key} requires an integer")
    if conf["aug_type"] not in {"Noise", "ED", "MD", "OP"}:
        raise ValueError("Unknown augmentation type")
    if conf["aug_type"] == "OP" and any(conf[k][0] != 0 for k in ("UB_ratios", "UI_ratios", "BI_ratios")):
        raise ValueError("OP requires zero view ratios")
    if conf["l2_regs"][0] < 0 or conf["c_lambdas"][0] < 0:
        raise ValueError("Regularization and contrastive weights cannot be negative")
    if any(not 0 <= conf[k][0] < 1 for k in ("UB_ratios", "UI_ratios", "BI_ratios")):
        raise ValueError("View ratios must be in [0, 1)")
    for key in ("epochs", "batch_size_train", "batch_size_test", "neg_num", "ed_interval", "test_interval"):
        if type(conf[key]) is not int or conf[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if conf["epochs"] < conf["test_interval"]:
        raise ValueError("The experiment would finish before its first evaluation")
    if (not isinstance(conf["topk"], list) or any(type(k) is not int or k <= 0 for k in conf["topk"])
            or len(set(conf["topk"])) != len(conf["topk"]) or not {20, 40}.issubset(conf["topk"])):
        raise ValueError("topk must include 20 and 40 for checkpoint selection")
    weights = conf["fusion_weights"]
    if len(weights["modal_weight"]) != 3 or any(v < 0 for v in weights["modal_weight"]):
        raise ValueError("modal_weight requires three nonnegative weights")
    if not math.isclose(sum(weights["modal_weight"]), 1, abs_tol=1e-8):
        raise ValueError("Modal weights must sum to 1")
    for view in ("UB", "UI", "BI"):
        if len(weights[view + "_layer"]) != conf["num_layerss"][0] + 1:
            raise ValueError(f"Wrong layer coefficient count for {view}")
    if type(conf["anchor_cl"]["enabled"]) is not bool or type(conf["latent_rebuild"]["enabled"]) is not bool:
        raise ValueError("Module enabled flags must be true/false booleans")
    if conf["anchor_cl"]["temp"] <= 0 or conf["anchor_cl"]["anchor_cl_lambda"] < 0:
        raise ValueError("Invalid pre-fusion contrastive parameters")
    data_dir = resolve_path(conf["data_path"]) / "iFashion"
    for filename in ("iFashion_data_size.txt", "user_item.txt", "bundle_item.txt",
                     "user_bundle_train.txt", "user_bundle_tune.txt", "user_bundle_test.txt"):
        if not (data_dir / filename).is_file():
            raise ValueError(f"Missing dataset file: {data_dir / filename}")
    sizes = [int(v) for v in (data_dir / "iFashion_data_size.txt").read_text().split()[:3]]
    if max(conf["topk"]) > sizes[1]:
        raise ValueError("topk exceeds the number of bundles")
    rebuild = conf["latent_rebuild"]
    if rebuild["enabled"]:
        for key in ("rebuild_k", "latent_diffusion_num_steps", "latent_diffusion_batch_size",
                    "latent_diffusion_infer_batch_size", "latent_diffusion_denoiser_layers"):
            if type(rebuild[key]) is not int or rebuild[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < rebuild["latent_diffusion_beta_start"] <= rebuild["latent_diffusion_beta_end"] < 1:
            raise ValueError("Invalid diffusion beta schedule")
        if not 0 <= rebuild["latent_diffusion_dropout"] < 1:
            raise ValueError("Invalid diffusion dropout")
        if (rebuild["latent_diffusion_lr"] <= 0 or rebuild["latent_diffusion_weight_decay"] < 0
                or rebuild["latent_diffusion_set_loss_weight"] < 0):
            raise ValueError("Invalid diffusion optimizer/loss settings")
        if type(rebuild["latent_diffusion_hidden_dim"]) is not int or rebuild["latent_diffusion_hidden_dim"] <= 0:
            raise ValueError("Diffusion hidden dimension must be a positive integer")
        time_dim = rebuild["latent_diffusion_time_dim"]
        if type(time_dim) is not int or time_dim < 0 or time_dim == 1:
            raise ValueError("Diffusion time dimension must be zero or an integer >= 2")
        for entity in ("user", "bundle"):
            path = resolve_path(rebuild[f"latent_diffusion_{entity}_embedding_path"])
            if not path.is_file():
                raise ValueError(f"Missing embedding: {path}")


def load_plan(path, gpu=None):
    with Path(path).open(encoding="utf-8") as stream:
        plan = yaml.safe_load(stream)
    allowed = {"version", "round_id", "note", "base_config", "seed", "gpu",
               "max_attempts", "result_file", "common_overrides", "experiments"}
    if not isinstance(plan, dict) or set(plan) - allowed:
        raise ValueError("Invalid or unknown plan fields")
    if plan.get("version") != 1:
        raise ValueError("Unsupported plan version")
    identifier(plan.get("round_id"), "round_id")
    if not isinstance(plan.get("note"), str) or not plan["note"].strip():
        raise ValueError("A round note is required")
    if type(plan.get("seed")) is not int or plan.get("max_attempts") != 3:
        raise ValueError("A fixed integer seed and max_attempts: 3 are required")
    plan["gpu"] = str(gpu if gpu is not None else plan.get("gpu", "0"))
    plan["result_file"] = str(resolve_path(plan["result_file"]))
    with resolve_path(plan["base_config"]).open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)["iFashion"]
    base = deep_merge(base, plan.get("common_overrides", {}))
    if base.get("seed") != plan["seed"]:
        raise ValueError("The base configuration seed must equal the round seed")
    experiments = plan.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("The plan needs a nonempty experiment list")
    source_hash = hashlib.sha256()
    for name in ("train.py", "models/model.py", "utility.py", "experiment_io.py"):
        source_hash.update(name.encode())
        source_hash.update((PROJECT_ROOT / name).read_bytes())
    seen = set()
    for experiment in experiments:
        if not isinstance(experiment, dict) or set(experiment) - {"id", "note", "overrides"}:
            raise ValueError("Invalid experiment fields")
        eid = identifier(experiment.get("id"), "experiment id")
        if eid in seen:
            raise ValueError(f"Duplicate experiment id: {eid}")
        seen.add(eid)
        if not isinstance(experiment.get("note"), str) or not experiment["note"].strip():
            raise ValueError(f"Experiment {eid} needs a note")
        conf = deep_merge(base, experiment.get("overrides", {}))
        validate_config(conf, plan["seed"])
        experiment["config"] = conf
        signature = {"config": conf, "gpu": plan["gpu"], "source": source_hash.hexdigest()}
        experiment["fingerprint"] = hashlib.sha256(
            json.dumps(signature, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    plan["source_hash"] = source_hash.hexdigest()
    return plan


def load_state(path):
    if not path.exists():
        return {"version": 1, "executions": []}
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if state.get("version") != 1 or not isinstance(state.get("executions"), list):
        raise ValueError(f"Invalid state file: {path}")
    return state


def latest_execution(state, experiment):
    return next((r for r in reversed(state["executions"])
                 if r["experiment_id"] == experiment["id"]
                 and r["fingerprint"] == experiment["fingerprint"]), None)


def select_experiments(plan, state, mode, only):
    ids = {e["id"] for e in plan["experiments"]}
    if mode == "selected":
        if not only or set(only) - ids:
            raise ValueError("selected mode requires valid experiment IDs with --only")
    elif only:
        raise ValueError("--only requires --mode selected")
    chosen = []
    for experiment in plan["experiments"]:
        previous = latest_execution(state, experiment)
        status = previous["status"] if previous else None
        if ((mode == "resume" and status != "SUCCESS")
                or (mode == "failed" and status == "FAILED")
                or (mode == "selected" and experiment["id"] in only)
                or mode == "all"):
            chosen.append(experiment)
    return chosen


class ConsoleProgress:
    """Print complete status snapshots that also remain readable in nohup logs."""

    def __init__(self, plan, state, selected, mode):
        self.plan = plan
        self.mode = mode
        self.selected_ids = {experiment["id"] for experiment in selected}
        self.rows = {}
        for experiment in plan["experiments"]:
            previous = latest_execution(state, experiment)
            status = previous["status"] if previous else None
            if experiment["id"] in self.selected_ids:
                detail = {"SUCCESS": "（主动重跑）", "FAILED": "（上次失败，本次重新尝试）",
                          "INTERRUPTED": "（上次中断，本次重新运行）"}.get(status, "")
                self.rows[experiment["id"]] = ("WAITING", "等待运行" + detail)
            else:
                label = {"SUCCESS": "已完成（历史成功，本次跳过）",
                         "FAILED": "失败：三次机会耗尽（历史记录，本次未选中）",
                         "INTERRUPTED": "已中断（本次未选中）"}.get(status, "未运行（本次未选中）")
                self.rows[experiment["id"]] = ("SKIPPED", label)

    def show(self):
        counts = {}
        for eid in self.selected_ids:
            status = self.rows[eid][0]
            counts[status] = counts.get(status, 0) + 1
        ended = sum(counts.get(status, 0) for status in TERMINAL)
        lines = ["", "=" * 72,
                 f"[{timestamp()}] 实验进度 | 轮次: {self.plan['round_id']}",
                 f"轮次备注: {self.plan['note']}",
                 f"模式: {self.mode} | GPU: {self.plan['gpu']} | 计划 {len(self.rows)} 组，本次运行 {len(self.selected_ids)} 组，"
                 f"跳过 {len(self.rows) - len(self.selected_ids)} 组",
                 f"本次进度: 已结束 {ended}/{len(self.selected_ids)} | 已完成 {counts.get('SUCCESS', 0)} | "
                 f"三次机会耗尽 {counts.get('FAILED', 0)} | 正在运行 {counts.get('RUNNING', 0)} | "
                 f"等待 {counts.get('WAITING', 0) + counts.get('RETRY', 0)} | 已中断 {counts.get('INTERRUPTED', 0)}"]
        for index, experiment in enumerate(self.plan["experiments"], 1):
            label = self.rows[experiment["id"]][1]
            value = experiment["config"]["c_lambdas"][0]
            lines.append(f"  {index}. {experiment['id']} | c_lambda={value} | {label} | {experiment['note']}")
        lines.append("=" * 72)
        print("\n".join(lines), flush=True)

    def update(self, experiment, status, label):
        self.rows[experiment["id"]] = (status, label)
        self.show()

    def finished(self, experiment, record):
        count = len(record["attempts"])
        label = {"SUCCESS": f"已完成（第 {count}/3 次尝试成功）",
                 "FAILED": "失败：三次机会耗尽（3/3）",
                 "INTERRUPTED": "已中断（下次启动重新运行）"}[record["status"]]
        self.update(experiment, record["status"], label)


@contextmanager
def round_lock(path):
    """An OS lock is automatically released even after the controller crashes."""
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
            raise RuntimeError("Another controller is already running this round") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def append_report(path, record):
    path = Path(path)
    end = f"END_EXECUTION: {record['execution_id']}"
    if path.exists() and end in path.read_text(encoding="utf-8").splitlines():
        return
    result = record.get("result")
    lines = ["", "=" * 72, f"EXECUTION_ID: {record['execution_id']}",
             f"轮次: {record['round_id']}", f"轮次备注: {record['round_note']}",
             f"实验: {record['experiment_id']}", f"实验备注: {record['note']}",
             f"状态: {record['status']}", f"GPU: {record['gpu']}",
             f"开始时间(UTC+08:00): {record['started_at']}",
             f"结束时间(UTC+08:00): {record['ended_at']}",
             f"实验总耗时（含重试）: {elapsed_text(record)}",
             f"尝试次数: {len(record['attempts'])}/3",
             f"代码版本: {record['git_commit']}", f"训练源码摘要: {record['source_hash']}",
             f"配置指纹: {record['fingerprint']}", "完整实际配置:",
             yaml.safe_dump(result["config"] if result else record["config"],
                            allow_unicode=True, sort_keys=False).rstrip()]
    if record.get("timing_note"):
        lines.append("计时说明: " + record["timing_note"])
    for attempt in record["attempts"]:
        lines.append(f"尝试 {attempt['number']}: {attempt['status']}；日志: {attempt['stdout']}")
        lines.extend([f"  开始时间(UTC+08:00): {attempt['started_at']}",
                      f"  结束时间(UTC+08:00): {attempt.get('ended_at', '未记录')}",
                      f"  本次尝试耗时: {elapsed_text(attempt)}"])
        if attempt.get("timing_note"):
            lines.append("  计时说明: " + attempt["timing_note"])
        if attempt.get("error"):
            lines.append("原因: " + attempt["error"])
    if result:
        lines.extend([f"best_epoch: {result['best_epoch']} (从0计数)",
                      f"最佳训练轮次: {result['best_epoch_1based']}",
                      f"验证选模分数: {result['selection_score']:.10f}"])
        for split in ("val", "test"):
            for k in sorted(map(int, result["metrics"][split]["recall"])):
                metric = result["metrics"][split]
                lines.append(f"{split} TOP {k}: Recall={metric['recall'][str(k)]:.8f}, "
                             f"NDCG={metric['ndcg'][str(k)]:.8f}")
        lines.append("训练产物: " + json.dumps(result["artifacts"], ensure_ascii=False))
    else:
        lines.append("最终原因: " + record.get("error", "Interrupted"))
    lines.extend([end, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())


def validate_result(path, config, info):
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict) or result.get("completed") is not True:
        raise ValueError("Training did not report successful completion")
    actual = result["config"]
    if not isinstance(actual, dict):
        raise ValueError("Invalid result configuration")
    if any(actual.get(key) != value for key, value in config.items()):
        raise ValueError("Reported configuration differs from the experiment snapshot")
    if actual.get("seed") != config["seed"] or actual.get("info") != info:
        raise ValueError("Result belongs to a different attempt")
    epoch = result["best_epoch"]
    if type(epoch) is not int or not 0 <= epoch < config["epochs"] or result["best_epoch_1based"] != epoch + 1:
        raise ValueError("Invalid best epoch")
    for split in ("val", "test"):
        for name in ("recall", "ndcg"):
            for k in config["topk"]:
                value = result["metrics"][split][name][str(k)]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Missing or non-finite best metric")
    score = validation_score(result["metrics"])
    if score <= 0 or not math.isclose(score, result["selection_score"], rel_tol=1e-12):
        raise ValueError("Invalid checkpoint selection score")
    for key in ("checkpoint_model", "checkpoint_config"):
        if not resolve_path(result["artifacts"][key]).is_file():
            raise ValueError(f"Missing best checkpoint artifact: {key}")
    return result


def process_alive(pid):
    if os.name == "nt":
        # os.kill(pid, 0) is not a portable Windows existence check.
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
                raise OSError("Cannot read worker exit status")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def training_command(config_path, result_path, info, gpu):
    return [sys.executable, "-u", str(PROJECT_ROOT / "train.py"),
            "--config", str(config_path), "--result-json", str(result_path),
            "-d", "iFashion", "-g", gpu, "-i", info]


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
            pass  # The worker may exit between poll() and killpg().
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
        process.wait()


def execute_experiment(plan, experiment, state, state_path, round_dir, commit, progress=None):
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = round_dir / experiment["id"] / execution_id
    directory.mkdir(parents=True)
    execution_started = time.monotonic()
    record = {
        "execution_id": execution_id, "round_id": plan["round_id"], "round_note": plan["note"],
        "experiment_id": experiment["id"], "note": experiment["note"],
        "fingerprint": experiment["fingerprint"], "source_hash": plan["source_hash"],
        "git_commit": commit, "config": experiment["config"], "gpu": plan["gpu"],
        "status": "RUNNING", "started_at": timestamp(), "attempts": [],
    }
    state["executions"].append(record)
    atomic_write_json(state_path, state)
    process = None
    try:
        for number in range(1, 4):
            config_path = directory / f"attempt_{number}_config.yaml"
            result_path = directory / f"attempt_{number}_result.json"
            stdout_path = directory / f"attempt_{number}_stdout.txt"
            info = f"tune_{execution_id}_{experiment['id']}_a{number}"
            config_path.write_text(yaml.safe_dump({"iFashion": experiment["config"]},
                                                 allow_unicode=True, sort_keys=False), encoding="utf-8")
            attempt_started = time.monotonic()
            attempt = {"number": number, "status": "RUNNING", "started_at": timestamp(),
                       "stdout": str(stdout_path), "config_snapshot": str(config_path)}
            record["attempts"].append(attempt)
            atomic_write_json(state_path, state)
            command = training_command(config_path, result_path, info, plan["gpu"])
            if progress is not None:
                progress.update(experiment, "RUNNING", f"正在运行（第 {number}/3 次尝试）")
            print(f"[{plan['round_id']}/{experiment['id']}] attempt {number}/3 -> {stdout_path}", flush=True)
            environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                               PYTHONHASHSEED=str(plan["seed"]), CUDA_VISIBLE_DEVICES=plan["gpu"])
            try:
                with stdout_path.open("w", encoding="utf-8") as output:
                    options = {"start_new_session": True} if os.name != "nt" else {
                        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                               stdout=output, stderr=subprocess.STDOUT, **options)
                    attempt["pid"] = process.pid
                    atomic_write_json(state_path, state)
                    attempt["return_code"] = process.wait()
                if attempt["return_code"] != 0:
                    raise RuntimeError(f"Training exited with code {attempt['return_code']}")
                record["result"] = validate_result(result_path, experiment["config"], info)
            except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
                if process is not None:
                    stop_process(process)
                attempt["status"] = "FAILED"
                tail = stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:] if stdout_path.exists() else ""
                attempt["error"] = str(error) + ("\nstdout tail:\n" + tail if tail else "")
                record["error"] = attempt["error"]
                print(f"  FAILED: {error}", flush=True)
                if progress is not None and number < 3:
                    progress.update(experiment, "RETRY", f"第 {number}/3 次尝试失败，等待第 {number + 1}/3 次尝试")
            else:
                attempt["status"] = "SUCCESS"
                record["status"] = "SUCCESS"
                finish_timing(record, execution_started)
                record.pop("error", None)
            finish_timing(attempt, attempt_started)
            atomic_write_json(state_path, state)
            if record["status"] == "SUCCESS":
                break
        else:
            record["status"] = "FAILED"
    except KeyboardInterrupt:
        if process is not None:
            stop_process(process)
        if record["attempts"]:
            attempt = record["attempts"][-1]
            if attempt["status"] == "RUNNING":
                attempt["status"] = "INTERRUPTED"
            if "ended_at" not in attempt:
                finish_timing(attempt, attempt_started)
        record.update(status="INTERRUPTED", error="Controller stopped; next resume starts this experiment from epoch 0")
        raise
    finally:
        finish_timing(record, execution_started)
        atomic_write_json(state_path, state)
        if record["status"] in TERMINAL:
            append_report(plan["result_file"], record)
            if progress is not None:
                progress.finished(experiment, record)
    return record["status"]


def run(plan_path, mode="resume", only=None, dry_run=False, gpu=None):
    plan = load_plan(plan_path, gpu=gpu)
    round_dir = PROJECT_ROOT / "log" / "ifashion_tuning" / plan["round_id"]
    state_path = round_dir / "state.json"
    if dry_run:
        state = load_state(state_path)
        selected = select_experiments(plan, state, mode, only or [])
        print(f"DRY RUN: {plan['round_id']} | {plan['note']} | GPU: {plan['gpu']} | selected={len(selected)}")
        for e in selected:
            c = e["config"]
            print(f"  {e['id']}: {e['note']} | c_lambdas={c['c_lambdas']} "
                  f"c_temps={c['c_temps']} anchor_cl={c['anchor_cl']['enabled']} seed={c['seed']}")
        return 0
    round_dir.mkdir(parents=True, exist_ok=True)
    with round_lock(round_dir / "controller.lock"):
        state = load_state(state_path)
        # A hard-killed controller may leave a worker alive. Never start a second
        # GPU worker until the previous PID has been checked by the operator.
        for r in state["executions"]:
            if r["status"] == "RUNNING":
                attempt = r["attempts"][-1] if r["attempts"] else None
                pid = attempt.get("pid") if attempt and attempt["status"] == "RUNNING" else None
                if pid and process_alive(pid):
                    raise RuntimeError(f"Previous worker PID {pid} is still alive; stop/check it before resuming")
                if len(r["attempts"]) == 3 and all(a["status"] == "FAILED" for a in r["attempts"]):
                    r["status"] = "FAILED"
                else:
                    r.update(status="INTERRUPTED", error="Previous controller stopped before completion")
                    if attempt and attempt["status"] == "RUNNING":
                        attempt.update(status="INTERRUPTED", ended_at=timestamp(), duration_seconds=None,
                                       timing_note="结束时间为恢复检测时间；实际停止时间及耗时无法确定。")
                r.update(ended_at=timestamp(), duration_seconds=None,
                         timing_note="结束时间为恢复检测时间；强制中断前未保存最终耗时，实际耗时无法确定。")
        atomic_write_json(state_path, state)
        for record in state["executions"]:
            if record["status"] in TERMINAL:
                append_report(plan["result_file"], record)
        selected = select_experiments(plan, state, mode, only or [])
        progress = ConsoleProgress(plan, state, selected, mode)
        progress.show()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip() or "unknown"
        statuses = [execute_experiment(plan, e, state, state_path, round_dir, commit, progress) for e in selected]
        print(f"Finished: success={statuses.count('SUCCESS')}, failed={statuses.count('FAILED')}", flush=True)
        return 1 if "FAILED" in statuses else 0


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="experiment plan YAML (relative to the current directory)")
    parser.add_argument("-g", "--gpu", default=None, help="GPU to use; overrides the plan's gpu field")
    parser.add_argument("--mode", choices=("resume", "failed", "selected", "all"), default="resume")
    parser.add_argument("--only", nargs="+", help="experiment IDs; requires --mode selected")
    parser.add_argument("--dry-run", action="store_true", help="validate and list work without writing files or launching training")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args.plan, args.mode, args.only, args.dry_run, gpu=args.gpu)
    except KeyboardInterrupt:
        print("Stopped. Resume with the same plan; the interrupted experiment restarts from epoch 0.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
