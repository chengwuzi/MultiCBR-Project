#!/usr/bin/env python3
"""Single-run MultiCBR reproduction, shared evaluation and read-only replay."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import yaml

if __package__:
    from .configuration import (MODEL_DIR, PROJECT_ROOT, BASE_KEYS, IMPLEMENTATION, SELECTION_RULE,
                                digest, file_hash, load_config, project_path, runtime_config,
                                source_fingerprints, validate)
else:
    from configuration import (MODEL_DIR, PROJECT_ROOT, BASE_KEYS, IMPLEMENTATION, SELECTION_RULE,
                               digest, file_hash, load_config, project_path, runtime_config,
                               source_fingerprints, validate)


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def components():
    if __package__:
        from .data import Datasets
        from .model import MultiCBR
    else:
        from data import Datasets
        from model import MultiCBR
    return Datasets, MultiCBR


def is_better_validation(current, saved):
    return (current["recall"][20] > saved["recall"][20]
            and current["ndcg"][20] > saved["ndcg"][20])


def get_recall(pred, grd, hit, k):
    num_pos = grd.sum(dim=1)
    return [(hit.sum(dim=1) / (num_pos + 1e-8)).sum().item(),
            pred.shape[0] - (num_pos == 0).sum().item()]


def get_ndcg(pred, grd, hit, k):
    import torch
    def dcg(values):
        return (values / torch.log2(torch.arange(2, k + 2, device=grd.device,
                                                 dtype=torch.float))).sum(-1)
    ideals = torch.empty(k + 1, dtype=torch.float)
    ideals[0] = 1
    for count in range(1, k + 1):
        ideal_hit = torch.zeros(k, dtype=torch.float)
        ideal_hit[:count] = 1
        ideals[count] = dcg(ideal_hit)
    num_pos = grd.sum(dim=1).clamp(0, k).long()
    ndcg = dcg(hit) / ideals[num_pos].to(grd.device)
    return [ndcg.sum().item(), pred.shape[0] - (num_pos == 0).sum().item()]


def get_metrics(metrics, grd, pred, topks):
    import torch
    # Separate topk calls match ReCoBR even when scores tie.
    for k in topks:
        columns = torch.topk(pred, k)[1]
        rows = torch.zeros_like(columns) + torch.arange(pred.shape[0], device=pred.device).view(-1, 1)
        hit = grd[rows.flatten().cpu(), columns.flatten().cpu()].view(-1, k)
        for name, values in (("recall", get_recall(pred, grd, hit, k)),
                             ("ndcg", get_ndcg(pred, grd, hit, k))):
            for index, value in enumerate(values):
                metrics[name][k][index] += value
    return metrics


def evaluate(model, loader, conf):
    import torch
    model.eval()
    totals = {name: {k: [0, 0] for k in conf["topk"]} for name in ("recall", "ndcg")}
    with torch.no_grad():
        representations = model.get_multi_modal_representations(test=True)
        for users, truth, original_train in loader:
            scores = model.evaluate(representations, users.to(conf["device"]))
            scores -= 1e8 * original_train.to(conf["device"])
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Non-finite evaluation scores")
            get_metrics(totals, truth, scores, conf["topk"])
    if any(pair[1] == 0 for metric in totals.values() for pair in metric.values()):
        raise ValueError("Evaluation split has no users with positives")
    return {name: {k: pair[0] / pair[1] for k, pair in values.items()}
            for name, values in totals.items()}


def train_epoch(model, optimizer, loader, conf, epoch, writer):
    import torch
    from tqdm import tqdm
    sums = np.zeros(3, dtype=np.float64)
    model.train()
    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{conf['epochs']}", disable=conf["quiet"])
    for index, batch in enumerate(progress):
        optimizer.zero_grad()
        step = epoch * len(loader) + index
        # Preserve author's pre-forward ED refresh schedule, including the last batch.
        ed_drop = conf["aug_type"] == "ED" and (step + 1) % (len(loader) * conf["ed_interval"]) == 0
        bpr, contrastive = model([value.to(conf["device"]) for value in batch], ED_drop=ed_drop)
        loss = bpr + conf["c_lambda"] * contrastive
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        values = [loss.item(), bpr.item(), contrastive.item()]
        sums += values
        for name, value in zip(("loss", "loss_bpr", "loss_c_unweighted"), values):
            writer.add_scalar(name, value, step)
        progress.set_postfix(loss=f"{values[0]:.6f}")
    return dict(zip(("loss", "bpr", "contrastive_unweighted"), (sums / len(loader)).tolist()))


@contextmanager
def record_lock(path):
    """Serialize complete TXT blocks if separate dataset consoles finish together."""
    lock = project_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if lock.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def append_record(path, record):
    # The canonical JSON block preserves full-precision metrics and complete provenance.
    lines = ["", "=" * 72, "EXECUTION_ID: " + record["execution_id"],
             "实验类别: 第一种情况（作者已有数据集配置）", "模型: MultiCBR",
             "数据集: " + record["dataset"], "状态: " + record["status"],
             "开始时间(UTC+08:00): " + record["started_at"],
             "结束时间(UTC+08:00): " + record["ended_at"],
             f"耗时（秒）: {record['duration_seconds']:.3f}", "实验备注: " + record["note"],
             "选模规则: " + SELECTION_RULE]
    best = record.get("result")
    if best is not None:
        lines += [f"best_epoch: {best['best_epoch']}（从0计数）",
                  f"最佳训练轮次: {best['best_epoch_1based']}"]
        for split in ("val", "test"):
            for k in record["config"]["topk"]:
                metrics = best["metrics"][split]
                lines.append(f"{split} TOP {k}: Recall={metrics['recall'][str(k)]:.8f}, "
                             f"NDCG={metrics['ndcg'][str(k)]:.8f}")
    lines += ["完整运行记录(JSON，包含完整实际配置与未舍入指标):",
              json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
              "END_EXECUTION: " + record["execution_id"], ""]
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with record_lock(str(path) + ".lock"):
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines))
            stream.flush()
            os.fsync(stream.fileno())


def run_training(base, args, device):
    import platform
    import torch
    from torch.utils.tensorboard import SummaryWriter
    Datasets, Model = components()
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = project_path(args.output_dir) / args.dataset / execution_id
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {name: str(directory / filename) for name, filename in {
        "checkpoint_model": "best.pt", "checkpoint_config": "best_config.json",
        "config": "config.yaml", "result_json": "result.json", "log": "training.log",
        "tensorboard": "tensorboard", "sources": "sources"}.items()}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    record = dict(execution_id=execution_id, dataset=args.dataset, model="MultiCBR", experiment_case="first",
                  implementation=IMPLEMENTATION, status="RUNNING", completed=False, completed_epochs=0,
                  started_at=timestamp(), note=args.note, gpu=args.gpu, device=str(device), git_commit=commit,
                  config=copy.deepcopy(base), sources=source_fingerprints(), data_files=None,
                  input_config_sha256=file_hash(args.config), artifacts=artifacts, selection_rule=SELECTION_RULE)
    record["environment"] = dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                                  numpy=np.__version__, scipy=__import__("scipy").__version__,
                                  cuda_runtime=torch.version.cuda,
                                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                  pythonhashseed=os.environ.get("PYTHONHASHSEED"))
    start = time.monotonic()
    writer = None
    log_stream = Path(artifacts["log"]).open("a", encoding="utf-8")

    def log(message):
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        log_stream.write(line + "\n")
        log_stream.flush()

    try:
        seed_everything(base["seed"])
        source_directory = Path(artifacts["sources"])
        source_directory.mkdir()
        for name, expected in record["sources"].items():
            payload = (MODEL_DIR / name).read_bytes()
            (source_directory / name).write_bytes(payload)
            if file_hash(source_directory / name) != expected:
                raise RuntimeError("Source changed before snapshot creation")
        dataset = Datasets(base, args.dataset)
        conf = runtime_config(base, args.dataset, dataset.sizes, args.gpu, device, args.quiet, args.info)
        record.update(config=conf, config_hash=digest(conf), data_files=dataset.data_files,
                      data_statistics=dataset.statistics)
        Path(artifacts["config"]).write_text(yaml.safe_dump({args.dataset: conf}, allow_unicode=True,
                                                            sort_keys=False), encoding="utf-8")
        atomic_json(artifacts["result_json"], record)
        log("Actual configuration=" + json.dumps(conf, sort_keys=True))
        log("Data=" + json.dumps(dataset.statistics))
        log("Checkpoint selection: " + SELECTION_RULE)
        model = Model(conf, dataset.graphs).to(device)
        # Author uses Adam weight_decay; no additional explicit L2 term is added.
        optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["l2_reg"])
        writer = SummaryWriter(artifacts["tensorboard"])
        saved = {"recall": {20: 0.0}, "ndcg": {20: 0.0}}
        for epoch in range(conf["epochs"]):
            epoch_start = time.monotonic()
            losses = train_epoch(model, optimizer, dataset.train_loader, conf, epoch, writer)
            record.update(completed_epochs=epoch + 1, last_losses=losses)
            log(f"Epoch {epoch+1}/{conf['epochs']} losses={losses} seconds={time.monotonic()-epoch_start:.3f}")
            if (epoch + 1) % conf["test_interval"]:
                continue
            metrics = {"val": evaluate(model, dataset.val_loader, conf),
                       "test": evaluate(model, dataset.test_loader, conf)}
            for split, values in metrics.items():
                for name, ks in values.items():
                    for k, value in ks.items():
                        writer.add_scalar(f"{split}/{name}@{k}", value, epoch + 1)
            log(f"Epoch {epoch+1} metrics=" + json.dumps(metrics))
            if is_better_validation(metrics["val"], saved):
                saved = copy.deepcopy(metrics["val"])
                best = dict(best_epoch=epoch, best_epoch_1based=epoch + 1, selection_rule=SELECTION_RULE,
                            selection_metrics={"recall@20": saved["recall"][20], "ndcg@20": saved["ndcg"][20]},
                            metrics=json.loads(json.dumps(metrics)))
                checkpoint = dict(implementation=IMPLEMENTATION, config=conf, result=best,
                                  state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                  data_files=dataset.data_files, sources=record["sources"], execution_id=execution_id)
                temporary = directory / "best.pt.tmp"
                try:
                    torch.save(checkpoint, temporary)
                    os.replace(temporary, artifacts["checkpoint_model"])
                finally:
                    temporary.unlink(missing_ok=True)
                atomic_json(artifacts["checkpoint_config"], conf)
                record["result"] = best
                log(f"Saved best checkpoint at epoch {epoch+1}")
            atomic_json(artifacts["result_json"], record)
            writer.flush()
        if "result" not in record:
            raise RuntimeError("No checkpoint passed the strict validation rule")
        if source_fingerprints() != record["sources"]:
            raise RuntimeError("Training source changed during this execution")
        directory_data = Path(base["data_path"]) / args.dataset
        if any(file_hash(directory_data / name) != expected for name, expected in dataset.data_files.items()):
            raise RuntimeError("Data changed during this execution")
        record.update(status="SUCCESS", completed=True)
        return record
    except BaseException as error:
        record.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                      completed=False, error=traceback.format_exc())
        log(record["error"])
        raise
    finally:
        record.update(ended_at=timestamp(), duration_seconds=round(time.monotonic() - start, 3))
        record["artifact_hashes"] = {key: file_hash(artifacts[key]) for key in
                                    ("checkpoint_model", "checkpoint_config", "config")
                                    if Path(artifacts[key]).is_file()}
        # Always attempt each finalization step, preserving a training exception.
        errors = []
        for action in (lambda: writer.close() if writer is not None else None,
                       lambda: atomic_json(artifacts["result_json"], record),
                       lambda: atomic_json(project_path(args.result_json), record) if args.result_json else None,
                       lambda: append_record(args.record_file or MODEL_DIR / "实验记录.txt", record)):
            try:
                action()
            except Exception as error:
                errors.append(error)
                log(f"Artifact finalization failed: {error}")
        log(f"Finished {record['status']} | {directory}")
        log_stream.close()
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError("Failed to finalize experiment artifacts") from errors[0]


def evaluate_checkpoint(path, args, device):
    import torch
    Datasets, Model = components()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        # Old author-era PyTorch predates weights_only. Replay only checkpoints
        # produced by this runner; current versions use restricted loading above.
        if "weights_only" not in str(error):
            raise
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("implementation") != IMPLEMENTATION:
        raise ValueError("Incompatible checkpoint implementation")
    if checkpoint.get("sources") != source_fingerprints():
        raise ValueError("Checkpoint source differs from the current training implementation")
    saved = checkpoint["config"]
    base = {key: copy.deepcopy(saved[key]) for key in BASE_KEYS}
    for key in ("data_path", "num_workers", "batch_size_test"):
        value = getattr(args, key)
        if value is not None:
            base[key] = value
    validate(base)
    dataset = Datasets(base, saved["dataset"])
    if dataset.data_files != checkpoint["data_files"]:
        raise ValueError("Dataset content differs from the checkpoint's fingerprints")
    if tuple(saved[key] for key in ("num_users", "num_bundles", "num_items")) != dataset.sizes:
        raise ValueError("Checkpoint entity counts differ")
    conf = runtime_config(base, saved["dataset"], dataset.sizes, args.gpu, device)
    seed_everything(conf["seed"])
    model = Model(conf, dataset.graphs).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    result = dict(checkpoint=str(Path(path).resolve()), execution_id=checkpoint["execution_id"],
                  best_epoch=checkpoint["result"]["best_epoch"],
                  best_epoch_1based=checkpoint["result"]["best_epoch_1based"], selection_rule=SELECTION_RULE,
                  metrics={"val": evaluate(model, dataset.val_loader, conf),
                           "test": evaluate(model, dataset.test_loader, conf)})
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.yaml")
    parser.add_argument("-d", "--dataset", default="NetEase")
    parser.add_argument("-m", "--model", choices=["MultiCBR"], default="MultiCBR")
    parser.add_argument("-g", "--gpu", default="0", help="One physical GPU ID; uses logical cuda:0")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("--data-path")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size-test", type=int)
    parser.add_argument("-i", "--info", default="")
    parser.add_argument("--note", default="Author dataset configuration under the shared evaluation protocol")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default=str(MODEL_DIR / "log"))
    parser.add_argument("--record-file")
    parser.add_argument("--result-json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Read-only full configuration/data validation")
    mode.add_argument("--evaluate", type=Path, metavar="CHECKPOINT", help="Read-only best-checkpoint evaluation")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if not re.fullmatch(r"[0-9]+", args.gpu):
        raise ValueError("Specify one nonnegative physical GPU ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.evaluate:
        overrides = {key: getattr(args, key) for key in
                     ("seed", "epochs", "data_path", "num_workers", "batch_size_test")}
        base = load_config(args.config, args.dataset, overrides)
        if args.check_config:
            Datasets, _ = components()
            dataset = Datasets(base, args.dataset)
            print(json.dumps(dict(config=base, dataset=args.dataset, data_statistics=dataset.statistics,
                                  data_files=dataset.data_files, selection_rule=SELECTION_RULE), indent=2))
            print("Configuration and original data validated; no training or output writes.")
            return 0
        os.environ["PYTHONHASHSEED"] = str(base["seed"])
    elif args.seed is not None or args.epochs is not None:
        raise ValueError("Checkpoint evaluation does not accept seed/epochs overrides")
    import torch
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if args.evaluate:
        evaluate_checkpoint(args.evaluate, args, device)
    else:
        run_training(base, args, device)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Training interrupted; status and available artifacts were saved.", file=sys.stderr)
        sys.exit(130)
