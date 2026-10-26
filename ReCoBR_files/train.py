#!/usr/bin/env python3
"""Run one ReCoBR configuration using the unchanged project model and helpers."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import uuid

from recobr_config import (MODEL_DIR, PROJECT_ROOT, RULE, normalize_config, runtime_config,
                           data_fingerprints, source_fingerprints, digest, file_hash, SINGLETONS)
from run_experiments import atomic_json, atomic_text, read_yaml, timestamp


def load_configuration(path, dataset):
    """Accept base configs and our own actual-config snapshots without silent typos."""
    raw = read_yaml(path)[dataset]
    for key, expected in (("dataset", dataset), ("model", "ReCoBR"), ("selection_rule", RULE)):
        if key in raw and raw[key] != expected:
            raise ValueError(f"Snapshot identity/protocol mismatch: {key}")
    for plural, single in SINGLETONS.items():
        if single in raw and raw.get(plural) != [raw[single]]:
            raise ValueError(f"Snapshot scalar disagrees with singleton setting: {single}")
    runtime = set(SINGLETONS.values()) | {"num_users", "num_bundles", "num_items", "dataset", "model",
                                        "gpu", "device", "info", "num_workers", "selection_rule", "quiet"}
    return normalize_config({key: value for key, value in raw.items() if key not in runtime}, dataset)


def reference_training():
    sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("recobr_training_reference", PROJECT_ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(reference, conf):
    dataset = reference.Datasets(conf)
    # Keep the root dataset/samplers/drop_last semantics. Explicit workers=0
    # makes Windows execution and negative-sampling random streams predictable.
    from torch.utils.data import DataLoader
    for name, data, batch, shuffle, drop in (
            ("train_loader", dataset.bundle_train_data, conf["batch_size_train"], True, True),
            ("val_loader", dataset.bundle_val_data, conf["batch_size_test"], False, False),
            ("test_loader", dataset.bundle_test_data, conf["batch_size_test"], False, False)):
        setattr(dataset, name, DataLoader(data, batch_size=batch, shuffle=shuffle,
                                        num_workers=conf["num_workers"], drop_last=drop))
    if not len(dataset.train_loader):
        raise ValueError("No complete training batch")
    return dataset


def train_one(reference, conf, dataset, artifacts, log, record):
    """Root run_recobr_training's scientific order, with isolated persistence."""
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    model = reference.ReCoBR(conf, dataset.graphs).to(conf["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["l2_reg"])
    rebuilder, diffusion_optimizer = reference.create_recobr_latent_rebuild_components(conf, dataset, conf["device"])
    rebuild_conf = conf["latent_rebuild"]
    best_val = {"recall": {20: 0.0}, "ndcg": {20: 0.0}}
    best = None
    batch_count = len(dataset.train_loader)
    with SummaryWriter(artifacts["tensorboard"]) as writer:
        for epoch in range(conf["epochs"]):
            start = time.monotonic()
            if (rebuild_conf["latent_diffusion_reset_after_epoch"] >= 0
                    and epoch == rebuild_conf["latent_diffusion_reset_after_epoch"] + 1):
                rebuilder, diffusion_optimizer = reference.reset_latent_diffusion_rebuilder(
                    rebuild_conf, rebuilder, diffusion_optimizer)
            diffusion_loss = reference.train_latent_diffusion_epoch(
                rebuilder, diffusion_optimizer, dataset, rebuild_conf, conf["device"])
            graph = reference.rebuild_ub_graph_with_latent_diffusion(
                rebuilder, dataset, rebuild_conf, conf["device"])
            if not math.isfinite(diffusion_loss):
                raise FloatingPointError("Non-finite diffusion loss")
            model.set_ub_graph(graph)
            sums = [0.0] * 4
            model.train(True)
            for index, batch in enumerate(tqdm(dataset.train_loader, desc=f"Epoch {epoch+1}/{conf['epochs']}",
                                               disable=conf.get("quiet", False))):
                model.train(True)
                optimizer.zero_grad()
                batch = [value.to(conf["device"]) for value in batch]
                step = epoch * batch_count + index
                ed_drop = conf["aug_type"] == "ED" and (step + 1) % (batch_count * conf["ed_interval"]) == 0
                bpr, contrastive, anchor = model(batch, ED_drop=ed_drop)
                loss = bpr + conf["c_lambda"] * contrastive + conf["anchor_cl"]["anchor_cl_lambda"] * anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite recommendation loss")
                loss.backward()
                optimizer.step()
                for i, (name, value) in enumerate(zip(("loss", "loss_bpr", "loss_c", "loss_anchor_cl"),
                                                      (loss, bpr, contrastive, anchor))):
                    scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                    sums[i] += scalar
                    writer.add_scalar(name, scalar, step)
            record["completed_epochs"] = epoch + 1
            writer.add_scalar("latent_rebuild/loss_epoch", diffusion_loss, epoch)
            writer.add_scalar("latent_rebuild/edges", graph.nnz, epoch)
            log(f"Epoch {epoch+1}/{conf['epochs']} losses={[v/batch_count for v in sums]} "
                f"diffusion={diffusion_loss:.8f} rebuilt_edges={graph.nnz} seconds={time.monotonic()-start:.3f}")
            if (epoch + 1) % conf["test_interval"]:
                continue
            metrics = {"val": reference.test(model, dataset.val_loader, conf),
                       "test": reference.test(model, dataset.test_loader, conf)}
            for split, split_metrics in metrics.items():
                for name, values in split_metrics.items():
                    for k, value in values.items():
                        if not math.isfinite(float(value)):
                            raise FloatingPointError("Non-finite evaluation metric")
                        writer.add_scalar(f"{split}/{name}@{k}", value, epoch + 1)
            log(f"Epoch {epoch+1} metrics=" + json.dumps(metrics, sort_keys=True))
            if reference.is_better_validation(metrics["val"], best_val):
                best_val = metrics["val"]
                best = {"best_epoch": epoch, "best_epoch_1based": epoch + 1, "selection_rule": RULE,
                        "selection_metrics": {"recall@20": float(best_val["recall"][20]),
                                              "ndcg@20": float(best_val["ndcg"][20])}, "metrics": metrics}
                # The reconstructed UB graph is not a registered model buffer.
                # Persist it with the state so re-evaluation uses this checkpoint's graph.
                payload = {"config": conf, "result": best, "model_state": model.state_dict(),
                           "ub_graph": graph, "optimizer_state": optimizer.state_dict(),
                           "diffusion_state": rebuilder.state_dict(),
                           "diffusion_optimizer_state": diffusion_optimizer.state_dict(),
                           "data_files": record["data_files"], "sources": record["sources"]}
                temporary = Path(artifacts["checkpoint_model"] + ".tmp")
                try:
                    torch.save(payload, temporary)
                    os.replace(temporary, artifacts["checkpoint_model"])
                finally:
                    temporary.unlink(missing_ok=True)
                atomic_json(artifacts["checkpoint_config"], conf)
                log(f"Saved best checkpoint at epoch {epoch+1}")
    if best is None:
        raise RuntimeError("No checkpoint passed the strict validation rule")
    return best


def evaluate_checkpoint(reference, path, conf, dataset, log, inputs):
    import torch
    checkpoint = torch.load(path, map_location=conf["device"], weights_only=False)
    if checkpoint["data_files"] != inputs:
        raise ValueError("Checkpoint dataset/pretrained embeddings differ")
    from recobr_config import SCIENTIFIC_SOURCES
    current = source_fingerprints()
    if any(checkpoint["sources"].get(key) != current[key] for key in SCIENTIFIC_SOURCES):
        raise ValueError("Checkpoint training implementation differs")
    # Permit only device, loader and path relocation during re-evaluation.
    excluded = {"device", "gpu", "info", "quiet", "num_workers", "data_path"}
    for key in set(conf) - excluded:
        if key == "latent_rebuild":
            from recobr_config import EMBEDDING_KEYS
            old = {k: v for k, v in checkpoint["config"][key].items() if k not in EMBEDDING_KEYS}
            new = {k: v for k, v in conf[key].items() if k not in EMBEDDING_KEYS}
            if old != new:
                raise ValueError("Checkpoint diffusion configuration differs")
        elif checkpoint["config"].get(key) != conf[key]:
            raise ValueError(f"Checkpoint configuration differs: {key}")
    model = reference.ReCoBR(conf, dataset.graphs).to(conf["device"])
    model.set_ub_graph(checkpoint["ub_graph"])
    model.load_state_dict(checkpoint["model_state"])
    metrics = {"val": reference.test(model, dataset.val_loader, conf),
               "test": reference.test(model, dataset.test_loader, conf)}
    log("Re-evaluated metrics=" + json.dumps(metrics, sort_keys=True))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.yaml")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--experiment-case", choices=("second",), default="second")
    parser.add_argument("--info", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record-file", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9]+", args.gpu) or args.num_workers < 0:
        parser.error("Specify one numeric GPU ID and nonnegative num-workers")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.dataset):
        parser.error("Invalid dataset identifier")
    conf = load_configuration(args.config, args.dataset)
    # No torch/CUDA import or artifact/state writes during configuration checks.
    if args.check_config:
        print(json.dumps({"dataset": args.dataset, "config": conf, "selection_rule": RULE}, indent=2))
        print("Configuration, dataset and embedding paths validated.")
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTHONHASHSEED"] = str(conf["seed"])
    import torch
    reference = reference_training()
    device = "cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and device == "cpu":
        raise RuntimeError("CUDA requested but unavailable")
    conf = runtime_config(conf, args.dataset, args.gpu, device, args.info, args.num_workers)
    conf["quiet"] = args.quiet
    reference.set_seed(conf["seed"])
    execution = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    output = (args.output_dir or MODEL_DIR / "outputs" / args.dataset / execution).resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {key: str(output / name) for key, name in {
        "checkpoint_model": "best.pt", "checkpoint_config": "best_config.json",
        "config": "config.yaml", "result_json": "result.json", "log": "training.log",
        "tensorboard": "tensorboard"}.items()}

    def log(message):
        line = f"[{timestamp()}] {message}"
        print(line, flush=True)
        with Path(artifacts["log"]).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    import yaml
    atomic_text(artifacts["config"], yaml.safe_dump({args.dataset: conf}, allow_unicode=True, sort_keys=False))
    record = {"execution_id": execution, "experiment_case": "second", "dataset": args.dataset,
              "status": "RUNNING", "completed": False, "completed_epochs": 0, "config": conf,
              "config_hash": digest(conf), "sources": source_fingerprints(), "gpu": args.gpu,
              "device": device, "data_files": data_fingerprints(conf, args.dataset),
              "note": args.note, "started_at": timestamp(), "artifacts": artifacts}
    import platform
    record["environment"] = {"python": sys.version, "platform": platform.platform(),
                             "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                             "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                             "numpy": reference.np.__version__, "scipy": __import__("scipy").__version__,
                             "cudnn_deterministic": torch.backends.cudnn.deterministic,
                             "cudnn_benchmark": torch.backends.cudnn.benchmark,
                             "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    record["source_hash"] = record["sources"]["train.py"]
    start = time.monotonic()
    code = 1
    try:
        log("Configuration=" + json.dumps(conf, sort_keys=True))
        dataset = make_dataset(reference, conf)
        log("Dataset=" + json.dumps({"UB_train": reference.summarize_sparse_graph(dataset.graphs[0]),
                                      "UI": reference.summarize_sparse_graph(dataset.graphs[1]),
                                      "BI": reference.summarize_sparse_graph(dataset.graphs[2])}))
        log("Checkpoint selection: " + RULE)
        if args.evaluate:
            record["evaluation_metrics"] = evaluate_checkpoint(reference, args.evaluate, conf, dataset, log,
                                                                 record["data_files"])
            record["status"] = "EVALUATED"
        else:
            record["result"] = train_one(reference, conf, dataset, artifacts, log, record)
            record["status"] = "SUCCESS"
        if source_fingerprints() != record["sources"] or data_fingerprints(conf, args.dataset) != record["data_files"]:
            raise RuntimeError("Source, dataset or pretrained embedding changed while running")
        record["completed"] = True
        code = 0
    except KeyboardInterrupt:
        record.update(status="INTERRUPTED", error="Training interrupted")
        code = 130
    except Exception:
        record.update(status="FAILED", error=traceback.format_exc())
        log(record["error"])
    finally:
        record.update(ended_at=timestamp(), duration_seconds=round(time.monotonic() - start, 3))
        record["artifact_hashes"] = {key: file_hash(path) for key, path in artifacts.items()
                                     if key in {"checkpoint_model", "checkpoint_config", "config"} and Path(path).is_file()}
        atomic_json(artifacts["result_json"], record)
        if args.result_json:
            atomic_json(args.result_json, record)
        if args.record_file:
            atomic_text(args.record_file, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        log(f"Finished: {record['status']} | {record['duration_seconds']} seconds | {output}")
    return code


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
