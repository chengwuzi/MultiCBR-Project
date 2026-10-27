"""DCDL single-run training and full-ranking evaluation; no tuning controller."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import uuid

import numpy as np
import yaml

if __package__:
    from .configuration import IMPLEMENTATION, MODEL_DIR, SELECTION_RULE, load_config, project_path, validate
else:
    from configuration import IMPLEMENTATION, MODEL_DIR, SELECTION_RULE, load_config, project_path, validate


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def improves(candidate, saved):
    return candidate["recall@20"] > saved["recall@20"] and candidate["ndcg@20"] > saved["ndcg@20"]


def worker_seed(worker_id):
    import torch
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Sparse CUDA scatter/reduction can remain nondeterministic; see README.


def modules():
    if __package__:
        from .data import Relations, TrainPairs, build_graphs, file_hash
        from .model import DCDL
    else:
        from data import Relations, TrainPairs, build_graphs, file_hash
        from model import DCDL
    return Relations, TrainPairs, build_graphs, file_hash, DCDL


def evaluate(model, relations, conf, split, representations=None):
    """Macro metrics on positive users; full candidates, original train mask only."""
    import torch
    truth = getattr(relations, split)
    if truth.nnz == 0:
        raise ValueError(f"{split} has no positives")
    model.eval()
    with torch.no_grad():
        if representations is None:
            representations = model.encode()[:2]
        users, bundles = representations
        device = users.device
        sums = {f"{metric}@{k}": 0.0 for k in conf["topk"] for metric in ("recall", "ndcg")}
        count = 0
        # Match root train.py arithmetic, including a separate topk call for each K
        # (prefixes of a larger topk may order equal scores differently).
        discounts, ideals = {}, {}
        for k in conf["topk"]:
            discount = torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
            ideal = torch.empty(k + 1, dtype=torch.float32)
            ideal[0] = 1
            for n in range(1, k + 1):
                hits = torch.zeros(k, dtype=torch.float32)
                hits[:n] = 1
                ideal[n] = (hits / discount).sum()
            discounts[k], ideals[k] = discount, ideal
        for start in range(0, relations.num_users, conf["batch_size_test"]):
            stop = min(start + conf["batch_size_test"], relations.num_users)
            scores = users[start:stop] @ bundles.T
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite full-ranking scores")
            mask = torch.from_numpy(relations.ub[start:stop].toarray()).to(device)
            scores = scores - 1e8 * mask
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite masked full-ranking scores")
            target = torch.from_numpy(truth[start:stop].toarray())
            num_positive = target.sum(1)
            valid = num_positive > 0
            count += int(valid.sum().item())
            for k in conf["topk"]:
                ranked = scores.topk(k, dim=1).indices.cpu()
                hits = target.gather(1, ranked)
                recall = hits.sum(1) / (num_positive + 1e-8)
                idcg = ideals[k][num_positive.clamp(max=k).long()]
                ndcg = (hits / discounts[k]).sum(1) / idcg
                sums[f"recall@{k}"] += recall.sum().item()
                sums[f"ndcg@{k}"] += ndcg.sum().item()
        if not count:
            raise ValueError(f"{split} has no users with positives")
        return {key: value / count for key, value in sums.items()}


def evaluate_both(model, relations, conf):
    import torch
    model.eval()
    with torch.no_grad():
        representations = model.encode()[:2]
        return {split: evaluate(model, relations, conf, split, representations) for split in ("val", "test")}


def source_fingerprints(file_hash):
    return {name: file_hash(MODEL_DIR / name) for name in
            ("configuration.py", "data.py", "model.py", "train.py", "config.yaml")}


def append_record(path, result):
    """One complete JSON object per run in an append-only UTF-8 TXT record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    block = (f"\n{'=' * 72}\nDCDL 单次实验 | {result['execution_id']} | {result['status']}\n"
             "配置性质：论文重建的工程配置，未经调参，不代表作者最优参数。\n"
             + json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block)


def run_training(conf, args, device):
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    Relations, TrainPairs, build_graphs, file_hash, Model = modules()
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = project_path(args.output_dir) / conf["dataset"] / execution_id
    directory.mkdir(parents=True, exist_ok=False)
    record_path = project_path(args.record_file) if args.record_file else MODEL_DIR / "实验记录.txt"
    logger = logging.getLogger("DCDL." + execution_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(directory / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    if not args.quiet:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream)
    result = dict(
        implementation=IMPLEMENTATION, configuration_kind="engineering_initial_not_author_best",
        execution_id=execution_id, status="RUNNING", started_at=now(), config=conf,
        selection_rule=SELECTION_RULE, completed_epochs=0, best_epoch=None,
        best_epoch_1based=None, selection_metrics=None, metrics=None,
        output_dir=str(directory), checkpoint=None, note=args.note,
        runtime=dict(device=str(device), torch=str(torch.__version__), numpy=np.__version__,
                     cuda=torch.version.cuda, gpu=(torch.cuda.get_device_name(device) if device.type == "cuda" else None)),
        source_files=source_fingerprints(file_hash), data_files=None,
        command=[sys.executable] + sys.argv,
    )
    writer = None
    started = time.monotonic()
    try:
        seed_everything(conf["seed"])
        (directory / "config.yaml").write_text(yaml.safe_dump({conf["dataset"]: conf}, sort_keys=False), encoding="utf-8")
        write_json(directory / "result.json", result)
        logger.info("Execution %s | device=%s | %s", execution_id, device, SELECTION_RULE)
        data = Relations(conf)
        result.update(data_files=data.fingerprints, data_statistics=data.statistics)
        logger.info("Data: %s", json.dumps(data.statistics))
        graphs = build_graphs(data, conf, logger.info)
        sizes = (data.num_users, data.num_bundles, data.num_items)
        model = Model(conf, sizes, graphs, len(data.pairs)).to(device)
        result["num_parameters"] = sum(p.numel() for p in model.parameters())
        result["relationship_norm"] = float(model.relation_norm.item())
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf["lr"], weight_decay=conf["adamw_weight_decay"])
        generator = torch.Generator().manual_seed(conf["seed"])
        loader = DataLoader(TrainPairs(data), batch_size=conf["batch_size_train"], shuffle=True,
                            num_workers=conf["num_workers"], drop_last=False,
                            worker_init_fn=worker_seed, generator=generator, pin_memory=device.type == "cuda")
        writer = SummaryWriter(str(directory / "tensorboard"))
        saved = {"recall@20": 0.0, "ndcg@20": 0.0}
        for epoch in range(conf["epochs"]):
            model.train()
            # Algorithm 1: assign a uniform t and Gaussian noise to EVERY user each epoch.
            steps = torch.randint(1, conf["diffusion_steps"] + 1, (data.num_users,), device=device)
            noise = torch.randn(data.num_users, conf["embedding_size"], device=device)
            sums = {"total": 0.0, "uib": 0.0, "diffusion": 0.0, "l2": 0.0}
            examples = 0
            for batch_index, (users, positive, negative) in enumerate(loader, start=1):
                users, positive, negative = [tensor.to(device, non_blocking=True) for tensor in (users, positive, negative)]
                optimizer.zero_grad(set_to_none=True)
                loss, terms = model(users, positive, negative, steps, noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
                optimizer.step()
                count = len(users)
                examples += count
                sums["total"] += loss.item() * count
                for name, value in terms.items():
                    sums[name] += value.item() * count
                if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                    logger.info("Epoch %d/%d batch %d/%d loss=%.8g", epoch + 1,
                                conf["epochs"], batch_index, len(loader), loss.item())
            result["completed_epochs"] = epoch + 1
            losses = {name: value / examples for name, value in sums.items()}
            result["last_losses"] = losses
            for name, value in losses.items():
                writer.add_scalar("loss/" + name, value, epoch + 1)
            logger.info("Epoch %d/%d losses=%s", epoch + 1, conf["epochs"], json.dumps(losses))
            if (epoch + 1) % conf["test_interval"]:
                continue
            metrics = evaluate_both(model, data, conf)
            for split, values in metrics.items():
                for name, value in values.items():
                    writer.add_scalar(split + "/" + name, value, epoch + 1)
            logger.info("Epoch %d metrics=%s", epoch + 1, json.dumps(metrics))
            if improves(metrics["val"], saved):
                saved = {key: metrics["val"][key] for key in saved}
                result.update(best_epoch=epoch, best_epoch_1based=epoch + 1,
                              selection_metrics=dict(saved), metrics=metrics,
                              checkpoint=str(directory / "best.pt"))
                checkpoint = dict(implementation=IMPLEMENTATION, config=conf, sizes=list(sizes),
                                  num_train=len(data.pairs), graphs=model.graph_state(),
                                  state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                                  data_files=data.fingerprints, source_files=result["source_files"],
                                  best_epoch=epoch, metrics=metrics)
                temporary = directory / "best.pt.tmp"
                torch.save(checkpoint, temporary)
                os.replace(temporary, directory / "best.pt")
                write_json(directory / "best_config.json", dict(config=conf, best_epoch=epoch,
                           best_epoch_1based=epoch + 1, selection_rule=SELECTION_RULE, metrics=metrics,
                           implementation=IMPLEMENTATION, execution_id=execution_id))
                logger.info("Saved best checkpoint at epoch %d", epoch + 1)
            write_json(directory / "result.json", result)
            writer.flush()
        if result["best_epoch"] is None:
            raise RuntimeError("Neither metric pair strictly exceeded the initial (0, 0); no best checkpoint was selected")
        result["status"] = "SUCCESS"
        return result
    except KeyboardInterrupt:
        result["status"] = "INTERRUPTED"
        result["error"] = "KeyboardInterrupt"
        logger.warning("Training interrupted")
        raise
    except Exception:
        result["status"] = "FAILED"
        result["error"] = traceback.format_exc()
        logger.exception("Training failed")
        raise
    finally:
        result.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        # Preserve the training exception if final output also encounters an IO error.
        output_errors = []
        for action in (
            lambda: write_json(directory / "result.json", result),
            lambda: append_record(record_path, result),
            lambda: write_json(project_path(args.result_json), result) if args.result_json else None,
            lambda: writer.close() if writer else None,
        ):
            try:
                action()
            except Exception as error:
                output_errors.append(error)
                logger.exception("Failed to finalize an artifact")
        logger.info("Status=%s | result=%s", result["status"], directory / "result.json")
        for current in list(logger.handlers):
            current.close()
            logger.removeHandler(current)
        if output_errors and sys.exc_info()[0] is None:
            raise RuntimeError(f"Failed to finalize {len(output_errors)} artifact(s)") from output_errors[0]


def evaluate_checkpoint(path, args, device):
    import torch
    Relations, _, _, file_hash, Model = modules()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("implementation") != IMPLEMENTATION:
        raise ValueError("Checkpoint implementation is incompatible")
    conf = dict(checkpoint["config"])
    if args.data_path:
        conf["data_path"] = str(project_path(args.data_path))
    if args.batch_size_test is not None:
        conf["batch_size_test"] = args.batch_size_test
    validate(conf)
    data = Relations(conf)
    if data.fingerprints != checkpoint["data_files"]:
        raise ValueError("Dataset contents differ from the saved checkpoint")
    sizes = [data.num_users, data.num_bundles, data.num_items]
    if sizes != checkpoint["sizes"] or len(data.pairs) != checkpoint["num_train"]:
        raise ValueError("Checkpoint entity counts or training size differ")
    seed_everything(conf["seed"])
    model = Model(conf, sizes, checkpoint["graphs"], checkpoint["num_train"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    result = dict(checkpoint=str(Path(path).resolve()), best_epoch=checkpoint["best_epoch"],
                  best_epoch_1based=checkpoint["best_epoch"] + 1, selection_rule=SELECTION_RULE,
                  metrics=evaluate_both(model, data, conf),
                  source_files_match=source_fingerprints(file_hash) == checkpoint["source_files"])
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.yaml")
    parser.add_argument("-d", "--dataset", default="Youshu")
    parser.add_argument("-g", "--gpu", default="0", help="CUDA_VISIBLE_DEVICES; use logical cuda:0")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("--data-path")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size-test", type=int)
    parser.add_argument("--note", default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default=str(MODEL_DIR / "log"))
    parser.add_argument("--record-file")
    parser.add_argument("--result-json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Read-only configuration and data validation")
    mode.add_argument("--evaluate", type=Path, metavar="CHECKPOINT", help="Read-only full-ranking checkpoint evaluation")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Delay importing PyTorch until after GPU visibility has been configured.
    if not args.evaluate:
        conf = load_config(args.config.resolve(), args.dataset,
                           {key: getattr(args, key) for key in ("seed", "epochs", "data_path", "num_workers", "batch_size_test")})
        if args.check_config:
            Relations, _, _, _, _ = modules()
            data = Relations(conf)
            print(yaml.safe_dump({args.dataset: conf}, sort_keys=False))
            print(json.dumps(dict(data_statistics=data.statistics, data_files=data.fingerprints), indent=2))
            print("Configuration and original data validated; no training or output writes. " + SELECTION_RULE)
            return 0
    import torch
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if args.evaluate:
        if any(value is not None for value in (args.seed, args.epochs, args.num_workers)):
            raise ValueError("Checkpoint evaluation does not accept seed/epochs/num-workers overrides")
        evaluate_checkpoint(args.evaluate.resolve(), args, device)
    else:
        if conf["epochs"] < conf["test_interval"]:
            raise ValueError("epochs must cover at least one scheduled evaluation")
        os.environ["PYTHONHASHSEED"] = str(conf["seed"])
        result = run_training(conf, args, device)
        print(json.dumps(dict(status=result["status"], best_epoch=result["best_epoch"],
                              metrics=result["metrics"], output_dir=result["output_dir"]), indent=2))
    return 0


if __name__ == "__main__":
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("DCDL interrupted; run status and available artifacts were saved.", file=sys.stderr)
        sys.exit(130)
