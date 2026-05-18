#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run NetEase Module2 analysis experiments and export analysis packages."""

import argparse
import csv
import json
import os
import shutil
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.model import AnchorViewBundleNet
from train import init_best_metrics, set_seed, test
from utility import Datasets


DATASET = "NetEase"
TOPK = 20
DEFAULT_OUTPUT_ROOT = Path("analysis_outputs") / DATASET


def build_run_specs():
    base = {
        "dataset": DATASET,
        "latent_rebuild_enabled": False,
        "anchor_cl": {
            "enabled": False,
            "anchor_cl_lambda": 0.005,
            "temp": 0.2,
            "user_cl": True,
            "bundle_cl": True,
            "ub_ui_user_weight": 1.0,
            "ub_bi_user_weight": 1.0,
            "ub_ui_bundle_weight": 1.0,
            "ub_bi_bundle_weight": 1.0,
        },
        "ui_bundle_user_agg_beta": 0.0,
        "bi_user_bundle_agg_beta": 0.0,
        "bi_drop_ratio": 0.0,
        "bi_drop_seed": 2026,
    }

    def spec(run_name, description, **overrides):
        item = deepcopy(base)
        item["run_name"] = run_name
        item["description"] = description
        for key, value in overrides.items():
            if key == "anchor_cl":
                item["anchor_cl"].update(value)
            else:
                item[key] = value
        return item

    full_anchor = {"enabled": True}
    ub_ui_only = {
        "enabled": True,
        "ub_ui_user_weight": 1.0,
        "ub_bi_user_weight": 0.0,
        "ub_ui_bundle_weight": 1.0,
        "ub_bi_bundle_weight": 0.0,
    }
    ub_bi_only = {
        "enabled": True,
        "ub_ui_user_weight": 0.0,
        "ub_bi_user_weight": 1.0,
        "ub_ui_bundle_weight": 0.0,
        "ub_bi_bundle_weight": 1.0,
    }
    user_only = {"enabled": True, "user_cl": True, "bundle_cl": False}
    bundle_only = {"enabled": True, "user_cl": False, "bundle_cl": True}

    runs = [
        spec("run_A_baseline", "Baseline: Module2 off"),
        spec(
            "run_B_full_module2",
            "Full Module2: beta plus full anchor_cl",
            anchor_cl=full_anchor,
            bi_user_bundle_agg_beta=0.35,
        ),
        spec(
            "run_C_beta_only",
            "Beta-only control",
            bi_user_bundle_agg_beta=0.35,
        ),
        spec(
            "run_D1_ub_ui_only_beta035",
            "UB-UI only with beta 0+0.35",
            anchor_cl=ub_ui_only,
            bi_user_bundle_agg_beta=0.35,
        ),
        spec(
            "run_D2_ub_ui_only_beta000",
            "UB-UI only with beta 0+0",
            anchor_cl=ub_ui_only,
            bi_user_bundle_agg_beta=0.0,
        ),
        spec(
            "run_E1_ub_bi_only_beta035",
            "UB-BI only with beta 0+0.35",
            anchor_cl=ub_bi_only,
            bi_user_bundle_agg_beta=0.35,
        ),
        spec(
            "run_E2_ub_bi_only_beta000",
            "UB-BI only with beta 0+0",
            anchor_cl=ub_bi_only,
            bi_user_bundle_agg_beta=0.0,
        ),
        spec(
            "run_F1_user_side_only_beta035",
            "User-side CL only with beta 0+0.35",
            anchor_cl=user_only,
            bi_user_bundle_agg_beta=0.35,
        ),
        spec(
            "run_F2_user_side_only_beta000",
            "User-side CL only with beta 0+0",
            anchor_cl=user_only,
            bi_user_bundle_agg_beta=0.0,
        ),
        spec(
            "run_G1_bundle_side_only_beta035",
            "Bundle-side CL only with beta 0+0.35",
            anchor_cl=bundle_only,
            bi_user_bundle_agg_beta=0.35,
        ),
        spec(
            "run_G2_bundle_side_only_beta000",
            "Bundle-side CL only with beta 0+0",
            anchor_cl=bundle_only,
            bi_user_bundle_agg_beta=0.0,
        ),
    ]

    for label, ratio in [("H", 0.1), ("J", 0.2), ("L", 0.3), ("N", 0.4)]:
        runs.append(
            spec(
                f"run_{label}_bi_drop_{ratio:.1f}_baseline",
                f"BI drop {ratio:.1f} baseline",
                bi_drop_ratio=ratio,
            )
        )
        next_label = chr(ord(label) + 1)
        runs.append(
            spec(
                f"run_{next_label}_bi_drop_{ratio:.1f}_full_module2",
                f"BI drop {ratio:.1f} Full Module2",
                anchor_cl=full_anchor,
                bi_user_bundle_agg_beta=0.35,
                bi_drop_ratio=ratio,
            )
        )
    return runs


RUN_SPECS = build_run_specs()
RUN_BY_NAME = {run["run_name"]: run for run in RUN_SPECS}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--runs", nargs="*", default=["all"], help="Run names, or all")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--bi-drop-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun even if package is complete")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned runs")
    return parser.parse_args()


def select_runs(names):
    if names == ["all"]:
        return RUN_SPECS
    missing = [name for name in names if name not in RUN_BY_NAME]
    if missing:
        raise ValueError(f"Unknown run names: {missing}")
    return [RUN_BY_NAME[name] for name in names]


def apply_run_spec(conf, run_spec, args):
    conf = deepcopy(conf)
    conf["dataset"] = DATASET
    conf["model"] = "AnchorViewBundleNet"
    conf["gpu"] = args.gpu
    conf["info"] = run_spec["run_name"]
    conf["run_name"] = run_spec["run_name"]
    conf["description"] = run_spec["description"]
    conf["train_style"] = "cbr"
    conf.setdefault("latent_rebuild", {})
    conf["latent_rebuild"]["enabled"] = False
    conf["anchor_cl"] = deepcopy(run_spec["anchor_cl"])
    conf["ui_bundle_user_agg_beta"] = float(run_spec["ui_bundle_user_agg_beta"])
    conf["bi_user_bundle_agg_beta"] = float(run_spec["bi_user_bundle_agg_beta"])
    conf["bi_drop_ratio"] = float(run_spec["bi_drop_ratio"])
    conf["bi_drop_seed"] = int(args.bi_drop_seed)
    conf["topk"] = sorted(set(conf.get("topk", []) + [TOPK]))
    if args.epochs is not None:
        conf["epochs"] = args.epochs
    return conf


def analysis_package_complete(run_dir):
    required = [
        "config.yaml",
        "metrics.json",
        "per_user_metrics.csv",
        "user_train_degree.csv",
        "bundle_train_degree.csv",
        "top20_recommendations.npy",
        "test_ground_truth.json",
        "view_similarity.json",
        "representations/user_UB.npy",
        "representations/user_UI.npy",
        "representations/user_BI.npy",
        "representations/bundle_UB.npy",
        "representations/bundle_UI.npy",
        "representations/bundle_BI.npy",
    ]
    return all((run_dir / path).is_file() for path in required)


def ensure_clean_run_dir(run_dir, force):
    if run_dir.exists() and force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "representations").mkdir(parents=True, exist_ok=True)


def drop_bi_graph(graph, drop_ratio, seed):
    if drop_ratio <= 0:
        return graph
    if not 0 <= drop_ratio < 1:
        raise ValueError(f"bi_drop_ratio must be in [0, 1), got {drop_ratio}")
    coo = graph.tocoo()
    rng = np.random.default_rng(seed)
    keep = rng.random(coo.data.shape[0]) >= drop_ratio
    return sp.coo_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=coo.shape,
        dtype=coo.dtype,
    ).tocsr()


def build_dataset_and_device(conf):
    dataset = Datasets(conf)
    if conf["bi_drop_ratio"] > 0:
        dataset.graphs[2] = drop_bi_graph(
            dataset.graphs[2],
            conf["bi_drop_ratio"],
            conf["bi_drop_seed"],
        )
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    return dataset, device


def run_single_training(conf, dataset, device, run_dir):
    lr = conf["lrs"][0]
    l2_reg = conf["l2_regs"][0]
    conf["l2_reg"] = l2_reg
    conf["embedding_size"] = conf["embedding_sizes"][0]
    conf["UB_ratio"] = conf["UB_ratios"][0]
    conf["UI_ratio"] = conf["UI_ratios"][0]
    conf["BI_ratio"] = conf["BI_ratios"][0]
    conf["num_layers"] = conf["num_layerss"][0]
    conf["c_lambda"] = conf["c_lambdas"][0]
    conf["c_temp"] = conf["c_temps"][0]

    set_seed(conf.get("seed", None))
    model = AnchorViewBundleNet(conf, dataset.graphs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

    batch_cnt = len(dataset.train_loader)
    test_interval_bs = max(1, int(batch_cnt * conf["test_interval"]))
    ed_interval_bs = max(1, int(batch_cnt * conf["ed_interval"]))

    best_metrics, best_perform = init_best_metrics(conf)
    best_epoch = -1
    best_score = float("-inf")
    best_model_path = run_dir / "best_model.pt"

    for epoch in range(conf["epochs"]):
        epoch_anchor = epoch * batch_cnt
        model.train(True)
        pbar = tqdm(enumerate(dataset.train_loader), total=batch_cnt, disable=True)

        for batch_i, batch in pbar:
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_anchor = epoch_anchor + batch_i
            ed_drop = conf["aug_type"] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0

            bpr_loss, c_loss, anchor_cl_loss, _ = model(batch, ED_drop=ed_drop)
            anchor_cl_lambda = conf.get("anchor_cl", {}).get("anchor_cl_lambda", 0.0)
            loss = bpr_loss + conf["c_lambda"] * c_loss + anchor_cl_lambda * anchor_cl_loss
            loss.backward()
            optimizer.step()

            if (batch_anchor + 1) % test_interval_bs == 0:
                metrics = {
                    "val": test(model, dataset.val_loader, conf),
                    "test": test(model, dataset.test_loader, conf),
                }
                current_score = metric_selection_score(metrics["val"])
                if current_score > best_score:
                    best_score = current_score
                    best_metrics = metrics
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)

    if best_epoch < 0:
        metrics = {
            "val": test(model, dataset.val_loader, conf),
            "test": test(model, dataset.test_loader, conf),
        }
        best_metrics = metrics
        best_epoch = conf["epochs"] - 1
        torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    return model, best_metrics, best_epoch, best_model_path


def metric_selection_score(metric_dict):
    return (
        metric_dict["recall"][20]
        + metric_dict["recall"][40]
        + metric_dict["ndcg"][20]
        + metric_dict["ndcg"][40]
    )


def sparse_rows_to_dict(graph):
    result = {}
    csr = graph.tocsr()
    for row in range(csr.shape[0]):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        result[str(row)] = [int(x) for x in csr.indices[start:end]]
    return result


def per_user_ndcg(top_items, positives):
    if not positives:
        return 0.0
    positive_set = set(positives)
    dcg = 0.0
    for rank, item in enumerate(top_items, start=1):
        if int(item) in positive_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(positive_set), len(top_items))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def export_analysis_package(conf, dataset, model, best_metrics, best_epoch, run_dir):
    device = conf["device"]
    model.eval()
    with torch.no_grad():
        representations = model.get_multi_modal_representations(test=True)
        users_rep, bundles_rep, pre_users, pre_bundles = representations

        save_representations(run_dir, pre_users, pre_bundles)
        view_similarity = compute_view_similarity(pre_users, pre_bundles)
        write_json(run_dir / "view_similarity.json", view_similarity)

        top20 = np.zeros((dataset.num_users, TOPK), dtype=np.int64)
        test_graph = dataset.bundle_test_data.u_b_graph.tocsr()
        train_graph = dataset.bundle_train_data.u_b_graph.tocsr()
        for users, _, train_mask_u_b in dataset.test_loader:
            user_ids = users.to(device)
            scores = model.evaluate(representations, user_ids)
            scores -= 1e8 * train_mask_u_b.to(device)
            _, indices = torch.topk(scores, TOPK)
            top20[users.numpy()] = indices.cpu().numpy()

    np.save(run_dir / "top20_recommendations.npy", top20)
    test_ground_truth = sparse_rows_to_dict(test_graph)
    write_json(run_dir / "test_ground_truth.json", test_ground_truth)

    user_train_degree = np.asarray(train_graph.getnnz(axis=1), dtype=np.int64)
    bundle_train_degree = np.asarray(train_graph.getnnz(axis=0), dtype=np.int64)
    bundle_test_count = np.asarray(test_graph.getnnz(axis=0), dtype=np.int64)
    write_degree_csv(run_dir / "user_train_degree.csv", "user_id", user_train_degree)
    write_bundle_degree_csv(run_dir / "bundle_train_degree.csv", bundle_train_degree, bundle_test_count)
    write_per_user_metrics(run_dir / "per_user_metrics.csv", top20, test_ground_truth, user_train_degree)

    metrics = {
        "run_name": conf["run_name"],
        "best_epoch": int(best_epoch),
        "recall@20": float(best_metrics["test"]["recall"][20]),
        "ndcg@20": float(best_metrics["test"]["ndcg"][20]),
        "val_recall@20": float(best_metrics["val"]["recall"][20]),
        "val_ndcg@20": float(best_metrics["val"]["ndcg"][20]),
        "all_metrics": best_metrics,
    }
    write_json(run_dir / "metrics.json", metrics)

    dump_conf = deepcopy(conf)
    dump_conf["device"] = str(dump_conf["device"])
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dump_conf, f, allow_unicode=True, sort_keys=False)


def save_representations(run_dir, pre_users, pre_bundles):
    names = ["UB", "UI", "BI"]
    rep_dir = run_dir / "representations"
    for name, tensor in zip(names, pre_users):
        np.save(rep_dir / f"user_{name}.npy", tensor.detach().cpu().numpy())
    for name, tensor in zip(names, pre_bundles):
        np.save(rep_dir / f"bundle_{name}.npy", tensor.detach().cpu().numpy())


def cosine_mean(left, right):
    left = F.normalize(left.detach(), dim=1)
    right = F.normalize(right.detach(), dim=1)
    return float((left * right).sum(dim=1).mean().cpu().item())


def compute_view_similarity(pre_users, pre_bundles):
    return {
        "user_UB_UI": cosine_mean(pre_users[0], pre_users[1]),
        "user_UB_BI": cosine_mean(pre_users[0], pre_users[2]),
        "user_UI_BI": cosine_mean(pre_users[1], pre_users[2]),
        "bundle_UB_UI": cosine_mean(pre_bundles[0], pre_bundles[1]),
        "bundle_UB_BI": cosine_mean(pre_bundles[0], pre_bundles[2]),
        "bundle_UI_BI": cosine_mean(pre_bundles[1], pre_bundles[2]),
        "scope": "all_users_and_all_bundles",
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_degree_csv(path, id_name, degrees):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([id_name, "train_ub_degree"])
        for idx, degree in enumerate(degrees):
            writer.writerow([idx, int(degree)])


def write_bundle_degree_csv(path, train_degrees, test_counts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bundle_id", "train_ub_degree", "test_pos_count"])
        for idx, degree in enumerate(train_degrees):
            writer.writerow([idx, int(degree), int(test_counts[idx])])


def write_per_user_metrics(path, top20, test_ground_truth, user_train_degree):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_id",
            "train_ub_degree",
            "test_pos_count",
            "recall@20",
            "ndcg@20",
            "top20_bundles",
            "test_bundles",
        ])
        for user_id in range(top20.shape[0]):
            positives = test_ground_truth[str(user_id)]
            hits = len(set(map(int, top20[user_id])) & set(positives))
            recall = hits / len(positives) if positives else 0.0
            ndcg = per_user_ndcg(top20[user_id], positives)
            writer.writerow([
                user_id,
                int(user_train_degree[user_id]),
                len(positives),
                recall,
                ndcg,
                " ".join(str(int(x)) for x in top20[user_id]),
                " ".join(str(int(x)) for x in positives),
            ])


def run_with_retries(base_conf, run_spec, args, output_root):
    run_dir = output_root / run_spec["run_name"]
    attempts = []
    if analysis_package_complete(run_dir) and not args.force:
        return {
            "run_name": run_spec["run_name"],
            "status": "skipped_complete",
            "run_dir": str(run_dir),
            "attempts": attempts,
        }

    for attempt_idx in range(1, args.max_retries + 1):
        started = datetime.now().isoformat(timespec="seconds")
        attempt_record = {"attempt": attempt_idx, "started_at": started, "status": "running"}
        attempts.append(attempt_record)
        try:
            ensure_clean_run_dir(run_dir, force=True)
            conf = apply_run_spec(base_conf, run_spec, args)
            dataset, device = build_dataset_and_device(conf)
            model, best_metrics, best_epoch, best_model_path = run_single_training(conf, dataset, device, run_dir)
            export_analysis_package(conf, dataset, model, best_metrics, best_epoch, run_dir)
            attempt_record.update({
                "status": "success",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "best_epoch": int(best_epoch),
                "recall@20": float(best_metrics["test"]["recall"][20]),
                "ndcg@20": float(best_metrics["test"]["ndcg"][20]),
                "best_model_path": str(best_model_path),
            })
            return {
                "run_name": run_spec["run_name"],
                "status": "success",
                "run_dir": str(run_dir),
                "attempts": attempts,
                "final_metrics": {
                    "best_epoch": int(best_epoch),
                    "recall@20": float(best_metrics["test"]["recall"][20]),
                    "ndcg@20": float(best_metrics["test"]["ndcg"][20]),
                },
            }
        except Exception as exc:
            attempt_record.update({
                "status": "failed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            write_json(run_dir / "attempts.json", {"run_name": run_spec["run_name"], "attempts": attempts})
            time.sleep(3)

    return {
        "run_name": run_spec["run_name"],
        "status": "failed",
        "run_dir": str(run_dir),
        "attempts": attempts,
    }


def write_report(output_root, selected_runs, results):
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET,
        "selected_runs": [run["run_name"] for run in selected_runs],
        "results": results,
    }
    write_json(output_root / "module2_experiment_report.json", report)
    with open(output_root / "module2_experiment_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Module2 experiment report: {report['created_at']}\n")
        f.write(f"Dataset: {DATASET}\n\n")
        for result in results:
            f.write(f"{result['run_name']}: {result['status']}\n")
            if "final_metrics" in result:
                metrics = result["final_metrics"]
                f.write(
                    f"  best_epoch={metrics['best_epoch']} "
                    f"recall@20={metrics['recall@20']:.6f} "
                    f"ndcg@20={metrics['ndcg@20']:.6f}\n"
                )
            for attempt in result["attempts"]:
                f.write(
                    f"  attempt {attempt['attempt']}: {attempt['status']} "
                    f"{attempt.get('started_at', '')} -> {attempt.get('finished_at', '')}\n"
                )
                if attempt.get("error"):
                    f.write(f"    error: {attempt['error']}\n")
            f.write(f"  output: {result['run_dir']}\n\n")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    selected_runs = select_runs(args.runs)
    if args.dry_run:
        for run in selected_runs:
            print(f"{run['run_name']}: {run['description']}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_conf = config[DATASET]
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for run_spec in selected_runs:
        print(f"Starting {run_spec['run_name']}: {run_spec['description']}")
        result = run_with_retries(base_conf, run_spec, args, output_root)
        results.append(result)
        write_report(output_root, selected_runs, results)
        if result["status"] == "failed":
            print(f"{run_spec['run_name']} failed after retries; continuing to next run.")
    write_report(output_root, selected_runs, results)


if __name__ == "__main__":
    main()
