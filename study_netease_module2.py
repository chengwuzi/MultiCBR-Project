#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

from models.MultiCBR import MultiCBR
from train import ensure_dir, init_best_metrics, set_seed, test
from utility import Datasets


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "analysis_outputs" / "netease_module2_study"
LOG_DIR = OUTPUT_ROOT / "logs"
TABLE_DIR = OUTPUT_ROOT / "tables"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
REPORT_PATH = OUTPUT_ROOT / "report.txt"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

TOPKS = [10, 20, 40, 80]
MAX_TOPK = max(TOPKS)
DEFAULT_SEED = 4096


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "a", encoding="utf-8") as log_file:
        tee = Tee(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_output_dirs():
    for path in [OUTPUT_ROOT, LOG_DIR, TABLE_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_manifest():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"experiments": {}}


def save_manifest(manifest):
    ensure_output_dirs()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def json_ready_metrics(metrics):
    return {
        split: {
            metric: {str(k): float(v) for k, v in values.items()}
            for metric, values in split_values.items()
        }
        for split, split_values in metrics.items()
    }


def build_base_conf(gpu, epochs):
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)["NetEase"]

    conf = copy.deepcopy(conf)
    conf["dataset"] = "NetEase"
    conf["model"] = "MultiCBR"
    conf["gpu"] = str(gpu)
    conf["seed"] = DEFAULT_SEED
    conf["epochs"] = int(epochs)
    conf["topk"] = TOPKS
    conf["train_style"] = "cbr"
    conf["ui_bundle_user_agg_beta"] = 0.0
    conf["train_connected_ratio_range"] = [0.0, 1.0]
    conf["user_ub_train_deg_min"] = 0

    latent_rebuild = copy.deepcopy(conf.get("latent_rebuild", {}))
    latent_rebuild["enabled"] = False
    conf["latent_rebuild"] = latent_rebuild
    return conf


def build_experiments(gpu, epochs):
    baseline = build_base_conf(gpu, epochs)
    baseline["info"] = "module2_study_baseline_seed4096"
    baseline["bi_user_bundle_agg_beta"] = 0.0
    baseline["anchor_cl"] = copy.deepcopy(baseline.get("anchor_cl", {}))
    baseline["anchor_cl"]["enabled"] = False

    module2 = build_base_conf(gpu, epochs)
    module2["info"] = "module2_study_module2_seed4096"
    module2["bi_user_bundle_agg_beta"] = 0.35
    module2["anchor_cl"] = copy.deepcopy(module2.get("anchor_cl", {}))
    module2["anchor_cl"]["enabled"] = True
    module2["anchor_cl"]["anchor_cl_lambda"] = 0.005
    module2["anchor_cl"]["temp"] = 0.2
    module2["anchor_cl"]["user_cl"] = True
    module2["anchor_cl"]["bundle_cl"] = True
    module2["anchor_cl"]["ub_ui_user_weight"] = 1.0
    module2["anchor_cl"]["ub_bi_user_weight"] = 1.0
    module2["anchor_cl"]["ub_ui_bundle_weight"] = 1.0
    module2["anchor_cl"]["ub_bi_bundle_weight"] = 1.0

    return [
        {"name": "baseline", "conf": baseline},
        {"name": "module2", "conf": module2},
    ]


def get_score(metric_dict):
    return (
        metric_dict["recall"][20]
        + metric_dict["recall"][40]
        + metric_dict["ndcg"][20]
        + metric_dict["ndcg"][40]
    )


def train_one_attempt(exp_name, conf, attempt_idx, force=False):
    if force:
        torch.cuda.empty_cache()

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    dataset = Datasets(conf)
    set_seed(conf.get("seed", None))

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    lr = float(conf["lrs"][0])
    l2_reg = float(conf["l2_regs"][0])
    conf["l2_reg"] = l2_reg
    conf["embedding_size"] = int(conf["embedding_sizes"][0])
    conf["UB_ratio"] = float(conf["UB_ratios"][0])
    conf["UI_ratio"] = float(conf["UI_ratios"][0])
    conf["BI_ratio"] = float(conf["BI_ratios"][0])
    conf["num_layers"] = int(conf["num_layerss"][0])
    conf["c_lambda"] = float(conf["c_lambdas"][0])
    conf["c_temp"] = float(conf["c_temps"][0])

    attempt_tag = f"{exp_name}_seed{conf['seed']}_attempt{attempt_idx}"
    checkpoint_path = CHECKPOINT_DIR / f"{attempt_tag}.pt"
    best_conf_path = CHECKPOINT_DIR / f"{attempt_tag}.json"
    run_path = OUTPUT_ROOT / "runs" / attempt_tag
    train_log_path = LOG_DIR / f"{attempt_tag}_train_metrics.txt"
    ensure_dir(str(run_path))

    print(f"[{now()}] Start experiment={exp_name} attempt={attempt_idx} device={device}")
    print(json.dumps(minimal_conf_for_report(conf), indent=2, ensure_ascii=False, default=str))

    run = SummaryWriter(str(run_path))
    model = MultiCBR(conf, dataset.graphs, dataset.user_beta_mask).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    batch_cnt = len(dataset.train_loader)
    test_interval = int(conf["test_interval"])
    test_interval_bs = int(batch_cnt * test_interval)
    ed_interval_bs = int(batch_cnt * conf["ed_interval"])
    best_metrics, _ = init_best_metrics(conf)
    best_epoch = -1
    best_batch_anchor = -1
    best_score = get_score(best_metrics["val"])

    for epoch in range(int(conf["epochs"])):
        model.train(True)
        epoch_anchor = epoch * batch_cnt
        epoch_loss = 0.0
        epoch_bpr = 0.0
        epoch_c = 0.0
        epoch_anchor_cl = 0.0
        epoch_steps = 0

        for batch_i, batch in enumerate(dataset.train_loader):
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_anchor = epoch_anchor + batch_i

            ED_drop = False
            if conf["aug_type"] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0:
                ED_drop = True

            bpr_loss, c_loss, anchor_cl_loss, _ = model(batch, ED_drop=ED_drop)
            anchor_cl_lambda = conf.get("anchor_cl", {}).get("anchor_cl_lambda", 0.0)
            loss = bpr_loss + conf["c_lambda"] * c_loss + anchor_cl_lambda * anchor_cl_loss

            loss.backward()
            optimizer.step()

            loss_scalar = float(loss.detach().cpu().item())
            bpr_scalar = float(bpr_loss.detach().cpu().item())
            c_scalar = float(c_loss.detach().cpu().item())
            anchor_scalar = float(anchor_cl_loss.detach().cpu().item())

            epoch_loss += loss_scalar
            epoch_bpr += bpr_scalar
            epoch_c += c_scalar
            epoch_anchor_cl += anchor_scalar
            epoch_steps += 1

            run.add_scalar("loss_bpr", bpr_loss.detach(), batch_anchor)
            run.add_scalar("loss_c", c_loss.detach(), batch_anchor)
            run.add_scalar("loss_anchor_cl", anchor_cl_loss.detach(), batch_anchor)
            run.add_scalar("loss", loss.detach(), batch_anchor)

            if (batch_anchor + 1) % test_interval_bs == 0:
                metrics = {
                    "val": test(model, dataset.val_loader, conf),
                    "test": test(model, dataset.test_loader, conf),
                }
                curr_score = get_score(metrics["val"])
                print_metric_block(
                    train_log_path,
                    f"[EVAL] epoch={epoch} batch_anchor={batch_anchor} score={curr_score:.8f}",
                    metrics,
                )
                if curr_score > best_score:
                    best_score = curr_score
                    best_metrics = copy.deepcopy(metrics)
                    best_epoch = epoch
                    best_batch_anchor = batch_anchor
                    torch.save(model.state_dict(), checkpoint_path)
                    dump_conf = copy.deepcopy(conf)
                    dump_conf.pop("device", None)
                    with open(best_conf_path, "w", encoding="utf-8") as f:
                        json.dump(dump_conf, f, indent=2, ensure_ascii=False)

        denom = max(epoch_steps, 1)
        print(
            f"[{now()}] epoch={epoch} "
            f"loss={epoch_loss / denom:.6f} "
            f"bpr={epoch_bpr / denom:.6f} "
            f"c={epoch_c / denom:.6f} "
            f"anchor_cl={epoch_anchor_cl / denom:.6f}"
        )

    run.close()

    if not checkpoint_path.exists():
        metrics = {
            "val": test(model, dataset.val_loader, conf),
            "test": test(model, dataset.test_loader, conf),
        }
        best_metrics = metrics
        best_epoch = int(conf["epochs"]) - 1
        best_batch_anchor = max(int(conf["epochs"]) * batch_cnt - 1, 0)
        torch.save(model.state_dict(), checkpoint_path)
        dump_conf = copy.deepcopy(conf)
        dump_conf.pop("device", None)
        with open(best_conf_path, "w", encoding="utf-8") as f:
            json.dump(dump_conf, f, indent=2, ensure_ascii=False)

    print(f"[{now()}] Finished experiment={exp_name} attempt={attempt_idx} best_epoch={best_epoch}")
    print_metric_block(
        train_log_path,
        f"[BEST] experiment={exp_name} attempt={attempt_idx} best_epoch={best_epoch}",
        best_metrics,
    )

    return {
        "status": "SUCCESS",
        "experiment": exp_name,
        "attempt": attempt_idx,
        "checkpoint_path": str(checkpoint_path),
        "conf_path": str(best_conf_path),
        "train_log_path": str(train_log_path),
        "run_path": str(run_path),
        "best_epoch": best_epoch,
        "best_batch_anchor": best_batch_anchor,
        "best_metrics": json_ready_metrics(best_metrics),
        "completed_at": now(),
    }


def print_metric_block(path, header, metrics):
    lines = [header]
    for topk in TOPKS:
        lines.append(
            "TOP%d VAL recall=%.6f ndcg=%.6f | TEST recall=%.6f ndcg=%.6f"
            % (
                topk,
                metrics["val"]["recall"][topk],
                metrics["val"]["ndcg"][topk],
                metrics["test"]["recall"][topk],
                metrics["test"]["ndcg"][topk],
            )
        )
    text = "\n".join(lines)
    print(text)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def minimal_conf_for_report(conf):
    return {
        "dataset": conf["dataset"],
        "seed": conf["seed"],
        "epochs": conf["epochs"],
        "batch_size_train": conf["batch_size_train"],
        "batch_size_test": conf["batch_size_test"],
        "topk": conf["topk"],
        "aug_type": conf["aug_type"],
        "UB_ratios": conf["UB_ratios"],
        "UI_ratios": conf["UI_ratios"],
        "BI_ratios": conf["BI_ratios"],
        "fusion_weights": conf["fusion_weights"],
        "lrs": conf["lrs"],
        "l2_regs": conf["l2_regs"],
        "c_lambdas": conf["c_lambdas"],
        "c_temps": conf["c_temps"],
        "latent_rebuild.enabled": conf.get("latent_rebuild", {}).get("enabled", False),
        "ui_bundle_user_agg_beta": conf.get("ui_bundle_user_agg_beta", 0.0),
        "bi_user_bundle_agg_beta": conf.get("bi_user_bundle_agg_beta", 0.0),
        "anchor_cl": conf.get("anchor_cl", {}),
    }


def run_experiments(experiments, max_retries, force):
    manifest = load_manifest()
    for exp in experiments:
        name = exp["name"]
        conf = exp["conf"]
        exp_record = manifest["experiments"].get(name)
        if (
            not force
            and exp_record
            and exp_record.get("status") == "SUCCESS"
            and Path(exp_record.get("checkpoint_path", "")).exists()
        ):
            print(f"[{now()}] Skip already successful experiment={name}")
            continue

        attempts = []
        success_record = None
        for attempt_idx in range(1, max_retries + 1):
            log_path = LOG_DIR / f"{name}_seed{conf['seed']}_attempt{attempt_idx}.txt"
            with tee_stdout(log_path):
                try:
                    record = train_one_attempt(name, copy.deepcopy(conf), attempt_idx, force=force)
                    attempts.append(record)
                    success_record = record
                    break
                except Exception as exc:
                    err = {
                        "status": "FAILED",
                        "experiment": name,
                        "attempt": attempt_idx,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "completed_at": now(),
                    }
                    attempts.append(err)
                    print(f"[{now()}] FAILED experiment={name} attempt={attempt_idx}: {repr(exc)}")
                    print(err["traceback"])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if success_record is None:
            manifest["experiments"][name] = {
                "status": "FAILED",
                "attempts": attempts,
                "updated_at": now(),
                "config": minimal_conf_for_report(conf),
            }
        else:
            success_record["attempts"] = attempts
            success_record["config"] = minimal_conf_for_report(conf)
            manifest["experiments"][name] = success_record
        save_manifest(manifest)

    return load_manifest()


def load_dataset_for_analysis(conf):
    dataset = Datasets(conf)
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    return dataset


def instantiate_model(conf, dataset, checkpoint_path, device):
    conf = copy.deepcopy(conf)
    conf["device"] = device
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items
    conf["l2_reg"] = float(conf["l2_regs"][0])
    conf["embedding_size"] = int(conf["embedding_sizes"][0])
    conf["UB_ratio"] = float(conf["UB_ratios"][0])
    conf["UI_ratio"] = float(conf["UI_ratios"][0])
    conf["BI_ratio"] = float(conf["BI_ratios"][0])
    conf["num_layers"] = int(conf["num_layerss"][0])
    conf["c_lambda"] = float(conf["c_lambdas"][0])
    conf["c_temp"] = float(conf["c_temps"][0])
    model = MultiCBR(conf, dataset.graphs, dataset.user_beta_mask).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, conf


def evaluate_user_level(model, dataset, conf, device):
    top_indices = np.zeros((dataset.num_users, MAX_TOPK), dtype=np.int64)
    per_user = {
        "num_pos": np.zeros(dataset.num_users, dtype=np.float32),
        "pos_score_sum": np.zeros(dataset.num_users, dtype=np.float64),
        "pos_score_mean": np.full(dataset.num_users, np.nan, dtype=np.float64),
        "pos_rank_mean": np.full(dataset.num_users, np.nan, dtype=np.float64),
        "pos_rank_min": np.full(dataset.num_users, np.nan, dtype=np.float64),
    }
    for topk in TOPKS:
        per_user[f"recall@{topk}"] = np.full(dataset.num_users, np.nan, dtype=np.float64)
        per_user[f"ndcg@{topk}"] = np.full(dataset.num_users, np.nan, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        propagate_result = model.get_multi_modal_representations(test=True)
        for users, ground_truth_u_b, train_mask_u_b in dataset.test_loader:
            user_ids = users.numpy().astype(np.int64)
            users_device = users.to(device)
            pred = model.evaluate(propagate_result, users_device)
            pred = pred - 1e8 * train_mask_u_b.to(device)
            _, batch_top = torch.topk(pred, MAX_TOPK, dim=1)
            batch_top_np = batch_top.cpu().numpy()
            top_indices[user_ids] = batch_top_np

            grd = ground_truth_u_b.float()
            num_pos = grd.sum(dim=1).numpy()
            per_user["num_pos"][user_ids] = num_pos

            pos_mask = grd.to(device).bool()
            score_sum = (pred * pos_mask.float()).sum(dim=1).detach().cpu().numpy()
            per_user["pos_score_sum"][user_ids] = score_sum
            valid = num_pos > 0
            mean_scores = np.full_like(num_pos, np.nan, dtype=np.float64)
            mean_scores[valid] = score_sum[valid] / num_pos[valid]
            per_user["pos_score_mean"][user_ids] = mean_scores

            for row_idx, user_id in enumerate(user_ids):
                positives = set(np.flatnonzero(ground_truth_u_b[row_idx].numpy() > 0))
                if not positives:
                    continue
                positive_tensor = torch.tensor(sorted(positives), dtype=torch.long, device=device)
                row_scores = pred[row_idx]
                positive_scores = row_scores[positive_tensor]
                positive_ranks = (row_scores.unsqueeze(0) > positive_scores.unsqueeze(1)).sum(dim=1).float() + 1.0
                per_user["pos_rank_mean"][user_id] = float(positive_ranks.mean().detach().cpu().item())
                per_user["pos_rank_min"][user_id] = float(positive_ranks.min().detach().cpu().item())
                ideal_len_cache = {}
                for topk in TOPKS:
                    rec_list = batch_top_np[row_idx, :topk]
                    hits = np.array([1.0 if int(item) in positives else 0.0 for item in rec_list], dtype=np.float64)
                    per_user[f"recall@{topk}"][user_id] = hits.sum() / max(len(positives), 1)
                    discounts = 1.0 / np.log2(np.arange(2, topk + 2, dtype=np.float64))
                    dcg = float((hits * discounts).sum())
                    ideal_hits = min(len(positives), topk)
                    if ideal_hits not in ideal_len_cache:
                        ideal_len_cache[ideal_hits] = float(discounts[:ideal_hits].sum()) if ideal_hits > 0 else 1.0
                    per_user[f"ndcg@{topk}"][user_id] = dcg / ideal_len_cache[ideal_hits]

    alignments = compute_alignment(model, device)
    return {
        "top_indices": top_indices,
        "per_user": per_user,
        "alignments": alignments,
    }


def compute_alignment(model, device):
    with torch.no_grad():
        users_rep, bundles_rep, users_feature, bundles_feature = model.get_multi_modal_representations(test=True)
        ub_u, ui_u, bi_u = users_feature
        ub_b, ui_b, bi_b = bundles_feature
        result = {
            "user_cos_UB_BI": cosine_array(ub_u, bi_u),
            "user_cos_UB_UI": cosine_array(ub_u, ui_u),
            "user_cos_UI_BI": cosine_array(ui_u, bi_u),
            "bundle_cos_UB_BI": cosine_array(ub_b, bi_b),
            "bundle_cos_UB_UI": cosine_array(ub_b, ui_b),
            "bundle_cos_UI_BI": cosine_array(ui_b, bi_b),
            "user_norm_fused": torch.linalg.norm(users_rep, dim=1).detach().cpu().numpy(),
            "bundle_norm_fused": torch.linalg.norm(bundles_rep, dim=1).detach().cpu().numpy(),
        }
    return result


def cosine_array(a, b):
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return cos.detach().cpu().numpy().astype(np.float64)


def load_connectivity_metrics(dataset_name="NetEase"):
    path = PROJECT_ROOT / "connectivity_analysis_outputs" / dataset_name / "user_connectivity_metrics.csv"
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = int(row["user_id"])
            rows[user_id] = {
                key: parse_float(value)
                for key, value in row.items()
                if key != "user_id"
            }
    return rows


def parse_float(value):
    try:
        if value == "":
            return float("nan")
        return float(value)
    except ValueError:
        return float("nan")


def connectivity_arrays(num_users, metrics):
    keys = [
        "train_connected_ratio",
        "avg_overlap_ratio_per_train_bundle",
        "user_test_recover_zero_ratio",
        "user_ub_train_deg",
        "user_ui_deg",
        "user_test_avg_recover_ratio",
    ]
    arrays = {key: np.full(num_users, np.nan, dtype=np.float64) for key in keys}
    for user_id in range(num_users):
        row = metrics.get(user_id, {})
        for key in keys:
            arrays[key][user_id] = row.get(key, np.nan)
    return arrays


def build_bucket_specs(connectivity):
    avg_overlap = connectivity["avg_overlap_ratio_per_train_bundle"]
    recover_zero = connectivity["user_test_recover_zero_ratio"]
    user_ui_deg = connectivity["user_ui_deg"]

    return {
        "train_connected_ratio": [
            ("ratio_0", lambda a: a == 0),
            ("ratio_0_0.25", lambda a: (a > 0) & (a <= 0.25)),
            ("ratio_0.25_0.5", lambda a: (a > 0.25) & (a <= 0.5)),
            ("ratio_0.5_0.75", lambda a: (a > 0.5) & (a <= 0.75)),
            ("ratio_gt_0.75", lambda a: a > 0.75),
        ],
        "avg_overlap_ratio_per_train_bundle": quantile_buckets(avg_overlap, "overlap_q"),
        "user_test_recover_zero_ratio": [
            ("recover_zero_0", lambda a: a == 0),
            ("recover_zero_0_0.5", lambda a: (a > 0) & (a <= 0.5)),
            ("recover_zero_0.5_1", lambda a: (a > 0.5) & (a < 1.0)),
            ("recover_zero_1", lambda a: a == 1.0),
        ],
        "user_ub_train_deg": [
            ("ub_deg_0_5", lambda a: (a >= 0) & (a <= 5)),
            ("ub_deg_6_8", lambda a: (a >= 6) & (a <= 8)),
            ("ub_deg_9_12", lambda a: (a >= 9) & (a <= 12)),
            ("ub_deg_13_19", lambda a: (a >= 13) & (a <= 19)),
            ("ub_deg_20_plus", lambda a: a >= 20),
        ],
        "user_ui_deg": quantile_buckets(user_ui_deg, "ui_deg_q"),
    }


def quantile_buckets(values, prefix):
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return []
    q25, q50, q75 = np.percentile(clean, [25, 50, 75])
    return [
        (f"{prefix}1_min_{q25:.6g}", lambda a, q25=q25: a <= q25),
        (f"{prefix}2_{q25:.6g}_{q50:.6g}", lambda a, q25=q25, q50=q50: (a > q25) & (a <= q50)),
        (f"{prefix}3_{q50:.6g}_{q75:.6g}", lambda a, q50=q50, q75=q75: (a > q50) & (a <= q75)),
        (f"{prefix}4_{q75:.6g}_max", lambda a, q75=q75: a > q75),
    ]


def summarize_array(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan")}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
    }


def mean_valid(values, mask):
    selected = np.asarray(values)[mask]
    selected = selected[~np.isnan(selected)]
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected))


def bucket_metric_rows(baseline_eval, module2_eval, connectivity, bucket_specs):
    rows = []
    valid_users = baseline_eval["per_user"]["num_pos"] > 0
    for feature_name, specs in bucket_specs.items():
        values = connectivity[feature_name]
        valid_feature = ~np.isnan(values)
        for bucket_name, fn in specs:
            mask = fn(values) & valid_feature & valid_users
            rows.extend(metric_rows_for_mask(feature_name, bucket_name, mask, baseline_eval, module2_eval, connectivity))
    return rows


def metric_rows_for_mask(feature_name, bucket_name, mask, baseline_eval, module2_eval, connectivity):
    rows = []
    row_base = {
        "feature": feature_name,
        "bucket": bucket_name,
        "users": int(mask.sum()),
        "avg_train_connected_ratio": mean_valid(connectivity["train_connected_ratio"], mask),
        "avg_user_ub_train_deg": mean_valid(connectivity["user_ub_train_deg"], mask),
        "avg_overlap_ratio_per_train_bundle": mean_valid(connectivity["avg_overlap_ratio_per_train_bundle"], mask),
        "avg_user_test_recover_zero_ratio": mean_valid(connectivity["user_test_recover_zero_ratio"], mask),
    }
    for topk in TOPKS:
        b_rec = mean_valid(baseline_eval["per_user"][f"recall@{topk}"], mask)
        m_rec = mean_valid(module2_eval["per_user"][f"recall@{topk}"], mask)
        b_ndcg = mean_valid(baseline_eval["per_user"][f"ndcg@{topk}"], mask)
        m_ndcg = mean_valid(module2_eval["per_user"][f"ndcg@{topk}"], mask)
        row = dict(row_base)
        row.update(
            {
                "topk": topk,
                "baseline_recall": b_rec,
                "module2_recall": m_rec,
                "delta_recall": m_rec - b_rec,
                "rel_recall": safe_rel(m_rec, b_rec),
                "baseline_ndcg": b_ndcg,
                "module2_ndcg": m_ndcg,
                "delta_ndcg": m_ndcg - b_ndcg,
                "rel_ndcg": safe_rel(m_ndcg, b_ndcg),
                "baseline_pos_score": mean_valid(baseline_eval["per_user"]["pos_score_mean"], mask),
                "module2_pos_score": mean_valid(module2_eval["per_user"]["pos_score_mean"], mask),
                "delta_pos_score": mean_valid(module2_eval["per_user"]["pos_score_mean"] - baseline_eval["per_user"]["pos_score_mean"], mask),
                "baseline_pos_rank_mean": mean_valid(baseline_eval["per_user"]["pos_rank_mean"], mask),
                "module2_pos_rank_mean": mean_valid(module2_eval["per_user"]["pos_rank_mean"], mask),
                "delta_pos_rank_mean": mean_valid(module2_eval["per_user"]["pos_rank_mean"] - baseline_eval["per_user"]["pos_rank_mean"], mask),
                "baseline_pos_rank_min": mean_valid(baseline_eval["per_user"]["pos_rank_min"], mask),
                "module2_pos_rank_min": mean_valid(module2_eval["per_user"]["pos_rank_min"], mask),
                "delta_pos_rank_min": mean_valid(module2_eval["per_user"]["pos_rank_min"] - baseline_eval["per_user"]["pos_rank_min"], mask),
            }
        )
        rows.append(row)
    return rows


def safe_rel(new, old):
    if old is None or np.isnan(old) or abs(old) < 1.0e-12:
        return float("nan")
    return float((new - old) / old)


def alignment_rows(baseline_eval, module2_eval, connectivity, bucket_specs):
    rows = []
    metrics = [
        "user_cos_UB_BI",
        "user_cos_UB_UI",
        "user_cos_UI_BI",
        "bundle_cos_UB_BI",
        "bundle_cos_UB_UI",
        "bundle_cos_UI_BI",
        "user_norm_fused",
        "bundle_norm_fused",
    ]
    for metric in metrics:
        if metric.startswith("bundle_"):
            b_stats = summarize_array(baseline_eval["alignments"][metric])
            m_stats = summarize_array(module2_eval["alignments"][metric])
            rows.append(alignment_summary_row("overall", "all_bundles", metric, b_stats, m_stats))
        else:
            b_values = baseline_eval["alignments"][metric]
            m_values = module2_eval["alignments"][metric]
            valid_users = ~np.isnan(b_values) & ~np.isnan(m_values)
            b_stats = summarize_array(b_values[valid_users])
            m_stats = summarize_array(m_values[valid_users])
            rows.append(alignment_summary_row("overall", "all_users", metric, b_stats, m_stats))

    for feature_name, specs in bucket_specs.items():
        values = connectivity[feature_name]
        valid_feature = ~np.isnan(values)
        for bucket_name, fn in specs:
            mask = fn(values) & valid_feature
            for metric in ["user_cos_UB_BI", "user_cos_UB_UI", "user_cos_UI_BI", "user_norm_fused"]:
                b_values = baseline_eval["alignments"][metric]
                m_values = module2_eval["alignments"][metric]
                valid = mask & ~np.isnan(b_values) & ~np.isnan(m_values)
                rows.append(
                    alignment_summary_row(
                        feature_name,
                        bucket_name,
                        metric,
                        summarize_array(b_values[valid]),
                        summarize_array(m_values[valid]),
                    )
                )
    return rows


def alignment_summary_row(feature, bucket, metric, b_stats, m_stats):
    return {
        "feature": feature,
        "bucket": bucket,
        "metric": metric,
        "count": min(b_stats["count"], m_stats["count"]),
        "baseline_mean": b_stats["mean"],
        "module2_mean": m_stats["mean"],
        "delta_mean": m_stats["mean"] - b_stats["mean"],
        "baseline_p25": b_stats["p25"],
        "module2_p25": m_stats["p25"],
        "delta_p25": m_stats["p25"] - b_stats["p25"],
        "baseline_p50": b_stats["p50"],
        "module2_p50": m_stats["p50"],
        "delta_p50": m_stats["p50"] - b_stats["p50"],
        "baseline_p75": b_stats["p75"],
        "module2_p75": m_stats["p75"],
        "delta_p75": m_stats["p75"] - b_stats["p75"],
    }


def two_dimensional_rows(baseline_eval, module2_eval, connectivity):
    ub_deg = connectivity["user_ub_train_deg"]
    conn = connectivity["train_connected_ratio"]
    valid_users = baseline_eval["per_user"]["num_pos"] > 0
    ub_specs = [
        ("ub_low_0_8", (ub_deg >= 0) & (ub_deg <= 8)),
        ("ub_mid_9_19", (ub_deg >= 9) & (ub_deg <= 19)),
        ("ub_high_20_plus", ub_deg >= 20),
    ]
    conn_specs = [
        ("conn_low_0_0.25", (conn >= 0) & (conn <= 0.25)),
        ("conn_high_gt_0.25", conn > 0.25),
    ]
    rows = []
    for ub_name, ub_mask in ub_specs:
        for conn_name, conn_mask in conn_specs:
            mask = ub_mask & conn_mask & valid_users
            rows.extend(metric_rows_for_mask("user_ub_train_deg_x_train_connected_ratio", f"{ub_name}|{conn_name}", mask, baseline_eval, module2_eval, connectivity))
    return rows


def load_graph_sets(dataset):
    ui_graph = dataset.graphs[1].tocsr()
    bi_graph = dataset.graphs[2].tocsr()
    test_graph = dataset.bundle_test_data.u_b_graph.tocsr()
    train_graph = dataset.bundle_train_data.u_b_graph.tocsr()
    bundle_pop = np.asarray(train_graph.sum(axis=0)).ravel()
    return ui_graph, bi_graph, test_graph, bundle_pop


def overlap_for_pair(ui_graph, bi_graph, user_id, bundle_id):
    user_items = ui_graph.indices[ui_graph.indptr[user_id]: ui_graph.indptr[user_id + 1]]
    bundle_items = bi_graph.indices[bi_graph.indptr[bundle_id]: bi_graph.indptr[bundle_id + 1]]
    if bundle_items.size == 0:
        return 0, 0.0
    count = int(np.intersect1d(user_items, bundle_items, assume_unique=False).size)
    return count, float(count / max(bundle_items.size, 1))


def new_hit_analysis(baseline_eval, module2_eval, dataset, connectivity, topk_values=(20, 80)):
    ui_graph, bi_graph, test_graph, bundle_pop = load_graph_sets(dataset)
    rows = []
    pair_rows = []
    for topk in topk_values:
        new_hit_features = []
        lost_hit_features = []
        both_hit_features = []
        for user_id in range(dataset.num_users):
            positives = test_graph.indices[test_graph.indptr[user_id]: test_graph.indptr[user_id + 1]]
            if positives.size == 0:
                continue
            positives_set = set(int(x) for x in positives)
            b_top = set(int(x) for x in baseline_eval["top_indices"][user_id, :topk])
            m_top = set(int(x) for x in module2_eval["top_indices"][user_id, :topk])
            new_hits = sorted((positives_set & m_top) - b_top)
            lost_hits = sorted((positives_set & b_top) - m_top)
            both_hits = sorted(positives_set & b_top & m_top)

            for hit_type, bundles, container in [
                ("new_hit", new_hits, new_hit_features),
                ("lost_hit", lost_hits, lost_hit_features),
                ("both_hit", both_hits, both_hit_features),
            ]:
                for bundle_id in bundles:
                    overlap_count, overlap_ratio = overlap_for_pair(ui_graph, bi_graph, user_id, bundle_id)
                    feature = {
                        "topk": topk,
                        "hit_type": hit_type,
                        "user_id": user_id,
                        "bundle_id": bundle_id,
                        "train_connected_ratio": connectivity["train_connected_ratio"][user_id],
                        "avg_overlap_ratio_per_train_bundle": connectivity["avg_overlap_ratio_per_train_bundle"][user_id],
                        "user_test_recover_zero_ratio": connectivity["user_test_recover_zero_ratio"][user_id],
                        "user_ub_train_deg": connectivity["user_ub_train_deg"][user_id],
                        "user_ui_deg": connectivity["user_ui_deg"][user_id],
                        "test_overlap_count": overlap_count,
                        "test_overlap_ratio": overlap_ratio,
                        "bundle_train_popularity": float(bundle_pop[bundle_id]),
                    }
                    container.append(feature)
                    if hit_type in ["new_hit", "lost_hit"]:
                        pair_rows.append(feature)

        for hit_type, features in [
            ("new_hit", new_hit_features),
            ("lost_hit", lost_hit_features),
            ("both_hit", both_hit_features),
        ]:
            rows.append(summarize_hit_features(topk, hit_type, features))
    return rows, pair_rows


def summarize_hit_features(topk, hit_type, features):
    row = {"topk": topk, "hit_type": hit_type, "count": len(features)}
    keys = [
        "train_connected_ratio",
        "avg_overlap_ratio_per_train_bundle",
        "user_test_recover_zero_ratio",
        "user_ub_train_deg",
        "user_ui_deg",
        "test_overlap_count",
        "test_overlap_ratio",
        "bundle_train_popularity",
    ]
    for key in keys:
        values = np.array([f[key] for f in features], dtype=np.float64) if features else np.array([])
        stats = summarize_array(values)
        row[f"{key}_mean"] = stats["mean"]
        row[f"{key}_p50"] = stats["p50"]
    return row


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def overall_rows(manifest):
    rows = []
    for name in ["baseline", "module2"]:
        exp = manifest["experiments"].get(name, {})
        metrics = exp.get("best_metrics", {})
        if not metrics:
            continue
        for topk in TOPKS:
            rows.append(
                {
                    "experiment": name,
                    "best_epoch": exp.get("best_epoch"),
                    "topk": topk,
                    "val_recall": float(metrics["val"]["recall"][str(topk)]),
                    "val_ndcg": float(metrics["val"]["ndcg"][str(topk)]),
                    "test_recall": float(metrics["test"]["recall"][str(topk)]),
                    "test_ndcg": float(metrics["test"]["ndcg"][str(topk)]),
                }
            )
    return rows


def delta_overall_rows(rows):
    by_name = {(row["experiment"], row["topk"]): row for row in rows}
    result = []
    for topk in TOPKS:
        base = by_name.get(("baseline", topk))
        mod = by_name.get(("module2", topk))
        if not base or not mod:
            continue
        result.append(
            {
                "topk": topk,
                "baseline_test_recall": base["test_recall"],
                "module2_test_recall": mod["test_recall"],
                "delta_test_recall": mod["test_recall"] - base["test_recall"],
                "rel_test_recall": safe_rel(mod["test_recall"], base["test_recall"]),
                "baseline_test_ndcg": base["test_ndcg"],
                "module2_test_ndcg": mod["test_ndcg"],
                "delta_test_ndcg": mod["test_ndcg"] - base["test_ndcg"],
                "rel_test_ndcg": safe_rel(mod["test_ndcg"], base["test_ndcg"]),
            }
        )
    return result


def analyze_successful_runs(manifest, experiments, force_analysis=False):
    for name in ["baseline", "module2"]:
        exp = manifest["experiments"].get(name)
        if not exp or exp.get("status") != "SUCCESS":
            raise RuntimeError(f"Experiment {name} is not successful; cannot run mechanism analysis.")

    if REPORT_PATH.exists() and not force_analysis:
        print(f"[{now()}] Existing report found at {REPORT_PATH}; use --force-analysis to rebuild it.")
        return

    base_conf = {exp["name"]: exp["conf"] for exp in experiments}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    analysis_conf = copy.deepcopy(base_conf["baseline"])
    analysis_conf["gpu"] = str(analysis_conf.get("gpu", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = analysis_conf["gpu"]
    dataset = load_dataset_for_analysis(analysis_conf)

    evals = {}
    for name in ["baseline", "module2"]:
        checkpoint_path = manifest["experiments"][name]["checkpoint_path"]
        model, model_conf = instantiate_model(base_conf[name], dataset, checkpoint_path, device)
        evals[name] = evaluate_user_level(model, dataset, model_conf, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    connectivity = connectivity_arrays(dataset.num_users, load_connectivity_metrics("NetEase"))
    bucket_specs = build_bucket_specs(connectivity)

    overall = overall_rows(manifest)
    delta_overall = delta_overall_rows(overall)
    bucket_rows = bucket_metric_rows(evals["baseline"], evals["module2"], connectivity, bucket_specs)
    two_d_rows = two_dimensional_rows(evals["baseline"], evals["module2"], connectivity)
    align_rows = alignment_rows(evals["baseline"], evals["module2"], connectivity, bucket_specs)
    hit_rows, hit_pair_rows = new_hit_analysis(evals["baseline"], evals["module2"], dataset, connectivity)

    write_csv(TABLE_DIR / "overall_metrics.csv", overall)
    write_csv(TABLE_DIR / "overall_delta.csv", delta_overall)
    write_csv(TABLE_DIR / "bucket_metrics.csv", bucket_rows)
    write_csv(TABLE_DIR / "two_dimensional_bucket_metrics.csv", two_d_rows)
    write_csv(TABLE_DIR / "alignment_metrics.csv", align_rows)
    write_csv(TABLE_DIR / "new_hit_summary.csv", hit_rows)
    write_csv(TABLE_DIR / "new_hit_pairs.csv", hit_pair_rows)

    report = build_report(manifest, overall, delta_overall, bucket_rows, two_d_rows, align_rows, hit_rows)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[{now()}] Report written to {REPORT_PATH}")


def fmt_float(value, digits=6):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def build_report(manifest, overall, delta_overall, bucket_rows, two_d_rows, align_rows, hit_rows):
    lines = []
    lines.append("===== NetEase Module2 Mechanism Study =====")
    lines.append(f"generated_at: {now()}")
    lines.append("")
    lines.append("This report compares exactly two conditions: baseline vs module2.")
    lines.append("module2 is treated as one complete module: BI-side user enhancement (beta=0.35) + pre-fusion anchor CL (lambda=0.005, temp=0.2).")
    lines.append("No beta/lambda tuning, no module splitting, no DWT, no latent rebuild.")
    lines.append("")

    lines.append("===== Experiment Configs =====")
    for name in ["baseline", "module2"]:
        exp = manifest["experiments"].get(name, {})
        lines.append(f"[{name}] status={exp.get('status')} best_epoch={exp.get('best_epoch')} checkpoint={exp.get('checkpoint_path')}")
        lines.append(json.dumps(exp.get("config", {}), indent=2, ensure_ascii=False))
    lines.append("")

    lines.append("===== Overall Test Metrics =====")
    for row in overall:
        lines.append(
            f"{row['experiment']} TOP{row['topk']} "
            f"VAL_REC={fmt_float(row['val_recall'])} VAL_NDCG={fmt_float(row['val_ndcg'])} "
            f"TEST_REC={fmt_float(row['test_recall'])} TEST_NDCG={fmt_float(row['test_ndcg'])}"
        )
    lines.append("")

    lines.append("===== Overall Delta: module2 - baseline =====")
    for row in delta_overall:
        lines.append(
            f"TOP{row['topk']} "
            f"DELTA_REC={fmt_float(row['delta_test_recall'])} REL_REC={fmt_float(row['rel_test_recall'] * 100, 2)}% "
            f"DELTA_NDCG={fmt_float(row['delta_test_ndcg'])} REL_NDCG={fmt_float(row['rel_test_ndcg'] * 100, 2)}%"
        )
    lines.append("")

    append_bucket_section(lines, "train_connected_ratio", bucket_rows)
    append_bucket_section(lines, "avg_overlap_ratio_per_train_bundle", bucket_rows)
    append_bucket_section(lines, "user_test_recover_zero_ratio", bucket_rows)
    append_bucket_section(lines, "user_ub_train_deg", bucket_rows)
    append_bucket_section(lines, "user_ui_deg", bucket_rows)
    append_bucket_section(lines, "user_ub_train_deg_x_train_connected_ratio", two_d_rows, title="2D Bucket: user_ub_train_deg x train_connected_ratio")

    lines.append("===== Alignment Overall =====")
    for row in align_rows:
        if row["feature"] == "overall":
            lines.append(
                f"{row['metric']} {row['bucket']} count={row['count']} "
                f"base_mean={fmt_float(row['baseline_mean'])} module2_mean={fmt_float(row['module2_mean'])} delta={fmt_float(row['delta_mean'])} "
                f"base_p50={fmt_float(row['baseline_p50'])} module2_p50={fmt_float(row['module2_p50'])}"
            )
    lines.append("")

    lines.append("===== Alignment By train_connected_ratio: focus on user_cos_UB_BI =====")
    for row in align_rows:
        if row["feature"] == "train_connected_ratio" and row["metric"] == "user_cos_UB_BI":
            delta_ndcg20 = find_bucket_delta(bucket_rows, "train_connected_ratio", row["bucket"], 20, "delta_ndcg")
            lines.append(
                f"{row['bucket']} count={row['count']} "
                f"base={fmt_float(row['baseline_mean'])} module2={fmt_float(row['module2_mean'])} "
                f"delta_cos={fmt_float(row['delta_mean'])} delta_ndcg20={fmt_float(delta_ndcg20)}"
            )
    lines.append("")

    lines.append("===== New Hit Analysis =====")
    for row in hit_rows:
        lines.append(
            f"TOP{row['topk']} {row['hit_type']} count={row['count']} "
            f"conn_mean={fmt_float(row['train_connected_ratio_mean'])} "
            f"train_overlap_mean={fmt_float(row['avg_overlap_ratio_per_train_bundle_mean'])} "
            f"test_overlap_mean={fmt_float(row['test_overlap_ratio_mean'])} "
            f"recover_zero_mean={fmt_float(row['user_test_recover_zero_ratio_mean'])} "
            f"bundle_pop_mean={fmt_float(row['bundle_train_popularity_mean'])}"
        )
    lines.append("")

    lines.append("===== Conclusion Hints =====")
    lines.extend(conclusion_hints(bucket_rows, align_rows, hit_rows))
    lines.append("")
    lines.append("Auxiliary CSV files are under analysis_outputs/netease_module2_study/tables/.")
    return "\n".join(lines) + "\n"


def append_bucket_section(lines, feature, rows, title=None):
    title = title or f"Bucket Metrics: {feature}"
    lines.append(f"===== {title} =====")
    filtered = [row for row in rows if row["feature"] == feature and row["topk"] in [20, 80]]
    for row in filtered:
        lines.append(
            f"{row['bucket']} TOP{row['topk']} users={row['users']} "
            f"base_R={fmt_float(row['baseline_recall'])} module2_R={fmt_float(row['module2_recall'])} dR={fmt_float(row['delta_recall'])} "
            f"base_N={fmt_float(row['baseline_ndcg'])} module2_N={fmt_float(row['module2_ndcg'])} dN={fmt_float(row['delta_ndcg'])} "
            f"rank_mean_base={fmt_float(row['baseline_pos_rank_mean'], 2)} rank_mean_module2={fmt_float(row['module2_pos_rank_mean'], 2)} dRank={fmt_float(row['delta_pos_rank_mean'], 2)} "
            f"conn={fmt_float(row['avg_train_connected_ratio'])} ub_deg={fmt_float(row['avg_user_ub_train_deg'])} overlap={fmt_float(row['avg_overlap_ratio_per_train_bundle'])}"
        )
    lines.append("")


def find_bucket_delta(rows, feature, bucket, topk, key):
    for row in rows:
        if row["feature"] == feature and row["bucket"] == bucket and row["topk"] == topk:
            return row.get(key)
    return float("nan")


def conclusion_hints(bucket_rows, align_rows, hit_rows):
    hints = []
    candidates = [
        row for row in bucket_rows
        if row["feature"] == "train_connected_ratio" and row["topk"] == 20 and row["users"] > 0
    ]
    if candidates:
        best = max(candidates, key=lambda r: -1e9 if np.isnan(r["delta_ndcg"]) else r["delta_ndcg"])
        hints.append(
            "largest_delta_ndcg20_by_train_connected_ratio="
            f"{best['bucket']} delta_ndcg20={fmt_float(best['delta_ndcg'])} users={best['users']}"
        )

    align_candidates = [
        row for row in align_rows
        if row["feature"] == "train_connected_ratio" and row["metric"] == "user_cos_UB_BI" and row["count"] > 0
    ]
    if align_candidates:
        best = max(align_candidates, key=lambda r: -1e9 if np.isnan(r["delta_mean"]) else r["delta_mean"])
        hints.append(
            "largest_delta_user_cos_UB_BI_by_train_connected_ratio="
            f"{best['bucket']} delta_cos={fmt_float(best['delta_mean'])} count={best['count']}"
        )

    hit20 = {row["hit_type"]: row for row in hit_rows if row["topk"] == 20}
    if "new_hit" in hit20 and "both_hit" in hit20:
        new_row = hit20["new_hit"]
        both_row = hit20["both_hit"]
        hints.append(
            "top20_new_hit_vs_both_hit="
            f"new_conn={fmt_float(new_row['train_connected_ratio_mean'])}, both_conn={fmt_float(both_row['train_connected_ratio_mean'])}; "
            f"new_test_overlap={fmt_float(new_row['test_overlap_ratio_mean'])}, both_test_overlap={fmt_float(both_row['test_overlap_ratio_mean'])}"
        )

    if not hints:
        hints.append("No automatic hints available; inspect the tables manually.")
    return hints


def parse_args():
    parser = argparse.ArgumentParser(description="Run NetEase module2 mechanism study.")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--max-retries", default=3, type=int)
    parser.add_argument("--force", action="store_true", help="rerun successful experiments")
    parser.add_argument("--force-analysis", action="store_true", help="rebuild report even if it exists")
    parser.add_argument("--analysis-only", action="store_true", help="skip training and analyze existing successful checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    experiments = build_experiments(args.gpu, args.epochs)
    if args.analysis_only:
        manifest = load_manifest()
    else:
        manifest = run_experiments(experiments, args.max_retries, args.force)
    analyze_successful_runs(manifest, experiments, force_analysis=args.force_analysis or args.force)


if __name__ == "__main__":
    main()
