#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize NetEase Module2 analysis packages into six paper experiments."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASET = "NetEase"
DEFAULT_INPUT_ROOT = Path("analysis_outputs") / DATASET
TOPK = 20

RUNS = {
    "A": "run_A_baseline",
    "B": "run_B_full_module2",
    "C": "run_C_beta_only",
    "D1": "run_D1_ub_ui_only_beta035",
    "D2": "run_D2_ub_ui_only_beta000",
    "E1": "run_E1_ub_bi_only_beta035",
    "E2": "run_E2_ub_bi_only_beta000",
    "F1": "run_F1_user_side_only_beta035",
    "F2": "run_F2_user_side_only_beta000",
    "G1": "run_G1_bundle_side_only_beta035",
    "G2": "run_G2_bundle_side_only_beta000",
    "H": "run_H_bi_drop_0.1_baseline",
    "I": "run_I_bi_drop_0.1_full_module2",
    "J": "run_J_bi_drop_0.2_baseline",
    "K": "run_K_bi_drop_0.2_full_module2",
    "L": "run_L_bi_drop_0.3_baseline",
    "M": "run_M_bi_drop_0.3_full_module2",
    "N": "run_N_bi_drop_0.4_baseline",
    "O": "run_O_bi_drop_0.4_full_module2",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_rel(new, old):
    old = float(old)
    new = float(new)
    if abs(old) < 1e-12:
        return None
    return (new - old) / old


def require_run(root, key, allow_missing=False):
    run_dir = root / RUNS[key]
    required = [run_dir / "metrics.json", run_dir / "config.yaml"]
    if all(path.exists() for path in required):
        return run_dir
    if allow_missing:
        return None
    raise FileNotFoundError(f"Missing analysis package for {key}: {run_dir}")


def load_metrics(root, key):
    return read_json(root / RUNS[key] / "metrics.json")


def load_similarity(root, key):
    return read_json(root / RUNS[key] / "view_similarity.json")


def load_per_user(root, key):
    rows = read_csv_dicts(root / RUNS[key] / "per_user_metrics.csv")
    parsed = {}
    for row in rows:
        user_id = int(row["user_id"])
        parsed[user_id] = {
            "train_ub_degree": int(row["train_ub_degree"]),
            "test_pos_count": int(row["test_pos_count"]),
            "recall@20": float(row["recall@20"]),
            "ndcg@20": float(row["ndcg@20"]),
        }
    return parsed


def load_bundle_degrees(root, key="A"):
    rows = read_csv_dicts(root / RUNS[key] / "bundle_train_degree.csv")
    bundle_ids = []
    train_degrees = []
    test_counts = []
    for row in rows:
        bundle_ids.append(int(row["bundle_id"]))
        train_degrees.append(int(row["train_ub_degree"]))
        test_counts.append(int(row["test_pos_count"]))
    return np.asarray(bundle_ids), np.asarray(train_degrees), np.asarray(test_counts)


def mean_group_metric(per_user, users, metric):
    values = [per_user[int(u)][metric] for u in users if per_user[int(u)]["test_pos_count"] > 0]
    return float(np.mean(values)) if values else 0.0


def split_by_degree(ids, degrees, n_groups):
    order = np.lexsort((ids, degrees))
    sorted_ids = ids[order]
    return np.array_split(sorted_ids, n_groups)


def experiment1_user_sparsity(root, out_dir):
    base = load_per_user(root, "A")
    full = load_per_user(root, "B")
    user_ids = np.asarray(sorted(base.keys()), dtype=np.int64)
    degrees = np.asarray([base[int(u)]["train_ub_degree"] for u in user_ids], dtype=np.int64)
    all_outputs = {}

    for n_groups in [3, 4, 5]:
        rows = []
        groups = split_by_degree(user_ids, degrees, n_groups)
        for group_idx, users in enumerate(groups):
            group_degrees = np.asarray([base[int(u)]["train_ub_degree"] for u in users], dtype=np.float64)
            b_recall = mean_group_metric(base, users, "recall@20")
            m_recall = mean_group_metric(full, users, "recall@20")
            b_ndcg = mean_group_metric(base, users, "ndcg@20")
            m_ndcg = mean_group_metric(full, users, "ndcg@20")
            row = {
                "group_id": group_idx + 1,
                "group_name": f"group_{group_idx + 1}_of_{n_groups}",
                "number_of_users": int(len(users)),
                "number_of_eval_users": int(sum(base[int(u)]["test_pos_count"] > 0 for u in users)),
                "train_ub_degree_min": int(group_degrees.min()) if len(group_degrees) else 0,
                "train_ub_degree_max": int(group_degrees.max()) if len(group_degrees) else 0,
                "train_ub_degree_mean": float(group_degrees.mean()) if len(group_degrees) else 0.0,
                "baseline_recall@20": b_recall,
                "module2_recall@20": m_recall,
                "relative_improvement_recall@20": safe_rel(m_recall, b_recall),
                "baseline_ndcg@20": b_ndcg,
                "module2_ndcg@20": m_ndcg,
                "relative_improvement_ndcg@20": safe_rel(m_ndcg, b_ndcg),
            }
            rows.append(row)
        all_outputs[f"{n_groups}_groups"] = rows
        fieldnames = list(rows[0].keys())
        write_csv(out_dir / f"experiment1_user_sparsity_{n_groups}groups.csv", rows, fieldnames)
        write_json(out_dir / f"experiment1_user_sparsity_{n_groups}groups.json", rows)
    return all_outputs


def experiment2_view_similarity(root, out_dir):
    base = load_similarity(root, "A")
    full = load_similarity(root, "B")
    rows = []
    for side, pairs in {
        "user": ["UB_UI", "UB_BI", "UI_BI"],
        "bundle": ["UB_UI", "UB_BI", "UI_BI"],
    }.items():
        for pair in pairs:
            key = f"{side}_{pair}"
            rows.append({
                "side": side,
                "view_pair": pair,
                "baseline_similarity": base[key],
                "module2_similarity": full[key],
                "gain": full[key] - base[key],
                "relative_gain": safe_rel(full[key], base[key]),
                "scope": full.get("scope", "all_users_and_all_bundles"),
            })
    write_csv(out_dir / "experiment2_view_similarity.csv", rows, list(rows[0].keys()))
    write_json(out_dir / "experiment2_view_similarity.json", rows)
    return rows


def build_variant_row(root, key, variant_name, beta_setting, extra):
    metrics = load_metrics(root, key)
    sim = load_similarity(root, key)
    row = {
        "variant_key": key,
        "variant_name": variant_name,
        "beta_setting": beta_setting,
        "recall@20": metrics["recall@20"],
        "ndcg@20": metrics["ndcg@20"],
        "user_UB_BI_similarity": sim["user_UB_BI"],
        "bundle_UB_BI_similarity": sim["bundle_UB_BI"],
    }
    row.update(extra)
    return row


def add_relative_columns(rows, baseline_row, beta_row=None):
    for row in rows:
        row["relative_improvement_vs_baseline_recall@20"] = safe_rel(row["recall@20"], baseline_row["recall@20"])
        row["relative_improvement_vs_baseline_ndcg@20"] = safe_rel(row["ndcg@20"], baseline_row["ndcg@20"])
        if beta_row is not None:
            row["relative_improvement_vs_beta_only_recall@20"] = safe_rel(row["recall@20"], beta_row["recall@20"])
            row["relative_improvement_vs_beta_only_ndcg@20"] = safe_rel(row["ndcg@20"], beta_row["ndcg@20"])
    return rows


def experiment3_alignment_paths(root, out_dir):
    rows = [
        build_variant_row(root, "A", "Baseline", "0+0", {
            "anchor_cl_enabled": False,
            "ub_ui_align_enabled": False,
            "ub_bi_align_enabled": False,
        }),
        build_variant_row(root, "C", "Beta-only", "0+0.35", {
            "anchor_cl_enabled": False,
            "ub_ui_align_enabled": False,
            "ub_bi_align_enabled": False,
        }),
        build_variant_row(root, "D1", "UB-UI only", "0+0.35", {
            "anchor_cl_enabled": True,
            "ub_ui_align_enabled": True,
            "ub_bi_align_enabled": False,
        }),
        build_variant_row(root, "E1", "UB-BI only", "0+0.35", {
            "anchor_cl_enabled": True,
            "ub_ui_align_enabled": False,
            "ub_bi_align_enabled": True,
        }),
        build_variant_row(root, "B", "Full Module2", "0+0.35", {
            "anchor_cl_enabled": True,
            "ub_ui_align_enabled": True,
            "ub_bi_align_enabled": True,
        }),
    ]
    rows = add_relative_columns(rows, rows[0], rows[1])
    write_csv(out_dir / "experiment3_alignment_paths_main.csv", rows, list(rows[0].keys()))
    write_json(out_dir / "experiment3_alignment_paths_main.json", rows)

    supplemental = [
        build_variant_row(root, "D2", "UB-UI only beta0", "0+0", {
            "anchor_cl_enabled": True,
            "ub_ui_align_enabled": True,
            "ub_bi_align_enabled": False,
        }),
        build_variant_row(root, "E2", "UB-BI only beta0", "0+0", {
            "anchor_cl_enabled": True,
            "ub_ui_align_enabled": False,
            "ub_bi_align_enabled": True,
        }),
    ]
    supplemental = add_relative_columns(supplemental, rows[0], rows[1])
    write_csv(out_dir / "experiment3_alignment_paths_supplemental_beta0.csv", supplemental, list(supplemental[0].keys()))
    write_json(out_dir / "experiment3_alignment_paths_supplemental_beta0.json", supplemental)
    return {"main": rows, "supplemental_beta0": supplemental}


def load_top20(root, key):
    return np.load(root / RUNS[key] / "top20_recommendations.npy")


def load_ground_truth(root, key="A"):
    payload = read_json(root / RUNS[key] / "test_ground_truth.json")
    return {int(k): [int(x) for x in v] for k, v in payload.items()}


def group_bundle_metric(top20, ground_truth, bundle_set):
    recalls = []
    ndcgs = []
    positives = 0
    eval_users = 0
    for user_id, user_gt in ground_truth.items():
        scoped_gt = [b for b in user_gt if b in bundle_set]
        if not scoped_gt:
            continue
        eval_users += 1
        positives += len(scoped_gt)
        hits = len(set(map(int, top20[user_id])) & set(scoped_gt))
        recalls.append(hits / len(scoped_gt))
        ndcgs.append(per_user_ndcg(top20[user_id], scoped_gt))
    return {
        "number_of_test_positives": int(positives),
        "number_of_eval_users": int(eval_users),
        "recall@20": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@20": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


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


def experiment4_bundle_tail(root, out_dir):
    bundle_ids, degrees, _ = load_bundle_degrees(root)
    degree_by_bundle = {int(b): int(d) for b, d in zip(bundle_ids, degrees)}
    groups = split_by_degree(bundle_ids, degrees, 4)
    names = ["Tail", "Mid-tail", "Mid-head", "Head"]
    base_top20 = load_top20(root, "A")
    full_top20 = load_top20(root, "B")
    gt = load_ground_truth(root)
    rows = []
    for idx, bundles in enumerate(groups):
        bundle_set = set(map(int, bundles))
        group_degrees = np.asarray([degree_by_bundle[int(b)] for b in bundles], dtype=np.float64)
        base_metrics = group_bundle_metric(base_top20, gt, bundle_set)
        full_metrics = group_bundle_metric(full_top20, gt, bundle_set)
        rows.append({
            "group_id": idx + 1,
            "group_name": names[idx],
            "number_of_bundles": int(len(bundles)),
            "number_of_test_positives": base_metrics["number_of_test_positives"],
            "number_of_eval_users": base_metrics["number_of_eval_users"],
            "bundle_train_degree_min": int(group_degrees.min()) if len(group_degrees) else 0,
            "bundle_train_degree_max": int(group_degrees.max()) if len(group_degrees) else 0,
            "bundle_train_degree_mean": float(group_degrees.mean()) if len(group_degrees) else 0.0,
            "baseline_recall@20": base_metrics["recall@20"],
            "module2_recall@20": full_metrics["recall@20"],
            "relative_improvement_recall@20": safe_rel(full_metrics["recall@20"], base_metrics["recall@20"]),
            "baseline_ndcg@20": base_metrics["ndcg@20"],
            "module2_ndcg@20": full_metrics["ndcg@20"],
            "relative_improvement_ndcg@20": safe_rel(full_metrics["ndcg@20"], base_metrics["ndcg@20"]),
        })
    write_csv(out_dir / "experiment4_bundle_tail.csv", rows, list(rows[0].keys()))
    write_json(out_dir / "experiment4_bundle_tail.json", rows)
    return rows


def experiment5_side_contribution(root, out_dir):
    rows = [
        build_variant_row(root, "A", "Baseline", "0+0", {
            "user_cl_enabled": False,
            "bundle_cl_enabled": False,
        }),
        build_variant_row(root, "C", "Beta-only", "0+0.35", {
            "user_cl_enabled": False,
            "bundle_cl_enabled": False,
        }),
        build_variant_row(root, "F1", "User-side only", "0+0.35", {
            "user_cl_enabled": True,
            "bundle_cl_enabled": False,
        }),
        build_variant_row(root, "G1", "Bundle-side only", "0+0.35", {
            "user_cl_enabled": False,
            "bundle_cl_enabled": True,
        }),
        build_variant_row(root, "B", "Full Module2", "0+0.35", {
            "user_cl_enabled": True,
            "bundle_cl_enabled": True,
        }),
    ]
    rows = add_relative_columns(rows, rows[0], rows[1])
    write_csv(out_dir / "experiment5_side_contribution_main.csv", rows, list(rows[0].keys()))
    write_json(out_dir / "experiment5_side_contribution_main.json", rows)

    supplemental = [
        build_variant_row(root, "F2", "User-side only beta0", "0+0", {
            "user_cl_enabled": True,
            "bundle_cl_enabled": False,
        }),
        build_variant_row(root, "G2", "Bundle-side only beta0", "0+0", {
            "user_cl_enabled": False,
            "bundle_cl_enabled": True,
        }),
    ]
    supplemental = add_relative_columns(supplemental, rows[0], rows[1])
    write_csv(out_dir / "experiment5_side_contribution_supplemental_beta0.csv", supplemental, list(supplemental[0].keys()))
    write_json(out_dir / "experiment5_side_contribution_supplemental_beta0.json", supplemental)
    return {"main": rows, "supplemental_beta0": supplemental}


def experiment6_bi_drop(root, out_dir):
    pairs = [
        (0.0, "A", "B"),
        (0.1, "H", "I"),
        (0.2, "J", "K"),
        (0.3, "L", "M"),
        (0.4, "N", "O"),
    ]
    rows = []
    for ratio, base_key, full_key in pairs:
        base = load_metrics(root, base_key)
        full = load_metrics(root, full_key)
        gap_recall = full["recall@20"] - base["recall@20"]
        gap_ndcg = full["ndcg@20"] - base["ndcg@20"]
        rows.append({
            "bi_drop_ratio": ratio,
            "baseline_recall@20": base["recall@20"],
            "module2_recall@20": full["recall@20"],
            "gap_recall@20": gap_recall,
            "baseline_ndcg@20": base["ndcg@20"],
            "module2_ndcg@20": full["ndcg@20"],
            "gap_ndcg@20": gap_ndcg,
            "relative_gap_recall@20": safe_rel(full["recall@20"], base["recall@20"]),
            "relative_gap_ndcg@20": safe_rel(full["ndcg@20"], base["ndcg@20"]),
        })
    write_csv(out_dir / "experiment6_bi_drop_robustness.csv", rows, list(rows[0].keys()))
    write_json(out_dir / "experiment6_bi_drop_robustness.json", rows)
    return rows


def verify_packages(root, allow_missing):
    required_keys = list(RUNS.keys())
    missing = []
    for key in required_keys:
        run_dir = root / RUNS[key]
        if not (run_dir / "metrics.json").exists():
            missing.append(key)
    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing required runs: {missing}")
    return missing


def main():
    args = parse_args()
    root = Path(args.input_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = verify_packages(root, args.allow_missing)
    if missing:
        print(f"Missing runs ignored because --allow-missing was set: {missing}")

    summary = {
        "dataset": DATASET,
        "experiment1_user_sparsity": experiment1_user_sparsity(root, out_dir),
        "experiment2_view_similarity": experiment2_view_similarity(root, out_dir),
        "experiment3_alignment_paths": experiment3_alignment_paths(root, out_dir),
        "experiment4_bundle_tail": experiment4_bundle_tail(root, out_dir),
        "experiment5_side_contribution": experiment5_side_contribution(root, out_dir),
        "experiment6_bi_drop": experiment6_bi_drop(root, out_dir),
    }
    write_json(out_dir / "module2_analysis_summary.json", summary)
    print(f"Analysis results written to {out_dir}")


if __name__ == "__main__":
    main()
