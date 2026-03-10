import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze feasibility of hard negatives for MultiCBR datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. NetEase / iFashion / Youshu")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Root path to datasets")
    parser.add_argument("--output_dir", type=str, default="./hard_negative_analysis", help="Output directory")
    parser.add_argument("--topk", type=int, default=200, help="Top-K truncation for hard candidate pool")
    parser.add_argument("--max_plot_points", type=int, default=200000, help="Max sampled points for plots")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def maybe_to_int(x):
    try:
        return int(x)
    except Exception:
        return x


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def safe_percentile(arr, q, default=0.0):
    if len(arr) == 0:
        return float(default)
    return float(np.percentile(arr, q))


def summarize_numeric(arr):
    arr = list(arr)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    arr_np = np.asarray(arr, dtype=np.float64)
    return {
        "count": int(len(arr_np)),
        "mean": float(np.mean(arr_np)),
        "median": float(np.median(arr_np)),
        "min": float(np.min(arr_np)),
        "max": float(np.max(arr_np)),
        "std": float(np.std(arr_np)),
        "p25": float(np.percentile(arr_np, 25)),
        "p75": float(np.percentile(arr_np, 75)),
        "p90": float(np.percentile(arr_np, 90)),
        "p95": float(np.percentile(arr_np, 95)),
    }


class ReservoirSampler:
    def __init__(self, k: int, seed: int = 2026):
        self.k = max(1, int(k))
        self.data = []
        self.n_seen = 0
        self.rng = random.Random(seed)

    def add(self, value):
        self.n_seen += 1
        if len(self.data) < self.k:
            self.data.append(value)
        else:
            j = self.rng.randint(1, self.n_seen)
            if j <= self.k:
                self.data[j - 1] = value

    def get(self):
        return self.data


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, ensure_ascii=False, indent=2)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def plot_histogram(data, title, xlabel, ylabel, filename, output_dir, bins=50, log=False):
    if data is None or len(data) == 0:
        print(f"[WARN] Skip plotting {filename}: empty data")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.75, edgecolor="black", log=log)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


# -----------------------------
# Import / dataset loading
# -----------------------------
def import_datasets_class():
    """
    Try importing Datasets from current working directory first.
    """
    try:
        from utility import Datasets  # noqa
        return Datasets
    except Exception as e:
        raise ImportError(
            "Failed to import `Datasets` from utility.py. "
            "Please run this script from the MultiCBR project root directory."
        ) from e


def read_config(config_path: str, dataset: str):
    conf = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict):
            if dataset in raw and isinstance(raw[dataset], dict):
                conf.update(raw[dataset])
            else:
                # fallback: merge top-level keys if config is flat
                for k, v in raw.items():
                    if not isinstance(v, dict):
                        conf[k] = v
    return conf


def build_analysis_conf(args):
    conf = read_config(args.config_path, args.dataset)

    # force critical fields
    conf["dataset"] = args.dataset
    conf["data_path"] = args.data_path

    # conservative defaults in case the repo config is missing fields
    conf.setdefault("batch_size_train", 2048)
    conf.setdefault("batch_size_test", 2048)
    conf.setdefault("neg_num", 1)

    return conf


def load_multicbr_dataset(args):
    print(f"[INFO] Loading dataset={args.dataset}, data_path={args.data_path}")
    Datasets = import_datasets_class()
    conf = build_analysis_conf(args)

    print("[INFO] Final analysis config keys:", sorted(conf.keys()))
    dataset = Datasets(conf)
    return dataset, conf


# -----------------------------
# Core structure extraction
# -----------------------------
def require_attr(obj, name: str):
    if not hasattr(obj, name):
        raise AttributeError(f"Object of type {type(obj)} missing required attribute: {name}")
    return getattr(obj, name)


def inspect_dataset(dataset):
    print("[INFO] Inspecting dataset object...")
    num_users = require_attr(dataset, "num_users")
    num_bundles = require_attr(dataset, "num_bundles")
    num_items = require_attr(dataset, "num_items")
    graphs = require_attr(dataset, "graphs")

    if not isinstance(graphs, (list, tuple)) or len(graphs) < 3:
        raise ValueError("dataset.graphs must exist and contain at least 3 matrices: [u_b_train, u_i, b_i]")

    u_b_graph_train = graphs[0]
    b_i_graph = graphs[2]

    print(f"[INFO] num_users={num_users}, num_bundles={num_bundles}, num_items={num_items}")
    print(f"[INFO] graphs len={len(graphs)}")
    print(f"[INFO] u_b_graph_train shape={u_b_graph_train.shape}, nnz={u_b_graph_train.nnz}")
    print(f"[INFO] b_i_graph shape={b_i_graph.shape}, nnz={b_i_graph.nnz}")

    return u_b_graph_train, b_i_graph, num_users, num_bundles, num_items


def extract_train_pairs(dataset):
    """
    Try several common locations used in recommendation repos.
    """
    candidates = []

    if hasattr(dataset, "bundle_train_data"):
        bundle_train_data = dataset.bundle_train_data
        if hasattr(bundle_train_data, "u_b_pairs"):
            candidates.append(("dataset.bundle_train_data.u_b_pairs", bundle_train_data.u_b_pairs))
        if hasattr(bundle_train_data, "pairs"):
            candidates.append(("dataset.bundle_train_data.pairs", bundle_train_data.pairs))

    if hasattr(dataset, "train_data") and hasattr(dataset.train_data, "u_b_pairs"):
        candidates.append(("dataset.train_data.u_b_pairs", dataset.train_data.u_b_pairs))

    for name, pairs in candidates:
        if pairs is None:
            continue
        try:
            pairs = list(pairs)
            if len(pairs) > 0:
                parsed = []
                for p in pairs:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        parsed.append((int(p[0]), int(p[1])))
                if len(parsed) > 0:
                    print(f"[INFO] train_pairs loaded from {name}, count={len(parsed)}")
                    return parsed
        except Exception:
            pass

    raise ValueError(
        "Failed to extract training pairs from dataset. "
        "Please inspect the actual MultiCBR dataset object fields."
    )


def build_core_structures(dataset):
    print("[INFO] Building core data structures...")
    u_b_graph_train, b_i_graph, num_users, num_bundles, num_items = inspect_dataset(dataset)

    # bundle2items
    bundle2items = {}
    for b_id in range(num_bundles):
        start, end = b_i_graph.indptr[b_id], b_i_graph.indptr[b_id + 1]
        bundle2items[b_id] = set(map(int, b_i_graph.indices[start:end]))

    # user2bundles
    user2bundles = {}
    for u_id in range(num_users):
        start, end = u_b_graph_train.indptr[u_id], u_b_graph_train.indptr[u_id + 1]
        user2bundles[u_id] = set(map(int, u_b_graph_train.indices[start:end]))

    # train_pairs
    train_pairs = extract_train_pairs(dataset)

    # sanity samples
    print(f"[INFO] train_pairs count={len(train_pairs)}")
    print(f"[INFO] sample train_pairs={train_pairs[:5]}")
    sample_bundle_sizes = [(b, len(bundle2items[b])) for b in range(min(5, num_bundles))]
    print(f"[INFO] sample bundle sizes={sample_bundle_sizes}")

    return bundle2items, user2bundles, train_pairs, num_users, num_bundles, num_items


def build_item2bundles(bundle2items):
    print("[INFO] Building item -> bundles inverted index...")
    item2bundles = defaultdict(set)
    for b_id, items in bundle2items.items():
        for item in items:
            item2bundles[item].add(b_id)
    print(f"[INFO] item2bundles size={len(item2bundles)}")
    return item2bundles


# -----------------------------
# Neighbor computation
# -----------------------------
def compute_bundle_neighbors(bundle2items, item2bundles, num_bundles, max_plot_points, seed):
    """
    Compute non-zero-overlap neighbors for each bundle.
    Returns:
      neighbor_dict[b] = sorted list of dicts with keys:
          other_bundle, intersection, union, jaccard
      nonzero_stats
      neighbor_counts_for_plot
      jaccards_for_plot
    """
    print("[INFO] Computing bundle neighbors with inverted index...")
    neighbor_dict = {}

    jaccard_sampler = ReservoirSampler(max_plot_points, seed=seed)
    total_nonzero_pairs = 0

    top1_list = []
    top5_mean_list = []
    top10_mean_list = []
    neighbor_counts = []

    for b_id in tqdm(range(num_bundles), desc="Bundles"):
        items = bundle2items[b_id]
        if not items:
            neighbor_dict[b_id] = []
            neighbor_counts.append(0)
            top1_list.append(0.0)
            top5_mean_list.append(0.0)
            top10_mean_list.append(0.0)
            continue

        candidate_intersections = defaultdict(int)
        for item in items:
            for other_b in item2bundles[item]:
                if other_b != b_id:
                    candidate_intersections[other_b] += 1

        neighbors = []
        len_b = len(items)

        for other_b, inter in candidate_intersections.items():
            len_other = len(bundle2items[other_b])
            union = len_b + len_other - inter
            if union <= 0:
                continue
            jac = inter / union
            neighbors.append(
                {
                    "other_bundle": int(other_b),
                    "intersection": int(inter),
                    "union": int(union),
                    "jaccard": float(jac),
                }
            )
            total_nonzero_pairs += 1
            jaccard_sampler.add(float(jac))

        neighbors.sort(key=lambda x: x["jaccard"], reverse=True)
        neighbor_dict[b_id] = neighbors

        ncnt = len(neighbors)
        neighbor_counts.append(ncnt)

        if ncnt == 0:
            top1_list.append(0.0)
            top5_mean_list.append(0.0)
            top10_mean_list.append(0.0)
        else:
            vals = [n["jaccard"] for n in neighbors]
            top1_list.append(float(vals[0]))
            top5_mean_list.append(float(np.mean(vals[:5])))
            top10_mean_list.append(float(np.mean(vals[:10])))

    num_with_neighbors = sum(1 for c in neighbor_counts if c > 0)
    coverage_stats = {
        "with_nonzero_neighbors_ratio": float(num_with_neighbors / num_bundles) if num_bundles > 0 else 0.0,
        "no_nonzero_neighbors_ratio": float(1.0 - num_with_neighbors / num_bundles) if num_bundles > 0 else 0.0,
        "total_nonzero_directed_pairs": int(total_nonzero_pairs),
    }

    # approximate jaccard distribution from reservoir sample for plotting
    sampled_jaccards = jaccard_sampler.get()
    jaccard_dist = summarize_numeric(sampled_jaccards)

    nonzero_stats = {
        "coverage": coverage_stats,
        "neighbor_counts": summarize_numeric(neighbor_counts),
        "jaccard_dist_sampled": jaccard_dist,
        "top_k_quality": {
            "top1": summarize_numeric(top1_list),
            "top5_mean": summarize_numeric(top5_mean_list),
            "top10_mean": summarize_numeric(top10_mean_list),
            "top1_mean_avg": float(np.mean(top1_list)) if len(top1_list) > 0 else 0.0,
            "top5_mean_avg": float(np.mean(top5_mean_list)) if len(top5_mean_list) > 0 else 0.0,
            "top10_mean_avg": float(np.mean(top10_mean_list)) if len(top10_mean_list) > 0 else 0.0,
        },
    }

    return neighbor_dict, nonzero_stats, neighbor_counts, sampled_jaccards, top1_list, top5_mean_list, top10_mean_list


# -----------------------------
# Threshold filtering / availability
# -----------------------------
def filter_neighbors_by_config(neighbors, min_int, low_jac, high_jac):
    out = []
    for n in neighbors:
        if n["intersection"] >= min_int and low_jac <= n["jaccard"] <= high_jac:
            out.append(n)
    return out


def summarize_threshold_coverage(neighbor_dict, threshold_configs, num_bundles, topk):
    print("[INFO] Summarizing threshold coverage...")
    threshold_stats = {}
    candidate_count_details = {}

    for name, cfg in threshold_configs.items():
        min_int = cfg["min_int"]
        low_jac = cfg["low_jac"]
        high_jac = cfg["high_jac"]

        candidate_counts = []
        truncated_counts = []

        for b_id, neighbors in neighbor_dict.items():
            filtered = filter_neighbors_by_config(neighbors, min_int, low_jac, high_jac)
            candidate_counts.append(len(filtered))
            truncated_counts.append(min(len(filtered), topk))

        threshold_stats[name] = {
            "config": {
                "min_intersection": int(min_int),
                "low_jaccard": float(low_jac),
                "high_jaccard": float(high_jac),
                "topk": int(topk),
            },
            "bundle_candidate_stats": summarize_numeric(candidate_counts),
            "bundle_candidate_ratios": {
                "at_least_1": float(sum(1 for c in candidate_counts if c >= 1) / num_bundles) if num_bundles > 0 else 0.0,
                "at_least_5": float(sum(1 for c in candidate_counts if c >= 5) / num_bundles) if num_bundles > 0 else 0.0,
                "at_least_10": float(sum(1 for c in candidate_counts if c >= 10) / num_bundles) if num_bundles > 0 else 0.0,
            },
            "bundle_hard_pool_after_topk": summarize_numeric(truncated_counts),
            "bundle_hard_pool_after_topk_ratios": {
                "eq_0": float(sum(1 for c in truncated_counts if c == 0) / num_bundles) if num_bundles > 0 else 0.0,
                "lt_5": float(sum(1 for c in truncated_counts if c < 5) / num_bundles) if num_bundles > 0 else 0.0,
            },
        }
        candidate_count_details[name] = candidate_counts

    return threshold_stats, candidate_count_details


def summarize_train_pair_availability(neighbor_dict, user2bundles, train_pairs, threshold_configs, topk):
    print("[INFO] Summarizing training-pair availability...")
    availability_stats = {}
    availability_count_details = {}

    for name, cfg in threshold_configs.items():
        min_int = cfg["min_int"]
        low_jac = cfg["low_jac"]
        high_jac = cfg["high_jac"]

        available_counts = []
        user_hit_flags = defaultdict(list)
        user_avg_counts = defaultdict(list)

        for u, b_pos in tqdm(train_pairs, desc=f"Pairs[{name}]"):
            neighbors = neighbor_dict.get(b_pos, [])
            filtered = filter_neighbors_by_config(neighbors, min_int, low_jac, high_jac)
            filtered = filtered[:topk]

            interacted = user2bundles.get(u, set())
            valid = [n["other_bundle"] for n in filtered if n["other_bundle"] not in interacted]
            count = len(valid)

            available_counts.append(count)
            user_hit_flags[u].append(1 if count > 0 else 0)
            user_avg_counts[u].append(count)

        pair_count = len(available_counts)
        fallback_rate = float(sum(1 for c in available_counts if c == 0) / pair_count) if pair_count > 0 else 0.0

        user_hit_ratios = [float(np.mean(v)) for v in user_hit_flags.values()] if len(user_hit_flags) > 0 else []
        user_mean_available = [float(np.mean(v)) for v in user_avg_counts.values()] if len(user_avg_counts) > 0 else []

        availability_stats[name] = {
            "available_counts": summarize_numeric(available_counts),
            "ratios": {
                "eq_0_fallback_rate": fallback_rate,
                "ge_1": float(sum(1 for c in available_counts if c >= 1) / pair_count) if pair_count > 0 else 0.0,
                "ge_5": float(sum(1 for c in available_counts if c >= 5) / pair_count) if pair_count > 0 else 0.0,
                "ge_10": float(sum(1 for c in available_counts if c >= 10) / pair_count) if pair_count > 0 else 0.0,
            },
            "user_level": {
                "hit_ratio": summarize_numeric(user_hit_ratios),
                "mean_available_count": summarize_numeric(user_mean_available),
            },
        }
        availability_count_details[name] = available_counts

    return availability_stats, availability_count_details


# -----------------------------
# Candidate pool export
# -----------------------------
def build_recommended_hard_candidates(neighbor_dict, cfg, topk):
    """
    Bundle-level candidate pool before user-specific filtering.
    """
    min_int = cfg["min_int"]
    low_jac = cfg["low_jac"]
    high_jac = cfg["high_jac"]

    hard_candidates = {}
    for b_id, neighbors in neighbor_dict.items():
        filtered = filter_neighbors_by_config(neighbors, min_int, low_jac, high_jac)
        hard_candidates[b_id] = [n["other_bundle"] for n in filtered[:topk]]
    return hard_candidates


# -----------------------------
# Reporting
# -----------------------------
def diagnose(availability_stats, recommended_key):
    stats = availability_stats[recommended_key]
    fb = stats["ratios"]["eq_0_fallback_rate"]
    ge1 = stats["ratios"]["ge_1"]
    ge10 = stats["ratios"]["ge_10"]

    if fb < 0.10 and ge10 > 0.80:
        return "EXCELLENT: 非常适合做 hard negatives，可直接优先尝试 H1/H2。"
    if fb < 0.30 and ge1 > 0.70:
        return "GOOD: 可以做，但更建议 hard + random 混采（H2）。"
    if fb < 0.50:
        return "MIXED: 能做，但要保守，阈值可适当放松，并保留强 fallback。"
    return "POOR: 纯 hard negatives 不合适，更适合作为辅助增强或需放松阈值。"


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    dataset, conf = load_multicbr_dataset(args)
    bundle2items, user2bundles, train_pairs, num_users, num_bundles, num_items = build_core_structures(dataset)
    item2bundles = build_item2bundles(bundle2items)

    # 1) basic stats
    bundle_sizes = [len(v) for v in bundle2items.values()]
    basic_stats = {
        "num_users": int(num_users),
        "num_bundles": int(num_bundles),
        "num_items": int(num_items),
        "num_train_interactions": int(len(train_pairs)),
        "bundle_size": summarize_numeric(bundle_sizes),
    }

    # 2) neighbors
    neighbor_dict, nonzero_stats, neighbor_counts, sampled_jaccards, top1_list, top5_list, top10_list = compute_bundle_neighbors(
        bundle2items=bundle2items,
        item2bundles=item2bundles,
        num_bundles=num_bundles,
        max_plot_points=args.max_plot_points,
        seed=args.seed,
    )

    # 3) threshold configs
    threshold_configs = {
        "jac_0.01_0.10_int_1": {"min_int": 1, "low_jac": 0.01, "high_jac": 0.10},
        "jac_0.05_0.20_int_1": {"min_int": 1, "low_jac": 0.05, "high_jac": 0.20},
        "jac_0.10_0.30_int_1": {"min_int": 1, "low_jac": 0.10, "high_jac": 0.30},
        "jac_0.05_0.20_int_2": {"min_int": 2, "low_jac": 0.05, "high_jac": 0.20},
    }
    recommended_key = "jac_0.05_0.20_int_1"

    threshold_stats, candidate_count_details = summarize_threshold_coverage(
        neighbor_dict, threshold_configs, num_bundles, args.topk
    )
    availability_stats, availability_count_details = summarize_train_pair_availability(
        neighbor_dict, user2bundles, train_pairs, threshold_configs, args.topk
    )

    # 4) export
    recommended_cfg = threshold_configs[recommended_key]
    hard_candidates = build_recommended_hard_candidates(neighbor_dict, recommended_cfg, args.topk)

    # full JSON report
    report = {
        "dataset": args.dataset,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "effective_conf_keys": sorted(list(conf.keys())),
        "basic_stats": basic_stats,
        "nonzero_neighbor_stats": nonzero_stats,
        "threshold_stats": threshold_stats,
        "availability_stats": availability_stats,
        "recommended_config": {
            "name": recommended_key,
            "jaccard_range": [recommended_cfg["low_jac"], recommended_cfg["high_jac"]],
            "min_intersection": recommended_cfg["min_int"],
            "topk": args.topk,
            "note": "bundle-level hard candidate pool is saved before user-specific filtering",
        },
        "diagnosis": diagnose(availability_stats, recommended_key),
    }

    save_json(report, os.path.join(args.output_dir, f"hard_negative_stats_{args.dataset}.json"))

    # candidate pools
    save_pickle(neighbor_dict, os.path.join(args.output_dir, f"bundle_nonzero_neighbors_{args.dataset}.pkl"))
    save_pickle(hard_candidates, os.path.join(args.output_dir, f"bundle_hard_candidates_{args.dataset}.pkl"))

    # extra lightweight details for recommended config
    save_json(
        {
            "config_name": recommended_key,
            "bundle_candidate_counts": candidate_count_details[recommended_key],
        },
        os.path.join(args.output_dir, f"bundle_candidate_counts_{args.dataset}_{recommended_key}.json"),
    )
    save_json(
        {
            "config_name": recommended_key,
            "train_available_counts": availability_count_details[recommended_key],
        },
        os.path.join(args.output_dir, f"train_available_counts_{args.dataset}_{recommended_key}.json"),
    )

    # 5) plots
    print("[INFO] Generating plots...")
    plot_histogram(
        bundle_sizes,
        f"{args.dataset} Bundle Size Distribution",
        "Bundle Size",
        "Count",
        "bundle_size_hist.png",
        args.output_dir,
        bins=50,
        log=True,
    )
    plot_histogram(
        neighbor_counts,
        f"{args.dataset} Non-zero Neighbor Count Distribution",
        "Non-zero Neighbor Count",
        "Bundle Count",
        "nonzero_neighbor_count_hist.png",
        args.output_dir,
        bins=50,
        log=True,
    )
    plot_histogram(
        sampled_jaccards,
        f"{args.dataset} Jaccard Distribution (Sampled Non-zero Pairs)",
        "Jaccard",
        "Pair Count",
        "jaccard_hist.png",
        args.output_dir,
        bins=100,
        log=False,
    )
    plot_histogram(
        candidate_count_details[recommended_key],
        f"{args.dataset} Bundle Candidate Count ({recommended_key})",
        "Candidate Count",
        "Bundle Count",
        "bundle_candidate_count_hist.png",
        args.output_dir,
        bins=50,
        log=True,
    )
    plot_histogram(
        availability_count_details[recommended_key],
        f"{args.dataset} Train Sample Available Hard Count ({recommended_key})",
        "Available Hard Negative Count",
        "Training Pair Count",
        "train_sample_available_hard_count_hist.png",
        args.output_dir,
        bins=50,
        log=True,
    )
    plot_histogram(
        top1_list,
        f"{args.dataset} Top1 Jaccard Distribution",
        "Top1 Jaccard",
        "Bundle Count",
        "bundle_top1_overlap_hist.png",
        args.output_dir,
        bins=100,
        log=False,
    )
    plot_histogram(
        top5_list,
        f"{args.dataset} Top5 Mean Jaccard Distribution",
        "Top5 Mean Jaccard",
        "Bundle Count",
        "bundle_top5_mean_overlap_hist.png",
        args.output_dir,
        bins=100,
        log=False,
    )
    plot_histogram(
        top10_list,
        f"{args.dataset} Top10 Mean Jaccard Distribution",
        "Top10 Mean Jaccard",
        "Bundle Count",
        "bundle_top10_mean_overlap_hist.png",
        args.output_dir,
        bins=100,
        log=False,
    )

    # 6) console summary
    print("\n" + "=" * 80)
    print(f"MultiCBR Hard Negative Analysis Summary | dataset={args.dataset}")
    print("=" * 80)
    print(f"Users={num_users}, Bundles={num_bundles}, Items={num_items}, TrainPairs={len(train_pairs)}")
    print(f"Avg Bundle Size={basic_stats['bundle_size']['mean']:.4f}, Median={basic_stats['bundle_size']['median']:.4f}")
    print(
        f"Bundles with >0 neighbors={nonzero_stats['coverage']['with_nonzero_neighbors_ratio'] * 100:.2f}% "
        f"| no-neighbor={nonzero_stats['coverage']['no_nonzero_neighbors_ratio'] * 100:.2f}%"
    )
    print("-" * 80)
    print(f"Recommended Config: {recommended_key} | topk={args.topk}")
    rec_th = threshold_stats[recommended_key]
    rec_av = availability_stats[recommended_key]
    print(
        f"Bundle candidate mean={rec_th['bundle_candidate_stats']['mean']:.4f}, "
        f"median={rec_th['bundle_candidate_stats']['median']:.4f}, "
        f"p95={rec_th['bundle_candidate_stats']['p95']:.4f}"
    )
    print(
        f"Bundle candidate coverage: >=1={rec_th['bundle_candidate_ratios']['at_least_1'] * 100:.2f}%, "
        f">=5={rec_th['bundle_candidate_ratios']['at_least_5'] * 100:.2f}%, "
        f">=10={rec_th['bundle_candidate_ratios']['at_least_10'] * 100:.2f}%"
    )
    print(
        f"Train availability mean={rec_av['available_counts']['mean']:.4f}, "
        f"median={rec_av['available_counts']['median']:.4f}, "
        f"p95={rec_av['available_counts']['p95']:.4f}"
    )
    print(
        f"Fallback rate={rec_av['ratios']['eq_0_fallback_rate'] * 100:.2f}%, "
        f">=1={rec_av['ratios']['ge_1'] * 100:.2f}%, "
        f">=5={rec_av['ratios']['ge_5'] * 100:.2f}%, "
        f">=10={rec_av['ratios']['ge_10'] * 100:.2f}%"
    )
    print("-" * 80)
    print("DIAGNOSIS:", diagnose(availability_stats, recommended_key))
    print("=" * 80)
    print("[INFO] Output directory:", os.path.abspath(args.output_dir))
    print("[INFO] Note: bundle_hard_candidates_*.pkl is bundle-level pool before user-specific filtering.")
    print("=" * 80)


if __name__ == "__main__":
    main()