#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility import load_external_embedding_tensor


DEFAULT_OUTPUT_DIR = Path("outputs/module1_ablation/edge_dynamics")
FREQUENCY_BINS = [
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-29", 21, 29),
    ("30", 30, 30),
]
POSITIONS = ["Selected Top-1", "Selected Top-2", "Selected Top-3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline mechanism analysis for iFashion Diffusion Top-K=3 edge dynamics."
    )
    parser.add_argument("--edge_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="iFashion")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis", choices=["frequency", "rank_percentile", "all"], default="all")
    parser.add_argument("--auto_find", action="store_true")
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args()


def find_edge_dir(dataset):
    bases = [
        Path("outputs/module1_ablation/diffusion_topk_edges") / dataset / "AnchorViewBundleNet",
        Path("module1_ablation/diffusion_topk_edges") / dataset / "AnchorViewBundleNet",
    ]
    matches = []
    for base in bases:
        if base.exists():
            matches.extend(path for path in base.iterdir() if path.is_dir() and path.name.startswith("M1A_diff_k3"))
    if not matches:
        raise FileNotFoundError("Could not find M1A_diff_k3 edge directory. Pass --edge_dir explicitly.")
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def resolve_edge_dir(args):
    if args.edge_dir is not None and not args.auto_find:
        return args.edge_dir
    return find_edge_dir(args.dataset)


def read_data_size(dataset):
    path = Path("datasets") / dataset / f"{dataset}_data_size.txt"
    with path.open("r", encoding="utf-8") as file:
        return [int(value) for value in file.readline().split("\t")[:3]]


def default_embedding_paths(dataset):
    if dataset != "iFashion":
        raise ValueError("Only iFashion default pretrained embedding paths are configured.")
    return (
        Path("datasets/iFashion_UB_emb/iFashion_UB-user-04261842.pt"),
        Path("datasets/iFashion_UB_emb/iFashion_UB-bundle-04261842.pt"),
    )


def read_epoch_edges(edge_dir):
    files = sorted(edge_dir.glob("epoch_*_edges.tsv"))
    if not files:
        raise FileNotFoundError(f"No epoch edge files found in {edge_dir}")

    epoch_edges = {}
    total_records = 0
    for path in files:
        epoch = int(path.stem.split("_")[1])
        user_to_bundles = {}
        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            if "user_id" not in reader.fieldnames or "bundle_id" not in reader.fieldnames:
                raise ValueError(f"{path} must contain user_id and bundle_id columns")
            for row in reader:
                user_id = int(row["user_id"])
                bundle_id = int(row["bundle_id"])
                user_to_bundles.setdefault(user_id, set()).add(bundle_id)
                total_records += 1
        epoch_edges[epoch] = user_to_bundles
    return epoch_edges, total_records


def read_train_history(dataset):
    path = Path("datasets") / dataset / "user_bundle_train.txt"
    history = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            user_id, bundle_id = [int(value) for value in line.rstrip("\n").split("\t")[:2]]
            history.setdefault(user_id, set()).add(bundle_id)
    return history


def compute_edge_frequencies(epoch_edges):
    counter = Counter()
    for user_to_bundles in epoch_edges.values():
        for user_id, bundles in user_to_bundles.items():
            for bundle_id in bundles:
                counter[(user_id, bundle_id)] += 1
    return counter


def frequency_bin_label(frequency):
    for label, lower, upper in FREQUENCY_BINS:
        if lower <= frequency <= upper:
            return label
    raise ValueError(f"Unexpected frequency {frequency}")


def write_frequency_outputs(edge_counter, output_dir):
    detail_path = output_dir / "edge_selection_frequency_detail.tsv"
    bins_path = output_dir / "edge_selection_frequency_bins.tsv"
    barplot_ratio_path = output_dir / "edge_selection_frequency_barplot.tsv"
    barplot_count_path = output_dir / "edge_selection_frequency_barplot_counts.tsv"

    with detail_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["user_id", "bundle_id", "frequency"])
        for (user_id, bundle_id), frequency in sorted(edge_counter.items()):
            writer.writerow([user_id, bundle_id, frequency])

    total_unique_edges = len(edge_counter)
    bin_counts = {label: 0 for label, _, _ in FREQUENCY_BINS}
    for frequency in edge_counter.values():
        bin_counts[frequency_bin_label(frequency)] += 1

    with bins_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["frequency_bin", "edge_count", "edge_ratio"])
        for label, _, _ in FREQUENCY_BINS:
            count = bin_counts[label]
            ratio = count / total_unique_edges if total_unique_edges else 0.0
            writer.writerow([label, count, f"{ratio:.6f}"])

    labels = [label for label, _, _ in FREQUENCY_BINS]
    with barplot_ratio_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["category"] + labels)
        writer.writerow(
            ["edge_ratio"]
            + [f"{(bin_counts[label] / total_unique_edges if total_unique_edges else 0.0):.6f}" for label in labels]
        )
    with barplot_count_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["category"] + labels)
        writer.writerow(["edge_count"] + [bin_counts[label] for label in labels])

    return {
        "detail": detail_path,
        "bins": bins_path,
        "barplot_ratio": barplot_ratio_path,
        "barplot_count": barplot_count_path,
        "bin_counts": bin_counts,
        "total_unique_edges": total_unique_edges,
    }


def plot_frequency(bin_counts, total_unique_edges, output_dir):
    import matplotlib.pyplot as plt

    labels = [label for label, _, _ in FREQUENCY_BINS]
    ratios = [bin_counts[label] / total_unique_edges if total_unique_edges else 0.0 for label in labels]
    plt.figure(figsize=(7.2, 4.5))
    plt.bar(labels, ratios, color="#2F6B9A")
    plt.xlabel("Selection frequency bin")
    plt.ylabel("Edge ratio")
    plt.title("Distribution of Edge Selection Frequency")
    plt.figtext(
        0.5,
        0.01,
        "Frequency denotes how many epochs a user-bundle edge is selected by the diffusion module.",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    png_path = output_dir / "edge_selection_frequency_distribution.png"
    pdf_path = output_dir / "edge_selection_frequency_distribution.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    return png_path, pdf_path


def build_pretrained_rank_maps(history, user_embedding, bundle_embedding):
    rank_maps = {}
    score_maps = {}
    for user_id, observed_set in history.items():
        observed = sorted(observed_set)
        if not observed:
            continue
        bundle_ids = torch.tensor(observed, dtype=torch.long)
        scores = torch.mv(bundle_embedding[bundle_ids].float(), user_embedding[user_id].float()).cpu().tolist()
        ranked = sorted(zip(observed, scores), key=lambda item: (-item[1], item[0]))
        rank_maps[user_id] = {}
        score_maps[user_id] = {}
        for rank, (bundle_id, score) in enumerate(ranked, start=1):
            rank_maps[user_id][bundle_id] = rank
            score_maps[user_id][bundle_id] = score
    return rank_maps, score_maps


def percentile_from_rank(rank, history_size):
    if history_size <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (history_size - 1)


def compute_rank_percentile_rows(epoch_edges, history, rank_maps, score_maps):
    long_rows = []
    wide_rows = []
    warnings = []
    skipped_incomplete = 0

    for epoch in sorted(epoch_edges):
        for user_id, selected_bundles in sorted(epoch_edges[epoch].items()):
            if len(selected_bundles) != 3:
                skipped_incomplete += 1
                continue
            if user_id not in history or user_id not in rank_maps:
                warnings.append(f"WARNING: user {user_id} is missing train history; epoch={epoch}")
                continue

            missing = sorted(bundle_id for bundle_id in selected_bundles if bundle_id not in rank_maps[user_id])
            if missing:
                warnings.append(
                    f"WARNING: epoch={epoch} user={user_id} selected bundles not in train history: {missing}"
                )
                continue

            history_size = len(history[user_id])
            selected_sorted = sorted(selected_bundles, key=lambda bundle_id: (rank_maps[user_id][bundle_id], bundle_id))
            wide_row = {}
            for slot_idx, bundle_id in enumerate(selected_sorted, start=1):
                position = f"Selected Top-{slot_idx}"
                rank = rank_maps[user_id][bundle_id]
                percentile = percentile_from_rank(rank, history_size)
                score = score_maps[user_id][bundle_id]
                long_rows.append(
                    {
                        "epoch": epoch,
                        "user_id": user_id,
                        "position": position,
                        "bundle_id": bundle_id,
                        "num_train_observed_bundles": history_size,
                        "pretrained_rank": rank,
                        "rank_quality_percentile": percentile,
                        "pretrained_score": score,
                    }
                )
                wide_row[position] = percentile
            wide_rows.append(wide_row)
    return long_rows, wide_rows, warnings, skipped_incomplete


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_percentiles(long_rows):
    by_position = {position: [] for position in POSITIONS}
    for row in long_rows:
        by_position[row["position"]].append(row["rank_quality_percentile"])

    summaries = []
    for position in POSITIONS:
        values = sorted(by_position[position])
        count = len(values)
        mean = sum(values) / count if count else 0.0
        median = quantile(values, 0.5)
        q1 = quantile(values, 0.25)
        q3 = quantile(values, 0.75)
        if count > 1:
            variance = sum((value - mean) ** 2 for value in values) / (count - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        summaries.append(
            {
                "position": position,
                "count": count,
                "mean": mean,
                "median": median,
                "std": std,
                "q1": q1,
                "q3": q3,
                "min": values[0] if values else 0.0,
                "max": values[-1] if values else 0.0,
            }
        )
    return summaries


def write_rank_percentile_outputs(long_rows, wide_rows, output_dir):
    long_path = output_dir / "selected_edges_pretrained_rank_percentile_long.tsv"
    wide_path = output_dir / "selected_edges_pretrained_rank_percentile_wide.tsv"
    summary_path = output_dir / "selected_edges_pretrained_rank_percentile_summary.tsv"

    with long_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "epoch",
                "user_id",
                "position",
                "bundle_id",
                "num_train_observed_bundles",
                "pretrained_rank",
                "rank_quality_percentile",
                "pretrained_score",
            ]
        )
        for row in long_rows:
            writer.writerow(
                [
                    row["epoch"],
                    row["user_id"],
                    row["position"],
                    row["bundle_id"],
                    row["num_train_observed_bundles"],
                    row["pretrained_rank"],
                    f"{row['rank_quality_percentile']:.6f}",
                    f"{row['pretrained_score']:.8f}",
                ]
            )

    with wide_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(POSITIONS)
        for row in wide_rows:
            writer.writerow([f"{row[position]:.6f}" for position in POSITIONS])

    summaries = summarize_percentiles(long_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["position", "count", "mean", "median", "std", "q1", "q3", "min", "max"])
        for row in summaries:
            writer.writerow(
                [
                    row["position"],
                    row["count"],
                    f"{row['mean']:.6f}",
                    f"{row['median']:.6f}",
                    f"{row['std']:.6f}",
                    f"{row['q1']:.6f}",
                    f"{row['q3']:.6f}",
                    f"{row['min']:.6f}",
                    f"{row['max']:.6f}",
                ]
            )

    return {
        "long": long_path,
        "wide": wide_path,
        "summary": summary_path,
        "summaries": summaries,
    }


def plot_rank_percentile(long_rows, output_dir):
    import matplotlib.pyplot as plt

    values = {position: [] for position in POSITIONS}
    for row in long_rows:
        values[row["position"]].append(row["rank_quality_percentile"])
    data = [values[position] for position in POSITIONS]

    plt.figure(figsize=(7.2, 4.8))
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_alpha(0.65)
    plt.xticks([1, 2, 3], POSITIONS)
    plt.ylim(0, 1)
    plt.xlabel("Selected position")
    plt.ylabel("Pretrained rank quality percentile")
    plt.title("Pretrained Rank Percentiles of Diffusion-selected Edges")
    plt.figtext(
        0.5,
        0.01,
        "Higher percentile indicates a better ranking position within the user's historical bundles.",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    png_path = output_dir / "selected_edges_pretrained_rank_percentile_violin.png"
    pdf_path = output_dir / "selected_edges_pretrained_rank_percentile_violin.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    return png_path, pdf_path


def run_frequency(epoch_edges, output_dir, skip_plot):
    edge_counter = compute_edge_frequencies(epoch_edges)
    outputs = write_frequency_outputs(edge_counter, output_dir)
    if not skip_plot:
        try:
            plot_frequency(outputs["bin_counts"], outputs["total_unique_edges"], output_dir)
        except Exception as exc:
            print(f"WARNING: failed to generate frequency plot: {exc}", file=sys.stderr)
    return outputs


def run_rank_percentile(args, epoch_edges, output_dir):
    num_users, num_bundles, _ = read_data_size(args.dataset)
    user_emb_path, bundle_emb_path = default_embedding_paths(args.dataset)
    history = read_train_history(args.dataset)
    user_embedding = load_external_embedding_tensor(str(user_emb_path), num_users, "user")
    bundle_embedding = load_external_embedding_tensor(str(bundle_emb_path), num_bundles, "bundle")
    rank_maps, score_maps = build_pretrained_rank_maps(history, user_embedding, bundle_embedding)
    long_rows, wide_rows, warnings, skipped_incomplete = compute_rank_percentile_rows(
        epoch_edges,
        history,
        rank_maps,
        score_maps,
    )
    for warning in warnings[:20]:
        print(warning, file=sys.stderr)
    if len(warnings) > 20:
        print(f"WARNING: suppressed {len(warnings) - 20} additional warnings", file=sys.stderr)
    outputs = write_rank_percentile_outputs(long_rows, wide_rows, output_dir)
    if not args.skip_plot:
        try:
            plot_rank_percentile(long_rows, output_dir)
        except Exception as exc:
            print(f"WARNING: failed to generate rank percentile plot: {exc}", file=sys.stderr)
    outputs["warnings"] = warnings
    outputs["skipped_incomplete"] = skipped_incomplete
    outputs["long_row_count"] = len(long_rows)
    outputs["wide_row_count"] = len(wide_rows)
    return outputs


def print_frequency_summary(outputs):
    print("Frequency distribution:")
    for label, _, _ in FREQUENCY_BINS:
        count = outputs["bin_counts"][label]
        ratio = count / outputs["total_unique_edges"] if outputs["total_unique_edges"] else 0.0
        print(f"  {label}: count={count}, ratio={ratio:.6f}")


def print_rank_summary(outputs):
    print("Rank percentile summary:")
    for row in outputs["summaries"]:
        print(
            f"  {row['position']}: count={row['count']}, "
            f"mean={row['mean']:.6f}, median={row['median']:.6f}, "
            f"q1={row['q1']:.6f}, q3={row['q3']:.6f}"
        )


def main():
    args = parse_args()
    edge_dir = resolve_edge_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epoch_edges, total_records = read_epoch_edges(edge_dir)

    print("Diffusion edge dynamics analysis")
    print(f"Edge directory: {edge_dir}")
    print(f"Epochs: {len(epoch_edges)}")
    print(f"Total selected edge records: {total_records}")

    frequency_outputs = None
    rank_outputs = None
    if args.analysis in {"frequency", "all"}:
        frequency_outputs = run_frequency(epoch_edges, args.output_dir, args.skip_plot)
        print(f"Unique selected edges: {frequency_outputs['total_unique_edges']}")
        print_frequency_summary(frequency_outputs)
        print(f"Frequency detail: {frequency_outputs['detail']}")
        print(f"Frequency bins: {frequency_outputs['bins']}")
        print(f"Frequency barplot ratio: {frequency_outputs['barplot_ratio']}")
        print(f"Frequency barplot count: {frequency_outputs['barplot_count']}")

    if args.analysis in {"rank_percentile", "all"}:
        rank_outputs = run_rank_percentile(args, epoch_edges, args.output_dir)
        print(f"Rank percentile long rows: {rank_outputs['long_row_count']}")
        print(f"Rank percentile wide rows: {rank_outputs['wide_row_count']}")
        print(f"Skipped incomplete user-epoch records: {rank_outputs['skipped_incomplete']}")
        print_rank_summary(rank_outputs)
        print(f"Rank percentile long: {rank_outputs['long']}")
        print(f"Rank percentile wide: {rank_outputs['wide']}")
        print(f"Rank percentile summary: {rank_outputs['summary']}")


if __name__ == "__main__":
    main()
