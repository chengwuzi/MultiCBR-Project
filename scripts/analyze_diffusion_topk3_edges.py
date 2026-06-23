#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility import load_external_embedding_tensor


DEFAULT_EDGE_DIR = Path(
    "module1_ablation/diffusion_topk_edges/iFashion/AnchorViewBundleNet/"
    "M1A_diff_k3_DWT_strategy_diffusion_k3"
)
DEFAULT_TRAIN_UB = Path("datasets/iFashion/user_bundle_train.txt")
DEFAULT_DATA_SIZE = Path("datasets/iFashion/iFashion_data_size.txt")
DEFAULT_USER_EMB = Path("datasets/iFashion_UB_emb/iFashion_UB-user-04261842.pt")
DEFAULT_BUNDLE_EMB = Path("datasets/iFashion_UB_emb/iFashion_UB-bundle-04261842.pt")
DEFAULT_OUTPUT_DIR = Path("module1_ablation/analysis/diffusion_topk3")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze M1A Diffusion Top-K=3 epoch edge dumps and generate "
            "CSV data for line/heatmap charts."
        )
    )
    parser.add_argument("--edge-dir", type=Path, default=DEFAULT_EDGE_DIR)
    parser.add_argument("--train-ub", type=Path, default=DEFAULT_TRAIN_UB)
    parser.add_argument("--data-size", type=Path, default=DEFAULT_DATA_SIZE)
    parser.add_argument("--user-embedding", type=Path, default=DEFAULT_USER_EMB)
    parser.add_argument("--bundle-embedding", type=Path, default=DEFAULT_BUNDLE_EMB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--representative-user-id", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--min-history-size", type=int, default=8)
    parser.add_argument("--min-unique-selected", type=int, default=4)
    parser.add_argument("--max-unique-selected", type=int, default=12)
    return parser.parse_args()


def read_data_size(path):
    with path.open("r", encoding="utf-8") as file:
        num_users, num_bundles, _ = [int(value) for value in file.readline().split("\t")[:3]]
    return num_users, num_bundles


def read_epoch_edges(edge_dir):
    epoch_files = sorted(edge_dir.glob("epoch_*_edges.tsv"))
    if not epoch_files:
        raise FileNotFoundError(f"No epoch edge files found in {edge_dir}")

    epoch_edges = {}
    for path in epoch_files:
        epoch = int(path.stem.split("_")[1])
        user_to_bundles = {}
        with path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            for row in reader:
                user_id = int(row["user_id"])
                bundle_id = int(row["bundle_id"])
                user_to_bundles.setdefault(user_id, set()).add(bundle_id)
        epoch_edges[epoch] = user_to_bundles
    return epoch_edges


def read_train_ub(path):
    user_history = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            user_id, bundle_id = [int(value) for value in line.rstrip("\n").split("\t")[:2]]
            user_history.setdefault(user_id, set()).add(bundle_id)
    return user_history


def jaccard(left, right):
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


def write_adjacent_overlap(epoch_edges, output_dir):
    epochs = sorted(epoch_edges)
    output_path = output_dir / "adjacent_epoch_overlap.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "epoch",
            "epoch_pair",
            "avg_jaccard_overlap",
            "avg_intersection_size",
            "avg_union_size",
            "num_users",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for prev_epoch, curr_epoch in zip(epochs, epochs[1:]):
            prev_edges = epoch_edges[prev_epoch]
            curr_edges = epoch_edges[curr_epoch]
            common_users = sorted(set(prev_edges) & set(curr_edges))
            overlap_sum = 0.0
            intersection_sum = 0
            union_sum = 0
            for user_id in common_users:
                prev_set = prev_edges[user_id]
                curr_set = curr_edges[user_id]
                intersection_size = len(prev_set & curr_set)
                union_size = len(prev_set | curr_set)
                overlap_sum += intersection_size / union_size if union_size else 0.0
                intersection_sum += intersection_size
                union_sum += union_size

            num_users = len(common_users)
            writer.writerow(
                {
                    "epoch": curr_epoch,
                    "epoch_pair": f"{prev_epoch}-{curr_epoch}",
                    "avg_jaccard_overlap": f"{overlap_sum / num_users:.6f}",
                    "avg_intersection_size": f"{intersection_sum / num_users:.6f}",
                    "avg_union_size": f"{union_sum / num_users:.6f}",
                    "num_users": num_users,
                }
            )
    return output_path


def user_epoch_sets(epoch_edges, user_id):
    return [epoch_edges[epoch].get(user_id, set()) for epoch in sorted(epoch_edges)]


def summarize_user_selection(epoch_edges, user_history, user_id):
    sets = user_epoch_sets(epoch_edges, user_id)
    complete_epoch_count = sum(1 for bundles in sets if len(bundles) == 3)
    unique_selected = set().union(*sets) if sets else set()
    pair_overlaps = [jaccard(left, right) for left, right in zip(sets, sets[1:])]
    changed_pairs = sum(1 for left, right in zip(sets, sets[1:]) if left != right)
    return {
        "user_id": user_id,
        "train_history_size": len(user_history.get(user_id, set())),
        "complete_epoch_count": complete_epoch_count,
        "unique_selected_bundles": len(unique_selected),
        "avg_adjacent_overlap": sum(pair_overlaps) / len(pair_overlaps) if pair_overlaps else 0.0,
        "min_adjacent_overlap": min(pair_overlaps) if pair_overlaps else 0.0,
        "max_adjacent_overlap": max(pair_overlaps) if pair_overlaps else 0.0,
        "changed_epoch_pairs": changed_pairs,
        "selected_bundle_ids": " ".join(str(bundle_id) for bundle_id in sorted(unique_selected)),
    }


def candidate_score(summary):
    unique_count = summary["unique_selected_bundles"]
    avg_overlap = summary["avg_adjacent_overlap"]
    history_size = summary["train_history_size"]
    changed_pairs = summary["changed_epoch_pairs"]

    unique_score = 1.0 - min(abs(unique_count - 7) / 7.0, 1.0)
    overlap_score = 1.0 - min(abs(avg_overlap - 0.65) / 0.65, 1.0)
    history_score = min(history_size / 30.0, 1.0)
    change_score = min(changed_pairs / 12.0, 1.0)
    return 0.40 * overlap_score + 0.30 * unique_score + 0.20 * history_score + 0.10 * change_score


def write_representative_candidates(
    epoch_edges,
    user_history,
    output_dir,
    candidate_count,
    min_history_size,
    min_unique_selected,
    max_unique_selected,
):
    epochs = sorted(epoch_edges)
    all_users = sorted(set.intersection(*(set(epoch_edges[epoch]) for epoch in epochs)))
    candidates = []
    for user_id in all_users:
        summary = summarize_user_selection(epoch_edges, user_history, user_id)
        if summary["complete_epoch_count"] != len(epochs):
            continue
        if summary["train_history_size"] < min_history_size:
            continue
        if not (min_unique_selected <= summary["unique_selected_bundles"] <= max_unique_selected):
            continue
        summary["candidate_score"] = candidate_score(summary)
        candidates.append(summary)

    candidates.sort(
        key=lambda row: (
            -row["candidate_score"],
            -row["avg_adjacent_overlap"],
            row["unique_selected_bundles"],
            row["user_id"],
        )
    )
    selected = candidates[:candidate_count]

    output_path = output_dir / "representative_user_candidates.csv"
    fieldnames = [
        "rank",
        "user_id",
        "candidate_score",
        "train_history_size",
        "complete_epoch_count",
        "unique_selected_bundles",
        "avg_adjacent_overlap",
        "min_adjacent_overlap",
        "max_adjacent_overlap",
        "changed_epoch_pairs",
        "selected_bundle_ids",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            out = dict(row)
            out["rank"] = rank
            out["candidate_score"] = f"{row['candidate_score']:.6f}"
            out["avg_adjacent_overlap"] = f"{row['avg_adjacent_overlap']:.6f}"
            out["min_adjacent_overlap"] = f"{row['min_adjacent_overlap']:.6f}"
            out["max_adjacent_overlap"] = f"{row['max_adjacent_overlap']:.6f}"
            writer.writerow(out)
    return output_path, selected


def build_pretrained_rank(user_id, user_history, user_embedding, bundle_embedding):
    observed_bundles = sorted(user_history[user_id])
    user_vector = user_embedding[user_id].float()
    bundle_ids = torch.tensor(observed_bundles, dtype=torch.long)
    bundle_vectors = bundle_embedding[bundle_ids].float()
    scores = torch.mv(bundle_vectors, user_vector).cpu().tolist()
    scored = sorted(zip(observed_bundles, scores), key=lambda item: (-item[1], item[0]))
    rank_by_bundle = {}
    score_by_bundle = {}
    for rank, (bundle_id, score) in enumerate(scored, start=1):
        rank_by_bundle[bundle_id] = rank
        score_by_bundle[bundle_id] = score
    return rank_by_bundle, score_by_bundle


def write_representative_heatmap(
    epoch_edges,
    user_history,
    user_id,
    user_embedding,
    bundle_embedding,
    output_dir,
):
    rank_by_bundle, score_by_bundle = build_pretrained_rank(
        user_id,
        user_history,
        user_embedding,
        bundle_embedding,
    )

    matrix_path = output_dir / "representative_user_top3_rank_matrix.csv"
    long_path = output_dir / "representative_user_top3_rank_long.csv"
    label_matrix_path = output_dir / "representative_user_top3_label_matrix.csv"

    epochs = sorted(epoch_edges)
    with matrix_path.open("w", newline="", encoding="utf-8") as matrix_file, long_path.open(
        "w", newline="", encoding="utf-8"
    ) as long_file, label_matrix_path.open("w", newline="", encoding="utf-8") as label_file:
        matrix_writer = csv.DictWriter(
            matrix_file,
            fieldnames=["epoch", "Selected Top-1", "Selected Top-2", "Selected Top-3"],
        )
        long_writer = csv.DictWriter(
            long_file,
            fieldnames=["epoch", "slot", "bundle_id", "pretrained_rank", "pretrained_score"],
        )
        label_writer = csv.DictWriter(
            label_file,
            fieldnames=["epoch", "Selected Top-1", "Selected Top-2", "Selected Top-3"],
        )
        matrix_writer.writeheader()
        long_writer.writeheader()
        label_writer.writeheader()

        for epoch in epochs:
            selected = sorted(
                epoch_edges[epoch][user_id],
                key=lambda bundle_id: (rank_by_bundle[bundle_id], bundle_id),
            )
            matrix_row = {"epoch": epoch}
            label_row = {"epoch": epoch}
            for slot, bundle_id in enumerate(selected, start=1):
                rank = rank_by_bundle[bundle_id]
                score = score_by_bundle[bundle_id]
                column = f"Selected Top-{slot}"
                matrix_row[column] = rank
                label_row[column] = f"{bundle_id} ({rank})"
                long_writer.writerow(
                    {
                        "epoch": epoch,
                        "slot": slot,
                        "bundle_id": bundle_id,
                        "pretrained_rank": rank,
                        "pretrained_score": f"{score:.8f}",
                    }
                )
            matrix_writer.writerow(matrix_row)
            label_writer.writerow(label_row)
    return matrix_path, long_path, label_matrix_path


def write_run_summary(output_dir, representative_user_id, paths):
    summary_path = output_dir / "README.txt"
    lines = [
        "Diffusion Top-K=3 mechanism analysis outputs",
        "=" * 48,
        "",
        f"Representative user id: {representative_user_id}",
        "",
        "Generated files:",
    ]
    for label, path in paths:
        lines.append(f"- {label}: {path}")
    lines.extend(
        [
            "",
            "Chart 1:",
            "Use adjacent_epoch_overlap.csv. X-axis can be epoch or epoch_pair; "
            "Y-axis is avg_jaccard_overlap.",
            "",
            "Chart 2:",
            "Use representative_user_top3_rank_matrix.csv for a 30x3 heatmap. "
            "Lower pretrained_rank means higher pretrained user-bundle dot-product score.",
            "Use representative_user_top3_rank_long.csv if the chart tool supports "
            "cell labels or richer annotations.",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_users, num_bundles = read_data_size(args.data_size)
    epoch_edges = read_epoch_edges(args.edge_dir)
    user_history = read_train_ub(args.train_ub)

    overlap_path = write_adjacent_overlap(epoch_edges, args.output_dir)
    candidate_path, candidates = write_representative_candidates(
        epoch_edges,
        user_history,
        args.output_dir,
        args.candidate_count,
        args.min_history_size,
        args.min_unique_selected,
        args.max_unique_selected,
    )
    if not candidates and args.representative_user_id is None:
        raise ValueError("No representative user candidates found. Relax the candidate filters.")

    representative_user_id = (
        args.representative_user_id
        if args.representative_user_id is not None
        else int(candidates[0]["user_id"])
    )
    if representative_user_id not in user_history:
        raise ValueError(f"Representative user {representative_user_id} has no train UB history")

    user_embedding = load_external_embedding_tensor(
        str(args.user_embedding),
        num_users,
        "user",
    )
    bundle_embedding = load_external_embedding_tensor(
        str(args.bundle_embedding),
        num_bundles,
        "bundle",
    )
    matrix_path, long_path, label_matrix_path = write_representative_heatmap(
        epoch_edges,
        user_history,
        representative_user_id,
        user_embedding,
        bundle_embedding,
        args.output_dir,
    )
    summary_path = write_run_summary(
        args.output_dir,
        representative_user_id,
        [
            ("Adjacent epoch overlap line data", overlap_path),
            ("Representative user candidates", candidate_path),
            ("Representative user heatmap rank matrix", matrix_path),
            ("Representative user heatmap long table", long_path),
            ("Representative user heatmap label matrix", label_matrix_path),
        ],
    )

    print(f"Wrote analysis outputs to {args.output_dir}")
    print(f"Representative user id: {representative_user_id}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
