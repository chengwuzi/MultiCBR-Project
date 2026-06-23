#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility import load_external_embedding_tensor


DEFAULT_OUTPUT_DIR = Path("outputs/module1_ablation/edge_dynamics")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate representative-user rank-quality line plot data from "
            "iFashion A Diffusion Top-K=3 epoch edge dumps."
        )
    )
    parser.add_argument("--edge_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="iFashion")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--auto_find", action="store_true", help="Automatically find the M1A_diff_k3 edge directory.")
    parser.add_argument("--representative_user_id", type=int, default=None)
    parser.add_argument("--candidate_count", type=int, default=10)
    parser.add_argument("--min_history_size", type=int, default=30)
    parser.add_argument("--min_unique_selected", type=int, default=4)
    parser.add_argument("--max_unique_selected", type=int, default=12)
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args()


def find_edge_dir(dataset):
    candidates = [
        Path("outputs/module1_ablation/diffusion_topk_edges")
        / dataset
        / "AnchorViewBundleNet",
        Path("module1_ablation/diffusion_topk_edges")
        / dataset
        / "AnchorViewBundleNet",
    ]
    matches = []
    for base in candidates:
        if not base.exists():
            continue
        matches.extend(path for path in base.iterdir() if path.is_dir() and path.name.startswith("M1A_diff_k3"))
    if not matches:
        raise FileNotFoundError("Could not find M1A_diff_k3 edge directory. Pass --edge_dir explicitly.")
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def read_data_size(dataset):
    path = Path("datasets") / dataset / f"{dataset}_data_size.txt"
    with path.open("r", encoding="utf-8") as file:
        return [int(value) for value in file.readline().split("\t")[:3]]


def default_embedding_paths(dataset):
    if dataset != "iFashion":
        raise ValueError("This script currently only knows the default iFashion pretrained embedding paths.")
    return (
        Path("datasets/iFashion_UB_emb/iFashion_UB-user-04261842.pt"),
        Path("datasets/iFashion_UB_emb/iFashion_UB-bundle-04261842.pt"),
    )


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


def read_epoch_edges(edge_dir):
    files = sorted(edge_dir.glob("epoch_*_edges.tsv"))
    if not files:
        raise FileNotFoundError(f"No epoch edge files found in {edge_dir}")
    epoch_edges = {}
    for path in files:
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


def jaccard(left, right):
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def build_pretrained_rank(user_id, history, user_embedding, bundle_embedding):
    observed = sorted(history[user_id])
    bundle_ids = torch.tensor(observed, dtype=torch.long)
    scores = torch.mv(bundle_embedding[bundle_ids].float(), user_embedding[user_id].float()).cpu().tolist()
    ranked = sorted(zip(observed, scores), key=lambda item: (-item[1], item[0]))
    rank_by_bundle = {}
    score_by_bundle = {}
    for rank, (bundle_id, score) in enumerate(ranked, start=1):
        rank_by_bundle[bundle_id] = rank
        score_by_bundle[bundle_id] = score
    return rank_by_bundle, score_by_bundle


def selected_rank_rows(user_id, epoch_edges, history, user_embedding, bundle_embedding):
    rank_by_bundle, score_by_bundle = build_pretrained_rank(user_id, history, user_embedding, bundle_embedding)
    max_rank = len(history[user_id])
    rows = []
    warnings = []
    for epoch in sorted(epoch_edges):
        selected = epoch_edges[epoch].get(user_id, set())
        if len(selected) != 3:
            raise ValueError(f"User {user_id} has {len(selected)} selected edges in epoch {epoch}, expected 3")
        missing = sorted(bundle_id for bundle_id in selected if bundle_id not in rank_by_bundle)
        if missing:
            warnings.append(f"WARNING: epoch {epoch} selected bundles not in train history: {missing}")
        selected = sorted(selected, key=lambda bundle_id: (rank_by_bundle[bundle_id], bundle_id))
        for slot_idx, bundle_id in enumerate(selected, start=1):
            pretrained_rank = rank_by_bundle[bundle_id]
            rows.append(
                {
                    "epoch": epoch,
                    "slot": f"Selected Top-{slot_idx}",
                    "bundle_id": bundle_id,
                    "pretrained_rank": pretrained_rank,
                    "rank_quality_score": max_rank + 1 - pretrained_rank,
                    "pretrained_score": score_by_bundle[bundle_id],
                }
            )
    return rows, warnings


def summarize_candidate(user_id, epoch_edges, history, user_embedding, bundle_embedding):
    rows, warnings = selected_rank_rows(user_id, epoch_edges, history, user_embedding, bundle_embedding)
    if warnings:
        return None

    by_epoch = {}
    for row in rows:
        by_epoch.setdefault(row["epoch"], []).append(row)
    top1_bundles = [sorted(epoch_rows, key=lambda row: int(row["slot"].split("-")[-1]))[0]["bundle_id"] for epoch_rows in by_epoch.values()]
    top1_counter = Counter(top1_bundles)
    top1_mode_bundle, top1_mode_frequency = top1_counter.most_common(1)[0]
    top1_mode_rank = next(row["pretrained_rank"] for row in rows if row["bundle_id"] == top1_mode_bundle)
    selected_sets = [
        {row["bundle_id"] for row in by_epoch[epoch]}
        for epoch in sorted(by_epoch)
    ]
    adjacent_overlaps = [jaccard(left, right) for left, right in zip(selected_sets, selected_sets[1:])]
    unique_selected = set().union(*selected_sets)
    top23_pairs = [
        tuple(row["bundle_id"] for row in sorted(by_epoch[epoch], key=lambda row: int(row["slot"].split("-")[-1]))[1:])
        for epoch in sorted(by_epoch)
    ]
    top23_unique_patterns = len(set(top23_pairs))

    top1_stability = top1_mode_frequency / len(by_epoch)
    avg_overlap = sum(adjacent_overlaps) / len(adjacent_overlaps)
    unique_count = len(unique_selected)
    unique_score = 1.0 - min(abs(unique_count - 7) / 7.0, 1.0)
    top23_variation_score = min(top23_unique_patterns / 8.0, 1.0)
    history_score = min(len(history[user_id]) / 50.0, 1.0)
    candidate_score = (
        0.35 * top1_stability
        + 0.25 * avg_overlap
        + 0.20 * unique_score
        + 0.10 * top23_variation_score
        + 0.10 * history_score
    )

    return {
        "user_id": user_id,
        "num_train_observed_bundles": len(history[user_id]),
        "num_unique_selected_bundles_across_epochs": unique_count,
        "top1_mode_bundle": top1_mode_bundle,
        "top1_mode_rank": top1_mode_rank,
        "top1_mode_frequency": top1_mode_frequency,
        "avg_adjacent_overlap": avg_overlap,
        "top23_unique_patterns": top23_unique_patterns,
        "candidate_score": candidate_score,
    }


def select_candidates(epoch_edges, history, user_embedding, bundle_embedding, args):
    epochs = sorted(epoch_edges)
    common_users = sorted(set.intersection(*(set(epoch_edges[epoch]) for epoch in epochs)))
    candidates = []
    for user_id in common_users:
        if len(history.get(user_id, set())) < args.min_history_size:
            continue
        selected_union = set()
        complete = True
        for epoch in epochs:
            selected = epoch_edges[epoch].get(user_id, set())
            if len(selected) != 3:
                complete = False
                break
            selected_union.update(selected)
        if not complete:
            continue
        if not (args.min_unique_selected <= len(selected_union) <= args.max_unique_selected):
            continue
        summary = summarize_candidate(user_id, epoch_edges, history, user_embedding, bundle_embedding)
        if summary is not None:
            candidates.append(summary)
    candidates.sort(
        key=lambda row: (
            -row["candidate_score"],
            -row["top1_mode_frequency"],
            -row["avg_adjacent_overlap"],
            row["num_unique_selected_bundles_across_epochs"],
            row["user_id"],
        )
    )
    return candidates[: args.candidate_count]


def write_candidates(path, candidates):
    fieldnames = [
        "user_id",
        "num_train_observed_bundles",
        "num_unique_selected_bundles_across_epochs",
        "top1_mode_bundle",
        "top1_mode_rank",
        "top1_mode_frequency",
        "avg_adjacent_overlap",
        "top23_unique_patterns",
        "candidate_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in candidates:
            out = dict(row)
            out["avg_adjacent_overlap"] = f"{row['avg_adjacent_overlap']:.6f}"
            out["candidate_score"] = f"{row['candidate_score']:.6f}"
            writer.writerow(out)


def write_detail(path, rows):
    fieldnames = [
        "epoch",
        "slot",
        "bundle_id",
        "pretrained_rank",
        "rank_quality_score",
        "pretrained_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["pretrained_score"] = f"{row['pretrained_score']:.8f}"
            writer.writerow(out)


def write_horizontal(path, rows):
    by_slot = {}
    for row in rows:
        by_slot.setdefault(row["slot"], {})[row["epoch"]] = row["rank_quality_score"]
    epochs = sorted({row["epoch"] for row in rows})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["series"] + epochs)
        for slot in ["Selected Top-1", "Selected Top-2", "Selected Top-3"]:
            writer.writerow([slot] + [by_slot[slot][epoch] for epoch in epochs])


def plot_line(path_png, path_pdf, rows, user_id):
    import matplotlib.pyplot as plt

    by_slot = {}
    for row in rows:
        by_slot.setdefault(row["slot"], []).append((row["epoch"], row["rank_quality_score"]))
    plt.figure(figsize=(8, 4.8))
    for slot in ["Selected Top-1", "Selected Top-2", "Selected Top-3"]:
        points = sorted(by_slot[slot])
        plt.plot(
            [epoch for epoch, _ in points],
            [score for _, score in points],
            marker="o",
            linewidth=2,
            markersize=4,
            label=slot,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Rank quality score")
    plt.title(f"Representative User {user_id}: Rank Quality Dynamics")
    plt.text(
        0.5,
        -0.20,
        "Higher score indicates better pretrained ranking.",
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_pdf)
    plt.close()


def main():
    args = parse_args()
    if args.edge_dir is None or args.auto_find:
        edge_dir = find_edge_dir(args.dataset)
    else:
        edge_dir = args.edge_dir

    num_users, num_bundles, _ = read_data_size(args.dataset)
    user_emb_path, bundle_emb_path = default_embedding_paths(args.dataset)
    history = read_train_history(args.dataset)
    epoch_edges = read_epoch_edges(edge_dir)
    user_embedding = load_external_embedding_tensor(str(user_emb_path), num_users, "user")
    bundle_embedding = load_external_embedding_tensor(str(bundle_emb_path), num_bundles, "bundle")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_candidates(epoch_edges, history, user_embedding, bundle_embedding, args)
    if not candidates and args.representative_user_id is None:
        raise ValueError("No representative user candidates found. Relax filtering parameters.")

    representative_user_id = (
        args.representative_user_id
        if args.representative_user_id is not None
        else candidates[0]["user_id"]
    )
    rows, warnings = selected_rank_rows(
        representative_user_id,
        epoch_edges,
        history,
        user_embedding,
        bundle_embedding,
    )
    if warnings:
        for warning in warnings:
            print(warning, file=sys.stderr)

    candidates_path = args.output_dir / "representative_user_candidates.tsv"
    horizontal_path = args.output_dir / "representative_user_rank_quality_lineplot.tsv"
    detail_path = args.output_dir / "representative_user_rank_dynamics_detail.tsv"
    png_path = args.output_dir / "representative_user_rank_quality_lineplot.png"
    pdf_path = args.output_dir / "representative_user_rank_quality_lineplot.pdf"

    write_candidates(candidates_path, candidates)
    write_horizontal(horizontal_path, rows)
    write_detail(detail_path, rows)
    if not args.skip_plot:
        try:
            plot_line(png_path, pdf_path, rows, representative_user_id)
        except Exception as exc:
            print(f"WARNING: failed to generate matplotlib figure: {exc}", file=sys.stderr)

    representative_summary = summarize_candidate(
        representative_user_id,
        epoch_edges,
        history,
        user_embedding,
        bundle_embedding,
    )
    print("Rank dynamics analysis finished.")
    print(f"Edge directory: {edge_dir}")
    print(f"Representative user id: {representative_user_id}")
    print(f"History size: {len(history[representative_user_id])}")
    print(f"Unique selected bundles: {representative_summary['num_unique_selected_bundles_across_epochs']}")
    print(
        "Top-1 mode: "
        f"bundle={representative_summary['top1_mode_bundle']}, "
        f"rank={representative_summary['top1_mode_rank']}, "
        f"frequency={representative_summary['top1_mode_frequency']}/30"
    )
    print(f"Average adjacent overlap: {representative_summary['avg_adjacent_overlap']:.6f}")
    print(f"Line plot TSV: {horizontal_path}")
    print(f"Detail TSV: {detail_path}")
    print(f"Candidates TSV: {candidates_path}")


if __name__ == "__main__":
    main()
