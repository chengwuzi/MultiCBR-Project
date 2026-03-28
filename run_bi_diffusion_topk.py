#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run script for the standalone BI diffusion top-k purifier.

Purpose
-------
This script is intentionally independent from the main MultiCBR training flow.
It only does the following:
1. Load the dataset and obtain the original B-I graph.
2. Build the BI diffusion purifier module.
3. Train the purifier on bundle ids.
4. Export the purified top-k B-I graph / pairs.
5. Print and optionally save basic graph statistics for later validation.

Typical usage
-------------
python run_bi_diffusion_topk.py --dataset Youshu --data_path ./datasets --epochs 20 --keep_k 4
python run_bi_diffusion_topk.py --dataset NetEase --data_path ./datasets --epochs 30 --keep_k 5 --device cuda:0

Notes
-----
- This script does NOT modify MultiCBR.py.
- It is designed to validate whether the BI purifier itself can learn useful
  item importance ranking inside bundles.
- Later, the purified graph can be plugged into MultiCBR as dataset.graphs[2].
"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
import random
from typing import Dict, Tuple, List

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from bi_diffusion_topk import build_bi_diffusion_module_from_sparse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BundleOnlyDataset(Dataset):
    """Simple dataset that only returns bundle ids for BI purifier training."""

    def __init__(self, num_bundles: int):
        self.num_bundles = int(num_bundles)

    def __len__(self) -> int:
        return self.num_bundles

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(idx, dtype=torch.long)


class BIDataLoader:
    """Load only the B-I graph and metadata from the existing dataset format."""

    def __init__(self, data_path: str, dataset_name: str):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(data_path, dataset_name)
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.num_users, self.num_bundles, self.num_items = self._load_data_size()
        self.bi_pairs, self.bi_graph = self._load_bi_graph()

    def _load_data_size(self) -> Tuple[int, int, int]:
        name = self.dataset_name.split("_")[0] if "_" in self.dataset_name else self.dataset_name
        size_file = os.path.join(self.dataset_dir, f"{name}_data_size.txt")
        if not os.path.isfile(size_file):
            raise FileNotFoundError(f"data size file not found: {size_file}")
        with open(size_file, "r", encoding="utf-8") as f:
            parts = f.readline().strip().split("\t")
        if len(parts) < 3:
            raise ValueError(f"invalid data size file format: {size_file}")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def _load_bi_graph(self) -> Tuple[List[Tuple[int, int]], sp.csr_matrix]:
        bi_file = os.path.join(self.dataset_dir, "bundle_item.txt")
        if not os.path.isfile(bi_file):
            raise FileNotFoundError(f"bundle_item file not found: {bi_file}")

        pairs: List[Tuple[int, int]] = []
        with open(bi_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                b, i = line.split("\t")[:2]
                pairs.append((int(b), int(i)))

        if len(pairs) == 0:
            raise ValueError(f"No B-I pairs found in {bi_file}")

        indice = np.asarray(pairs, dtype=np.int32)
        values = np.ones(len(pairs), dtype=np.float32)
        graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_bundles, self.num_items),
        ).tocsr()
        graph.eliminate_zeros()
        return pairs, graph


def graph_stats(graph: sp.csr_matrix) -> Dict[str, float]:
    graph = graph.tocsr()
    row_nnz = np.diff(graph.indptr)
    col_nnz = np.asarray((graph > 0).sum(axis=0)).reshape(-1)
    total_possible = graph.shape[0] * graph.shape[1]
    nonzero_rows = float(np.mean(row_nnz > 0)) if graph.shape[0] > 0 else 0.0
    nonzero_cols = float(np.mean(col_nnz > 0)) if graph.shape[1] > 0 else 0.0
    return {
        "num_rows": int(graph.shape[0]),
        "num_cols": int(graph.shape[1]),
        "nnz": int(graph.nnz),
        "density": float(graph.nnz / max(total_possible, 1)),
        "avg_items_per_bundle": float(row_nnz.mean()) if len(row_nnz) > 0 else 0.0,
        "median_items_per_bundle": float(np.median(row_nnz)) if len(row_nnz) > 0 else 0.0,
        "max_items_per_bundle": int(row_nnz.max()) if len(row_nnz) > 0 else 0,
        "min_items_per_bundle": int(row_nnz.min()) if len(row_nnz) > 0 else 0,
        "avg_bundles_per_item": float(col_nnz.mean()) if len(col_nnz) > 0 else 0.0,
        "nonzero_rows_ratio": nonzero_rows,
        "nonzero_cols_ratio": nonzero_cols,
    }


def print_graph_stats(title: str, stats: Dict[str, float]) -> None:
    print("=" * 20 + f" {title} " + "=" * 20)
    for k, v in stats.items():
        print(f"{k}: {v}")


def save_pairs_txt(pairs: List[Tuple[int, int]], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for b, i in pairs:
            f.write(f"{b}\t{i}\n")


def train_one_epoch(model, loader, optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_diff = 0.0
    total_blcc = 0.0
    total_valid = 0.0
    step_count = 0

    for batch_bundle_ids in loader:
        batch_bundle_ids = batch_bundle_ids.to(device)
        optimizer.zero_grad()
        out = model(batch_bundle_ids)
        loss = out["total_loss"]
        loss.backward()
        optimizer.step()

        total_loss += float(out["total_loss"].detach().cpu())
        total_diff += float(out["diffusion_loss"].detach().cpu())
        total_blcc += float(out["blcc_loss"].detach().cpu())
        total_valid += float(out["num_valid_bundles"].detach().cpu())
        step_count += 1

    denom = max(step_count, 1)
    return {
        "loss": total_loss / denom,
        "diffusion_loss": total_diff / denom,
        "blcc_loss": total_blcc / denom,
        "avg_valid_bundles_per_step": total_valid / denom,
    }


@torch.no_grad()
def inspect_score_distribution(model, bundle_ids: List[int], max_print: int = 5) -> List[Dict[str, float]]:
    model.eval()
    out = []
    for b in bundle_ids[:max_print]:
        scores = model.score_original_bundle_items(bundle_id=b)
        if scores.numel() == 0:
            continue
        s = scores.detach().cpu().numpy()
        out.append({
            "bundle_id": int(b),
            "bundle_len": int(len(s)),
            "score_min": float(np.min(s)),
            "score_max": float(np.max(s)),
            "score_mean": float(np.mean(s)),
            "score_std": float(np.std(s)),
            "top1_minus_last": float(np.max(s) - np.min(s)),
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Youshu or NetEase")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Root dataset directory")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda:0 / cuda:1 ...")
    parser.add_argument("--seed", type=int, default=2026)

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)

    # module hyperparameters
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--time_embed_size", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--max_context_items", type=int, default=32)
    parser.add_argument("--num_negative", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_layernorm", action="store_true")

    # top-k purification
    parser.add_argument("--keep_k", type=int, default=None, help="Keep top-k original items per bundle")
    parser.add_argument("--keep_ratio", type=float, default=None, help="Keep a ratio of original items per bundle")
    parser.add_argument("--min_keep", type=int, default=5, help="Minimum number of items to keep when using keep_ratio")
    parser.add_argument("--keep_all_if_len_le_k", action="store_true")

    # optional BLCC
    parser.add_argument("--blcc_enabled", action="store_true")
    parser.add_argument("--blcc_lambda", type=float, default=0.1)

    # outputs
    parser.add_argument("--output_dir", type=str, default="./bi_diffusion_outputs")
    parser.add_argument("--save_pairs", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.keep_k is None and args.keep_ratio is None:
        raise ValueError("Either --keep_k or --keep_ratio must be provided.")
    
    if args.keep_ratio is not None:
        print(f"Using ratio truncation: keep_ratio={args.keep_ratio}, min_keep={args.min_keep}")
    else:
        print(f"Using fixed keep_k={args.keep_k}")

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    data = BIDataLoader(data_path=args.data_path, dataset_name=args.dataset)
    original_stats = graph_stats(data.bi_graph)
    print_graph_stats("Original BI Graph", original_stats)

    model = build_bi_diffusion_module_from_sparse(
        bi_graph=data.bi_graph,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        time_embed_size=args.time_embed_size,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        max_context_items=args.max_context_items,
        num_negative=args.num_negative,
        blcc_enabled=args.blcc_enabled,
        blcc_lambda=args.blcc_lambda,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
        device=str(device),
    ).to(device)

    bundle_dataset = BundleOnlyDataset(num_bundles=data.num_bundles)
    bundle_loader = DataLoader(
        bundle_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Start training BI diffusion purifier...")
    history = []
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = train_one_epoch(model, bundle_loader, optimizer, device)
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={epoch_metrics['loss']:.6f} | "
            f"diff={epoch_metrics['diffusion_loss']:.6f} | "
            f"blcc={epoch_metrics['blcc_loss']:.6f} | "
            f"valid/step={epoch_metrics['avg_valid_bundles_per_step']:.2f}"
        )
    train_time = time.time() - train_start
    print(f"Training finished in {train_time:.2f}s")

    print(f"Building purified BI graph...")
    model.eval()
    purified_graph = model.build_purified_bi_graph(
        keep_k=args.keep_k,
        keep_ratio=args.keep_ratio,
        min_keep=args.min_keep,
        keep_all_if_len_le_k=args.keep_all_if_len_le_k,
    )
    purified_stats = graph_stats(purified_graph)
    print_graph_stats("Purified BI Graph", purified_stats)

    density_ratio = purified_stats["density"] / max(original_stats["density"], 1e-12)
    nnz_ratio = purified_stats["nnz"] / max(original_stats["nnz"], 1)
    print("=" * 20 + " Purification Summary " + "=" * 20)
    if args.keep_ratio is not None:
        print(f"keep_ratio: {args.keep_ratio}, min_keep: {args.min_keep}")
    else:
        print(f"keep_k: {args.keep_k}")
    print(f"nnz ratio purified/original: {nnz_ratio:.6f}")
    print(f"density ratio purified/original: {density_ratio:.6f}")

    sample_bundle_ids = list(range(min(10, data.num_bundles)))
    score_summary = inspect_score_distribution(model, sample_bundle_ids, max_print=10)
    print("=" * 20 + " Sample Score Distribution " + "=" * 20)
    for row in score_summary:
        print(row)

    if args.keep_ratio is not None:
        run_name = (
            f"{args.dataset}_ratio{args.keep_ratio}_min{args.min_keep}_ep{args.epochs}_neg{args.num_negative}_"
            f"ds{args.diffusion_steps}_hs{args.hidden_size}"
        )
    else:
        run_name = (
            f"{args.dataset}_k{args.keep_k}_ep{args.epochs}_neg{args.num_negative}_"
            f"ds{args.diffusion_steps}_hs{args.hidden_size}"
        )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary = {
        "args": vars(args),
        "train_history": history,
        "original_stats": original_stats,
        "purified_stats": purified_stats,
        "nnz_ratio": nnz_ratio,
        "density_ratio": density_ratio,
        "sample_score_summary": score_summary,
        "train_time_sec": train_time,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_pairs:
        pairs = model.export_topk_pairs(
            keep_k=args.keep_k,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            keep_all_if_len_le_k=args.keep_all_if_len_le_k,
        )
        save_pairs_txt(pairs, os.path.join(run_dir, "purified_bundle_item.txt"))
        print(f"Saved top-k pairs to: {os.path.join(run_dir, 'purified_bundle_item.txt')}")

    if args.save_npz:
        sp.save_npz(os.path.join(run_dir, "purified_bi_graph.npz"), purified_graph)
        print(f"Saved purified graph to: {os.path.join(run_dir, 'purified_bi_graph.npz')}")

    if args.save_checkpoint:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "num_bundles": data.num_bundles,
            "num_items": data.num_items,
        }
        ckpt_path = os.path.join(run_dir, "bi_diffusion_topk.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to: {ckpt_path}")

    print(f"All outputs are under: {run_dir}")


if __name__ == "__main__":
    main()
