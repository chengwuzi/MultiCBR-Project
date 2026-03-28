#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze whether BI diffusion top-k behaves like popularity top-k.

Main goal
---------
For each bundle on a given dataset (default: Youshu), compare the actual kept
items of three strategies:
1. diffusion top-k
2. popularity top-k
3. random top-k

This script reports:
- average overlap@k between methods
- average Jaccard between methods
- exact-match rate
- per-bundle disagreement ranking (largest diffusion vs popularity gap)
- optional export of chosen item ids per bundle

It supports two modes:
A. load a trained purifier checkpoint
B. train a purifier on the fly, then analyze it

Typical usage
-------------
# analyze from a saved checkpoint
python analyze_bi_topk_overlap.py \
  --dataset Youshu \
  --data_path ./datasets \
  --checkpoint ./bi_validation_outputs/Youshu_validate_xxx/bi_diffusion_topk.pt \
  --keep_k 15 \
  --device cpu

# if no checkpoint was saved, retrain first, then analyze
python analyze_bi_topk_overlap.py \
  --dataset Youshu \
  --data_path ./datasets \
  --keep_k 15 \
  --epochs 20 \
  --batch_size 256 \
  --num_negative 8 \
  --diffusion_steps 20 \
  --hidden_size 128 \
  --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from bi_diffusion_topk import build_bi_diffusion_module_from_sparse, resolve_keep_k


# -----------------------------------------------------------------------------
# basic utils
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BundleOnlyDataset(Dataset):
    def __init__(self, num_bundles: int):
        self.num_bundles = int(num_bundles)

    def __len__(self) -> int:
        return self.num_bundles

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(idx, dtype=torch.long)


class BIDataLoader:
    def __init__(self, data_path: str, dataset_name: str):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(data_path, dataset_name)
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.num_users, self.num_bundles, self.num_items = self._load_data_size()
        self.bi_pairs, self.bi_graph = self._load_bi_graph()
        self.bundle_items = self._build_bundle_items_cache(self.bi_graph)
        self.item_popularity = np.asarray((self.bi_graph > 0).sum(axis=0)).reshape(-1).astype(np.int64)

    def _load_data_size(self) -> Tuple[int, int, int]:
        name = self.dataset_name.split("_")[0] if "_" in self.dataset_name else self.dataset_name
        size_file = os.path.join(self.dataset_dir, f"{name}_data_size.txt")
        with open(size_file, "r", encoding="utf-8") as f:
            parts = f.readline().strip().split("\t")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def _load_bi_graph(self) -> Tuple[List[Tuple[int, int]], sp.csr_matrix]:
        bi_file = os.path.join(self.dataset_dir, "bundle_item.txt")
        pairs: List[Tuple[int, int]] = []
        with open(bi_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                b, i = line.split("\t")[:2]
                pairs.append((int(b), int(i)))

        indice = np.asarray(pairs, dtype=np.int32)
        values = np.ones(len(pairs), dtype=np.float32)
        graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])),
            shape=(self.num_bundles, self.num_items),
        ).tocsr()
        graph.eliminate_zeros()
        return pairs, graph

    @staticmethod
    def _build_bundle_items_cache(bi_graph: sp.csr_matrix) -> List[np.ndarray]:
        items: List[np.ndarray] = []
        indptr = bi_graph.indptr
        indices = bi_graph.indices
        for b in range(bi_graph.shape[0]):
            items.append(indices[indptr[b]:indptr[b + 1]].copy())
        return items


# -----------------------------------------------------------------------------
# train helper
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_diff = 0.0
    total_blcc = 0.0
    total_valid = 0.0
    steps = 0

    for batch_bundle_ids in loader:
        batch_bundle_ids = batch_bundle_ids.to(device)
        optimizer.zero_grad()
        out = model(batch_bundle_ids)
        out["total_loss"].backward()
        optimizer.step()

        total_loss += float(out["total_loss"].detach().cpu())
        total_diff += float(out["diffusion_loss"].detach().cpu())
        total_blcc += float(out["blcc_loss"].detach().cpu())
        total_valid += float(out["num_valid_bundles"].detach().cpu())
        steps += 1

    denom = max(steps, 1)
    return {
        "loss": total_loss / denom,
        "diffusion_loss": total_diff / denom,
        "blcc_loss": total_blcc / denom,
        "avg_valid_bundles_per_step": total_valid / denom,
    }


def maybe_train_or_load_model(args, bi_graph: sp.csr_matrix, num_bundles: int, device: torch.device):
    model = build_bi_diffusion_module_from_sparse(
        bi_graph=bi_graph,
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

    train_history = []
    if args.checkpoint and os.path.isfile(args.checkpoint):
        payload = torch.load(args.checkpoint, map_location=device)
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")
        return model, train_history

    print("No valid checkpoint provided. Start training purifier before overlap analysis...")
    loader = DataLoader(
        BundleOnlyDataset(num_bundles),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, loader, optimizer, device)
        metrics["epoch"] = epoch
        train_history.append(metrics)
        print(
            f"[Epoch {epoch:03d}] loss={metrics['loss']:.6f} | "
            f"diff={metrics['diffusion_loss']:.6f} | blcc={metrics['blcc_loss']:.6f} | "
            f"valid/step={metrics['avg_valid_bundles_per_step']:.2f}"
        )

    return model, train_history


# -----------------------------------------------------------------------------
# top-k builders
# -----------------------------------------------------------------------------

def build_random_topk_selection(bundle_items: Sequence[np.ndarray], keep_k: Optional[int], seed: int, keep_ratio: Optional[float] = None, min_keep: int = 5) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: List[np.ndarray] = []
    for items in bundle_items:
        L = len(items)
        if L == 0:
            selected.append(np.empty(0, dtype=np.int64))
            continue
        k = resolve_keep_k(L, keep_k=keep_k, keep_ratio=keep_ratio, min_keep=min_keep)
        chosen = items if L <= k else rng.choice(items, size=k, replace=False)
        selected.append(np.sort(chosen.astype(np.int64)))
    return selected


def build_popularity_topk_selection(bundle_items: Sequence[np.ndarray], item_popularity: np.ndarray, keep_k: Optional[int], keep_ratio: Optional[float] = None, min_keep: int = 5) -> List[np.ndarray]:
    selected: List[np.ndarray] = []
    for items in bundle_items:
        L = len(items)
        if L == 0:
            selected.append(np.empty(0, dtype=np.int64))
            continue
        k = resolve_keep_k(L, keep_k=keep_k, keep_ratio=keep_ratio, min_keep=min_keep)
        if L <= k:
            chosen = items
        else:
            pop = item_popularity[items]
            order = np.argsort(-pop, kind="stable")[:k]
            chosen = items[order]
        selected.append(np.sort(chosen.astype(np.int64)))
    return selected


@torch.no_grad()
def build_diffusion_topk_selection(model, num_bundles: int, keep_k: Optional[int], keep_ratio: Optional[float] = None, min_keep: int = 5) -> List[np.ndarray]:
    selected: List[np.ndarray] = []
    model.eval()
    for b in range(num_bundles):
        item_ids = model.bundle_items[b]
        L = len(item_ids)
        if L == 0:
            selected.append(np.empty(0, dtype=np.int64))
            continue
        k = resolve_keep_k(L, keep_k=keep_k, keep_ratio=keep_ratio, min_keep=min_keep)
        if L <= k:
            selected.append(np.sort(item_ids.astype(np.int64)))
            continue
        scores = model.score_original_bundle_items(bundle_id=b)
        top_idx = torch.topk(scores, k=k, dim=0).indices.cpu().numpy()
        chosen = np.sort(item_ids[top_idx].astype(np.int64))
        selected.append(chosen)
    return selected


# -----------------------------------------------------------------------------
# overlap metrics
# -----------------------------------------------------------------------------

def pair_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    sa = set(map(int, a.tolist()))
    sb = set(map(int, b.tolist()))
    inter = len(sa & sb)
    union = len(sa | sb)
    denom = max(min(len(sa), len(sb)), 1)
    overlap = inter / denom
    jaccard = inter / max(union, 1)
    exact = 1.0 if sa == sb else 0.0
    return {
        "intersection": float(inter),
        "union": float(union),
        "overlap": float(overlap),
        "jaccard": float(jaccard),
        "exact": float(exact),
    }


def aggregate_pairwise(name_a: str, name_b: str, sel_a: Sequence[np.ndarray], sel_b: Sequence[np.ndarray]) -> Dict[str, float]:
    rows = [pair_metrics(a, b) for a, b in zip(sel_a, sel_b)]
    return {
        "pair": f"{name_a}_vs_{name_b}",
        "num_bundles": len(rows),
        "avg_intersection": float(np.mean([x["intersection"] for x in rows])),
        "avg_union": float(np.mean([x["union"] for x in rows])),
        "avg_overlap@k": float(np.mean([x["overlap"] for x in rows])),
        "avg_jaccard": float(np.mean([x["jaccard"] for x in rows])),
        "exact_match_rate": float(np.mean([x["exact"] for x in rows])),
    }


def top_disagreement_bundles(
    bundle_items: Sequence[np.ndarray],
    diffusion_sel: Sequence[np.ndarray],
    popularity_sel: Sequence[np.ndarray],
    random_sel: Sequence[np.ndarray],
    limit: int = 30,
) -> List[Dict[str, object]]:
    rows = []
    for b, (full_items, d, p, r) in enumerate(zip(bundle_items, diffusion_sel, popularity_sel, random_sel)):
        dp = pair_metrics(d, p)
        dr = pair_metrics(d, r)
        rows.append({
            "bundle_id": int(b),
            "bundle_len": int(len(full_items)),
            "diff_vs_pop_overlap": dp["overlap"],
            "diff_vs_pop_jaccard": dp["jaccard"],
            "diff_vs_pop_exact": dp["exact"],
            "diff_vs_rand_overlap": dr["overlap"],
            "diffusion_items": d.tolist(),
            "popularity_items": p.tolist(),
            "random_items": r.tolist(),
        })
    rows.sort(key=lambda x: (x["diff_vs_pop_overlap"], x["diff_vs_pop_jaccard"], x["bundle_len"]))
    return rows[:limit]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze overlap between diffusion-topk and popularity-topk on BI graph")
    parser.add_argument("--dataset", type=str, default="Youshu")
    parser.add_argument("--data_path", type=str, default="./datasets")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--checkpoint", type=str, default="", help="Optional trained purifier checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)

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
    parser.add_argument("--blcc_enabled", action="store_true")
    parser.add_argument("--blcc_lambda", type=float, default=0.1)

    parser.add_argument("--keep_k", type=int, default=None, help="Keep fixed top-k")
    parser.add_argument("--keep_ratio", type=float, default=None, help="Keep a ratio of items")
    parser.add_argument("--min_keep", type=int, default=5, help="Minimum keep threshold for ratio")
    parser.add_argument("--random_seed", type=int, default=3407)
    parser.add_argument("--save_topk_lists", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./bi_overlap_outputs")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.keep_k is None and args.keep_ratio is None:
        raise ValueError("Either --keep_k or --keep_ratio must be provided.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    data = BIDataLoader(args.data_path, args.dataset)
    device = torch.device(args.device)

    t0 = time.time()
    model, train_history = maybe_train_or_load_model(args, data.bi_graph, data.num_bundles, device)
    train_time_sec = time.time() - t0

    if args.keep_ratio is not None:
        print(f"Building top-k selections with keep_ratio={args.keep_ratio}, min_keep={args.min_keep} ...")
    else:
        print(f"Building top-k selections with keep_k={args.keep_k} ...")
        
    diffusion_sel = build_diffusion_topk_selection(model, data.num_bundles, keep_k=args.keep_k, keep_ratio=args.keep_ratio, min_keep=args.min_keep)
    popularity_sel = build_popularity_topk_selection(data.bundle_items, data.item_popularity, keep_k=args.keep_k, keep_ratio=args.keep_ratio, min_keep=args.min_keep)
    random_sel = build_random_topk_selection(data.bundle_items, keep_k=args.keep_k, keep_ratio=args.keep_ratio, min_keep=args.min_keep, seed=args.random_seed)

    metrics_dp = aggregate_pairwise("diffusion", "popularity", diffusion_sel, popularity_sel)
    metrics_dr = aggregate_pairwise("diffusion", "random", diffusion_sel, random_sel)
    metrics_pr = aggregate_pairwise("popularity", "random", popularity_sel, random_sel)

    print("=" * 20 + " Overlap Summary " + "=" * 20)
    for block in [metrics_dp, metrics_dr, metrics_pr]:
        print(block)

    disagreement_rows = top_disagreement_bundles(
        bundle_items=data.bundle_items,
        diffusion_sel=diffusion_sel,
        popularity_sel=popularity_sel,
        random_sel=random_sel,
        limit=30,
    )

    result = {
        "args": vars(args),
        "truncation_mode": "ratio" if args.keep_ratio is not None else "fixed_k",
        "keep_ratio": args.keep_ratio,
        "keep_k": args.keep_k,
        "min_keep": args.min_keep,
        "train_history": train_history,
        "train_time_sec": train_time_sec,
        "overlap_summary": {
            "diffusion_vs_popularity": metrics_dp,
            "diffusion_vs_random": metrics_dr,
            "popularity_vs_random": metrics_pr,
        },
        "top_disagreement_bundles": disagreement_rows,
    }

    if args.save_topk_lists:
        def pack(sel: Sequence[np.ndarray]) -> Dict[str, List[int]]:
            return {str(i): arr.tolist() for i, arr in enumerate(sel)}
        result["topk_lists"] = {
            "diffusion": pack(diffusion_sel),
            "popularity": pack(popularity_sel),
            "random": pack(random_sel),
        }

    if args.keep_ratio is not None:
        out_name = f"{args.dataset}_overlap_ratio{args.keep_ratio}_min{args.min_keep}_seed{args.seed}.json"
    else:
        out_name = f"{args.dataset}_overlap_k{args.keep_k}_seed{args.seed}.json"
        
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved overlap analysis to: {out_path}")


if __name__ == "__main__":
    main()
