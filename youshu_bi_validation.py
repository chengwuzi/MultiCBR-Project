#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Youshu BI module validation script.

Purpose
-------
Validate whether the standalone BI diffusion purifier really learns
"important vs less-important" items inside each bundle on the Youshu dataset.

This script intentionally focuses on the BI module itself and does NOT touch
MultiCBR training.

It performs five things in one run:
1. Load Youshu BI graph.
2. Train the BI diffusion purifier once.
3. Sweep multiple keep_k values (recommended for Youshu: 10-20).
4. Compare against baselines:
   - full BI (structure stats only)
   - random top-k
   - popularity top-k
   - diffusion top-k
5. Run leave-one-out reconstruction evaluation to test whether the module
   understands bundle composition structure.

Outputs
-------
Under output_dir/run_name/, it saves:
- summary.json                 : all consolidated results
- leave_one_out_details.json   : per-method / per-k reconstruction metrics
- optional purified graph/pairs/checkpoint if requested

Typical usage
-------------
python youshu_bi_validation.py \
  --data_path ./datasets \
  --epochs 20 \
  --k_values 10 12 15 18 20 \
  --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

from bi_diffusion_topk import build_bi_diffusion_module_from_sparse


# -----------------------------------------------------------------------------
# utils
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


@dataclass
class BundleLenStats:
    avg: float
    median: float
    p75: float
    p90: float
    p95: float
    min_len: int
    max_len: int
    le_5: int
    le_10: int
    le_15: int
    le_20: int



def bundle_length_stats(bundle_items: Sequence[np.ndarray]) -> BundleLenStats:
    lens = np.asarray([len(x) for x in bundle_items], dtype=np.int64)
    return BundleLenStats(
        avg=float(lens.mean()),
        median=float(np.median(lens)),
        p75=float(np.percentile(lens, 75)),
        p90=float(np.percentile(lens, 90)),
        p95=float(np.percentile(lens, 95)),
        min_len=int(lens.min()),
        max_len=int(lens.max()),
        le_5=int((lens <= 5).sum()),
        le_10=int((lens <= 10).sum()),
        le_15=int((lens <= 15).sum()),
        le_20=int((lens <= 20).sum()),
    )



def graph_stats(graph: sp.csr_matrix) -> Dict[str, float]:
    graph = graph.tocsr()
    row_nnz = np.diff(graph.indptr)
    col_nnz = np.asarray((graph > 0).sum(axis=0)).reshape(-1)
    total_possible = graph.shape[0] * graph.shape[1]
    return {
        "num_rows": int(graph.shape[0]),
        "num_cols": int(graph.shape[1]),
        "nnz": int(graph.nnz),
        "density": float(graph.nnz / max(total_possible, 1)),
        "avg_items_per_bundle": float(row_nnz.mean()) if len(row_nnz) > 0 else 0.0,
        "median_items_per_bundle": float(np.median(row_nnz)) if len(row_nnz) > 0 else 0.0,
        "p75_items_per_bundle": float(np.percentile(row_nnz, 75)) if len(row_nnz) > 0 else 0.0,
        "p90_items_per_bundle": float(np.percentile(row_nnz, 90)) if len(row_nnz) > 0 else 0.0,
        "max_items_per_bundle": int(row_nnz.max()) if len(row_nnz) > 0 else 0,
        "min_items_per_bundle": int(row_nnz.min()) if len(row_nnz) > 0 else 0,
        "avg_bundles_per_item": float(col_nnz.mean()) if len(col_nnz) > 0 else 0.0,
    }



def print_graph_stats(title: str, stats: Dict[str, float]) -> None:
    print("=" * 20 + f" {title} " + "=" * 20)
    for k, v in stats.items():
        print(f"{k}: {v}")



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


@torch.no_grad()
def inspect_score_distribution(model, bundle_ids: List[int], max_print: int = 10) -> List[Dict[str, float]]:
    model.eval()
    rows = []
    for b in bundle_ids[:max_print]:
        scores = model.score_original_bundle_items(bundle_id=b)
        if scores.numel() == 0:
            continue
        s = scores.detach().cpu().numpy()
        rows.append({
            "bundle_id": int(b),
            "bundle_len": int(len(s)),
            "score_min": float(np.min(s)),
            "score_max": float(np.max(s)),
            "score_mean": float(np.mean(s)),
            "score_std": float(np.std(s)),
            "top1_minus_last": float(np.max(s) - np.min(s)),
        })
    return rows


# -----------------------------------------------------------------------------
# baselines and purified graph builders
# -----------------------------------------------------------------------------

def build_random_topk_graph(bundle_items: Sequence[np.ndarray], num_bundles: int, num_items: int, keep_k: int, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for b, items in enumerate(bundle_items):
        L = len(items)
        if L == 0:
            continue
        k = min(keep_k, L)
        chosen = items if L <= k else rng.choice(items, size=k, replace=False)
        rows.extend([b] * len(chosen))
        cols.extend(chosen.tolist())
    vals = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((vals, (np.asarray(rows), np.asarray(cols))), shape=(num_bundles, num_items)).tocsr()



def build_popularity_topk_graph(bundle_items: Sequence[np.ndarray], item_popularity: np.ndarray, num_bundles: int, num_items: int, keep_k: int) -> sp.csr_matrix:
    rows, cols = [], []
    for b, items in enumerate(bundle_items):
        L = len(items)
        if L == 0:
            continue
        k = min(keep_k, L)
        if L <= k:
            chosen = items
        else:
            pop = item_popularity[items]
            order = np.argsort(-pop, kind="stable")[:k]
            chosen = items[order]
        rows.extend([b] * len(chosen))
        cols.extend(chosen.tolist())
    vals = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((vals, (np.asarray(rows), np.asarray(cols))), shape=(num_bundles, num_items)).tocsr()


# -----------------------------------------------------------------------------
# leave-one-out evaluation
# -----------------------------------------------------------------------------

def sample_eval_bundles(bundle_items: Sequence[np.ndarray], max_eval_bundles: int, min_len: int, seed: int) -> List[int]:
    candidates = [b for b, items in enumerate(bundle_items) if len(items) >= min_len]
    rng = np.random.default_rng(seed)
    if len(candidates) <= max_eval_bundles:
        return candidates
    return rng.choice(np.asarray(candidates), size=max_eval_bundles, replace=False).tolist()



def leave_one_out_eval_diffusion(model, bundle_items: Sequence[np.ndarray], num_items: int, eval_bundles: Sequence[int], num_negative_eval: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    hits1 = hits3 = hits5 = 0.0
    rr_sum = 0.0
    total = 0
    rank_sum = 0.0

    item_universe = np.arange(num_items, dtype=np.int64)

    model.eval()
    for b in eval_bundles:
        items = bundle_items[b]
        if len(items) < 2:
            continue
        target_pos = int(rng.choice(items))
        context = items[items != target_pos]
        if len(context) == 0:
            continue

        neg_pool_mask = np.ones(num_items, dtype=bool)
        neg_pool_mask[items] = False
        neg_pool = item_universe[neg_pool_mask]
        if len(neg_pool) == 0:
            continue
        nneg = min(num_negative_eval, len(neg_pool))
        negs = rng.choice(neg_pool, size=nneg, replace=False)
        candidates = np.concatenate([[target_pos], negs])

        # build a context-aware score using the same internal scorer;
        # context is the masked bundle, candidates include true held-out item.
        scores = score_candidates_with_context(model, bundle_id=b, context_item_ids=context, candidate_item_ids=candidates)
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1  # target is candidates[0]

        hits1 += float(rank <= 1)
        hits3 += float(rank <= 3)
        hits5 += float(rank <= 5)
        rr_sum += 1.0 / rank
        rank_sum += rank
        total += 1

    denom = max(total, 1)
    return {
        "evaluated_bundles": int(total),
        "Hit@1": hits1 / denom,
        "Hit@3": hits3 / denom,
        "Hit@5": hits5 / denom,
        "MRR": rr_sum / denom,
        "avg_rank": rank_sum / denom,
    }


@torch.no_grad()
def score_candidates_with_context(model, bundle_id: int, context_item_ids: np.ndarray, candidate_item_ids: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    bundle_emb_table = model.bundle_embedding_table.weight
    item_emb_table = model.item_embedding_table.weight

    context_repr, time_repr = model._build_context_for_inference(
        bundle_id=bundle_id,
        item_ids=context_item_ids,
        bundle_emb_table=bundle_emb_table,
        item_emb_table=item_emb_table,
    )
    cand_ids = torch.as_tensor(candidate_item_ids, device=device, dtype=torch.long)
    cand_item_emb = item_emb_table[cand_ids]
    bundle_emb = bundle_emb_table[bundle_id].unsqueeze(0).expand(cand_item_emb.size(0), -1)
    context_expand = context_repr.unsqueeze(0).expand(cand_item_emb.size(0), -1)
    time_expand = time_repr.unsqueeze(0).expand(cand_item_emb.size(0), -1)
    score_input = torch.cat([cand_item_emb, context_expand, bundle_emb, time_expand], dim=-1)
    logits = model.scorer(score_input).squeeze(-1)
    return logits.detach().cpu().numpy()



def leave_one_out_eval_popularity(bundle_items: Sequence[np.ndarray], item_popularity: np.ndarray, num_items: int, eval_bundles: Sequence[int], num_negative_eval: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    hits1 = hits3 = hits5 = 0.0
    rr_sum = 0.0
    total = 0
    rank_sum = 0.0
    item_universe = np.arange(num_items, dtype=np.int64)

    for b in eval_bundles:
        items = bundle_items[b]
        if len(items) < 2:
            continue
        target_pos = int(rng.choice(items))
        neg_pool_mask = np.ones(num_items, dtype=bool)
        neg_pool_mask[items] = False
        neg_pool = item_universe[neg_pool_mask]
        if len(neg_pool) == 0:
            continue
        nneg = min(num_negative_eval, len(neg_pool))
        negs = rng.choice(neg_pool, size=nneg, replace=False)
        candidates = np.concatenate([[target_pos], negs])
        scores = item_popularity[candidates]
        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == 0)[0][0]) + 1

        hits1 += float(rank <= 1)
        hits3 += float(rank <= 3)
        hits5 += float(rank <= 5)
        rr_sum += 1.0 / rank
        rank_sum += rank
        total += 1

    denom = max(total, 1)
    return {
        "evaluated_bundles": int(total),
        "Hit@1": hits1 / denom,
        "Hit@3": hits3 / denom,
        "Hit@5": hits5 / denom,
        "MRR": rr_sum / denom,
        "avg_rank": rank_sum / denom,
    }



def leave_one_out_eval_random(bundle_items: Sequence[np.ndarray], num_items: int, eval_bundles: Sequence[int], num_negative_eval: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    hits1 = hits3 = hits5 = 0.0
    rr_sum = 0.0
    total = 0
    rank_sum = 0.0
    item_universe = np.arange(num_items, dtype=np.int64)

    for b in eval_bundles:
        items = bundle_items[b]
        if len(items) < 2:
            continue
        target_pos = int(rng.choice(items))
        neg_pool_mask = np.ones(num_items, dtype=bool)
        neg_pool_mask[items] = False
        neg_pool = item_universe[neg_pool_mask]
        if len(neg_pool) == 0:
            continue
        nneg = min(num_negative_eval, len(neg_pool))
        negs = rng.choice(neg_pool, size=nneg, replace=False)
        candidates = np.concatenate([[target_pos], negs])
        scores = rng.random(len(candidates))
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1

        hits1 += float(rank <= 1)
        hits3 += float(rank <= 3)
        hits5 += float(rank <= 5)
        rr_sum += 1.0 / rank
        rank_sum += rank
        total += 1

    denom = max(total, 1)
    return {
        "evaluated_bundles": int(total),
        "Hit@1": hits1 / denom,
        "Hit@3": hits3 / denom,
        "Hit@5": hits5 / denom,
        "MRR": rr_sum / denom,
        "avg_rank": rank_sum / denom,
    }


# -----------------------------------------------------------------------------
# argument parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Youshu")
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=2026)

    # training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=0)

    # module hparams
    p.add_argument("--embedding_size", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--time_embed_size", type=int, default=64)
    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--max_context_items", type=int, default=32)
    p.add_argument("--num_negative", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_layernorm", action="store_true")
    p.add_argument("--blcc_enabled", action="store_true")
    p.add_argument("--blcc_lambda", type=float, default=0.1)

    # validation sweep
    p.add_argument("--k_values", type=int, nargs="+", default=[10, 12, 15, 18, 20])
    p.add_argument("--random_baseline_seed", type=int, default=3407)

    # leave-one-out
    p.add_argument("--max_eval_bundles", type=int, default=3000)
    p.add_argument("--min_eval_bundle_len", type=int, default=2)
    p.add_argument("--num_negative_eval", type=int, default=99)

    # outputs
    p.add_argument("--output_dir", type=str, default="./bi_validation_outputs")
    p.add_argument("--save_pairs", action="store_true")
    p.add_argument("--save_npz", action="store_true")
    p.add_argument("--save_checkpoint", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.dataset != "Youshu":
        print(f"[Warning] This script is designed for Youshu-first validation, current dataset={args.dataset}")

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    data = BIDataLoader(data_path=args.data_path, dataset_name=args.dataset)
    original_stats = graph_stats(data.bi_graph)
    length_stats = asdict(bundle_length_stats(data.bundle_items))
    print_graph_stats("Original BI Graph", original_stats)
    print("=" * 20 + " Bundle Length Stats " + "=" * 20)
    for k, v in length_stats.items():
        print(f"{k}: {v}")

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
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, bundle_loader, optimizer, device)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"[Epoch {epoch:03d}] loss={metrics['loss']:.6f} | "
            f"diff={metrics['diffusion_loss']:.6f} | "
            f"blcc={metrics['blcc_loss']:.6f} | "
            f"valid/step={metrics['avg_valid_bundles_per_step']:.2f}"
        )
    train_time = time.time() - t0
    print(f"Training finished in {train_time:.2f}s")

    sample_bundle_ids = list(range(min(10, data.num_bundles)))
    score_summary = inspect_score_distribution(model, sample_bundle_ids, max_print=10)
    print("=" * 20 + " Sample Score Distribution " + "=" * 20)
    for row in score_summary:
        print(row)

    print("Preparing leave-one-out evaluation bundles...")
    eval_bundles = sample_eval_bundles(
        bundle_items=data.bundle_items,
        max_eval_bundles=args.max_eval_bundles,
        min_len=args.min_eval_bundle_len,
        seed=args.seed,
    )
    print(f"Number of eval bundles: {len(eval_bundles)}")

    # LOO evaluation for ranking capability
    print("Running leave-one-out evaluation baselines...")
    loo_random = leave_one_out_eval_random(
        bundle_items=data.bundle_items,
        num_items=data.num_items,
        eval_bundles=eval_bundles,
        num_negative_eval=args.num_negative_eval,
        seed=args.seed,
    )
    loo_pop = leave_one_out_eval_popularity(
        bundle_items=data.bundle_items,
        item_popularity=data.item_popularity,
        num_items=data.num_items,
        eval_bundles=eval_bundles,
        num_negative_eval=args.num_negative_eval,
        seed=args.seed,
    )
    loo_diff = leave_one_out_eval_diffusion(
        model=model,
        bundle_items=data.bundle_items,
        num_items=data.num_items,
        eval_bundles=eval_bundles,
        num_negative_eval=args.num_negative_eval,
        seed=args.seed,
    )

    print("=" * 20 + " Leave-One-Out Metrics " + "=" * 20)
    print({"random": loo_random, "popularity": loo_pop, "diffusion": loo_diff})

    # sweep ks and compare structural baselines
    sweep_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for k in args.k_values:
        print(f"\n===== Sweeping keep_k={k} =====")
        diffusion_graph = model.build_purified_bi_graph(keep_k=k, keep_all_if_len_le_k=True)
        random_graph = build_random_topk_graph(
            bundle_items=data.bundle_items,
            num_bundles=data.num_bundles,
            num_items=data.num_items,
            keep_k=k,
            seed=args.random_baseline_seed + k,
        )
        popularity_graph = build_popularity_topk_graph(
            bundle_items=data.bundle_items,
            item_popularity=data.item_popularity,
            num_bundles=data.num_bundles,
            num_items=data.num_items,
            keep_k=k,
        )

        diffusion_stats = graph_stats(diffusion_graph)
        random_stats = graph_stats(random_graph)
        popularity_stats = graph_stats(popularity_graph)

        sweep_results[str(k)] = {
            "full_bi": original_stats,
            "random_topk": random_stats,
            "popularity_topk": popularity_stats,
            "diffusion_topk": diffusion_stats,
        }

        print_graph_stats(f"k={k} | diffusion_topk", diffusion_stats)
        print_graph_stats(f"k={k} | popularity_topk", popularity_stats)
        print_graph_stats(f"k={k} | random_topk", random_stats)

    run_name = (
        f"{args.dataset}_validate_ep{args.epochs}_neg{args.num_negative}_"
        f"ds{args.diffusion_steps}_hs{args.hidden_size}_seed{args.seed}"
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary = {
        "args": vars(args),
        "train_history": history,
        "train_time_sec": train_time,
        "original_stats": original_stats,
        "bundle_length_stats": length_stats,
        "sample_score_summary": score_summary,
        "leave_one_out": {
            "random": loo_random,
            "popularity": loo_pop,
            "diffusion": loo_diff,
        },
        "k_sweep_structural_stats": sweep_results,
        "num_eval_bundles": len(eval_bundles),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "leave_one_out_details.json"), "w", encoding="utf-8") as f:
        json.dump(summary["leave_one_out"], f, indent=2, ensure_ascii=False)

    # optionally save per-k diffusion outputs for later MultiCBR integration
    if args.save_pairs or args.save_npz:
        per_k_dir = os.path.join(run_dir, "per_k_outputs")
        os.makedirs(per_k_dir, exist_ok=True)
        for k in args.k_values:
            diffusion_graph = model.build_purified_bi_graph(keep_k=k, keep_all_if_len_le_k=True)
            k_dir = os.path.join(per_k_dir, f"k_{k}")
            os.makedirs(k_dir, exist_ok=True)
            if args.save_npz:
                sp.save_npz(os.path.join(k_dir, "purified_bi_graph.npz"), diffusion_graph)
            if args.save_pairs:
                coo = diffusion_graph.tocoo()
                with open(os.path.join(k_dir, "purified_bundle_item.txt"), "w", encoding="utf-8") as f:
                    for b, i in zip(coo.row.tolist(), coo.col.tolist()):
                        f.write(f"{b}\t{i}\n")

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
