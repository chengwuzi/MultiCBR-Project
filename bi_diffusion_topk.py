#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone BI-view diffusion-style purifier for MultiCBR.

Design goal
-----------
This file is intentionally self-contained and minimally invasive:
- It does NOT modify train.py / utility.py / MultiCBR.py.
- It focuses only on BI-view denoising / purification.
- It outputs a purified binary B-I graph by scoring items for each bundle
  and keeping top-k original items.

Practical interpretation
------------------------
This module treats bundle-item purification as a *conditional diffusion-style
scoring* problem instead of a full generative diffusion over the whole item
vocabulary.

For each bundle b:
1. Build a noisy bundle context from its original items.
2. Learn to distinguish positive items in the bundle from sampled negatives.
3. Score original bundle items and keep top-k.

Why this formulation?
---------------------
If we only train on original bundle items with target 1, the model cannot learn
meaningful importance differences. Therefore, training uses positive / negative
item discrimination conditioned on a noised bundle context. Inference only ranks
*original* bundle items and prunes weak edges.

Main APIs
---------
- BIDiffusionTopK(...)
- module.forward(bundle_ids) -> dict(losses)
- module.build_purified_bi_graph(keep_k=..., ...) -> scipy.sparse.csr_matrix
- module.export_topk_pairs(keep_k=..., ...) -> List[Tuple[bundle_id, item_id]]

Later integration idea
----------------------
After training this module, replace the original bi_graph in MultiCBR with the
purified graph returned by build_purified_bi_graph(...).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_keep_k(bundle_len: int, keep_k: Optional[int] = None, keep_ratio: Optional[float] = None, min_keep: int = 5) -> int:
    """
    Resolve the number of items to keep for a given bundle based on truncation mode.
    If keep_ratio is provided, it uses ratio truncation: min(bundle_len, max(min_keep, ceil(keep_ratio * bundle_len)))
    Otherwise, it uses fixed top-k: min(bundle_len, keep_k)
    """
    if keep_ratio is not None:
        k_b = max(min_keep, math.ceil(keep_ratio * bundle_len))
        return min(k_b, bundle_len)
    elif keep_k is not None:
        return min(keep_k, bundle_len)
    else:
        raise ValueError("Either keep_k or keep_ratio must be provided.")



@dataclass
class BIDiffusionConfig:
    # model size
    embedding_size: int = 64
    hidden_size: int = 128
    time_embed_size: int = 64

    # diffusion-style schedule
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # training sampling
    max_context_items: int = 32
    num_negative: int = 16
    keep_all_if_len_le_k: bool = True

    # optional BLCC
    blcc_enabled: bool = False
    blcc_lambda: float = 0.1

    # misc
    dropout: float = 0.1
    use_layernorm: bool = True
    device: str = "cpu"


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """timesteps: [B] int64/float -> [B, dim]"""
        device = timesteps.device
        half = self.dim // 2
        if half == 0:
            return timesteps.float().unsqueeze(-1)
        factor = math.log(10000.0) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=device) * -factor)
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, use_layernorm: bool):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)])
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BIDiffusionTopK(nn.Module):
    """
    Conditional BI diffusion-style purifier.

    Training objective:
    - Build a noisy bundle context from bundle's positive items.
    - Distinguish in-bundle positives vs out-of-bundle negatives.
    - Optional BLCC regularization to keep purified bundle semantics close to
      the raw bundle semantics.

    Inference:
    - Score original items in each bundle.
    - Keep top-k items per bundle.
    - Rebuild a purified binary BI graph.
    """

    def __init__(
        self,
        num_bundles: int,
        num_items: int,
        bi_graph: sp.csr_matrix,
        config: Optional[BIDiffusionConfig] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if not sp.isspmatrix_csr(bi_graph):
            bi_graph = bi_graph.tocsr()

        self.num_bundles = int(num_bundles)
        self.num_items = int(num_items)
        self.bi_graph = bi_graph
        self.cfg = config or BIDiffusionConfig()
        self.device_name = self.cfg.device

        self.bundle_items: List[np.ndarray] = self._build_bundle_items_cache(bi_graph)
        self.bundle_item_sets: List[set] = [set(arr.tolist()) for arr in self.bundle_items]

        emb_dim = self.cfg.embedding_size
        hid_dim = self.cfg.hidden_size
        time_dim = self.cfg.time_embed_size

        # learnable fallback embeddings; can be replaced externally during forward()
        self.bundle_embedding_table = nn.Embedding(num_bundles, emb_dim)
        self.item_embedding_table = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_normal_(self.bundle_embedding_table.weight)
        nn.init.xavier_normal_(self.item_embedding_table.weight)

        if bundle_embeddings is not None:
            assert bundle_embeddings.shape == (num_bundles, emb_dim)
            with torch.no_grad():
                self.bundle_embedding_table.weight.copy_(bundle_embeddings)
        if item_embeddings is not None:
            assert item_embeddings.shape == (num_items, emb_dim)
            with torch.no_grad():
                self.item_embedding_table.weight.copy_(item_embeddings)

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
        )

        # context encoder: consumes noised item embeddings + bundle embedding + time embedding
        self.context_mlp = MLP(
            in_dim=emb_dim + hid_dim,
            hidden_dim=hid_dim,
            out_dim=emb_dim,
            dropout=self.cfg.dropout,
            use_layernorm=self.cfg.use_layernorm,
        )

        # candidate scorer: candidate item vs bundle context
        self.scorer = MLP(
            in_dim=emb_dim * 3 + hid_dim,
            hidden_dim=hid_dim,
            out_dim=1,
            dropout=self.cfg.dropout,
            use_layernorm=self.cfg.use_layernorm,
        )

        betas = torch.linspace(self.cfg.beta_start, self.cfg.beta_end, self.cfg.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def from_bi_graph(
        bi_graph: sp.csr_matrix,
        embedding_size: int = 64,
        hidden_size: int = 128,
        time_embed_size: int = 64,
        diffusion_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        max_context_items: int = 32,
        num_negative: int = 16,
        blcc_enabled: bool = False,
        blcc_lambda: float = 0.1,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        device: str = "cpu",
    ) -> "BIDiffusionTopK":
        cfg = BIDiffusionConfig(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            time_embed_size=time_embed_size,
            diffusion_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            max_context_items=max_context_items,
            num_negative=num_negative,
            blcc_enabled=blcc_enabled,
            blcc_lambda=blcc_lambda,
            dropout=dropout,
            use_layernorm=use_layernorm,
            device=device,
        )
        return BIDiffusionTopK(
            num_bundles=bi_graph.shape[0],
            num_items=bi_graph.shape[1],
            bi_graph=bi_graph,
            config=cfg,
        )

    def set_external_embeddings(
        self,
        bundle_embeddings: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        """Copy external embeddings into internal embedding tables."""
        with torch.no_grad():
            if bundle_embeddings is not None:
                assert bundle_embeddings.shape == self.bundle_embedding_table.weight.shape
                self.bundle_embedding_table.weight.copy_(bundle_embeddings)
            if item_embeddings is not None:
                assert item_embeddings.shape == self.item_embedding_table.weight.shape
                self.item_embedding_table.weight.copy_(item_embeddings)

    # ---------------------------------------------------------------------
    # training forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        bundle_ids: torch.Tensor,
        bundle_embeddings: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        bundle_ids: [B] or [B, 1]

        Returns dict with keys:
        - total_loss
        - diffusion_loss
        - blcc_loss
        - num_valid_bundles
        """
        if bundle_ids.dim() == 2:
            bundle_ids = bundle_ids.squeeze(-1)
        bundle_ids = bundle_ids.long().to(self.alpha_bars.device)

        bundle_emb_table, item_emb_table = self._resolve_embeddings(bundle_embeddings, item_embeddings)

        diffusion_losses = []
        blcc_losses = []
        valid = 0

        for b in bundle_ids.tolist():
            pos_items = self.bundle_items[b]
            if len(pos_items) == 0:
                continue
            # bundle length 1 is still trainable, but not very informative.
            valid += 1
            diffusion_loss, blcc_loss = self._bundle_loss(
                bundle_id=b,
                pos_items=pos_items,
                bundle_emb_table=bundle_emb_table,
                item_emb_table=item_emb_table,
            )
            diffusion_losses.append(diffusion_loss)
            blcc_losses.append(blcc_loss)

        if valid == 0:
            zero = torch.zeros([], device=self.alpha_bars.device)
            return {
                "total_loss": zero,
                "diffusion_loss": zero,
                "blcc_loss": zero,
                "num_valid_bundles": torch.zeros([], device=zero.device),
            }

        diffusion_loss = torch.stack(diffusion_losses).mean()
        if self.cfg.blcc_enabled and len(blcc_losses) > 0:
            blcc_loss = torch.stack(blcc_losses).mean()
            total_loss = diffusion_loss + self.cfg.blcc_lambda * blcc_loss
        else:
            blcc_loss = torch.zeros([], device=diffusion_loss.device)
            total_loss = diffusion_loss

        return {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "blcc_loss": blcc_loss,
            "num_valid_bundles": torch.tensor(float(valid), device=total_loss.device),
        }

    # ---------------------------------------------------------------------
    # graph rebuilding / inference
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def build_purified_bi_graph(
        self,
        keep_k: Optional[int] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
        keep_all_if_len_le_k: Optional[bool] = None,
        batch_size: int = 1024,
        keep_ratio: Optional[float] = None,
        min_keep: int = 5,
    ) -> sp.csr_matrix:
        """
        Score original items in each bundle and keep top-k or top-ratio.

        Returns
        -------
        scipy.sparse.csr_matrix of shape [num_bundles, num_items]
        """
        if keep_k is None and keep_ratio is None:
            raise ValueError("Either keep_k or keep_ratio must be provided.")
        if keep_k is not None and keep_k <= 0:
            raise ValueError("keep_k must be >= 1")

        keep_all = self.cfg.keep_all_if_len_le_k if keep_all_if_len_le_k is None else keep_all_if_len_le_k
        bundle_emb_table, item_emb_table = self._resolve_embeddings(bundle_embeddings, item_embeddings)

        rows: List[int] = []
        cols: List[int] = []

        for start in range(0, self.num_bundles, batch_size):
            end = min(start + batch_size, self.num_bundles)
            for b in range(start, end):
                item_ids = self.bundle_items[b]
                L = len(item_ids)
                if L == 0:
                    continue
                
                k = resolve_keep_k(L, keep_k=keep_k, keep_ratio=keep_ratio, min_keep=min_keep)

                if keep_all and L <= k:
                    rows.extend([b] * L)
                    cols.extend(item_ids.tolist())
                    continue

                scores = self.score_original_bundle_items(
                    bundle_id=b,
                    bundle_embeddings=bundle_emb_table,
                    item_embeddings=item_emb_table,
                )
                top_idx = torch.topk(scores, k=k, dim=0).indices.cpu().numpy()
                kept_items = item_ids[top_idx]
                rows.extend([b] * len(kept_items))
                cols.extend(kept_items.tolist())

        values = np.ones(len(rows), dtype=np.float32)
        purified = sp.coo_matrix(
            (values, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(self.num_bundles, self.num_items),
        ).tocsr()
        purified.eliminate_zeros()
        return purified

    @torch.no_grad()
    def export_topk_pairs(
        self,
        keep_k: Optional[int] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
        keep_all_if_len_le_k: Optional[bool] = None,
        keep_ratio: Optional[float] = None,
        min_keep: int = 5,
    ) -> List[Tuple[int, int]]:
        purified = self.build_purified_bi_graph(
            keep_k=keep_k,
            bundle_embeddings=bundle_embeddings,
            item_embeddings=item_embeddings,
            keep_all_if_len_le_k=keep_all_if_len_le_k,
            keep_ratio=keep_ratio,
            min_keep=min_keep,
        )
        coo = purified.tocoo()
        return list(zip(coo.row.tolist(), coo.col.tolist()))

    @torch.no_grad()
    def score_original_bundle_items(
        self,
        bundle_id: int,
        bundle_embeddings: Optional[torch.Tensor] = None,
        item_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns scores for ORIGINAL items only, shape [bundle_len].
        Higher means more important / more trustworthy edge.
        """
        bundle_emb_table, item_emb_table = self._resolve_embeddings(bundle_embeddings, item_embeddings)
        item_ids = self.bundle_items[bundle_id]
        if len(item_ids) == 0:
            return torch.empty(0, device=self.alpha_bars.device)

        # inference uses the full raw bundle context with t=0 and x=1
        context_repr, time_repr = self._build_context_for_inference(
            bundle_id=bundle_id,
            item_ids=item_ids,
            bundle_emb_table=bundle_emb_table,
            item_emb_table=item_emb_table,
        )
        cand_item_emb = item_emb_table[torch.as_tensor(item_ids, device=self.alpha_bars.device, dtype=torch.long)]
        bundle_emb = bundle_emb_table[bundle_id].unsqueeze(0).expand(cand_item_emb.size(0), -1)
        context_expand = context_repr.unsqueeze(0).expand(cand_item_emb.size(0), -1)
        time_expand = time_repr.unsqueeze(0).expand(cand_item_emb.size(0), -1)
        score_input = torch.cat([cand_item_emb, context_expand, bundle_emb, time_expand], dim=-1)
        scores = self.scorer(score_input).squeeze(-1)
        return scores

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _build_bundle_items_cache(self, bi_graph: sp.csr_matrix) -> List[np.ndarray]:
        items: List[np.ndarray] = []
        indptr = bi_graph.indptr
        indices = bi_graph.indices
        for b in range(bi_graph.shape[0]):
            items.append(indices[indptr[b]:indptr[b + 1]].copy())
        return items

    def _resolve_embeddings(
        self,
        bundle_embeddings: Optional[torch.Tensor],
        item_embeddings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bundle_embeddings is None:
            bundle_emb_table = self.bundle_embedding_table.weight
        else:
            bundle_emb_table = bundle_embeddings.to(self.alpha_bars.device)
        if item_embeddings is None:
            item_emb_table = self.item_embedding_table.weight
        else:
            item_emb_table = item_embeddings.to(self.alpha_bars.device)
        return bundle_emb_table, item_emb_table

    def _sample_negatives(self, bundle_id: int, num_negative: int) -> np.ndarray:
        negs: List[int] = []
        pos_set = self.bundle_item_sets[bundle_id]
        while len(negs) < num_negative:
            cand = random.randint(0, self.num_items - 1)
            if cand not in pos_set:
                negs.append(cand)
        return np.asarray(negs, dtype=np.int64)

    def _sample_context_items(self, pos_items: np.ndarray) -> np.ndarray:
        if len(pos_items) <= self.cfg.max_context_items:
            return pos_items
        perm = np.random.permutation(len(pos_items))[: self.cfg.max_context_items]
        return pos_items[perm]

    def _bundle_loss(
        self,
        bundle_id: int,
        pos_items: np.ndarray,
        bundle_emb_table: torch.Tensor,
        item_emb_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.alpha_bars.device
        pos_items = self._sample_context_items(pos_items)
        neg_items = self._sample_negatives(bundle_id, num_negative=max(self.cfg.num_negative, len(pos_items)))

        context_ids = torch.as_tensor(pos_items, device=device, dtype=torch.long)
        pos_ids = torch.as_tensor(pos_items, device=device, dtype=torch.long)
        neg_ids = torch.as_tensor(neg_items, device=device, dtype=torch.long)
        cand_ids = torch.cat([pos_ids, neg_ids], dim=0)
        labels = torch.cat(
            [torch.ones(pos_ids.size(0), device=device), torch.zeros(neg_ids.size(0), device=device)], dim=0
        )

        t = torch.randint(1, self.cfg.diffusion_steps + 1, size=(1,), device=device).long()
        context_repr, time_repr = self._build_noisy_context(
            bundle_id=bundle_id,
            context_ids=context_ids,
            t=t,
            bundle_emb_table=bundle_emb_table,
            item_emb_table=item_emb_table,
        )

        cand_item_emb = item_emb_table[cand_ids]
        bundle_emb = bundle_emb_table[bundle_id].unsqueeze(0).expand(cand_item_emb.size(0), -1)
        context_expand = context_repr.unsqueeze(0).expand(cand_item_emb.size(0), -1)
        time_expand = time_repr.unsqueeze(0).expand(cand_item_emb.size(0), -1)
        score_input = torch.cat([cand_item_emb, context_expand, bundle_emb, time_expand], dim=-1)
        logits = self.scorer(score_input).squeeze(-1)
        diffusion_loss = F.binary_cross_entropy_with_logits(logits, labels)

        if self.cfg.blcc_enabled:
            blcc_loss = self._compute_blcc(
                bundle_id=bundle_id,
                context_repr=context_repr,
                time_repr=time_repr,
                bundle_emb_table=bundle_emb_table,
                item_emb_table=item_emb_table,
            )
        else:
            blcc_loss = torch.zeros([], device=device)

        return diffusion_loss, blcc_loss

    def _build_noisy_context(
        self,
        bundle_id: int,
        context_ids: torch.Tensor,
        t: torch.Tensor,
        bundle_emb_table: torch.Tensor,
        item_emb_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build noisy context representation from original positive items.
        xt is applied to per-item scalar gates.
        """
        device = self.alpha_bars.device
        item_emb = item_emb_table[context_ids]  # [L, D]
        x0 = torch.ones(context_ids.size(0), device=device)
        xt = self._q_sample(x0=x0, t=t.expand(context_ids.size(0)))
        gated_item_emb = item_emb * xt.unsqueeze(-1)

        bundle_emb = bundle_emb_table[bundle_id]
        time_repr = self.time_proj(self.time_embedding(t.float())).squeeze(0)
        per_item_input = torch.cat(
            [gated_item_emb, time_repr.unsqueeze(0).expand(gated_item_emb.size(0), -1)], dim=-1
        )
        encoded_items = self.context_mlp(per_item_input)
        context_repr = encoded_items.mean(dim=0) + bundle_emb
        return context_repr, time_repr

    def _build_context_for_inference(
        self,
        bundle_id: int,
        item_ids: np.ndarray,
        bundle_emb_table: torch.Tensor,
        item_emb_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.alpha_bars.device
        ids = torch.as_tensor(item_ids, device=device, dtype=torch.long)
        item_emb = item_emb_table[ids]
        t = torch.zeros(1, device=device)
        time_repr = self.time_proj(self.time_embedding(t)).squeeze(0)
        per_item_input = torch.cat([item_emb, time_repr.unsqueeze(0).expand(item_emb.size(0), -1)], dim=-1)
        encoded_items = self.context_mlp(per_item_input)
        context_repr = encoded_items.mean(dim=0) + bundle_emb_table[bundle_id]
        return context_repr, time_repr

    def _compute_blcc(
        self,
        bundle_id: int,
        context_repr: torch.Tensor,
        time_repr: torch.Tensor,
        bundle_emb_table: torch.Tensor,
        item_emb_table: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional bundle latent consistency constraint.

        We compare:
        - raw bundle semantic vector: mean original item embeddings
        - purified semantic vector: top-k-weighted combination using scorer over
          original items (softmax weights, NOT hard top-k inside the loss)
        """
        device = self.alpha_bars.device
        item_ids_np = self.bundle_items[bundle_id]
        item_ids = torch.as_tensor(item_ids_np, device=device, dtype=torch.long)
        item_emb = item_emb_table[item_ids]

        raw_bundle_repr = item_emb.mean(dim=0)
        bundle_emb = bundle_emb_table[bundle_id].unsqueeze(0).expand(item_emb.size(0), -1)
        context_expand = context_repr.unsqueeze(0).expand(item_emb.size(0), -1)
        time_expand = time_repr.unsqueeze(0).expand(item_emb.size(0), -1)
        score_input = torch.cat([item_emb, context_expand, bundle_emb, time_expand], dim=-1)
        scores = self.scorer(score_input).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        purified_bundle_repr = torch.sum(weights.unsqueeze(-1) * item_emb, dim=0)
        return F.mse_loss(purified_bundle_repr, raw_bundle_repr)

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        DDPM-style forward noising for scalar gates.
        x0: [L]
        t:  [L] integer in [1, T]
        """
        if t.dtype != torch.long:
            t = t.long()
        alpha_bar_t = self.alpha_bars[t - 1]
        noise = torch.randn_like(x0)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


def build_bi_diffusion_module_from_sparse(
    bi_graph: sp.csr_matrix,
    embedding_size: int = 64,
    hidden_size: int = 128,
    time_embed_size: int = 64,
    diffusion_steps: int = 50,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    max_context_items: int = 32,
    num_negative: int = 16,
    blcc_enabled: bool = False,
    blcc_lambda: float = 0.1,
    dropout: float = 0.1,
    use_layernorm: bool = True,
    device: str = "cpu",
) -> BIDiffusionTopK:
    return BIDiffusionTopK.from_bi_graph(
        bi_graph=bi_graph,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        time_embed_size=time_embed_size,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        max_context_items=max_context_items,
        num_negative=num_negative,
        blcc_enabled=blcc_enabled,
        blcc_lambda=blcc_lambda,
        dropout=dropout,
        use_layernorm=use_layernorm,
        device=device,
    )


if __name__ == "__main__":
    # Minimal smoke test
    num_bundles, num_items = 4, 10
    rows = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3], dtype=np.int32)
    cols = np.array([1, 3, 5, 0, 2, 2, 4, 7, 9], dtype=np.int32)
    vals = np.ones(len(rows), dtype=np.float32)
    bi = sp.coo_matrix((vals, (rows, cols)), shape=(num_bundles, num_items)).tocsr()

    model = build_bi_diffusion_module_from_sparse(
        bi_graph=bi,
        embedding_size=32,
        hidden_size=64,
        diffusion_steps=20,
        num_negative=8,
        blcc_enabled=False,
    )
    model.train()
    out = model(torch.tensor([0, 1, 2, 3]))
    print({k: float(v.detach().cpu()) if torch.is_tensor(v) and v.numel() == 1 else v for k, v in out.items()})

    model.eval()
    purified = model.build_purified_bi_graph(keep_k=2)
    print("purified nnz:", purified.nnz)
