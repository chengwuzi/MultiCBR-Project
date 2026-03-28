#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone UB-view diffusion-style purifier for MultiCBR.

Design goal
-----------
This file is intentionally self-contained and minimally invasive:
- It does NOT modify train.py / utility.py / MultiCBR.py.
- It focuses only on UB-view denoising / purification of existing training edges.
- It outputs a purified binary U-B graph by scoring bundles for each user
  and keeping top-k / top-ratio original positive bundles.

Practical interpretation
------------------------
This module treats user-bundle purification as a *conditional diffusion-style
scoring* problem.

For each user u:
1. Build a noisy user context from their original positive bundles.
2. Learn to distinguish positive bundles from sampled negative bundles.
3. Score original positive bundles and keep top-k or top-ratio.

Why this formulation?
---------------------
If we only train on original user bundles with target 1, the model cannot learn
meaningful importance differences. Therefore, training uses positive / negative
bundle discrimination conditioned on a noised user context. Inference only ranks
*original* positive bundles and prunes weak edges.
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


def resolve_keep_k(bundle_len: int, keep_k: Optional[int] = None, keep_ratio: Optional[float] = None, min_keep: int = 1) -> int:
    """
    Resolve the number of bundles to keep for a given user based on truncation mode.
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
class UBDiffusionConfig:
    # model size
    embedding_size: int = 64
    hidden_size: int = 128
    time_embed_size: int = 64

    # diffusion-style schedule
    diffusion_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # training sampling
    max_context_bundles: int = 32
    num_negative: int = 8
    keep_all_if_len_le_k: bool = True

    # optional BLCC (Bundle Latent Consistency Constraint)
    blcc_enabled: bool = False
    blcc_lambda: float = 0.1

    # misc
    dropout: float = 0.1
    use_layernorm: bool = False
    device: str = "cpu"


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
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


class UBDiffusionTopK(nn.Module):
    """
    Conditional UB diffusion-style purifier.
    """

    def __init__(
        self,
        num_users: int,
        num_bundles: int,
        ub_graph: sp.csr_matrix,
        config: Optional[UBDiffusionConfig] = None,
        user_embeddings: Optional[torch.Tensor] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if not sp.isspmatrix_csr(ub_graph):
            ub_graph = ub_graph.tocsr()

        self.num_users = int(num_users)
        self.num_bundles = int(num_bundles)
        self.ub_graph = ub_graph
        self.cfg = config or UBDiffusionConfig()
        self.device_name = self.cfg.device

        self.user_bundles: List[np.ndarray] = self._build_user_bundles_cache(ub_graph)
        self.user_bundle_sets: List[set] = [set(arr.tolist()) for arr in self.user_bundles]

        emb_dim = self.cfg.embedding_size
        hid_dim = self.cfg.hidden_size
        time_dim = self.cfg.time_embed_size

        self.user_embedding_table = nn.Embedding(num_users, emb_dim)
        self.bundle_embedding_table = nn.Embedding(num_bundles, emb_dim)
        nn.init.xavier_normal_(self.user_embedding_table.weight)
        nn.init.xavier_normal_(self.bundle_embedding_table.weight)

        if user_embeddings is not None:
            assert user_embeddings.shape == (num_users, emb_dim)
            with torch.no_grad():
                self.user_embedding_table.weight.copy_(user_embeddings)
        if bundle_embeddings is not None:
            assert bundle_embeddings.shape == (num_bundles, emb_dim)
            with torch.no_grad():
                self.bundle_embedding_table.weight.copy_(bundle_embeddings)

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.context_mlp = MLP(
            in_dim=emb_dim + hid_dim,
            hidden_dim=hid_dim,
            out_dim=emb_dim,
            dropout=self.cfg.dropout,
            use_layernorm=self.cfg.use_layernorm,
        )

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

    @staticmethod
    def from_ub_graph(
        ub_graph: sp.csr_matrix,
        embedding_size: int = 64,
        hidden_size: int = 128,
        time_embed_size: int = 64,
        diffusion_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        max_context_bundles: int = 32,
        num_negative: int = 8,
        blcc_enabled: bool = False,
        blcc_lambda: float = 0.1,
        dropout: float = 0.1,
        use_layernorm: bool = False,
        device: str = "cpu",
    ) -> "UBDiffusionTopK":
        cfg = UBDiffusionConfig(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            time_embed_size=time_embed_size,
            diffusion_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            max_context_bundles=max_context_bundles,
            num_negative=num_negative,
            blcc_enabled=blcc_enabled,
            blcc_lambda=blcc_lambda,
            dropout=dropout,
            use_layernorm=use_layernorm,
            device=device,
        )
        return UBDiffusionTopK(
            num_users=ub_graph.shape[0],
            num_bundles=ub_graph.shape[1],
            ub_graph=ub_graph,
            config=cfg,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        user_embeddings: Optional[torch.Tensor] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if user_ids.dim() == 2:
            user_ids = user_ids.squeeze(-1)
        user_ids = user_ids.long().to(self.alpha_bars.device)

        user_emb_table, bundle_emb_table = self._resolve_embeddings(user_embeddings, bundle_embeddings)

        diffusion_losses = []
        blcc_losses = []
        valid = 0

        for u in user_ids.tolist():
            pos_bundles = self.user_bundles[u]
            if len(pos_bundles) == 0:
                continue
            valid += 1
            diffusion_loss, blcc_loss = self._user_loss(
                user_id=u,
                pos_bundles=pos_bundles,
                user_emb_table=user_emb_table,
                bundle_emb_table=bundle_emb_table,
            )
            diffusion_losses.append(diffusion_loss)
            blcc_losses.append(blcc_loss)

        if valid == 0:
            zero = torch.zeros([], device=self.alpha_bars.device)
            return {
                "total_loss": zero,
                "diffusion_loss": zero,
                "blcc_loss": zero,
                "num_valid_users": torch.zeros([], device=zero.device),
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
            "num_valid_users": torch.tensor(float(valid), device=total_loss.device),
        }

    @torch.no_grad()
    def build_purified_ub_graph(
        self,
        keep_k: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        min_keep: int = 1,
        user_embeddings: Optional[torch.Tensor] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
        keep_all_if_len_le_k: Optional[bool] = None,
        batch_size: int = 1024,
    ) -> sp.csr_matrix:
        if keep_k is None and keep_ratio is None:
            raise ValueError("Either keep_k or keep_ratio must be provided.")
        if keep_k is not None and keep_k <= 0:
            raise ValueError("keep_k must be >= 1")

        keep_all = self.cfg.keep_all_if_len_le_k if keep_all_if_len_le_k is None else keep_all_if_len_le_k
        user_emb_table, bundle_emb_table = self._resolve_embeddings(user_embeddings, bundle_embeddings)

        rows: List[int] = []
        cols: List[int] = []

        for start in range(0, self.num_users, batch_size):
            end = min(start + batch_size, self.num_users)
            for u in range(start, end):
                bundle_ids = self.user_bundles[u]
                L = len(bundle_ids)
                if L == 0:
                    continue
                
                k = resolve_keep_k(L, keep_k=keep_k, keep_ratio=keep_ratio, min_keep=min_keep)

                if keep_all and L <= k:
                    rows.extend([u] * L)
                    cols.extend(bundle_ids.tolist())
                    continue

                scores = self.score_original_user_bundles(
                    user_id=u,
                    user_embeddings=user_emb_table,
                    bundle_embeddings=bundle_emb_table,
                )
                top_idx = torch.topk(scores, k=k, dim=0).indices.cpu().numpy()
                kept_bundles = bundle_ids[top_idx]
                rows.extend([u] * len(kept_bundles))
                cols.extend(kept_bundles.tolist())

        values = np.ones(len(rows), dtype=np.float32)
        purified = sp.coo_matrix(
            (values, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(self.num_users, self.num_bundles),
        ).tocsr()
        purified.eliminate_zeros()
        return purified

    @torch.no_grad()
    def export_topk_pairs(
        self,
        keep_k: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        min_keep: int = 1,
        user_embeddings: Optional[torch.Tensor] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
        keep_all_if_len_le_k: Optional[bool] = None,
    ) -> List[Tuple[int, int]]:
        purified = self.build_purified_ub_graph(
            keep_k=keep_k,
            keep_ratio=keep_ratio,
            min_keep=min_keep,
            user_embeddings=user_embeddings,
            bundle_embeddings=bundle_embeddings,
            keep_all_if_len_le_k=keep_all_if_len_le_k,
        )
        coo = purified.tocoo()
        return list(zip(coo.row.tolist(), coo.col.tolist()))

    @torch.no_grad()
    def score_original_user_bundles(
        self,
        user_id: int,
        user_embeddings: Optional[torch.Tensor] = None,
        bundle_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        user_emb_table, bundle_emb_table = self._resolve_embeddings(user_embeddings, bundle_embeddings)
        bundle_ids = self.user_bundles[user_id]
        if len(bundle_ids) == 0:
            return torch.empty(0, device=self.alpha_bars.device)

        context_repr, time_repr = self._build_context_for_inference(
            user_id=user_id,
            bundle_ids=bundle_ids,
            user_emb_table=user_emb_table,
            bundle_emb_table=bundle_emb_table,
        )
        cand_bundle_emb = bundle_emb_table[torch.as_tensor(bundle_ids, device=self.alpha_bars.device, dtype=torch.long)]
        user_emb = user_emb_table[user_id].unsqueeze(0).expand(cand_bundle_emb.size(0), -1)
        context_expand = context_repr.unsqueeze(0).expand(cand_bundle_emb.size(0), -1)
        time_expand = time_repr.unsqueeze(0).expand(cand_bundle_emb.size(0), -1)
        score_input = torch.cat([cand_bundle_emb, context_expand, user_emb, time_expand], dim=-1)
        scores = self.scorer(score_input).squeeze(-1)
        return scores

    def _build_user_bundles_cache(self, ub_graph: sp.csr_matrix) -> List[np.ndarray]:
        bundles: List[np.ndarray] = []
        indptr = ub_graph.indptr
        indices = ub_graph.indices
        for u in range(ub_graph.shape[0]):
            bundles.append(indices[indptr[u]:indptr[u + 1]].copy())
        return bundles

    def _resolve_embeddings(
        self,
        user_embeddings: Optional[torch.Tensor],
        bundle_embeddings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if user_embeddings is None:
            user_emb_table = self.user_embedding_table.weight
        else:
            user_emb_table = user_embeddings.to(self.alpha_bars.device)
        if bundle_embeddings is None:
            bundle_emb_table = self.bundle_embedding_table.weight
        else:
            bundle_emb_table = bundle_embeddings.to(self.alpha_bars.device)
        return user_emb_table, bundle_emb_table

    def _sample_negatives(self, user_id: int, num_negative: int) -> np.ndarray:
        negs: List[int] = []
        pos_set = self.user_bundle_sets[user_id]
        while len(negs) < num_negative:
            cand = random.randint(0, self.num_bundles - 1)
            if cand not in pos_set:
                negs.append(cand)
        return np.asarray(negs, dtype=np.int64)

    def _sample_context_bundles(self, pos_bundles: np.ndarray) -> np.ndarray:
        if len(pos_bundles) <= self.cfg.max_context_bundles:
            return pos_bundles
        perm = np.random.permutation(len(pos_bundles))[: self.cfg.max_context_bundles]
        return pos_bundles[perm]

    def _user_loss(
        self,
        user_id: int,
        pos_bundles: np.ndarray,
        user_emb_table: torch.Tensor,
        bundle_emb_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.alpha_bars.device
        pos_bundles = self._sample_context_bundles(pos_bundles)
        neg_bundles = self._sample_negatives(user_id, num_negative=max(self.cfg.num_negative, len(pos_bundles)))

        context_ids = torch.as_tensor(pos_bundles, device=device, dtype=torch.long)
        pos_ids = torch.as_tensor(pos_bundles, device=device, dtype=torch.long)
        neg_ids = torch.as_tensor(neg_bundles, device=device, dtype=torch.long)
        cand_ids = torch.cat([pos_ids, neg_ids], dim=0)
        labels = torch.cat(
            [torch.ones(pos_ids.size(0), device=device), torch.zeros(neg_ids.size(0), device=device)], dim=0
        )

        t = torch.randint(1, self.cfg.diffusion_steps + 1, size=(1,), device=device).long()
        context_repr, time_repr = self._build_noisy_context(
            user_id=user_id,
            context_ids=context_ids,
            t=t,
            user_emb_table=user_emb_table,
            bundle_emb_table=bundle_emb_table,
        )

        cand_bundle_emb = bundle_emb_table[cand_ids]
        user_emb = user_emb_table[user_id].unsqueeze(0).expand(cand_bundle_emb.size(0), -1)
        context_expand = context_repr.unsqueeze(0).expand(cand_bundle_emb.size(0), -1)
        time_expand = time_repr.unsqueeze(0).expand(cand_bundle_emb.size(0), -1)
        score_input = torch.cat([cand_bundle_emb, context_expand, user_emb, time_expand], dim=-1)
        logits = self.scorer(score_input).squeeze(-1)
        diffusion_loss = F.binary_cross_entropy_with_logits(logits, labels)

        if self.cfg.blcc_enabled:
            blcc_loss = self._compute_blcc(
                user_id=user_id,
                context_repr=context_repr,
                time_repr=time_repr,
                user_emb_table=user_emb_table,
                bundle_emb_table=bundle_emb_table,
            )
        else:
            blcc_loss = torch.zeros([], device=device)

        return diffusion_loss, blcc_loss

    def _build_noisy_context(
        self,
        user_id: int,
        context_ids: torch.Tensor,
        t: torch.Tensor,
        user_emb_table: torch.Tensor,
        bundle_emb_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.alpha_bars.device
        bundle_emb = bundle_emb_table[context_ids]
        x0 = torch.ones(context_ids.size(0), device=device)
        xt = self._q_sample(x0=x0, t=t.expand(context_ids.size(0)))
        gated_bundle_emb = bundle_emb * xt.unsqueeze(-1)

        user_emb = user_emb_table[user_id]
        time_repr = self.time_proj(self.time_embedding(t.float())).squeeze(0)
        per_bundle_input = torch.cat(
            [gated_bundle_emb, time_repr.unsqueeze(0).expand(gated_bundle_emb.size(0), -1)], dim=-1
        )
        encoded_bundles = self.context_mlp(per_bundle_input)
        context_repr = encoded_bundles.mean(dim=0) + user_emb
        return context_repr, time_repr

    def _build_context_for_inference(
        self,
        user_id: int,
        bundle_ids: np.ndarray,
        user_emb_table: torch.Tensor,
        bundle_emb_table: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.alpha_bars.device
        ids = torch.as_tensor(bundle_ids, device=device, dtype=torch.long)
        bundle_emb = bundle_emb_table[ids]
        t = torch.zeros(1, device=device)
        time_repr = self.time_proj(self.time_embedding(t)).squeeze(0)
        per_bundle_input = torch.cat([bundle_emb, time_repr.unsqueeze(0).expand(bundle_emb.size(0), -1)], dim=-1)
        encoded_bundles = self.context_mlp(per_bundle_input)
        context_repr = encoded_bundles.mean(dim=0) + user_emb_table[user_id]
        return context_repr, time_repr

    def _compute_blcc(
        self,
        user_id: int,
        context_repr: torch.Tensor,
        time_repr: torch.Tensor,
        user_emb_table: torch.Tensor,
        bundle_emb_table: torch.Tensor,
    ) -> torch.Tensor:
        device = self.alpha_bars.device
        bundle_ids_np = self.user_bundles[user_id]
        bundle_ids = torch.as_tensor(bundle_ids_np, device=device, dtype=torch.long)
        bundle_emb = bundle_emb_table[bundle_ids]

        raw_user_repr = bundle_emb.mean(dim=0)
        user_emb = user_emb_table[user_id].unsqueeze(0).expand(bundle_emb.size(0), -1)
        context_expand = context_repr.unsqueeze(0).expand(bundle_emb.size(0), -1)
        time_expand = time_repr.unsqueeze(0).expand(bundle_emb.size(0), -1)
        score_input = torch.cat([bundle_emb, context_expand, user_emb, time_expand], dim=-1)
        scores = self.scorer(score_input).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        purified_user_repr = torch.sum(weights.unsqueeze(-1) * bundle_emb, dim=0)
        return F.mse_loss(purified_user_repr, raw_user_repr)

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.long()
        alpha_bar_t = self.alpha_bars[t - 1]
        noise = torch.randn_like(x0)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


def build_ub_diffusion_module_from_sparse(
    ub_graph: sp.csr_matrix,
    embedding_size: int = 64,
    hidden_size: int = 128,
    time_embed_size: int = 64,
    diffusion_steps: int = 20,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    max_context_bundles: int = 32,
    num_negative: int = 8,
    blcc_enabled: bool = False,
    blcc_lambda: float = 0.1,
    dropout: float = 0.1,
    use_layernorm: bool = False,
    device: str = "cpu",
) -> UBDiffusionTopK:
    return UBDiffusionTopK.from_ub_graph(
        ub_graph=ub_graph,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        time_embed_size=time_embed_size,
        diffusion_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        max_context_bundles=max_context_bundles,
        num_negative=num_negative,
        blcc_enabled=blcc_enabled,
        blcc_lambda=blcc_lambda,
        dropout=dropout,
        use_layernorm=use_layernorm,
        device=device,
    )
