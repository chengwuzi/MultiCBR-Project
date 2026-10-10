#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReCoBR auxiliary contrastive losses used by the DWT compatibility path.

The CBR implementation remains untouched.  These helpers reproduce its
post-fusion and UB-anchored pre-fusion loss definitions without coupling DWT
to ``AnchorViewBundleNet`` or changing DWT graph propagation/fusion logic.
"""

import torch
import torch.nn.functional as F


def post_fusion_c_loss(users_embedding, bundles_embedding, temperature, device):
    """Return the CBR-style post-fusion contrastive loss for one training batch."""
    user_loss = paired_info_nce(
        users_embedding,
        users_embedding,
        temperature,
        device,
    )
    bundle_loss = paired_info_nce(
        bundles_embedding,
        bundles_embedding,
        temperature,
        device,
    )
    return (user_loss + bundle_loss) / 2


def ub_anchored_pre_fusion_loss(
    pre_fusion_users,
    pre_fusion_bundles,
    users,
    positive_bundles,
    anchor_conf,
    device,
):
    """Return the CBR-style UB-anchored contrastive loss before fusion.

    ``pre_fusion_users`` and ``pre_fusion_bundles`` are ordered as UB, UI,
    BI.  Only unique ids in the batch are used so repeated positive examples
    do not become false negatives.
    """
    ub_users, ui_users, bi_users = pre_fusion_users
    ub_bundles, ui_bundles, bi_bundles = pre_fusion_bundles
    temperature = anchor_conf.get("temp", 0.2)
    user_indices = torch.unique(users.squeeze(-1))
    bundle_indices = torch.unique(positive_bundles.squeeze(-1))
    loss = torch.tensor(0.0, device=device)

    if anchor_conf.get("user_cl", True) and user_indices.size(0) > 1:
        ub = F.normalize(ub_users[user_indices], p=2, dim=1)
        ui = F.normalize(ui_users[user_indices], p=2, dim=1)
        bi = F.normalize(bi_users[user_indices], p=2, dim=1)
        loss = loss + (
            anchor_conf.get("ub_ui_user_weight", 1.0) * normalized_info_nce(ub, ui, temperature, device)
            + anchor_conf.get("ub_bi_user_weight", 1.0) * normalized_info_nce(ub, bi, temperature, device)
        )

    if anchor_conf.get("bundle_cl", True) and bundle_indices.size(0) > 1:
        ub = F.normalize(ub_bundles[bundle_indices], p=2, dim=1)
        ui = F.normalize(ui_bundles[bundle_indices], p=2, dim=1)
        bi = F.normalize(bi_bundles[bundle_indices], p=2, dim=1)
        loss = loss + (
            anchor_conf.get("ub_ui_bundle_weight", 1.0) * normalized_info_nce(ub, ui, temperature, device)
            + anchor_conf.get("ub_bi_bundle_weight", 1.0) * normalized_info_nce(ub, bi, temperature, device)
        )

    return loss


def paired_info_nce(pos, aug, temperature, device):
    """CBR's paired InfoNCE for tensors shaped ``[batch, views, dim]``."""
    pos = pos[:, 0, :]
    aug = aug[:, 0, :]
    pos = F.normalize(pos, p=2, dim=1)
    aug = F.normalize(aug, p=2, dim=1)
    return normalized_info_nce(pos, aug, temperature, device)


def normalized_info_nce(anchor, positive, temperature, device):
    """InfoNCE after callers have arranged and normalized paired vectors."""
    if anchor.size(0) <= 1:
        return torch.tensor(0.0, device=device)
    positive_scores = torch.exp(torch.sum(anchor * positive, dim=1) / temperature)
    all_scores = torch.sum(torch.exp(torch.matmul(anchor, positive.transpose(0, 1)) / temperature), dim=1)
    return -torch.mean(torch.log(positive_scores / all_scores))
