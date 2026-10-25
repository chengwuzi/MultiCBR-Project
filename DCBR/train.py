#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent DCBR port and single-experiment runner.

Model/DNN/diffusion: ported from raw/models/DCBR.py; no runtime raw imports.
Data sampling and metrics: aligned with the root ReCoBR utility.py/train.py.
Protocol changes and commands are documented in DCBR/README.md.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
SELECTION_RULE = "validation Recall@20 and NDCG@20 both strictly improve over the saved checkpoint"


class DCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        self.device = conf["device"]

        self.embedding_size = conf["embedding_size"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.num_layers = conf["num_layers"]
        self.gamma_1 = conf["gamma_1"]
        self.gamma_2 = conf["gamma_2"]
        self.tau = conf["tau"]
        self.lambda_1 = conf["lambda_1"]
        self.lambda_2 = conf["lambda_2"]

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph)
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph)

        self.upsilon_dict = {"UB": conf["upsilon_UB"], "UI": conf["upsilon_UI"], "BI": conf["upsilon_BI"]}
        self.modal_coefs = torch.FloatTensor([conf["omega"], 1 - conf["omega"]]).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.UB_layer_coefs = torch.FloatTensor(conf["xi_UB"]).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = torch.FloatTensor(conf["xi_UI"]).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = torch.FloatTensor(conf["xi_BI"]).unsqueeze(0).unsqueeze(-1).to(self.device)

        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

        self.restore_user_e = None
        self.restore_bundle_e = None
        self.other_params_name = ["restore_user_e", "restore_bundle_e"]


    def other_params(self):
        if hasattr(self, "other_params_name"):
            return {key: getattr(self, key) for key in self.other_params_name}
        return dict()


    def getUserEmbeds(self):
        return self.users_feature


    def getBundleEmbeds(self):
        return self.bundles_feature


    def get_propagation_graph(self, bipartite_graph):
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])
        rowsum_sqrt = sp.diags(1/(np.sqrt(np.asarray(propagation_graph.sum(axis=1)).ravel()) + 1e-8))
        colsum_sqrt = sp.diags(1/(np.sqrt(np.asarray(propagation_graph.sum(axis=0)).ravel()) + 1e-8))
        propagation_graph = rowsum_sqrt @ propagation_graph @ colsum_sqrt
        propagation_graph = propagation_graph.tocoo()
        values = propagation_graph.data
        indices = np.vstack((propagation_graph.row, propagation_graph.col))
        propagation_graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(propagation_graph.shape)).to(self.device)
        return propagation_graph


    def get_aggregation_graph(self, bipartite_graph):
        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/np.asarray(bundle_size).ravel()) @ bipartite_graph
        bipartite_graph = bipartite_graph.tocoo()
        values = bipartite_graph.data
        indices = np.vstack((bipartite_graph.row, bipartite_graph.col))
        bipartite_graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(bipartite_graph.shape)).to(self.device)
        return bipartite_graph


    def graph_propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), dim=0)
        all_features = [features]
        for _ in range(self.num_layers):
            features = torch.spmm(graph, features)
            if not test:
                random_noise = torch.rand_like(features).to(self.device)
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * self.upsilon_dict[graph_type]
            all_features.append(F.normalize(features, p=2, dim=1))
        all_features = torch.stack(all_features, dim=1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), dim=0)
        return A_feature, B_feature


    def graph_aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)
        if not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * self.upsilon_dict[graph_type]
        return aggregated_feature


    def propagate(self, UB_propagation_graph=None, test=False):
        if UB_propagation_graph is not None and not test:
            UB_users_feature, UB_bundles_feature = self.graph_propagate(UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        UI_users_feature, UI_items_feature = self.graph_propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
        UI_bundles_feature = self.graph_aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
        BI_bundles_feature, BI_items_feature = self.graph_propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
        BI_users_feature = self.graph_aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)

        users_rep = torch.sum(torch.stack([UI_users_feature, BI_users_feature], dim=0) * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(torch.stack([UI_bundles_feature, BI_bundles_feature], dim=0) * self.modal_coefs, dim=0)

        if test:
            return users_rep, bundles_rep
        else:
            users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
            bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]
            return users_rep, bundles_rep, users_feature, bundles_feature


    def cal_reg_loss(self):
        reg_loss = 0
        for W in self.parameters():
            reg_loss += W.norm(2).square()
        return reg_loss


    def cal_bpr_loss(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, dim=2)
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0].unsqueeze(1).expand_as(negs)
        else:
            negs = pred[:, 1].unsqueeze(1)
            pos = pred[:, 0].unsqueeze(1)
        bpr_loss = - torch.mean(torch.log(torch.sigmoid(pos - negs)))
        return bpr_loss + self.lambda_2 * self.cal_reg_loss()


    def cal_cl_loss(self, pos, aug):
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.matmul(pos, aug.transpose(0, 1))
        ttl_score = torch.sum(torch.exp(ttl_score / self.tau), dim=1)
        cl_loss = - torch.mean(torch.log(pos_score / ttl_score))
        return cl_loss


    def forward(self, UB_propagation_graph, batch):
        users, bundles = batch
        users_rep, bundles_rep, users_feature, bundles_feature = self.propagate(UB_propagation_graph)

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]
        bpr_loss = self.cal_bpr_loss(users_embedding, bundles_embedding)

        user = users[:, 0]
        bundle = bundles[:, 0]
        ub_users_embedding = users_feature[0][user]
        ub_bundles_embedding = bundles_feature[0][bundle]
        ui_users_embedding = users_feature[1][user]
        ui_bundles_embedding = bundles_feature[1][bundle]
        bi_users_embedding = users_feature[2][user]
        bi_bundles_embedding = bundles_feature[2][bundle]

        u_cl_inter = self.cal_cl_loss(ub_users_embedding, ui_users_embedding) \
                   + self.cal_cl_loss(ub_users_embedding, bi_users_embedding) \
                   + self.cal_cl_loss(ui_users_embedding, bi_users_embedding)

        b_cl_inter = self.cal_cl_loss(ub_bundles_embedding, ui_bundles_embedding) \
                   + self.cal_cl_loss(ub_bundles_embedding, bi_bundles_embedding) \
                   + self.cal_cl_loss(ui_bundles_embedding, bi_bundles_embedding)

        u_cl_intra = self.cal_cl_loss(ub_users_embedding, ub_users_embedding) \
                   + self.cal_cl_loss(ui_users_embedding, ui_users_embedding) \
                   + self.cal_cl_loss(bi_users_embedding, bi_users_embedding) \

        b_cl_intra = self.cal_cl_loss(ub_bundles_embedding, ub_bundles_embedding) \
                   + self.cal_cl_loss(ui_bundles_embedding, ui_bundles_embedding) \
                   + self.cal_cl_loss(bi_bundles_embedding, bi_bundles_embedding)

        cl_loss = self.gamma_1 * (u_cl_inter + b_cl_inter) + self.gamma_2 * (u_cl_intra + b_cl_intra)
        cl_loss *= self.lambda_1

        return bpr_loss, cl_loss


    def evaluate(self, propagate_result, users):
        self.restore_user_e, self.restore_bundle_e = propagate_result
        scores = torch.matmul(self.restore_user_e[users], self.restore_bundle_e.transpose(0, 1))
        return scores


class DNN(nn.Module):
    def __init__(self, in_dims, out_dims, time_emb_dim, norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = time_emb_dim
        self.norm = norm
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        self.drop = nn.Dropout(dropout)
        self.init_weights()


    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)


    def forward(self, x, timesteps, use_dropout=True, max_period=10000):
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=self.time_emb_dim // 2, dtype=torch.float32) / (self.time_emb_dim // 2)).to(x.device)
        args = timesteps[:, None].float() * freqs[None]
        timestep_embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_emb_dim % 2:
            timestep_embedding = torch.cat([timestep_embedding, torch.zeros_like(timestep_embedding[:, :1])], dim=-1)
        emb = self.emb_layer(timestep_embedding)
        if self.norm:
            x = F.normalize(x)
        if use_dropout:
            x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True, device="cpu"):
        super(GaussianDiffusion, self).__init__()
        self.device = torch.device(device)
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.0001
            self.calculate_for_diffusion()
            # Persist the original float64 schedule without changing its arithmetic.
            for name, value in list(vars(self).items()):
                if torch.is_tensor(value):
                    delattr(self, name)
                    self.register_buffer(name, value)


    def get_betas(self, max_beta=0.999):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas)


    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)


    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(self.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(self.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t


    def training_CBDM_losses(self, model, x_start, U_Embeds, B_Embeds, batch_user_index):
        batch_size = x_start.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(self.device)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((x_start - model_output) ** 2)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        elbo_loss = weight * mse

        B_new_embeds = torch.mm(model_output.transpose(0, 1), U_Embeds[batch_user_index])
        blcc_loss = self.mean_flat((B_new_embeds - B_Embeds) ** 2)

        return elbo_loss.mean(), blcc_loss.mean()


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise


    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        return posterior_mean


    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        model_mean = self.q_posterior_mean_variance(x_start=model_output, x_t=x, t=t)
        return {
            "mean": model_mean,
            "log_variance": model_log_variance
        }


    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(self.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))


# ReCoBR-compatible data and evaluation, independently implemented here so the
# frozen root entrypoint and the reference raw tree are never imported/edited.
class BundleTrainDataset(Dataset):
    def __init__(self, pairs, graph, num_bundles, neg_num):
        self.pairs = pairs
        self.graph = graph
        self.num_bundles = num_bundles
        self.neg_num = neg_num
        active = np.unique(pairs[:, 0])
        if np.any(num_bundles - graph.getnnz(axis=1)[active] < neg_num):
            raise ValueError("A training user has fewer unobserved bundles than neg_num")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        user, positive = self.pairs[index]
        bundles = [int(positive)]
        while len(bundles) < self.neg_num + 1:
            candidate = np.random.randint(self.num_bundles)
            if self.graph[user, candidate] == 0 and candidate not in bundles:
                bundles.append(candidate)
        return torch.LongTensor([user]), torch.LongTensor(bundles)


class BundleTestDataset(Dataset):
    def __init__(self, ground_truth, train_graph):
        self.ground_truth = ground_truth
        self.train_graph = train_graph

    def __len__(self):
        return self.ground_truth.shape[0]

    def __getitem__(self, index):
        return (index, torch.from_numpy(self.ground_truth[index].toarray()).squeeze(0),
                torch.from_numpy(self.train_graph[index].toarray()).squeeze(0))


class DiffusionDataset(Dataset):
    """All users, including zero-degree rows, as in the author's dense dataset.

    Materialize one identical float32 row at a time instead of allocating the
    entire iFashion U x B dense matrix (~5.56 GiB) plus its tensor copy.
    """
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return self.graph.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.graph[index].toarray().ravel()), index


class Datasets:
    def __init__(self, conf, num_workers=None):
        directory = Path(conf["data_path"]) / conf["dataset"]
        self.num_users, self.num_bundles, self.num_items = read_data_size(directory, conf["dataset"])
        sizes = self.num_users, self.num_bundles, self.num_items
        if max(conf["topk"]) > self.num_bundles:
            raise ValueError("topk exceeds the number of bundles")
        self.data_files = {}

        def relation(name, shape):
            path = directory / (name + ".txt")
            pairs = np.loadtxt(path, dtype=np.int64, delimiter="\t", ndmin=2, comments=None)
            if pairs.shape[1] != 2 or not len(pairs):
                raise ValueError(f"Expected nonempty two-column interactions: {path}")
            if np.any(pairs < 0) or np.any(pairs >= np.asarray(shape)):
                raise ValueError(f"Interaction IDs out of bounds: {path}")
            graph = sp.coo_matrix((np.ones(len(pairs), dtype=np.float32),
                                   (pairs[:, 0], pairs[:, 1])), shape=shape).tocsr()
            if graph.nnz != len(pairs):
                raise ValueError(f"Duplicate interactions: {path}")
            self.data_files[path.name] = file_hash(path)
            return pairs, graph

        # Same graph load order and training-pair order as the reference project.
        _, bi = relation("bundle_item", (sizes[1], sizes[2]))
        _, ui = relation("user_item", (sizes[0], sizes[2]))
        pairs, ub = relation("user_bundle_train", (sizes[0], sizes[1]))
        _, val = relation("user_bundle_tune", (sizes[0], sizes[1]))
        _, test = relation("user_bundle_test", (sizes[0], sizes[1]))
        for left, right in ((ub, val), (ub, test), (val, test)):
            if left.multiply(right).nnz:
                raise ValueError("User-bundle train/tune/test splits overlap")
        self.data_files[conf["dataset"] + "_data_size.txt"] = file_hash(
            directory / (conf["dataset"] + "_data_size.txt"))
        self.graphs = [ub, ui, bi]
        self.graph_statistics = {name: {"shape": list(g.shape), "edges": int(g.nnz)}
                                 for name, g in zip(("UB_train", "UI", "BI", "UB_val", "UB_test"),
                                                    (ub, ui, bi, val, test))}
        workers = (0 if os.name == "nt" else 10) if num_workers is None else num_workers
        self.train_loader = DataLoader(
            BundleTrainDataset(pairs, ub, self.num_bundles, conf["neg_num"]),
            batch_size=conf["batch_size_train"], shuffle=True, num_workers=workers, drop_last=True)
        if len(self.train_loader) == 0:
            raise ValueError("Training set is smaller than batch_size_train with drop_last=True")
        self.val_loader = DataLoader(BundleTestDataset(val, ub), batch_size=conf["batch_size_test"],
                                     shuffle=False, num_workers=workers)
        self.test_loader = DataLoader(BundleTestDataset(test, ub), batch_size=conf["batch_size_test"],
                                      shuffle=False, num_workers=workers)


def get_metrics(metrics, grd, pred, topks):
    # Exact arithmetic/statistics from the frozen root train.py.
    for k in topks:
        _, columns = torch.topk(pred, k)
        rows = torch.zeros_like(columns) + torch.arange(pred.shape[0], device=pred.device).view(-1, 1)
        hit = grd[rows.flatten().cpu(), columns.flatten().cpu()].view(-1, k)
        for name, result in (("recall", get_recall(pred, grd, hit, k)),
                             ("ndcg", get_ndcg(pred, grd, hit, k))):
            for index, value in enumerate(result):
                metrics[name][k][index] += value
    return metrics


def get_recall(pred, grd, is_hit, topk):
    num_pos = grd.sum(dim=1)
    return [(is_hit.sum(dim=1) / (num_pos + 1e-8)).sum().item(),
            pred.shape[0] - (num_pos == 0).sum().item()]


def get_ndcg(pred, grd, is_hit, topk):
    def dcg(hit):
        return (hit / torch.log2(torch.arange(2, topk + 2, device=grd.device,
                                            dtype=torch.float))).sum(-1)
    idcgs = torch.empty(1 + topk, dtype=torch.float)
    idcgs[0] = 1
    for index in range(1, topk + 1):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:index] = 1
        idcgs[index] = dcg(hit)
    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    ndcg = dcg(is_hit) / idcgs[num_pos].to(grd.device)
    return [ndcg.sum().item(), pred.shape[0] - (num_pos == 0).sum().item()]


@torch.no_grad()
def evaluate(model, loader, conf, representations=None):
    model.eval()
    if representations is None:
        representations = model.propagate(test=True)
    totals = {name: {k: [0, 0] for k in conf["topk"]} for name in ("recall", "ndcg")}
    for users, ground_truth, train_mask in loader:
        scores = model.evaluate(representations, users.to(conf["device"]))
        # Always mask ORIGINAL observed training edges, never the rebuilt graph.
        scores -= 1e8 * train_mask.to(conf["device"])
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite prediction scores")
        get_metrics(totals, ground_truth, scores, conf["topk"])
    if any(value[1] == 0 for metric in totals.values() for value in metric.values()):
        raise ValueError("Evaluation split has no positive users")
    return {name: {k: value[0] / value[1] for k, value in metric.items()}
            for name, metric in totals.items()}


def selection_metrics(metrics):
    return {"recall@20": metrics["recall"][20], "ndcg@20": metrics["ndcg"][20]}


def is_better_validation(current, best):
    return (current["recall"][20] > best["recall"][20]
            and current["ndcg"][20] > best["ndcg"][20])


def build_rebuilt_graph(rows, columns, num_users, num_bundles, device):
    """Author's all-bundle Top-K graph, WITH self loops and degree normalization."""
    ub = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)),
                       shape=(num_users, num_bundles), dtype=np.float32)
    adjacency = sp.bmat([[sp.csr_matrix((num_users, num_users)), ub],
                         [ub.T, sp.csr_matrix((num_bundles, num_bundles))]])
    adjacency = adjacency + sp.eye(adjacency.shape[0])
    degree = np.power(np.asarray(adjacency.sum(axis=1)), -0.5).flatten()
    degree[np.isinf(degree)] = 0.0
    normalized = (sp.diags(degree) @ adjacency @ sp.diags(degree)).tocoo()
    return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((normalized.row, normalized.col))),
                                   torch.FloatTensor(normalized.data), normalized.shape).to(device)


def train_diffusion_epoch(model, denoiser, diffusion, optimizer, loader, conf):
    # The recommendation embeddings condition BLCC but receive NO gradients here.
    denoiser.train()
    sums = np.zeros(3, dtype=np.float64)
    progress = tqdm(loader, desc="Diffusion", disable=conf.get("quiet", False))
    for interaction, users in progress:
        interaction, users = interaction.to(conf["device"]), users.to(conf["device"])
        optimizer.zero_grad()
        elbo, blcc = diffusion.training_CBDM_losses(
            denoiser, interaction, model.getUserEmbeds().detach(), model.getBundleEmbeds().detach(), users)
        loss = elbo + conf["lambda_0"] * blcc
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite diffusion loss")
        loss.backward()
        optimizer.step()
        values = [loss.item(), elbo.item(), blcc.item()]
        sums += values
        progress.set_postfix(loss=f"{values[0]:.6f}")
    return dict(zip(("diffusion", "elbo", "blcc_unweighted"), (sums / len(loader)).tolist()))


@torch.no_grad()
def rebuild_graph(denoiser, diffusion, loader, conf):
    # Same shuffled loader is iterated again, as in raw/train.py; all users and
    # ALL bundle candidates remain eligible. p_sample disables DNN dropout.
    rows, columns = [], []
    for interaction, users in loader:
        denoised = diffusion.p_sample(denoiser, interaction.to(conf["device"]),
                                      conf["sampling_steps"], conf["sampling_noise"])
        if not torch.isfinite(denoised).all():
            raise FloatingPointError("Non-finite diffusion predictions")
        selected = torch.topk(denoised, k=conf["rebuild_k"])[1].cpu().numpy()
        rows.extend(np.repeat(users.numpy(), conf["rebuild_k"]).tolist())
        columns.extend(selected.ravel().tolist())
    return build_rebuilt_graph(rows, columns, conf["num_users"], conf["num_bundles"], conf["device"])


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_data_size(directory, dataset):
    values = (directory / (dataset + "_data_size.txt")).read_text(encoding="utf-8").split()
    sizes = tuple(map(int, values[:3]))
    if len(sizes) != 3 or min(sizes) <= 0:
        raise ValueError("Dataset size must contain positive user, bundle and item counts")
    return sizes


def load_config(path, dataset, args=None):
    with Path(path).open(encoding="utf-8") as stream:
        configs = yaml.safe_load(stream)
    if not isinstance(configs, dict) or dataset not in configs:
        raise ValueError(f"No configuration for dataset {dataset}")
    conf = copy.deepcopy(configs[dataset])
    # Allow a saved actual-config snapshot to be replayed. Dimensions/device and
    # execution metadata are recomputed rather than accepted as model settings.
    for key in ("device", "gpu", "info", "quiet", "num_users", "num_bundles", "num_items", "selection_rule"):
        conf.pop(key, None)
    if args is not None:
        for name in ("seed", "epochs", "data_path"):
            value = getattr(args, name, None)
            if value is not None:
                conf[name] = value
    conf.update(dataset=dataset, model="DCBR")
    conf["data_path"] = str(resolve_project_path(conf["data_path"]))
    validate_config(conf)
    return conf


def validate_config(conf):
    required = {"data_path", "embedding_size", "batch_size_train", "batch_size_test", "topk",
                "neg_num", "lr", "test_interval", "seed", "dims", "norm", "time_emb_dim", "steps",
                "noise_scale", "noise_min", "noise_max", "sampling_noise", "sampling_steps", "rebuild_k",
                "lambda_0", "num_layers", "upsilon_UB", "upsilon_UI", "upsilon_BI", "xi_UB", "xi_UI",
                "xi_BI", "omega", "gamma_1", "gamma_2", "tau", "lambda_1", "lambda_2", "epochs"}
    missing = required - conf.keys()
    unknown = conf.keys() - required - {"dataset", "model", "num_workers"}
    if missing or unknown:
        raise ValueError(f"Configuration missing={sorted(missing)} unknown={sorted(unknown)}")
    if "num_workers" in conf and (type(conf["num_workers"]) is not int or conf["num_workers"] < 0):
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("embedding_size", "batch_size_train", "batch_size_test", "neg_num", "test_interval",
                "time_emb_dim", "steps", "rebuild_k", "num_layers", "epochs"):
        if type(conf[key]) is not int or conf[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if conf["time_emb_dim"] < 2 or conf["steps"] < 2:
        raise ValueError("Author diffusion requires steps >= 2 and time_emb_dim >= 2")
    if type(conf["seed"]) is not int or not 0 <= conf["seed"] <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    if type(conf["sampling_steps"]) is not int or not 0 <= conf["sampling_steps"] <= conf["steps"]:
        raise ValueError("sampling_steps must be an integer between 0 and steps")
    if any(type(conf[key]) is not bool for key in ("norm", "sampling_noise")):
        raise ValueError("norm and sampling_noise must be booleans")
    if not isinstance(conf["dims"], list) or not conf["dims"] or any(
            type(v) is not int or v <= 0 for v in conf["dims"]):
        raise ValueError("dims must be a nonempty list of positive integers")
    for key in ("lr", "tau", "noise_scale", "noise_min", "noise_max", "lambda_0", "lambda_1",
                "lambda_2", "gamma_1", "gamma_2", "upsilon_UB", "upsilon_UI", "upsilon_BI", "omega"):
        value = conf[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
    if min(conf["lr"], conf["tau"], conf["noise_scale"], conf["noise_min"]) <= 0:
        raise ValueError("lr, tau, noise_scale and noise_min must be positive")
    if not conf["noise_min"] <= conf["noise_max"] or conf["noise_scale"] * conf["noise_max"] >= 1:
        raise ValueError("Invalid diffusion variance range")
    if not 0 <= conf["omega"] <= 1:
        raise ValueError("omega must be in [0, 1]")
    for view in ("UB", "UI", "BI"):
        weights = conf["xi_" + view]
        if not isinstance(weights, list) or len(weights) != conf["num_layers"] + 1 or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights):
            raise ValueError(f"xi_{view} must have num_layers+1 finite nonnegative weights")
    topk = conf["topk"]
    if not isinstance(topk, list) or any(type(k) is not int or k <= 0 for k in topk) or (
            len(set(topk)) != len(topk) or 20 not in topk):
        raise ValueError("topk must contain unique positive integers including 20")
    if conf["epochs"] < conf["test_interval"]:
        raise ValueError("Training must reach at least one scheduled evaluation")
    directory = Path(conf["data_path"]) / conf["dataset"]
    sizes = read_data_size(directory, conf["dataset"])
    if max(topk) > sizes[1] or conf["rebuild_k"] > sizes[1]:
        raise ValueError("topk or rebuild_k exceeds the number of bundles")
    for name in ("bundle_item", "user_item", "user_bundle_train", "user_bundle_tune", "user_bundle_test"):
        if not (directory / (name + ".txt")).is_file():
            raise ValueError(f"Missing data file: {name}.txt")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def serializable_config(conf):
    return {key: str(value) if isinstance(value, torch.device) else value for key, value in conf.items()}


def append_experiment_record(path, record):
    result = record.get("result")
    lines = ["", "=" * 72, "EXECUTION_ID: " + record["execution_id"],
             "实验类别: " + ("第二种情况（新数据集参数搜索）" if record.get("experiment_case") == "second"
                           else "第一种情况（作者已有数据集配置）"), "模型: DCBR", "数据集: " + record["dataset"],
             "实验备注: " + record["note"], "状态: " + record["status"],
             "开始时间(UTC+08:00): " + record["started_at"], "结束时间(UTC+08:00): " + record["ended_at"],
             f"耗时（秒）: {record['duration_seconds']:.3f}",
             "GPU: " + record["gpu"], "运行设备: " + record["device"],
             "代码版本: " + record["git_commit"], "训练源码摘要: " + record["source_hash"],
             "配置指纹: " + record["config_hash"], "选模规则: " + SELECTION_RULE,
             "完整实际配置:", yaml.safe_dump(record["config"], allow_unicode=True, sort_keys=False).rstrip()]
    if result is not None:
        lines += [f"best_epoch: {result['best_epoch']}（从0计数）",
                  f"最佳训练轮次: {result['best_epoch_1based']}",
                  f"最佳验证 Recall@20: {result['selection_metrics']['recall@20']:.10f}",
                  f"最佳验证 NDCG@20: {result['selection_metrics']['ndcg@20']:.10f}"]
        for split in ("val", "test"):
            for k in sorted(map(int, result["metrics"][split]["recall"])):
                metrics = result["metrics"][split]
                lines.append(f"{split} TOP {k}: Recall={metrics['recall'][str(k)]:.8f}, "
                             f"NDCG={metrics['ndcg'][str(k)]:.8f}")
    if record.get("error"):
        lines.append("原因: " + record["error"])
    lines += ["训练产物: " + json.dumps(record["artifacts"], ensure_ascii=False),
              "END_EXECUTION: " + record["execution_id"], ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush()
        os.fsync(stream.fileno())


class RunLog:
    def __init__(self, path):
        self.stream = Path(path).open("a", encoding="utf-8")

    def write(self, message):
        text = f"[{timestamp()}] {message}"
        print(text, flush=True)
        self.stream.write(text + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def run_training(conf, args):
    experiment_case = args.experiment_case or ("second" if conf["dataset"] == "Youshu" else "first")
    conf["selection_rule"] = SELECTION_RULE
    execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = resolve_project_path(args.output_dir) / conf["dataset"] / execution_id
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {"log": str(directory / "training.log"), "config": str(directory / "config.yaml"),
                 "checkpoint_model": str(directory / "best.pt"),
                 "checkpoint_config": str(directory / "best_config.json"),
                 "result_json": str(directory / "result.json"),
                 "tensorboard": str(directory / "tensorboard")}
    started = time.monotonic()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    source_hash = file_hash(Path(__file__))
    config_hash = hashlib.sha256(json.dumps(serializable_config(conf), sort_keys=True).encode()).hexdigest()
    record = {"execution_id": execution_id, "dataset": conf["dataset"],
              "experiment_case": experiment_case,
              "note": args.note or f"DCBR {conf['dataset']} under the shared evaluation protocol",
              "status": "RUNNING", "started_at": timestamp(), "gpu": args.gpu,
              "device": str(conf["device"]), "git_commit": commit, "source_hash": source_hash,
              "config_hash": config_hash, "config": serializable_config(conf), "artifacts": artifacts}
    atomic_json(artifacts["result_json"], record)
    log = RunLog(artifacts["log"])
    writer = None
    try:
        set_seed(conf["seed"])
        workers = args.num_workers if args.num_workers is not None else conf.get("num_workers")
        dataset = Datasets(conf, workers)
        conf.update(num_users=dataset.num_users, num_bundles=dataset.num_bundles, num_items=dataset.num_items,
                    gpu=args.gpu, info=args.info, quiet=args.quiet,
                    num_workers=dataset.train_loader.num_workers)
        record["config"] = serializable_config(conf)
        record["config_hash"] = hashlib.sha256(json.dumps(record["config"], sort_keys=True).encode()).hexdigest()
        record["data_files"] = dataset.data_files
        with Path(artifacts["config"]).open("w", encoding="utf-8") as stream:
            yaml.safe_dump({conf["dataset"]: record["config"]}, stream, allow_unicode=True, sort_keys=False)
        log.write(f"DCBR {experiment_case}-case experiment " + execution_id)
        log.write("Actual configuration:\n" + yaml.safe_dump(record["config"], sort_keys=False))
        log.write("Dataset: " + json.dumps(dataset.graph_statistics))
        log.write("Checkpoint selection: " + SELECTION_RULE)
        model = DCBR(conf, dataset.graphs).to(conf["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=0)
        out_dims = conf["dims"] + [conf["num_bundles"]]
        denoiser = DNN(out_dims[::-1], out_dims, conf["time_emb_dim"], norm=conf["norm"]).to(conf["device"])
        diffusion = GaussianDiffusion(conf["noise_scale"], conf["noise_min"], conf["noise_max"],
                                      conf["steps"], device=conf["device"]).to(conf["device"])
        cbdm_optimizer = torch.optim.Adam(denoiser.parameters(), lr=conf["lr"], weight_decay=0)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(artifacts["tensorboard"])
        best_validation = {"recall": {20: 0.0}, "ndcg": {20: 0.0}}
        best_result = None
        for epoch in range(conf["epochs"]):
            epoch_start = time.monotonic()
            loader = DataLoader(DiffusionDataset(dataset.graphs[0]), batch_size=conf["batch_size_train"],
                                shuffle=True, num_workers=0)
            diffusion_losses = train_diffusion_epoch(model, denoiser, diffusion, cbdm_optimizer, loader, conf)
            ub_graph = rebuild_graph(denoiser, diffusion, loader, conf)
            model.train()
            losses = np.zeros(3, dtype=np.float64)
            progress = tqdm(dataset.train_loader, desc=f"Epoch {epoch + 1}/{conf['epochs']}", disable=args.quiet)
            for batch_index, batch in enumerate(progress):
                optimizer.zero_grad()
                bpr, cl = model(ub_graph, [value.to(conf["device"]) for value in batch])
                loss = bpr + cl  # Both explicit L2 and lambda_1 already applied inside DCBR.
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite recommendation loss")
                loss.backward()
                optimizer.step()
                values = [loss.item(), bpr.item(), cl.item()]
                losses += values
                step = epoch * len(dataset.train_loader) + batch_index
                for tag, value in zip(("loss", "loss_bpr_with_l2", "loss_cl_weighted"), values):
                    writer.add_scalar(tag, value, step)
                progress.set_postfix(loss=f"{values[0]:.6f}")
            for tag, value in diffusion_losses.items():
                writer.add_scalar("diffusion/" + tag, value, epoch)
            log.write(f"Epoch {epoch + 1}/{conf['epochs']} losses={(losses / len(dataset.train_loader)).tolist()} "
                      f"diffusion={diffusion_losses} seconds={time.monotonic() - epoch_start:.3f}")
            if (epoch + 1) % conf["test_interval"] != 0:
                continue
            metrics = {"val": evaluate(model, dataset.val_loader, conf),
                       "test": evaluate(model, dataset.test_loader, conf)}
            selected_metrics = selection_metrics(metrics["val"])
            if any(not math.isfinite(value) for value in selected_metrics.values()):
                raise FloatingPointError("Non-finite validation selection metrics")
            log.write(f"Epoch {epoch + 1} selection_metrics={json.dumps(selected_metrics)} metrics={json.dumps(metrics)}")
            for split, split_metrics in metrics.items():
                for name, values in split_metrics.items():
                    for k, value in values.items():
                        writer.add_scalar(f"{name}_{k}/{split}", value, step)
            if is_better_validation(metrics["val"], best_validation):
                best_validation = copy.deepcopy(metrics["val"])
                best_result = {"best_epoch": epoch, "best_epoch_1based": epoch + 1,
                               "selection_metrics": selected_metrics, "selection_rule": SELECTION_RULE,
                               "metrics": json.loads(json.dumps(metrics))}
                checkpoint = {"config": record["config"], "result": best_result,
                              "state_dict": model.state_dict(), "denoiser_state_dict": denoiser.state_dict(),
                              "diffusion_state_dict": diffusion.state_dict(), "rebuilt_ub_graph": ub_graph,
                              "optimizer_state_dict": optimizer.state_dict(),
                              "diffusion_optimizer_state_dict": cbdm_optimizer.state_dict(),
                              "data_files": dataset.data_files, "source_hash": source_hash,
                              "execution_id": execution_id}
                temporary = Path(artifacts["checkpoint_model"] + ".tmp")
                torch.save(checkpoint, temporary)
                os.replace(temporary, artifacts["checkpoint_model"])
                atomic_json(artifacts["checkpoint_config"], record["config"])
                record["result"] = best_result
                atomic_json(artifacts["result_json"], record)
                log.write(f"Saved best checkpoint at epoch {epoch + 1}")
        if best_result is None:
            raise RuntimeError("Validation Recall@20 and NDCG@20 never both exceeded zero; no best checkpoint was selected")
        record["artifact_hashes"] = {key: file_hash(artifacts[key])
                                    for key in ("checkpoint_model", "checkpoint_config", "config")}
        record.update(status="SUCCESS", completed=True, completed_epochs=conf["epochs"], result=best_result)
        log.write("Best result: " + json.dumps(best_result))
    except BaseException as error:
        record.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                      completed=False, error=f"{type(error).__name__}: {error}")
        log.write(traceback.format_exc())
        raise
    finally:
        record.update(ended_at=timestamp(), duration_seconds=round(time.monotonic() - started, 3))
        if writer is not None:
            writer.close()
        log.close()
        atomic_json(artifacts["result_json"], record)
        if args.result_json:
            atomic_json(Path(args.result_json).resolve(), record)
        default_record = MODEL_DIR / ("搜参结果.txt" if experiment_case == "second" else "实验记录.txt")
        append_experiment_record(resolve_project_path(args.record_file or default_record), record)
    return record


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as error:
        # Author's torch 1.11 predates weights_only. Only load trusted checkpoints
        # produced by this runner; modern versions use restricted loading above.
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location=device)


def evaluate_checkpoint(path, args, device):
    checkpoint = load_checkpoint(path, device)
    conf = copy.deepcopy(checkpoint["config"])
    if args.data_path is not None:
        conf["data_path"] = str(resolve_project_path(args.data_path))
    conf["device"] = device
    workers = args.num_workers if args.num_workers is not None else conf.get("num_workers")
    dataset = Datasets(conf, workers)
    if dataset.data_files != checkpoint["data_files"]:
        raise ValueError("Dataset content differs from the checkpoint's data fingerprints")
    model = DCBR(conf, dataset.graphs).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics = {"val": evaluate(model, dataset.val_loader, conf),
               "test": evaluate(model, dataset.test_loader, conf)}
    result = {"checkpoint": str(Path(path).resolve()), "best_epoch": checkpoint["result"]["best_epoch"],
              "selection_rule": checkpoint["result"].get("selection_rule", "unknown"),
              "selection_metrics": selection_metrics(metrics["val"]), "metrics": metrics}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def get_cmd(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODEL_DIR / "config.yaml")
    parser.add_argument("-d", "--dataset", default="iFashion")
    parser.add_argument("-m", "--model", choices=["DCBR"], default="DCBR")
    parser.add_argument("-g", "--gpu", default="0", help="CUDA_VISIBLE_DEVICES; logical cuda:0 is used")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("-s", "--seed", type=int)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("--data-path", dest="data_path")
    parser.add_argument("-i", "--info", default="")
    parser.add_argument("--note", default=None)
    parser.add_argument("--experiment-case", choices=["first", "second"], default=None,
                        help="Record category; defaults to second for Youshu, first otherwise")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", default=str(MODEL_DIR / "log"))
    parser.add_argument("--record-file", default=None, help="Defaults to the TXT for the experiment category")
    parser.add_argument("--result-json", help="Optional additional machine-readable result path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="Validate configuration and paths, without training or writing")
    mode.add_argument("--evaluate", type=Path, metavar="CHECKPOINT", help="Recompute metrics from a saved best checkpoint")
    return parser.parse_args(argv)


def main(argv=None):
    args = get_cmd(argv)
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.evaluate:
        conf = load_config(args.config.resolve(), args.dataset, args)
        if args.check_config:
            print(yaml.safe_dump({args.dataset: conf}, allow_unicode=True, sort_keys=False))
            print("Configuration and dataset paths validated. Selection: " + SELECTION_RULE)
            return 0
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if args.evaluate:
        evaluate_checkpoint(args.evaluate.resolve(), args, device)
    else:
        conf["device"] = device
        # Set before DataLoader workers are spawned (PYTHONHASHSEED applies to children).
        os.environ["PYTHONHASHSEED"] = str(conf["seed"])
        run_training(conf, args)
    return 0


if __name__ == "__main__":
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Training interrupted; available artifacts and execution status have been saved.", file=sys.stderr)
        sys.exit(130)
