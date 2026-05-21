#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 


class DWT(nn.Module):
    """DWT branch for bundle recommendation.

    The branch keeps UB as a training-time structural view, while the final
    recommendation representation is fused from the UI and BI views. The UB
    view is therefore available to the contrastive objective without changing
    the inference-time scoring surface.
    """

    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        self.device = conf["device"]

        dwt_conf = conf.get("dwt", {})
        if not dwt_conf:
            raise ValueError("DWT train_style requires a non-empty dwt config block.")

        self.embedding_size = conf["embedding_size"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.num_layers = conf["num_layers"]
        self.gamma_1 = dwt_conf["gamma_1"]
        self.gamma_2 = dwt_conf["gamma_2"]
        self.tau = dwt_conf["tau"]
        self.lambda_1 = dwt_conf["lambda_1"]
        self.lambda_2 = dwt_conf["lambda_2"]

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.ub_training_graph = None

        graph_conf = resolve_dwt_graph_config(conf)
        self._init_graph_operators()
        self._init_view_controls(graph_conf)

        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

    def set_training_ub_graph(self, ub_graph):
        self.ub_training_graph = ub_graph

    def _init_graph_operators(self):
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph)
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph)

    def _init_view_controls(self, graph_conf):
        self.upsilon_dict = {
            "UB": graph_conf["upsilon"]["UB"],
            "UI": graph_conf["upsilon"]["UI"],
            "BI": graph_conf["upsilon"]["BI"],
        }
        self.modal_coefs = torch.FloatTensor(
            [graph_conf["omega"], 1 - graph_conf["omega"]]
        ).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.UB_layer_coefs = self._build_layer_coef(graph_conf["layer_coefs"]["UB"], "UB")
        self.UI_layer_coefs = self._build_layer_coef(graph_conf["layer_coefs"]["UI"], "UI")
        self.BI_layer_coefs = self._build_layer_coef(graph_conf["layer_coefs"]["BI"], "BI")

    def _build_layer_coef(self, values, view_name):
        layer_coef = torch.FloatTensor(values).unsqueeze(0).unsqueeze(-1).to(self.device)
        expected_layer_coef_len = self.num_layers + 1
        if layer_coef.shape[1] != expected_layer_coef_len:
            raise ValueError(f"DWT expects fusion_weights.{view_name}_layer to have {expected_layer_coef_len} values")
        return layer_coef

    def get_propagation_graph(self, bipartite_graph):
        propagation_graph = sp.bmat(
            [
                [sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph],
                [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))],
            ]
        )
        rowsum_sqrt = sp.diags(1 / (np.sqrt(np.asarray(propagation_graph.sum(axis=1)).ravel()) + 1e-8))
        colsum_sqrt = sp.diags(1 / (np.sqrt(np.asarray(propagation_graph.sum(axis=0)).ravel()) + 1e-8))
        propagation_graph = rowsum_sqrt @ propagation_graph @ colsum_sqrt
        return _dwt_sparse_tensor(propagation_graph, self.device)

    def get_aggregation_graph(self, bipartite_graph):
        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1 / np.asarray(bundle_size).ravel()) @ bipartite_graph
        return _dwt_sparse_tensor(bipartite_graph, self.device)

    def graph_propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        return self._propagate_view_embeddings(graph, A_feature, B_feature, graph_type, layer_coef, test)

    def _propagate_view_embeddings(self, graph, source_feature, target_feature, graph_type, layer_coef, test):
        features = torch.cat((source_feature, target_feature), dim=0)
        all_features = [features]
        for _ in range(self.num_layers):
            features = torch.spmm(graph, features)
            if not test:
                features = self._add_view_noise(features, graph_type)
            all_features.append(F.normalize(features, p=2, dim=1))
        all_features = torch.stack(all_features, dim=1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        return torch.split(all_features, (source_feature.shape[0], target_feature.shape[0]), dim=0)

    def graph_aggregate(self, agg_graph, node_feature, graph_type, test):
        return self._aggregate_view_embeddings(agg_graph, node_feature, graph_type, test)

    def _aggregate_view_embeddings(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)
        if not test:
            aggregated_feature = self._add_view_noise(aggregated_feature, graph_type)
        return aggregated_feature

    def _add_view_noise(self, features, graph_type):
        random_noise = torch.rand_like(features).to(self.device)
        return features + torch.sign(features) * F.normalize(random_noise, dim=-1) * self.upsilon_dict[graph_type]

    def get_multi_modal_representations(self, test=False):
        if test:
            return self._propagate_for_eval()
        return self._propagate_for_train()

    def _propagate_for_train(self):
        if self.ub_training_graph is None:
            raise ValueError("DWT training UB graph has not been set.")

        UB_users_feature, UB_bundles_feature = self._build_ub_training_view()
        UI_users_feature, UI_bundles_feature = self._build_ui_view(test=False)
        BI_users_feature, BI_bundles_feature = self._build_bi_view(test=False)

        users_rep, bundles_rep = self._fuse_recommendation_views(
            UI_users_feature,
            BI_users_feature,
            UI_bundles_feature,
            BI_bundles_feature,
        )

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]
        return users_rep, bundles_rep, users_feature, bundles_feature

    def _propagate_for_eval(self):
        UI_users_feature, UI_bundles_feature = self._build_ui_view(test=True)
        BI_users_feature, BI_bundles_feature = self._build_bi_view(test=True)

        users_rep, bundles_rep = self._fuse_recommendation_views(
            UI_users_feature,
            BI_users_feature,
            UI_bundles_feature,
            BI_bundles_feature,
        )
        return users_rep, bundles_rep, None, None

    def _build_ub_training_view(self):
        return self.graph_propagate(
            self.ub_training_graph,
            self.users_feature,
            self.bundles_feature,
            "UB",
            self.UB_layer_coefs,
            False,
        )

    def _build_ui_view(self, test):
        UI_users_feature, UI_items_feature = self.graph_propagate(
            self.UI_propagation_graph,
            self.users_feature,
            self.items_feature,
            "UI",
            self.UI_layer_coefs,
            test,
        )
        UI_bundles_feature = self.graph_aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
        return UI_users_feature, UI_bundles_feature

    def _build_bi_view(self, test):
        BI_bundles_feature, BI_items_feature = self.graph_propagate(
            self.BI_propagation_graph,
            self.bundles_feature,
            self.items_feature,
            "BI",
            self.BI_layer_coefs,
            test,
        )
        BI_users_feature = self.graph_aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)
        return BI_users_feature, BI_bundles_feature

    def _fuse_recommendation_views(self, UI_users_feature, BI_users_feature, UI_bundles_feature, BI_bundles_feature):
        users_rep = torch.sum(
            torch.stack([UI_users_feature, BI_users_feature], dim=0) * self.modal_coefs,
            dim=0,
        )
        bundles_rep = torch.sum(
            torch.stack([UI_bundles_feature, BI_bundles_feature], dim=0) * self.modal_coefs,
            dim=0,
        )
        return users_rep, bundles_rep

    def cal_reg_loss(self):
        reg_loss = 0
        for parameter in self.parameters():
            reg_loss += parameter.norm(2).square()
        return reg_loss

    def cal_bpr_loss(self, users_feature, bundles_feature):
        pred = torch.sum(users_feature * bundles_feature, dim=2)
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0].unsqueeze(1).expand_as(negs)
        else:
            negs = pred[:, 1].unsqueeze(1)
            pos = pred[:, 0].unsqueeze(1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos - negs)))
        return bpr_loss + self.lambda_2 * self.cal_reg_loss()

    def cal_cl_loss(self, pos, aug):
        return self._paired_info_nce(pos, aug)

    def _paired_info_nce(self, pos, aug):
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.matmul(pos, aug.transpose(0, 1))
        ttl_score = torch.sum(torch.exp(ttl_score / self.tau), dim=1)
        return -torch.mean(torch.log(pos_score / ttl_score))

    def cal_dwt_cl_loss(self, users, bundles, users_feature, bundles_feature):
        if self.lambda_1 == 0 or (self.gamma_1 == 0 and self.gamma_2 == 0):
            return torch.tensor(0.0, device=self.device)

        user = users[:, 0]
        bundle = bundles[:, 0]
        ub_user_view = users_feature[0][user]
        ub_bundle_view = bundles_feature[0][bundle]
        ui_user_view = users_feature[1][user]
        ui_bundle_view = bundles_feature[1][bundle]
        bi_user_view = users_feature[2][user]
        bi_bundle_view = bundles_feature[2][bundle]

        inter_cl = torch.tensor(0.0, device=self.device)
        if self.gamma_1 != 0:
            inter_cl = (
                self.cal_cl_loss(ub_user_view, ui_user_view)
                + self.cal_cl_loss(ub_user_view, bi_user_view)
                + self.cal_cl_loss(ui_user_view, bi_user_view)
                + self.cal_cl_loss(ub_bundle_view, ui_bundle_view)
                + self.cal_cl_loss(ub_bundle_view, bi_bundle_view)
                + self.cal_cl_loss(ui_bundle_view, bi_bundle_view)
            )

        intra_cl = torch.tensor(0.0, device=self.device)
        if self.gamma_2 != 0:
            intra_cl = (
                self.cal_cl_loss(ub_user_view, ub_user_view)
                + self.cal_cl_loss(ui_user_view, ui_user_view)
                + self.cal_cl_loss(bi_user_view, bi_user_view)
                + self.cal_cl_loss(ub_bundle_view, ub_bundle_view)
                + self.cal_cl_loss(ui_bundle_view, ui_bundle_view)
                + self.cal_cl_loss(bi_bundle_view, bi_bundle_view)
            )

        return self.lambda_1 * (self.gamma_1 * inter_cl + self.gamma_2 * intra_cl)

    def forward(self, batch):
        users, bundles = batch
        users_rep, bundles_rep, users_feature, bundles_feature = self._propagate_for_train()

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]
        bpr_loss = self.cal_bpr_loss(users_embedding, bundles_embedding)
        dwt_cl_loss = self.cal_dwt_cl_loss(users, bundles, users_feature, bundles_feature)
        return bpr_loss, dwt_cl_loss

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result[:2]
        return torch.matmul(users_feature[users], bundles_feature.transpose(0, 1))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        dims = [input_dim]
        if num_layers == 1:
            dims.append(output_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(output_dim)

        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = F.relu(x, inplace=False)
                x = self.dropout(x)
        return x


class LatentDiffusionRebuilder(nn.Module):
    def __init__(self, conf, user_embeddings, bundle_embeddings):
        super().__init__()

        if user_embeddings.ndim != 2 or bundle_embeddings.ndim != 2:
            raise ValueError("External user and bundle embeddings must be 2D")
        if user_embeddings.shape[1] != bundle_embeddings.shape[1]:
            raise ValueError("User and bundle embeddings must share the same embedding dimension")

        self.embedding_dim = user_embeddings.shape[1]
        self.hidden_dim = conf["latent_diffusion_hidden_dim"]
        self.time_dim = conf["latent_diffusion_time_dim"]
        self.num_steps = conf["latent_diffusion_num_steps"]
        self.dropout = conf["latent_diffusion_dropout"]
        self.set_loss_weight = conf["latent_diffusion_set_loss_weight"]

        model_input_dim = self.embedding_dim * 4 + self.time_dim
        self.denoiser = MLP(
            model_input_dim,
            self.hidden_dim,
            self.embedding_dim,
            num_layers=conf["latent_diffusion_denoiser_layers"],
            dropout=self.dropout,
        )

        self.register_buffer("user_embeddings", user_embeddings.detach().float())
        self.register_buffer("bundle_embeddings", bundle_embeddings.detach().float())

        betas = torch.linspace(
            conf["latent_diffusion_beta_start"],
            conf["latent_diffusion_beta_end"],
            self.num_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1.0e-8)),
        )

    def reset_parameters(self):
        self.denoiser.reset_parameters()

    def _time_embedding(self, timesteps):
        device = timesteps.device
        half_dim = self.time_dim // 2
        if half_dim == 0:
            return torch.zeros((timesteps.shape[0], 0), device=device)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=device)
            / max(half_dim, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
        return emb

    def _pool_observed(self, observed_bundle_indices, observed_mask):
        safe_indices = observed_bundle_indices.clamp(min=0)
        observed_vecs = self.bundle_embeddings[safe_indices]
        observed_mask = observed_mask.unsqueeze(-1)
        summed = (observed_vecs * observed_mask).sum(dim=1)
        counts = observed_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _model_features(self, z_t, user_vec, timesteps):
        return torch.cat(
            [
                z_t,
                user_vec,
                z_t * user_vec,
                torch.abs(z_t - user_vec),
                self._time_embedding(timesteps),
            ],
            dim=-1,
        )

    def _predict_clean(self, z_t, user_vec, timesteps):
        features = self._model_features(z_t, user_vec, timesteps)
        return self.denoiser(features)

    def _extract(self, values, timesteps, reference):
        out = values[timesteps].view(-1, 1)
        return out.expand_as(reference)

    def q_sample(self, z0, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        coeff1 = self._extract(self.sqrt_alphas_cumprod, timesteps, z0)
        coeff2 = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, z0)
        return coeff1 * z0 + coeff2 * noise

    def training_loss(self, batch):
        user_indices = batch["user_indices"]
        observed_indices = batch["observed_indices"]
        observed_mask = batch["observed_mask"]

        z0 = self._pool_observed(observed_indices, observed_mask)
        user_vec = self.user_embeddings[user_indices]
        timesteps = torch.randint(0, self.num_steps, (z0.shape[0],), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, timesteps, noise)
        z_pred = self._predict_clean(z_t, user_vec, timesteps)

        diff_loss = F.mse_loss(z_pred, z0)

        safe_indices = observed_indices.clamp(min=0)
        observed_vecs = self.bundle_embeddings[safe_indices]
        pred_norm = F.normalize(z_pred, dim=-1).unsqueeze(1)
        observed_norm = F.normalize(observed_vecs, dim=-1)
        cosine = (pred_norm * observed_norm).sum(dim=-1)
        cosine = cosine * observed_mask
        mean_cosine = cosine.sum(dim=1) / observed_mask.sum(dim=1).clamp(min=1.0)
        set_loss = (1.0 - mean_cosine).mean()

        return diff_loss + self.set_loss_weight * set_loss

    def refine_latent(self, user_indices, observed_bundle_indices, observed_mask):
        z = self._pool_observed(observed_bundle_indices, observed_mask)
        user_vec = self.user_embeddings[user_indices]
        for step in reversed(range(self.num_steps)):
            timesteps = torch.full((z.shape[0],), step, dtype=torch.long, device=z.device)
            z = self._predict_clean(z, user_vec, timesteps)
        return z

    def rebuild_topk(self, user_indices, observed_bundle_indices, rebuild_k=1):
        if rebuild_k <= 0:
            raise ValueError("rebuild_k must be positive")

        device = self.user_embeddings.device
        observed_bundle_indices = observed_bundle_indices.to(device)
        valid_mask = observed_bundle_indices.ge(0)
        observed_mask = valid_mask.float()
        safe_indices = observed_bundle_indices.clamp(min=0)

        z_star = self.refine_latent(user_indices, safe_indices, observed_mask)
        z_norm = F.normalize(z_star, dim=-1).unsqueeze(1)
        observed_vecs = self.bundle_embeddings[safe_indices]
        observed_norm = F.normalize(observed_vecs, dim=-1)
        scores = (z_norm * observed_norm).sum(dim=-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        topk = min(rebuild_k, observed_bundle_indices.shape[1])
        _, selected_pos = torch.topk(scores, k=topk, dim=1)
        selected_bundles = torch.gather(safe_indices, 1, selected_pos)
        selected_valid = torch.gather(valid_mask, 1, selected_pos)
        return selected_bundles, selected_valid


class UserPreferenceDiffusionRebuilder(nn.Module):
    def __init__(self, conf, user_embeddings, bundle_embeddings):
        super().__init__()

        if user_embeddings.ndim != 2 or bundle_embeddings.ndim != 2:
            raise ValueError("External user and bundle embeddings must be 2D")
        if user_embeddings.shape[1] != bundle_embeddings.shape[1]:
            raise ValueError("User and bundle embeddings must share the same embedding dimension")

        self.embedding_dim = user_embeddings.shape[1]
        self.hidden_dim = conf["hidden_dim"]
        self.time_dim = conf["time_dim"]
        self.num_steps = conf["num_steps"]
        self.dropout = conf["dropout"]
        self.tau_select = conf["tau_select"]
        self.target_num = conf["target_num"]
        self.neg_num_for_core = conf.get("neg_num_for_core", 1)
        if self.neg_num_for_core < 1:
            raise ValueError("neg_num_for_core must be at least 1")
        self.neg_sampling = conf.get("neg_sampling", "random")
        if self.neg_sampling not in {"random", "hard"}:
            raise ValueError(f"Unsupported neg_sampling: {self.neg_sampling}")

        self.lambda_query = conf.get("lambda_query", 0.0)
        self.lambda_consistency = conf.get("lambda_consistency", 0.0)
        self.lambda_anchor = conf.get("lambda_anchor", 0.0)
        self.anchor_type = conf.get("anchor_type", "cosine")
        self.infer_mode = conf.get("infer_mode", "one_step")
        self.infer_step = int(conf.get("infer_step", 0))

        model_input_dim = self.embedding_dim * 4 + self.time_dim
        self.denoiser = MLP(
            model_input_dim,
            self.hidden_dim,
            self.embedding_dim,
            num_layers=conf["denoiser_layers"],
            dropout=self.dropout,
        )

        self.register_buffer("user_embeddings", user_embeddings.detach().float())
        self.register_buffer("bundle_embeddings", bundle_embeddings.detach().float())

        betas = torch.linspace(
            conf["beta_start"],
            conf["beta_end"],
            self.num_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1.0e-8)),
        )

    def reset_parameters(self):
        self.denoiser.reset_parameters()

    def _time_embedding(self, timesteps):
        device = timesteps.device
        half_dim = self.time_dim // 2
        if half_dim == 0:
            return torch.zeros((timesteps.shape[0], 0), device=device)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32, device=device)
            / max(half_dim, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
        return emb

    def _pool_observed(self, observed_bundle_indices, observed_mask):
        safe_indices = observed_bundle_indices.clamp(min=0)
        observed_vecs = self.bundle_embeddings[safe_indices]
        observed_mask = observed_mask.unsqueeze(-1)
        summed = (observed_vecs * observed_mask).sum(dim=1)
        counts = observed_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _model_features(self, e_t, z0, timesteps):
        return torch.cat(
            [
                e_t,
                z0,
                e_t * z0,
                torch.abs(e_t - z0),
                self._time_embedding(timesteps),
            ],
            dim=-1,
        )

    def _predict_query(self, e_t, z0, timesteps):
        features = self._model_features(e_t, z0, timesteps)
        return self.denoiser(features)

    def _extract(self, values, timesteps, reference):
        out = values[timesteps].view(-1, 1)
        return out.expand_as(reference)

    def q_sample(self, e0, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(e0)
        coeff1 = self._extract(self.sqrt_alphas_cumprod, timesteps, e0)
        coeff2 = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, e0)
        return coeff1 * e0 + coeff2 * noise

    def _sample_target_mask(self, observed_mask):
        valid_mask = observed_mask.bool()
        batch_size, max_len = valid_mask.shape
        target_mask = torch.zeros_like(valid_mask)
        trainable_rows = torch.zeros(batch_size, dtype=torch.bool, device=observed_mask.device)

        for row_idx in range(batch_size):
            valid_positions = torch.nonzero(valid_mask[row_idx], as_tuple=False).flatten()
            observed_count = valid_positions.numel()
            if observed_count <= 1:
                continue
            effective_target_num = min(self.target_num, observed_count - 1)
            perm = torch.randperm(observed_count, device=observed_mask.device)[:effective_target_num]
            target_positions = valid_positions[perm]
            target_mask[row_idx, target_positions] = True
            trainable_rows[row_idx] = True

        return target_mask, trainable_rows

    def _sample_negative_bundles(self, observed_indices, observed_mask):
        device = observed_indices.device
        batch_size = observed_indices.shape[0]
        neg_ids = torch.empty(batch_size, self.neg_num_for_core, dtype=torch.long, device=device)
        num_bundles = self.bundle_embeddings.shape[0]

        observed_cpu = observed_indices.detach().cpu()
        observed_mask_cpu = observed_mask.detach().cpu().bool()
        for row_idx in range(batch_size):
            observed_set = set(observed_cpu[row_idx][observed_mask_cpu[row_idx]].tolist())
            sampled = set()
            neg_col = 0
            while neg_col < self.neg_num_for_core:
                candidate = int(torch.randint(0, num_bundles, (1,), device=device).item())
                if candidate not in observed_set and candidate not in sampled:
                    neg_ids[row_idx, neg_col] = candidate
                    sampled.add(candidate)
                    neg_col += 1
        return neg_ids

    def training_loss(self, batch):
        user_indices = batch["user_indices"]
        observed_indices = batch["observed_indices"]
        observed_mask = batch["observed_mask"]
        device = observed_indices.device

        z0 = self._pool_observed(observed_indices, observed_mask)
        user_vec = self.user_embeddings[user_indices]
        timesteps = torch.randint(0, self.num_steps, (user_vec.shape[0],), device=device)
        e_t = self.q_sample(user_vec, timesteps)
        q = self._predict_query(e_t, z0, timesteps)

        safe_indices = observed_indices.clamp(min=0)
        observed_vecs = self.bundle_embeddings[safe_indices]
        target_mask, trainable_rows = self._sample_target_mask(observed_mask)
        if not trainable_rows.any():
            zero = q.sum() * 0.0
            return {
                "loss": zero,
                "core": zero.detach(),
                "query": zero.detach(),
                "consistency": zero.detach(),
                "anchor": zero.detach(),
            }

        candidate_mask = observed_mask.bool() & ~target_mask
        row_filter = trainable_rows
        q_valid = q[row_filter]
        user_vec_valid = user_vec[row_filter]
        observed_vecs_valid = observed_vecs[row_filter]
        observed_indices_valid = observed_indices[row_filter]
        observed_mask_valid = observed_mask[row_filter]
        target_mask_valid = target_mask[row_filter]
        candidate_mask_valid = candidate_mask[row_filter]

        target_weight = target_mask_valid.float().unsqueeze(-1)
        target_counts = target_weight.sum(dim=1).clamp(min=1.0)
        target_center = (observed_vecs_valid * target_weight).sum(dim=1) / target_counts

        q_norm = F.normalize(q_valid, dim=-1).unsqueeze(1)
        observed_norm = F.normalize(observed_vecs_valid, dim=-1)
        candidate_scores = (q_norm * observed_norm).sum(dim=-1)
        candidate_scores = candidate_scores.masked_fill(~candidate_mask_valid, float("-inf"))
        selection_weights = torch.softmax(candidate_scores / self.tau_select, dim=1)
        core_rep = (selection_weights.unsqueeze(-1) * observed_vecs_valid).sum(dim=1)

        neg_ids = self._sample_negative_bundles(observed_indices_valid, observed_mask_valid)
        neg_vec = self.bundle_embeddings[neg_ids]

        core_norm = F.normalize(core_rep, dim=-1)
        target_norm = F.normalize(target_center, dim=-1)
        neg_norm = F.normalize(neg_vec, dim=-1)
        score_pos = (core_norm * target_norm).sum(dim=-1)
        score_neg_all = (core_norm.unsqueeze(1) * neg_norm).sum(dim=-1)
        if self.neg_sampling == "hard":
            score_neg = score_neg_all.max(dim=1).values
        else:
            score_neg = score_neg_all.mean(dim=1)
        core_loss = F.softplus(-(score_pos - score_neg)).mean()

        query_loss = torch.zeros((), dtype=core_loss.dtype, device=device)
        if self.lambda_query != 0.0:
            q_direct_norm = F.normalize(q_valid, dim=-1)
            score_pos_query = (q_direct_norm * target_norm).sum(dim=-1)
            score_neg_query_all = (q_direct_norm.unsqueeze(1) * neg_norm).sum(dim=-1)
            if self.neg_sampling == "hard":
                score_neg_query = score_neg_query_all.max(dim=1).values
            else:
                score_neg_query = score_neg_query_all.mean(dim=1)
            query_loss = F.softplus(-(score_pos_query - score_neg_query)).mean()

        consistency_loss = torch.zeros((), dtype=core_loss.dtype, device=device)
        if self.lambda_consistency != 0.0:
            timesteps_2 = torch.randint(0, self.num_steps, (user_vec.shape[0],), device=device)
            e_t_2 = self.q_sample(user_vec, timesteps_2)
            q_2 = self._predict_query(e_t_2, z0, timesteps_2)[row_filter]
            consistency_loss = (1.0 - (F.normalize(q_valid, dim=-1) * F.normalize(q_2, dim=-1)).sum(dim=-1)).mean()

        anchor_loss = torch.zeros((), dtype=core_loss.dtype, device=device)
        if self.lambda_anchor != 0.0:
            if self.anchor_type == "mse":
                anchor_loss = F.mse_loss(q_valid, user_vec_valid)
            else:
                anchor_loss = (
                    1.0
                    - (F.normalize(q_valid, dim=-1) * F.normalize(user_vec_valid, dim=-1)).sum(dim=-1)
                ).mean()

        total_loss = (
            core_loss
            + self.lambda_query * query_loss
            + self.lambda_consistency * consistency_loss
            + self.lambda_anchor * anchor_loss
        )

        return {
            "loss": total_loss,
            "core": core_loss.detach(),
            "query": query_loss.detach(),
            "consistency": consistency_loss.detach(),
            "anchor": anchor_loss.detach(),
        }

    def refine_user_query(self, user_indices, observed_bundle_indices, observed_mask):
        z0 = self._pool_observed(observed_bundle_indices, observed_mask)
        user_vec = self.user_embeddings[user_indices]

        if self.infer_mode == "iterative":
            q = user_vec
            for step in reversed(range(self.num_steps)):
                timesteps = torch.full((q.shape[0],), step, dtype=torch.long, device=q.device)
                q = self._predict_query(q, z0, timesteps)
            return q

        if self.infer_mode != "one_step":
            raise ValueError(f"Unsupported user preference diffusion infer_mode: {self.infer_mode}")

        infer_step = min(max(self.infer_step, 0), self.num_steps - 1)
        timesteps = torch.full((user_vec.shape[0],), infer_step, dtype=torch.long, device=user_vec.device)
        return self._predict_query(user_vec, z0, timesteps)

    def rebuild_topk(self, user_indices, observed_bundle_indices, rebuild_k=1):
        if rebuild_k <= 0:
            raise ValueError("rebuild_k must be positive")

        device = self.user_embeddings.device
        user_indices = user_indices.to(device)
        observed_bundle_indices = observed_bundle_indices.to(device)
        valid_mask = observed_bundle_indices.ge(0)
        observed_mask = valid_mask.float()
        safe_indices = observed_bundle_indices.clamp(min=0)

        q_star = self.refine_user_query(user_indices, safe_indices, observed_mask)
        q_norm = F.normalize(q_star, dim=-1).unsqueeze(1)
        observed_vecs = self.bundle_embeddings[safe_indices]
        observed_norm = F.normalize(observed_vecs, dim=-1)
        scores = (q_norm * observed_norm).sum(dim=-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        topk = min(rebuild_k, observed_bundle_indices.shape[1])
        _, selected_pos = torch.topk(scores, k=topk, dim=1)
        selected_bundles = torch.gather(safe_indices, 1, selected_pos)
        selected_valid = torch.gather(valid_mask, 1, selected_pos)
        return selected_bundles, selected_valid


class AnchorViewBundleNet(nn.Module):
    """Anchor-guided multi-view bundle recommendation model.

    The model keeps three recommendation views over user-bundle, user-item,
    and bundle-item relations. Two project-specific components live on top of
    the base propagation path: cross-view residual aggregation and UB-anchored
    pre-fusion contrastive learning.
    """

    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        self.fusion_weights = conf['fusion_weights']

        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        self.ui_bundle_user_agg_beta = self.conf.get("ui_bundle_user_agg_beta", 0.1)
        self.bi_user_bundle_agg_beta = self.conf.get("bi_user_bundle_agg_beta", 0.1)

        self._init_static_view_graphs()
        self._init_training_view_graphs()

        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()

    def _init_static_view_graphs(self):
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)

        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        self.BU_aggregation_graph_ori = self.get_aggregation_graph(self.ub_graph.T)
        self.UB_aggregation_graph_ori = self.get_aggregation_graph(self.ub_graph)

    def _init_training_view_graphs(self):
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        self.BU_aggregation_graph = self.get_aggregation_graph(self.ub_graph.T, self.conf["UB_ratio"])
        self.UB_aggregation_graph = self.get_aggregation_graph(self.ub_graph, self.conf["UB_ratio"])

    def set_ub_graph(self, ub_graph):
        self.ub_graph = ub_graph
        self._refresh_anchor_view_graphs()

    def _refresh_anchor_view_graphs(self):
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
        self.BU_aggregation_graph_ori = self.get_aggregation_graph(self.ub_graph.T)
        self.UB_aggregation_graph_ori = self.get_aggregation_graph(self.ub_graph)

        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])
        self.BU_aggregation_graph = self.get_aggregation_graph(self.ub_graph.T, self.conf["UB_ratio"])
        self.UB_aggregation_graph = self.get_aggregation_graph(self.ub_graph, self.conf["UB_ratio"])

    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.conf["UB_ratio"], True)
        self.UI_dropout = nn.Dropout(self.conf["UI_ratio"], True)
        self.BI_dropout = nn.Dropout(self.conf["BI_ratio"], True)
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }

    def init_noise_eps(self):
        self.UB_eps = self.conf["UB_ratio"]
        self.UI_eps = self.conf["UI_ratio"]
        self.BI_eps = self.conf["BI_ratio"]
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

    def init_fusion_weights(self):
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1), \
            "The number of layer fusion weights does not correspond to number of layers"

        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)

    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat(
            [[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph],
             [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)

    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1 / bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)

    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        return self._propagate_view_pair(graph, A_feature, B_feature, graph_type, layer_coef, test)

    def _propagate_view_pair(self, graph, source_feature, target_feature, graph_type, layer_coef, test):
        features = torch.cat((source_feature, target_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.conf["aug_type"] == "Noise" and not test:
                features = self._add_feature_noise(features, graph_type)

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (source_feature.shape[0], target_feature.shape[0]), 0)

        return A_feature, B_feature

    def aggregate(self, agg_graph, node_feature, graph_type, test):
        return self._aggregate_neighbor_view(agg_graph, node_feature, graph_type, test)

    def _aggregate_neighbor_view(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        if self.conf["aug_type"] == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.conf["aug_type"] == "Noise" and not test:
            aggregated_feature = self._add_feature_noise(aggregated_feature, graph_type)

        return aggregated_feature

    def _add_feature_noise(self, feature, graph_type):
        random_noise = torch.rand_like(feature).to(self.device)
        eps = self.eps_dict[graph_type]
        return feature + torch.sign(feature) * F.normalize(random_noise, dim=-1) * eps

    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        users_feature = torch.stack(users_feature, dim=0)
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # Modal aggregation
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        return users_rep, bundles_rep

    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature,
                                                                  self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature,
                                                                  self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature,
                                                                self.items_feature, "UI", self.UI_layer_coefs, test)
            # Original: Bundle from Items (via BI aggregation graph)
            UI_bundles_feature_from_items = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
            # New Supplement: Bundle from Users (via BU aggregation graph)
            UI_bundles_feature_from_users = self.aggregate(self.BU_aggregation_graph_ori, UI_users_feature, "UB", test)
        else:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature,
                                                                self.items_feature, "UI", self.UI_layer_coefs, test)
            # Original: Bundle from Items (via BI aggregation graph)
            UI_bundles_feature_from_items = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
            # New Supplement: Bundle from Users (via BU aggregation graph)
            UI_bundles_feature_from_users = self.aggregate(self.BU_aggregation_graph, UI_users_feature, "UB", test)

        # UI view: Residual Combination
        UI_bundles_feature = UI_bundles_feature_from_items + self.ui_bundle_user_agg_beta * UI_bundles_feature_from_users

        #  =============================  BI graph propagation  =============================
        if test:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature,
                                                                  self.items_feature, "BI", self.BI_layer_coefs, test)
            # Original: User from Items (via UI aggregation graph)
            BI_users_feature_from_items = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
            # New Supplement: User from Bundles (via UB aggregation graph)
            BI_users_feature_from_bundles = self.aggregate(self.UB_aggregation_graph_ori, BI_bundles_feature, "UB",
                                                           test)
        else:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature,
                                                                  self.items_feature, "BI", self.BI_layer_coefs, test)
            # Original: User from Items (via UI aggregation graph)
            BI_users_feature_from_items = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)
            # New Supplement: User from Bundles (via UB aggregation graph)
            BI_users_feature_from_bundles = self.aggregate(self.UB_aggregation_graph, BI_bundles_feature, "UB", test)

        # BI view: Residual Combination
        BI_users_feature = BI_users_feature_from_items + self.bi_user_bundle_agg_beta * BI_users_feature_from_bundles

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]

        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)

        # 显式返回融合前(pre-fusion)的三视图表示，用于 anchor_cl 计算
        return users_rep, bundles_rep, users_feature, bundles_feature

    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)  # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0))  # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp)  # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)  # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def cal_pre_fusion_anchor_cl_loss(self, pre_fusion_users, pre_fusion_bundles, users, pos_bundles):
        # 取出配置
        anchor_cl_conf = self.conf.get("anchor_cl", {})
        temp = anchor_cl_conf.get("temp", 0.2)

        UB_u, UI_u, BI_u = pre_fusion_users
        UB_b, UI_b, BI_b = pre_fusion_bundles

        # 对当前 batch 内的节点表示进行严格去重 (unique)，避免同 ID 样本在 InfoNCE 中被当作负样本 (false negatives)
        u_idx = torch.unique(users.squeeze(-1))  # [unique_u_num]
        b_idx = torch.unique(pos_bundles.squeeze(-1))  # [unique_b_num] 仅使用 batch 内的 unique positive bundle 做对比

        loss = torch.tensor(0.0, device=self.device)
        # Keep the return shape unchanged for the training loop, but skip
        # per-batch detailed logging to avoid unnecessary host synchronization.
        loss_dict = {}

        # 辅助函数：计算标准 batch 内 InfoNCE (使用 F.cross_entropy 实现)
        def info_nce(anchor, positive):
            # 如果去重后只剩 1 个（或 0 个）节点，无法形成有效的 batch 内负样本，直接返回 0
            if anchor.size(0) <= 1:
                return torch.tensor(0.0, device=self.device)
            # [bs, bs] 相似度矩阵，对角线为正样本，其余为负样本
            sim_matrix = torch.matmul(anchor, positive.T) / temp
            labels = torch.arange(anchor.size(0)).to(self.device)
            return F.cross_entropy(sim_matrix, labels)

        # User 侧的 UB-UI 和 UB-BI 跨视图对比
        if anchor_cl_conf.get("user_cl", True) and u_idx.size(0) > 1:
            # 以 UB 为 anchor，提取 unique user 的三视图 pre-fusion 表示并做 L2 正则化
            batch_u_ub = F.normalize(UB_u[u_idx], p=2, dim=1)
            batch_u_ui = F.normalize(UI_u[u_idx], p=2, dim=1)
            batch_u_bi = F.normalize(BI_u[u_idx], p=2, dim=1)

            l_u_ub_ui = info_nce(batch_u_ub, batch_u_ui)
            l_u_ub_bi = info_nce(batch_u_ub, batch_u_bi)

            w_ui = anchor_cl_conf.get("ub_ui_user_weight", 1.0)
            w_bi = anchor_cl_conf.get("ub_bi_user_weight", 1.0)

            loss += w_ui * l_u_ub_ui + w_bi * l_u_ub_bi

        # Bundle 侧的 UB-UI 和 UB-BI 跨视图对比
        if anchor_cl_conf.get("bundle_cl", True) and b_idx.size(0) > 1:
            # 以 UB 为 anchor，提取 unique positive bundle 的三视图 pre-fusion 表示并做 L2 正则化
            batch_b_ub = F.normalize(UB_b[b_idx], p=2, dim=1)
            batch_b_ui = F.normalize(UI_b[b_idx], p=2, dim=1)
            batch_b_bi = F.normalize(BI_b[b_idx], p=2, dim=1)

            l_b_ub_ui = info_nce(batch_b_ub, batch_b_ui)
            l_b_ub_bi = info_nce(batch_b_ub, batch_b_bi)

            w_ui = anchor_cl_conf.get("ub_ui_bundle_weight", 1.0)
            w_bi = anchor_cl_conf.get("ub_bi_bundle_weight", 1.0)

            loss += w_ui * l_b_ub_ui + w_bi * l_b_ub_bi

        return loss, loss_dict

    def cal_loss(self, users_feature, bundles_feature, compute_c_loss=True):
        # users_feature / bundles_feature: [bs, 1+neg_num, emb_size]
        pred = torch.sum(users_feature * bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        if compute_c_loss:
            # cl is abbr. of "contrastive loss"
            u_view_cl = self.cal_c_loss(users_feature, users_feature)
            b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)

            c_losses = [u_view_cl, b_view_cl]
            c_loss = sum(c_losses) / len(c_losses)
        else:
            c_loss = torch.tensor(0.0, device=self.device)

        return bpr_loss, c_loss

    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

            self.BU_aggregation_graph = self.get_aggregation_graph(self.ub_graph.T, self.conf["UB_ratio"])
            self.UB_aggregation_graph = self.get_aggregation_graph(self.ub_graph, self.conf["UB_ratio"])

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_rep, bundles_rep, pre_fusion_users, pre_fusion_bundles = self.get_multi_modal_representations()

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]

        compute_c_loss = self.conf.get("c_lambda", 0.0) != 0
        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding, compute_c_loss=compute_c_loss)

        # 计算新增的 UB-anchored pre-fusion cross-view contrastive loss
        anchor_cl_loss = torch.tensor(0.0, device=self.device)
        anchor_cl_dict = {}
        if self.conf.get("anchor_cl", {}).get("enabled", False):
            pos_bundles = bundles[:, 0:1]  # 仅使用 positive bundles 进行对比 [bs, 1]
            anchor_cl_loss, anchor_cl_dict = self.cal_pre_fusion_anchor_cl_loss(
                pre_fusion_users, pre_fusion_bundles, users, pos_bundles
            )

        return bpr_loss, c_loss, anchor_cl_loss, anchor_cl_dict

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature, _, _ = propagate_result
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    return _as_sparse_tensor(graph)


def _as_sparse_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(graph.shape),
    )

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


def _read_singleton_ratio(conf, key):
    values = conf.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise ValueError(f"DWT expects {key} to be a single-value list, got {values!r}")
    return float(values[0])


def _dwt_sparse_tensor(sparse_matrix, device):
    sparse_matrix = sparse_matrix.tocoo()
    values = sparse_matrix.data
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(sparse_matrix.shape),
    ).to(device)


def resolve_dwt_graph_config(conf):
    fusion_weights = conf.get("fusion_weights", {})
    modal_weight = fusion_weights.get("modal_weight")
    if not isinstance(modal_weight, (list, tuple)) or len(modal_weight) != 3:
        raise ValueError(
            "DWT expects fusion_weights.modal_weight to contain exactly three values "
            "for [UB, UI, BI]"
        )

    ub_modal, ui_modal, bi_modal = [float(x) for x in modal_weight]
    if abs(ub_modal) > 1.0e-8:
        raise ValueError(
            "DWT expects fusion_weights.modal_weight[0] to be 0.0 because UB is not "
            "part of the final DWT fusion"
        )
    if abs((ui_modal + bi_modal) - 1.0) > 1.0e-8:
        raise ValueError(
            "DWT expects fusion_weights.modal_weight[1] + fusion_weights.modal_weight[2] == 1.0"
        )

    return {
        "upsilon": {
            "UB": _read_singleton_ratio(conf, "UB_ratios"),
            "UI": _read_singleton_ratio(conf, "UI_ratios"),
            "BI": _read_singleton_ratio(conf, "BI_ratios"),
        },
        "layer_coefs": {
            "UB": list(fusion_weights["UB_layer"]),
            "UI": list(fusion_weights["UI_layer"]),
            "BI": list(fusion_weights["BI_layer"]),
        },
        "omega": ui_modal,
    }

