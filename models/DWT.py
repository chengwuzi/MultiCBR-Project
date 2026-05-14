#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


def _read_singleton_ratio(conf, key):
    values = conf.get(key)
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise ValueError(f"DWT expects {key} to be a single-value list, got {values!r}")
    return float(values[0])


def _as_sparse_tensor(sparse_matrix, device):
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
        return _as_sparse_tensor(propagation_graph, self.device)

    def get_aggregation_graph(self, bipartite_graph):
        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1 / np.asarray(bundle_size).ravel()) @ bipartite_graph
        return _as_sparse_tensor(bipartite_graph, self.device)

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
