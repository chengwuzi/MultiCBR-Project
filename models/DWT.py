#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
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
        self.noise_eps_dict = {
            "UB": conf["UB_ratio"],
            "UI": conf["UI_ratio"],
            "BI": conf["BI_ratio"],
        }
        fusion_weights = conf["fusion_weights"]

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.ub_training_graph = None

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph)
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph)
        modal_weight = fusion_weights["modal_weight"]
        if len(modal_weight) != 3:
            raise ValueError("DWT expects fusion_weights.modal_weight to have three entries for UB/UI/BI.")
        if abs(float(modal_weight[0])) > 1.0e-8:
            raise ValueError("DWT requires fusion_weights.modal_weight[0] to be 0 because UB is not fused into final scores.")

        self.modal_coefs = torch.FloatTensor(
            [modal_weight[1], modal_weight[2]]
        ).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.UB_layer_coefs = torch.FloatTensor(fusion_weights["UB_layer"]).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = torch.FloatTensor(fusion_weights["UI_layer"]).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = torch.FloatTensor(fusion_weights["BI_layer"]).unsqueeze(0).unsqueeze(-1).to(self.device)

        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)

    def set_training_ub_graph(self, ub_graph):
        self.ub_training_graph = ub_graph

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
        propagation_graph = propagation_graph.tocoo()
        values = propagation_graph.data
        indices = np.vstack((propagation_graph.row, propagation_graph.col))
        return torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(values),
            torch.Size(propagation_graph.shape),
        ).to(self.device)

    def get_aggregation_graph(self, bipartite_graph):
        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1 / np.asarray(bundle_size).ravel()) @ bipartite_graph
        bipartite_graph = bipartite_graph.tocoo()
        values = bipartite_graph.data
        indices = np.vstack((bipartite_graph.row, bipartite_graph.col))
        return torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(values),
            torch.Size(bipartite_graph.shape),
        ).to(self.device)

    def graph_propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), dim=0)
        all_features = [features]
        for _ in range(self.num_layers):
            features = torch.spmm(graph, features)
            if not test:
                random_noise = torch.rand_like(features).to(self.device)
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * self.noise_eps_dict[graph_type]
            all_features.append(F.normalize(features, p=2, dim=1))
        all_features = torch.stack(all_features, dim=1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        return torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), dim=0)

    def graph_aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)
        if not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * self.noise_eps_dict[graph_type]
        return aggregated_feature

    def get_multi_modal_representations(self, test=False):
        if test:
            return self._propagate_for_eval()
        return self._propagate_for_train()

    def _propagate_for_train(self):
        if self.ub_training_graph is None:
            raise ValueError("DWT training UB graph has not been set.")

        UB_users_feature, UB_bundles_feature = self.graph_propagate(
            self.ub_training_graph,
            self.users_feature,
            self.bundles_feature,
            "UB",
            self.UB_layer_coefs,
            False,
        )
        UI_users_feature, UI_items_feature = self.graph_propagate(
            self.UI_propagation_graph,
            self.users_feature,
            self.items_feature,
            "UI",
            self.UI_layer_coefs,
            False,
        )
        UI_bundles_feature = self.graph_aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", False)
        BI_bundles_feature, BI_items_feature = self.graph_propagate(
            self.BI_propagation_graph,
            self.bundles_feature,
            self.items_feature,
            "BI",
            self.BI_layer_coefs,
            False,
        )
        BI_users_feature = self.graph_aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", False)

        users_rep = torch.sum(
            torch.stack([UI_users_feature, BI_users_feature], dim=0) * self.modal_coefs,
            dim=0,
        )
        bundles_rep = torch.sum(
            torch.stack([UI_bundles_feature, BI_bundles_feature], dim=0) * self.modal_coefs,
            dim=0,
        )

        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]
        return users_rep, bundles_rep, users_feature, bundles_feature

    def _propagate_for_eval(self):
        UI_users_feature, UI_items_feature = self.graph_propagate(
            self.UI_propagation_graph,
            self.users_feature,
            self.items_feature,
            "UI",
            self.UI_layer_coefs,
            True,
        )
        UI_bundles_feature = self.graph_aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", True)
        BI_bundles_feature, BI_items_feature = self.graph_propagate(
            self.BI_propagation_graph,
            self.bundles_feature,
            self.items_feature,
            "BI",
            self.BI_layer_coefs,
            True,
        )
        BI_users_feature = self.graph_aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", True)

        users_rep = torch.sum(
            torch.stack([UI_users_feature, BI_users_feature], dim=0) * self.modal_coefs,
            dim=0,
        )
        bundles_rep = torch.sum(
            torch.stack([UI_bundles_feature, BI_bundles_feature], dim=0) * self.modal_coefs,
            dim=0,
        )
        return users_rep, bundles_rep, None, None

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
        ub_users_embedding = users_feature[0][user]
        ub_bundles_embedding = bundles_feature[0][bundle]
        ui_users_embedding = users_feature[1][user]
        ui_bundles_embedding = bundles_feature[1][bundle]
        bi_users_embedding = users_feature[2][user]
        bi_bundles_embedding = bundles_feature[2][bundle]

        inter_cl = torch.tensor(0.0, device=self.device)
        if self.gamma_1 != 0:
            inter_cl = (
                self.cal_cl_loss(ub_users_embedding, ui_users_embedding)
                + self.cal_cl_loss(ub_users_embedding, bi_users_embedding)
                + self.cal_cl_loss(ui_users_embedding, bi_users_embedding)
                + self.cal_cl_loss(ub_bundles_embedding, ui_bundles_embedding)
                + self.cal_cl_loss(ub_bundles_embedding, bi_bundles_embedding)
                + self.cal_cl_loss(ui_bundles_embedding, bi_bundles_embedding)
            )

        intra_cl = torch.tensor(0.0, device=self.device)
        if self.gamma_2 != 0:
            intra_cl = (
                self.cal_cl_loss(ub_users_embedding, ub_users_embedding)
                + self.cal_cl_loss(ui_users_embedding, ui_users_embedding)
                + self.cal_cl_loss(bi_users_embedding, bi_users_embedding)
                + self.cal_cl_loss(ub_bundles_embedding, ub_bundles_embedding)
                + self.cal_cl_loss(ui_bundles_embedding, ui_bundles_embedding)
                + self.cal_cl_loss(bi_bundles_embedding, bi_bundles_embedding)
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
