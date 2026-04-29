#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 


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


class MultiCBR(nn.Module):
    def __init__(self, conf, raw_graph, user_beta_mask=None):
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
        
        # Register user_beta_mask as a buffer so it's part of the model state but not a parameter
        if user_beta_mask is None:
             # Fallback to all-ones if not provided
             user_beta_mask = torch.ones(self.num_users, dtype=torch.float32)
        self.register_buffer("user_beta_mask", user_beta_mask.to(device))

        self.fusion_weights = conf['fusion_weights']

        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        # New Simple Aggregation Supplement coefficients
        self.ui_bundle_user_agg_beta = self.conf.get("ui_bundle_user_agg_beta", 0.1)
        self.bi_user_bundle_agg_beta = self.conf.get("bi_user_bundle_agg_beta", 0.1)

        # generate the graph without any dropouts for testing
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)

        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)
        
        # New Aggregation Graphs for Supplement (Simple Aggregation)
        # UI View: Bundle <- Users (using UB graph transpose)
        self.BU_aggregation_graph_ori = self.get_aggregation_graph(self.ub_graph.T)
        # BI View: User <- Bundles (using UB graph)
        self.UB_aggregation_graph_ori = self.get_aggregation_graph(self.ub_graph)

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])
        
        # New Aggregation Graphs for Supplement (with dropout if needed, using UB_ratio for consistency)
        self.BU_aggregation_graph = self.get_aggregation_graph(self.ub_graph.T, self.conf["UB_ratio"])
        self.UB_aggregation_graph = self.get_aggregation_graph(self.ub_graph, self.conf["UB_ratio"])

        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()

    def set_ub_graph(self, ub_graph):
        self.ub_graph = ub_graph

        # Evaluation-time graphs should follow the same rebuilt UB structure so
        # inference uses the same message-passing graph as training.
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

        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
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
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

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
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)


    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.conf["aug_type"] == "Noise" and not test:
                random_noise = torch.rand_like(features).to(self.device)
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["aug_type"] == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.conf["aug_type"] == "Noise" and not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature


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
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            # Original: Bundle from Items (via BI aggregation graph)
            UI_bundles_feature_from_items = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
            # New Supplement: Bundle from Users (via BU aggregation graph)
            UI_bundles_feature_from_users = self.aggregate(self.BU_aggregation_graph_ori, UI_users_feature, "UB", test)
        else:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            # Original: Bundle from Items (via BI aggregation graph)
            UI_bundles_feature_from_items = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
            # New Supplement: Bundle from Users (via BU aggregation graph)
            UI_bundles_feature_from_users = self.aggregate(self.BU_aggregation_graph, UI_users_feature, "UB", test)

        # UI view: Residual Combination
        UI_bundles_feature = UI_bundles_feature_from_items + self.ui_bundle_user_agg_beta * UI_bundles_feature_from_users

        #  =============================  BI graph propagation  =============================
        if test:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            # Original: User from Items (via UI aggregation graph)
            BI_users_feature_from_items = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
            # New Supplement: User from Bundles (via UB aggregation graph)
            BI_users_feature_from_bundles = self.aggregate(self.UB_aggregation_graph_ori, BI_bundles_feature, "UB", test)
        else:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            # Original: User from Items (via UI aggregation graph)
            BI_users_feature_from_items = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)
            # New Supplement: User from Bundles (via UB aggregation graph)
            BI_users_feature_from_bundles = self.aggregate(self.UB_aggregation_graph, BI_bundles_feature, "UB", test)

        # BI view: Residual Combination
        # Apply user_beta_mask: only users in the mask get the bundle supplement
        supplement = self.bi_user_bundle_agg_beta * BI_users_feature_from_bundles
        conditional_supplement = supplement * self.user_beta_mask.unsqueeze(1)
        BI_users_feature = BI_users_feature_from_items + conditional_supplement

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
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_pre_fusion_anchor_cl_loss(self, pre_fusion_users, pre_fusion_bundles, users, pos_bundles):
        # 取出配置
        anchor_cl_conf = self.conf.get("anchor_cl", {})
        temp = anchor_cl_conf.get("temp", 0.2)
        
        UB_u, UI_u, BI_u = pre_fusion_users
        UB_b, UI_b, BI_b = pre_fusion_bundles
        
        # 对当前 batch 内的节点表示进行严格去重 (unique)，避免同 ID 样本在 InfoNCE 中被当作负样本 (false negatives)
        u_idx = torch.unique(users.squeeze(-1)) # [unique_u_num]
        b_idx = torch.unique(pos_bundles.squeeze(-1)) # [unique_b_num] 仅使用 batch 内的 unique positive bundle 做对比
        
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
            pos_bundles = bundles[:, 0:1] # 仅使用 positive bundles 进行对比 [bs, 1]
            anchor_cl_loss, anchor_cl_dict = self.cal_pre_fusion_anchor_cl_loss(
                pre_fusion_users, pre_fusion_bundles, users, pos_bundles
            )

        return bpr_loss, c_loss, anchor_cl_loss, anchor_cl_dict


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature, _, _ = propagate_result
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores
