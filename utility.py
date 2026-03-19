#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.u_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_pairs, b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [
            u_b_graph_train, 
            u_i_graph, 
            b_i_graph
        ]

        if conf.get('use_bi_weighted_graph', False):
            self.bi_weighted_raw_graph = self.build_bi_weighted_raw_graph(conf, b_i_graph, u_i_graph, u_b_graph_train)
            self.graphs.append(self.bi_weighted_raw_graph)
        else:
            self.bi_weighted_raw_graph = None
            self.graphs.append(None)

        if conf.get('use_core_item_boost', False):
            self.core_item_boost_matrix = self.build_core_item_boost_matrix(conf, b_i_graph, u_i_graph, u_b_graph_train)
            self.graphs.append(self.core_item_boost_matrix)
        else:
            self.core_item_boost_matrix = None
            self.graphs.append(None)

        # Windows compatibility: reduce num_workers to avoid overhead/errors
        num_workers = 4 if os.name == 'nt' else 10
        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)


    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_pairs, b_i_graph


    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph


    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph


    def build_bi_weighted_raw_graph(self, conf, b_i_graph, u_i_graph, u_b_graph_train):
        import time
        print("Building BI weighted raw graph...")
        start_time = time.time()
        
        num_users = self.num_users
        num_bundles = self.num_bundles
        num_items = self.num_items
        
        user_threshold = conf.get('core_item_user_threshold', 10)
        valid_item_threshold = conf.get('core_item_valid_item_threshold', 3)
        topk = conf.get('core_item_topk', 2)
        boost = conf.get('core_item_boost', 2.0)
        use_item_idf_reweight = conf.get('use_item_idf_reweight', True)

        bundle_user_graph = u_b_graph_train.T.tocsr() 
        item_df = np.array(u_i_graph.sum(axis=0)).flatten()
        
        row_indices = []
        col_indices = []
        values = []
        
        b_i_csr = b_i_graph.tocsr()
        u_i_csr = u_i_graph.tocsr()
        
        boosted_bundle_cnt = 0
        total_boosted_items = 0

        for b in range(num_bundles):
            b_start = b_i_csr.indptr[b]
            b_end = b_i_csr.indptr[b+1]
            b_items = b_i_csr.indices[b_start:b_end]
            
            if len(b_items) == 0:
                continue
            
            b_weights = np.ones(len(b_items), dtype=np.float32)
            
            u_start = bundle_user_graph.indptr[b]
            u_end = bundle_user_graph.indptr[b+1]
            users_b = bundle_user_graph.indices[u_start:u_end]
            
            if len(users_b) >= user_threshold:
                users_b_items_sum = np.array(u_i_csr[users_b].sum(axis=0)).flatten()
                cnt_b_i = users_b_items_sum[b_items]
                
                valid_local_indices = np.where(cnt_b_i > 0)[0]
                if len(valid_local_indices) >= valid_item_threshold:
                    coverage = cnt_b_i[valid_local_indices] / len(users_b)
                    
                    if use_item_idf_reweight:
                        valid_items = b_items[valid_local_indices]
                        idf = np.log((num_users + 1.0) / (item_df[valid_items] + 1.0))
                        valid_scores = coverage * idf
                    else:
                        valid_scores = coverage
                    
                    actual_topk = min(topk, len(valid_local_indices))
                    
                    if actual_topk > 0:
                        top_in_valid = np.argsort(-valid_scores)[:actual_topk]
                        top_local_indices = valid_local_indices[top_in_valid]
                        
                        b_weights[top_local_indices] = float(boost)
                        boosted_bundle_cnt += 1
                        total_boosted_items += actual_topk
            
            row_indices.extend([b] * len(b_items))
            col_indices.extend(b_items)
            values.extend(b_weights)

        weighted_raw_graph = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_bundles, num_items), dtype=np.float32)
        
        print(f"BI Weighted Raw Graph built in {time.time() - start_time:.2f}s")
        print(f"Boosted bundles: {boosted_bundle_cnt} / {num_bundles} ({(boosted_bundle_cnt/num_bundles)*100:.2f}%)")
        if boosted_bundle_cnt > 0:
            print(f"Average boosted items per boosted bundle: {total_boosted_items / boosted_bundle_cnt:.2f}")
        print(f"Graph nnz: {weighted_raw_graph.nnz}")
        
        return weighted_raw_graph


    def build_core_item_boost_matrix(self, conf, b_i_graph, u_i_graph, u_b_graph_train):
        import time
        print("Building core item boost matrix...")
        start_time = time.time()
        
        num_users = self.num_users
        num_bundles = self.num_bundles
        num_items = self.num_items
        
        user_threshold = conf.get('core_item_user_threshold', 10)
        valid_item_threshold = conf.get('core_item_valid_item_threshold', 3)
        topk = conf.get('core_item_topk', 2)
        boost = conf.get('core_item_boost', 2.0)
        use_item_idf_reweight = conf.get('use_item_idf_reweight', True)

        bundle_user_graph = u_b_graph_train.T.tocsr() 
        item_df = np.array(u_i_graph.sum(axis=0)).flatten()
        
        row_indices = []
        col_indices = []
        values = []
        
        b_i_csr = b_i_graph.tocsr()
        u_i_csr = u_i_graph.tocsr()
        
        boosted_bundle_cnt = 0
        total_boosted_items = 0

        for b in range(num_bundles):
            b_start = b_i_csr.indptr[b]
            b_end = b_i_csr.indptr[b+1]
            b_items = b_i_csr.indices[b_start:b_end]
            
            if len(b_items) == 0:
                continue
            
            b_weights = np.ones(len(b_items), dtype=np.float32)
            
            u_start = bundle_user_graph.indptr[b]
            u_end = bundle_user_graph.indptr[b+1]
            users_b = bundle_user_graph.indices[u_start:u_end]
            
            if len(users_b) >= user_threshold:
                users_b_items_sum = np.array(u_i_csr[users_b].sum(axis=0)).flatten()
                cnt_b_i = users_b_items_sum[b_items]
                
                valid_local_indices = np.where(cnt_b_i > 0)[0]
                if len(valid_local_indices) >= valid_item_threshold:
                    coverage = cnt_b_i[valid_local_indices] / len(users_b)
                    
                    if use_item_idf_reweight:
                        valid_items = b_items[valid_local_indices]
                        idf = np.log((num_users + 1.0) / (item_df[valid_items] + 1.0))
                        valid_scores = coverage * idf
                    else:
                        valid_scores = coverage
                    
                    actual_topk = min(topk, len(valid_local_indices))
                    
                    if actual_topk > 0:
                        # 在有效得分中选取 topk
                        top_in_valid = np.argsort(-valid_scores)[:actual_topk]
                        # 映射回 b_items 的局部索引
                        top_local_indices = valid_local_indices[top_in_valid]
                        
                        b_weights[top_local_indices] = float(boost)
                        boosted_bundle_cnt += 1
                        total_boosted_items += actual_topk
            
            row_sum = np.sum(b_weights)
            if row_sum > 0:
                b_weights = b_weights / row_sum
            
            row_indices.extend([b] * len(b_items))
            col_indices.extend(b_items)
            values.extend(b_weights)

        boost_matrix = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_bundles, num_items), dtype=np.float32)
        
        print(f"Core Item Boost Matrix built in {time.time() - start_time:.2f}s")
        print(f"Boosted bundles: {boosted_bundle_cnt} / {num_bundles}")
        if boosted_bundle_cnt > 0:
            print(f"Average boosted items per boosted bundle: {total_boosted_items / boosted_bundle_cnt:.2f}")
        
        return boost_matrix
