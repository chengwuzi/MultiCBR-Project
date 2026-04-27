#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader


import csv

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


def load_external_embedding_tensor(path, num_expected, entity_name):
    if not path:
        raise ValueError(f"{entity_name} embedding path is required")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        embedding = payload.detach().float()
    elif isinstance(payload, np.ndarray):
        embedding = torch.from_numpy(payload).float()
    elif isinstance(payload, dict):
        rows = []
        for idx in range(num_expected):
            if idx in payload:
                value = payload[idx]
            elif str(idx) in payload:
                value = payload[str(idx)]
            else:
                raise KeyError(
                    f"{entity_name} embedding file {path} is missing id {idx}; "
                    "expected a full mapping aligned with current dataset ids"
                )
            value = torch.as_tensor(value).detach().float().view(-1)
            rows.append(value)
        embedding = torch.stack(rows, dim=0)
    else:
        raise TypeError(
            f"Unsupported payload type {type(payload)} in {path}; "
            "expected a torch tensor, numpy array, or id->embedding dict"
        )

    if embedding.ndim != 2:
        raise ValueError(f"{entity_name} embedding tensor from {path} must be 2D")
    if embedding.shape[0] != num_expected:
        raise ValueError(
            f"{entity_name} embedding tensor from {path} has {embedding.shape[0]} rows, "
            f"expected {num_expected}"
        )
    return embedding.contiguous()


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
        self.conf = conf
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

        # Load connectivity metrics and generate user_beta_mask
        self.user_beta_mask = self.get_user_beta_mask(conf)

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        # High-order substitution pre-computation (Legacy - Disabled)
        # if conf.get('enable_high_order_replace', False):
        #    ... (Logic removed)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]
        self.user_observed_bundles = self.build_user_observed_bundles(u_b_graph_train)

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


    def get_user_beta_mask(self, conf):
        # Default range is [0.0, 1.0], which means all users are included
        ratio_range = conf.get("train_connected_ratio_range", [0.0, 1.0])
        min_ratio, max_ratio = ratio_range[0], ratio_range[1]
        
        # Min user_ub_train_deg filter
        min_deg = conf.get("user_ub_train_deg_min", 0)
        
        print(f"Generating user_beta_mask with train_connected_ratio in [{min_ratio}, {max_ratio}] AND user_ub_train_deg >= {min_deg}")

        mask = torch.zeros(self.num_users, dtype=torch.float32)
        
        # Path to the metrics CSV file
        # Assuming the structure is connectivity_analysis_outputs/<dataset>/user_connectivity_metrics.csv
        # We need to handle the path carefully as 'self.path' is './datasets'
        # But the output is in './connectivity_analysis_outputs'
        
        # Construct path relative to project root
        project_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(project_root, 'connectivity_analysis_outputs', self.name, 'user_connectivity_metrics.csv')
        
        if not os.path.exists(csv_path):
            print(f"Warning: Connectivity metrics file not found at {csv_path}. Using all-ones mask.")
            return torch.ones(self.num_users, dtype=torch.float32)
            
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    user_id = int(row['user_id'])
                    if user_id >= self.num_users:
                        continue
                        
                    ratio = float(row['train_connected_ratio'])
                    deg = int(row['user_ub_train_deg'])
                    
                    # Check if ratio is within range (inclusive) AND degree is above threshold
                    if (min_ratio <= ratio <= max_ratio) and (deg >= min_deg):
                        mask[user_id] = 1.0
                        count += 1
                        
            print(f"Mask generated: {count}/{self.num_users} users selected ({(count/self.num_users)*100:.2f}%)")
            
        except Exception as e:
            print(f"Error reading connectivity metrics: {e}. Using all-ones mask.")
            return torch.ones(self.num_users, dtype=torch.float32)
            
        return mask


    def build_user_observed_bundles(self, u_b_graph):
        user_observed_bundles = []
        indptr = u_b_graph.indptr
        indices = u_b_graph.indices
        for user_id in range(self.num_users):
            start = indptr[user_id]
            end = indptr[user_id + 1]
            user_observed_bundles.append(indices[start:end].astype(np.int64).tolist())
        return user_observed_bundles
