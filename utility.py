#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 
from collections import defaultdict

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


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1, alignment_modulation=None):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample
        self.alignment_modulation = alignment_modulation

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

        if self.alignment_modulation is None:
            return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

        alignment_modulation = torch.tensor(self.alignment_modulation[index], dtype=torch.float32)
        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles), alignment_modulation


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
        u_b_graph_train_for_ub_view = self.get_ub_graph_for_ub_view(conf, u_b_pairs_train, u_b_graph_train)

        # Load connectivity metrics and generate user_beta_mask
        self.user_beta_mask = self.get_user_beta_mask(conf)
        self.train_alignment_modulation = self.get_alignment_modulation(conf, u_b_pairs_train)

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(
            conf,
            u_b_pairs_train,
            u_b_graph_train,
            self.num_bundles,
            u_b_for_neg_sample,
            b_b_for_neg_sample,
            conf["neg_num"],
            alignment_modulation=self.train_alignment_modulation,
        )
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        # High-order substitution pre-computation (Legacy - Disabled)
        # if conf.get('enable_high_order_replace', False):
        #    ... (Logic removed)
        
        # Keep the full train graph for BPR sampling and evaluation masking,
        # while allowing the UB view graph itself to use a filtered version.
        self.graphs = [u_b_graph_train_for_ub_view, u_i_graph, b_i_graph]

        # Allow explicit override while keeping the original platform-specific defaults.
        num_workers = conf.get("num_workers")
        if num_workers is None:
            num_workers = 4 if os.name == 'nt' else 10
        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)


    def resolve_project_path(self, path_value, config_name):
        if not path_value:
            raise ValueError(f"{config_name} is enabled, but weight_path is not configured.")

        if os.path.isabs(path_value):
            resolved_path = path_value
        else:
            project_root = os.path.dirname(os.path.abspath(__file__))
            resolved_path = os.path.join(project_root, path_value)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Weight file not found for {config_name}: {resolved_path}")

        return resolved_path


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


    def get_ub_graph_for_ub_view(self, conf, u_b_pairs_train, u_b_graph_train):
        graph_filter_conf = conf.get("ub_graph_topk_filter", {})
        if not graph_filter_conf.get("enabled", False):
            return u_b_graph_train

        return self.build_topk_filtered_ub_graph(u_b_pairs_train, graph_filter_conf)


    def build_topk_filtered_ub_graph(self, u_b_pairs_train, graph_filter_conf):
        topk = graph_filter_conf.get("topk")
        if topk is None:
            raise ValueError("ub_graph_topk_filter.enabled is true, but topk is not configured.")
        if topk < 0:
            raise ValueError("ub_graph_topk_filter.topk must be >= 0.")

        resolved_weight_path = self.resolve_project_path(
            graph_filter_conf.get("weight_path"),
            "ub_graph_topk_filter",
        )

        print(f"Applying UB graph top-k filter for {self.name}: topk={topk}, weight_path={resolved_weight_path}")

        user_entries = defaultdict(list)
        expected_lines = len(u_b_pairs_train)

        with open(resolved_weight_path, "r", encoding="utf-8") as f:
            for idx, pair in enumerate(u_b_pairs_train):
                line = f.readline()
                if not line:
                    raise ValueError(
                        f"Weight file ended early at line {idx + 1}; expected {expected_lines} lines."
                    )

                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid weight file format at line {idx + 1}: {line.strip()}")

                user_id = int(parts[0])
                bundle_id = int(parts[1])
                weight = float(parts[2])

                if pair[0] != user_id or pair[1] != bundle_id:
                    raise ValueError(
                        "Weight file is not aligned with user_bundle_train.txt at line "
                        f"{idx + 1}: train_pair={pair}, weight_pair=({user_id}, {bundle_id})"
                    )

                user_entries[user_id].append((weight, idx))

            extra_line = f.readline()
            if extra_line:
                raise ValueError("Weight file has more lines than user_bundle_train.txt.")

        keep_mask = np.zeros(expected_lines, dtype=np.bool_)
        kept_edges = 0

        for entries in user_entries.values():
            if topk == 0:
                continue
            # Smaller weights are treated as more important; keep the lowest-weight
            # edges for each user and preserve original train-file order for ties.
            entries.sort(key=lambda x: (x[0], x[1]))
            for _, idx in entries[:topk]:
                keep_mask[idx] = True
                kept_edges += 1

        kept_pairs = [u_b_pairs_train[idx] for idx, keep in enumerate(keep_mask) if keep]
        if kept_pairs:
            indice = np.array(kept_pairs, dtype=np.int32)
            values = np.ones(len(kept_pairs), dtype=np.float32)
            filtered_graph = sp.coo_matrix(
                (values, (indice[:, 0], indice[:, 1])),
                shape=(self.num_users, self.num_bundles),
            ).tocsr()
        else:
            filtered_graph = sp.csr_matrix((self.num_users, self.num_bundles), dtype=np.float32)

        pruned_edges = expected_lines - kept_edges
        print(
            f"UB top-k filter kept {kept_edges}/{expected_lines} train edges "
            f"({(kept_edges / expected_lines) * 100:.2f}%), pruned {pruned_edges}."
        )
        print_statistics(filtered_graph, f"U-B statistics in train (UB-view topk={topk})")

        return filtered_graph


    def get_alignment_modulation(self, conf, u_b_pairs_train):
        ali_uni_conf = conf.get("alignment_uniformity", {})
        weighted_alignment_conf = ali_uni_conf.get("weighted_alignment", {})
        if not ali_uni_conf.get("enabled", False) or not weighted_alignment_conf.get("enabled", False):
            return None

        gamma = float(weighted_alignment_conf.get("gamma", 0.0))
        resolved_weight_path = self.resolve_project_path(
            weighted_alignment_conf.get("weight_path"),
            "alignment_uniformity.weighted_alignment",
        )

        expected_lines = len(u_b_pairs_train)
        raw_weights = np.empty(expected_lines, dtype=np.float32)
        user_ids = np.empty(expected_lines, dtype=np.int64)
        user_weight_sums = np.zeros(self.num_users, dtype=np.float64)
        user_weight_counts = np.zeros(self.num_users, dtype=np.int64)

        print(
            f"Loading alignment weight matrix for {self.name}: "
            f"weight_path={resolved_weight_path}, gamma={gamma}"
        )

        with open(resolved_weight_path, "r", encoding="utf-8") as f:
            for idx, pair in enumerate(u_b_pairs_train):
                line = f.readline()
                if not line:
                    raise ValueError(
                        f"Alignment weight file ended early at line {idx + 1}; expected {expected_lines} lines."
                    )

                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid alignment weight file format at line {idx + 1}: {line.strip()}")

                user_id = int(parts[0])
                bundle_id = int(parts[1])
                raw_weight = float(parts[2])

                if pair[0] != user_id or pair[1] != bundle_id:
                    raise ValueError(
                        "Alignment weight file is not aligned with user_bundle_train.txt at line "
                        f"{idx + 1}: train_pair={pair}, weight_pair=({user_id}, {bundle_id})"
                    )

                raw_weights[idx] = raw_weight
                user_ids[idx] = user_id
                user_weight_sums[user_id] += raw_weight
                user_weight_counts[user_id] += 1

            extra_line = f.readline()
            if extra_line:
                raise ValueError("Alignment weight file has more lines than user_bundle_train.txt.")

        user_weight_means = np.zeros(self.num_users, dtype=np.float32)
        nonzero_mask = user_weight_counts > 0
        user_weight_means[nonzero_mask] = (
            user_weight_sums[nonzero_mask] / user_weight_counts[nonzero_mask]
        ).astype(np.float32)

        transformed_weights = 1.0 + gamma * (raw_weights - user_weight_means[user_ids])
        transformed_weights = transformed_weights.astype(np.float32)

        print(
            "Alignment weight matrix transformed with "
            "g(w)=1+gamma*(w-user_mean), "
            f"min={transformed_weights.min():.6f}, "
            f"max={transformed_weights.max():.6f}, "
            f"mean={transformed_weights.mean():.6f}"
        )

        return transformed_weights


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
