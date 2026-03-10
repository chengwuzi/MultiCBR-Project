#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import pickle
from collections import defaultdict
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
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, neg_sample=1, hard_candidates=None):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        # Hard Negative Configuration
        self.hard_neg_enable = conf.get("hard_neg_enable", False)
        self.hard_neg_strategy = conf.get("hard_neg_strategy", "random")
        self.hard_sample_ratio = conf.get("hard_sample_ratio", 0.5)
        self.hard_candidates = hard_candidates
        
        # Debug counters (optional)
        self.hard_attempt_count = 0
        self.hard_success_count = 0
        self.random_fallback_count = 0


    def sample_random_negative(self, user_b, banned_bundles):
        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and i not in banned_bundles:
                return i

    def sample_hard_negative(self, user_b, pos_bundle, banned_bundles):
        if not self.hard_candidates or pos_bundle not in self.hard_candidates:
            return None
            
        candidates = self.hard_candidates[pos_bundle]
        # Filter candidates:
        # 1. Not interacted by user
        # 2. Not in banned_bundles (which includes pos_bundle and other negs)
        
        # Optimization: We can just sample and check validness instead of filtering all if the list is long
        # But for correctness and simplicity, let's try to filter.
        # Since candidates are pre-filtered for overlap, we just check user interaction.
        
        valid_candidates = []
        for cand in candidates:
            if cand not in banned_bundles and self.u_b_graph[user_b, cand] == 0:
                valid_candidates.append(cand)
                
        if not valid_candidates:
            return None
            
        return random.choice(valid_candidates)

    def sample_one_negative(self, user_b, pos_bundle, banned_bundles):
        if not self.hard_neg_enable or self.hard_neg_strategy == "random":
            return self.sample_random_negative(user_b, banned_bundles)
        
        use_hard = False
        if self.hard_neg_strategy == "hard":
            use_hard = True
        elif self.hard_neg_strategy == "mix":
            if random.random() < self.hard_sample_ratio:
                use_hard = True
                
        if use_hard:
            self.hard_attempt_count += 1
            hard_neg = self.sample_hard_negative(user_b, pos_bundle, banned_bundles)
            if hard_neg is not None:
                self.hard_success_count += 1
                return hard_neg
            else:
                self.random_fallback_count += 1
                # Fallback to random
                return self.sample_random_negative(user_b, banned_bundles)
        else:
            return self.sample_random_negative(user_b, banned_bundles)


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]
        
        # Use a set for faster lookup of banned bundles (pos + already sampled negs)
        banned_bundles = set(all_bundles)

        for _ in range(self.neg_sample):
            neg_bundle = self.sample_one_negative(user_b, pos_bundle, banned_bundles)
            all_bundles.append(neg_bundle)
            banned_bundles.add(neg_bundle)

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

        # Load Hard Candidate Pool
        # Fix: use b_i_graph (which is the 2nd return from get_bi()) instead of self.graphs[2]
        self.hard_candidates = load_or_build_hard_candidate_pool(conf, b_i_graph, self.num_bundles)

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, conf["neg_num"], self.hard_candidates)
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)


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


def build_item2bundles(b_i_graph, num_bundles):
    print("Building item2bundles inverted index...")
    item2bundles = defaultdict(set)
    # b_i_graph is CSR matrix
    for b_id in range(num_bundles):
        start = b_i_graph.indptr[b_id]
        end = b_i_graph.indptr[b_id+1]
        items = b_i_graph.indices[start:end]
        for item in items:
            item2bundles[item].add(b_id)
    return item2bundles


def load_or_build_hard_candidate_pool(conf, b_i_graph, num_bundles):
    # Only proceed if hard negatives are enabled
    if not conf.get("hard_neg_enable", False):
        return None

    dataset_name = conf["dataset"]
    cache_dir = conf.get("hard_pool_cache_dir", "./hard_negative_analysis")
    use_precomputed = conf.get("hard_pool_use_precomputed", True)
    force_rebuild = conf.get("hard_pool_force_rebuild", False)
    
    # Paths to check
    # 1. Inside dataset subfolder
    path1 = os.path.join(cache_dir, dataset_name, f"bundle_hard_candidates_{dataset_name}.pkl")
    # 2. Directly in cache dir
    path2 = os.path.join(cache_dir, f"bundle_hard_candidates_{dataset_name}.pkl")
    
    candidates = None
    
    # Try loading if allowed and not forced to rebuild
    if use_precomputed and not force_rebuild:
        if os.path.exists(path1):
            print(f"Loading hard candidate pool from {path1}")
            try:
                with open(path1, 'rb') as f:
                    candidates = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {path1}: {e}")
        elif os.path.exists(path2):
            print(f"Loading hard candidate pool from {path2}")
            try:
                with open(path2, 'rb') as f:
                    candidates = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {path2}: {e}")
    
    if candidates is not None:
        if conf.get("hard_debug_print", False):
             print(f"Hard pool loaded. Size: {len(candidates)}")
             # Enhanced debug stats
             non_empty = sum(1 for c in candidates.values() if c)
             pool_sizes = [len(c) for c in candidates.values()]
             print(f"[Hard Pool Stats]")
             print(f"  Non-empty bundles: {non_empty} / {len(candidates)} ({non_empty/len(candidates):.2%})")
             if pool_sizes:
                 print(f"  Pool Size - Mean: {np.mean(pool_sizes):.2f}, Median: {np.median(pool_sizes):.2f}")
             print(f"  Strategy: {conf.get('hard_neg_strategy')} (Ratio: {conf.get('hard_sample_ratio')})")
             print(f"  Window: {conf.get('hard_window_low')} - {conf.get('hard_window_high')}")
             print(f"  TopK: {conf.get('hard_topk')}")
        return candidates
        
    # If not loaded, build it
    print("Building hard candidate pool from scratch (this may take a while)...")
    
    # Parameters
    window_low = conf.get("hard_window_low", 0.05)
    window_high = conf.get("hard_window_high", 0.20)
    min_intersection = conf.get("hard_min_intersection", 1)
    topk = conf.get("hard_topk", 200)
    
    # 1. Build item2bundles
    item2bundles = build_item2bundles(b_i_graph, num_bundles)
    
    # 2. Build bundle2items (already implicitly in b_i_graph, but dict is faster for set ops)
    bundle2items = {}
    for b_id in range(num_bundles):
        start = b_i_graph.indptr[b_id]
        end = b_i_graph.indptr[b_id+1]
        bundle2items[b_id] = set(b_i_graph.indices[start:end])
        
    candidates = {}
    
    # 3. Compute overlaps
    # We iterate all bundles, find neighbors via item2bundles to avoid O(B^2)
    from tqdm import tqdm
    for b_id in tqdm(range(num_bundles), desc="Building Hard Pool"):
        items = bundle2items[b_id]
        if not items:
            continue
            
        # Find raw candidates
        raw_counts = defaultdict(int)
        for item in items:
            for other_b in item2bundles[item]:
                if other_b != b_id:
                    raw_counts[other_b] += 1
        
        # Filter and calculate Jaccard
        valid_neighbors = []
        len_b = len(items)
        for other_b, intersection in raw_counts.items():
            if intersection < min_intersection:
                continue
                
            len_other = len(bundle2items[other_b])
            union = len_b + len_other - intersection
            jaccard = intersection / union
            
            if window_low <= jaccard <= window_high:
                valid_neighbors.append((other_b, jaccard))
        
        # Sort and TopK
        valid_neighbors.sort(key=lambda x: x[1], reverse=True)
        candidates[b_id] = [x[0] for x in valid_neighbors[:topk]]
        
    # Save cache if possible
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        
    # Default to dataset subfolder structure to match analysis script
    dataset_subdir = os.path.join(cache_dir, dataset_name)
    if not os.path.exists(dataset_subdir):
        os.makedirs(dataset_subdir, exist_ok=True)
    
    save_path = os.path.join(dataset_subdir, f"bundle_hard_candidates_{dataset_name}.pkl")
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(candidates, f)
        print(f"Saved hard candidate pool to {save_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")
        
    return candidates
