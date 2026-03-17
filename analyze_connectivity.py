import os
import argparse
import numpy as np
import json
import csv
from collections import defaultdict
from tqdm import tqdm
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze User Connectivity Metrics")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Data root directory")
    parser.add_argument("--dataset", type=str, default="NetEase", help="Dataset name")
    return parser.parse_args()

def load_txt(path):
    data = []
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return data
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            data.append((int(parts[0]), int(parts[1])))
    return data

def load_data(data_root, dataset_name):
    base_path = os.path.join(data_root, dataset_name)
    
    # Handle dataset name variations (e.g. NetEase_Small -> NetEase)
    # But based on user input, we assume standard naming or correct folder structure
    
    print(f"Loading data from {base_path}...")
    
    # Load User-Item
    ui_path = os.path.join(base_path, 'user_item.txt')
    ui_data = load_txt(ui_path)
    user_items = defaultdict(set)
    for u, i in ui_data:
        user_items[u].add(i)
    print(f"Loaded {len(ui_data)} user-item interactions.")

    # Load Bundle-Item
    bi_path = os.path.join(base_path, 'bundle_item.txt')
    bi_data = load_txt(bi_path)
    bundle_items = defaultdict(set)
    for b, i in bi_data:
        bundle_items[b].add(i)
    print(f"Loaded {len(bi_data)} bundle-item interactions.")

    # Load User-Bundle Train
    ub_train_path = os.path.join(base_path, 'user_bundle_train.txt')
    ub_train_data = load_txt(ub_train_path)
    user_train_bundles = defaultdict(list)
    for u, b in ub_train_data:
        user_train_bundles[u].append(b)
    print(f"Loaded {len(ub_train_data)} user-bundle train interactions.")

    # Load User-Bundle Test
    ub_test_path = os.path.join(base_path, 'user_bundle_test.txt')
    ub_test_data = load_txt(ub_test_path)
    user_test_bundles = defaultdict(list)
    for u, b in ub_test_data:
        user_test_bundles[u].append(b)
    print(f"Loaded {len(ub_test_data)} user-bundle test interactions.")
    
    # Load Data Size (to get all user ids)
    # File name might be <dataset>_data_size.txt
    # Try constructing name
    size_file_name = f"{dataset_name}_data_size.txt"
    if "_" in dataset_name:
         # Try split if needed, but utility.py logic suggests:
         # if "_" in name: name = name.split("_")[0]
         # But file system check showed 'NetEase_data_size.txt' in 'NetEase/'
         # Let's try the direct name first.
         pass
         
    # Check for file existence
    size_path = os.path.join(base_path, size_file_name)
    if not os.path.exists(size_path):
         # Try split approach
         prefix = dataset_name.split('_')[0]
         size_path = os.path.join(base_path, f"{prefix}_data_size.txt")
    
    num_users = 0
    if os.path.exists(size_path):
        with open(size_path, 'r') as f:
            line = f.readline()
            parts = line.strip().split('\t')
            num_users = int(parts[0])
        print(f"Total users defined in size file: {num_users}")
    else:
        print("Warning: data_size file not found. Inferring num_users from max ID.")
        all_users = set(user_items.keys()) | set(user_train_bundles.keys()) | set(user_test_bundles.keys())
        num_users = max(all_users) + 1 if all_users else 0
        print(f"Inferred num_users: {num_users}")

    return num_users, user_items, bundle_items, user_train_bundles, user_test_bundles

def compute_user_metrics(user_id, u_items, u_train_bundles, u_test_bundles, bundle_items_map):
    metrics = {}
    metrics['user_id'] = user_id
    
    # Basic Degrees
    metrics['user_ub_train_deg'] = len(u_train_bundles)
    metrics['user_ui_deg'] = len(u_items)
    metrics['user_test_pos_count'] = len(u_test_bundles)
    
    # --- Train Bundle Connectivity ---
    m = len(u_train_bundles)
    n_connected = 0
    total_overlap_count = 0
    total_overlap_ratio = 0.0
    
    if m > 0:
        for b in u_train_bundles:
            b_items = bundle_items_map.get(b, set())
            overlap = len(u_items.intersection(b_items))
            
            if overlap > 0:
                n_connected += 1
            
            total_overlap_count += overlap
            b_size = len(b_items)
            if b_size > 0:
                total_overlap_ratio += overlap / b_size
            else:
                # If bundle has no items, overlap ratio is technically undefined or 0. 
                # Assuming 0 if bundle is empty.
                total_overlap_ratio += 0.0
                
        metrics['train_bundle_count'] = m
        metrics['train_connected_bundle_count'] = n_connected
        metrics['train_disconnected_bundle_count'] = m - n_connected
        metrics['train_connected_ratio'] = n_connected / m
        metrics['train_disconnected_ratio'] = 1.0 - metrics['train_connected_ratio']
        metrics['avg_overlap_count_per_train_bundle'] = total_overlap_count / m
        metrics['avg_overlap_ratio_per_train_bundle'] = total_overlap_ratio / m
    else:
        metrics['train_bundle_count'] = 0
        metrics['train_connected_bundle_count'] = 0
        metrics['train_disconnected_bundle_count'] = 0
        metrics['train_connected_ratio'] = 0.0
        metrics['train_disconnected_ratio'] = 0.0 # Or 1.0? If no bundles, effectively disconnected? 
                                                  # But strictly, ratio is undefined. Let's keep 0 for consistency with "count".
                                                  # Actually, if m=0, connected_ratio=0 is safe. 
        metrics['avg_overlap_count_per_train_bundle'] = 0.0
        metrics['avg_overlap_ratio_per_train_bundle'] = 0.0

    # --- Test Positive Sample Recovery ---
    test_count = len(u_test_bundles)
    total_recover_count = 0
    total_recover_ratio = 0.0
    recover_zero_count = 0
    
    if test_count > 0:
        for b in u_test_bundles:
            b_items = bundle_items_map.get(b, set())
            recover_cnt = len(u_items.intersection(b_items))
            
            if recover_cnt == 0:
                recover_zero_count += 1
                
            total_recover_count += recover_cnt
            b_size = len(b_items)
            if b_size > 0:
                total_recover_ratio += recover_cnt / b_size
            else:
                total_recover_ratio += 0.0
        
        metrics['user_test_avg_recover_count'] = total_recover_count / test_count
        metrics['user_test_avg_recover_ratio'] = total_recover_ratio / test_count
        metrics['user_test_recover_zero_count'] = recover_zero_count
        metrics['user_test_recover_zero_ratio'] = recover_zero_count / test_count
    else:
        metrics['user_test_avg_recover_count'] = 0.0
        metrics['user_test_avg_recover_ratio'] = 0.0
        metrics['user_test_recover_zero_count'] = 0
        metrics['user_test_recover_zero_ratio'] = 0.0 # No test samples

    # --- Optional: UB/UI Balance ---
    # log(1 + user_ub_train_deg) / log(1 + user_ui_deg)
    num = math.log(1 + metrics['user_ub_train_deg'])
    den = math.log(1 + metrics['user_ui_deg'])
    if den > 0:
        metrics['ub_ui_balance'] = num / den
    else:
        metrics['ub_ui_balance'] = 0.0 # If UI deg is 0, balance is undefined/0
        
    return metrics

def calculate_statistics(metrics_list, columns):
    stats = {}
    # Convert list of dicts to dict of lists/arrays for easier processing
    data_arrays = defaultdict(list)
    for m in metrics_list:
        for col in columns:
            if col in m:
                data_arrays[col].append(m[col])
    
    for col in columns:
        if col not in data_arrays or not data_arrays[col]:
            continue
            
        arr = np.array(data_arrays[col], dtype=np.float64)
        
        # Handle empty or all-nan cases if any (though we initialized with 0.0)
        if len(arr) == 0:
            continue
            
        stats[col] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)) if len(arr) > 1 else 0.0,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p10': float(np.percentile(arr, 10)),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.median(arr)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90))
        }
    return stats

def analyze_buckets(metrics_list):
    # Convert to easier structure
    # We need to filter based on 'train_connected_ratio'
    
    # Define buckets
    buckets_def = [
        ('ratio_0', lambda x: x == 0),
        ('ratio_0_0.25', lambda x: 0 < x <= 0.25),
        ('ratio_0.25_0.5', lambda x: 0.25 < x <= 0.5),
        ('ratio_0.5_0.75', lambda x: 0.5 < x <= 0.75),
        ('ratio_gt_0.75', lambda x: x > 0.75)
    ]
    
    bucket_stats = {}
    total_users = len(metrics_list)
    total_test_pos = sum(m.get('user_test_pos_count', 0) for m in metrics_list)
    
    # Pre-group users into buckets
    grouped_metrics = defaultdict(list)
    for m in metrics_list:
        ratio = m.get('train_connected_ratio', 0.0)
        for name, func in buckets_def:
            if func(ratio):
                grouped_metrics[name].append(m)
                break 
    
    # Calculate stats for each bucket
    for name, _ in buckets_def:
        group = grouped_metrics[name]
        count = len(group)
        pct = count / total_users if total_users > 0 else 0.0
        
        # Aggregates
        sub_test_pos = sum(m.get('user_test_pos_count', 0) for m in group)
        test_pos_pct = sub_test_pos / total_test_pos if total_test_pos > 0 else 0.0
        
        if count > 0:
            avg_ub_deg = np.mean([m.get('user_ub_train_deg', 0) for m in group])
            avg_ui_deg = np.mean([m.get('user_ui_deg', 0) for m in group])
            avg_test_recover = np.mean([m.get('user_test_avg_recover_ratio', 0) for m in group])
            avg_test_zero = np.mean([m.get('user_test_recover_zero_ratio', 0) for m in group])
        else:
            avg_ub_deg = 0.0
            avg_ui_deg = 0.0
            avg_test_recover = 0.0
            avg_test_zero = 0.0
            
        stats = {
            'user_count': count,
            'user_ratio': float(pct),
            'avg_user_ub_train_deg': float(avg_ub_deg),
            'avg_user_ui_deg': float(avg_ui_deg),
            'avg_user_test_avg_recover_ratio': float(avg_test_recover),
            'avg_user_test_recover_zero_ratio': float(avg_test_zero),
            'test_pos_total_share': float(test_pos_pct)
        }
        bucket_stats[name] = stats
        
    return bucket_stats

def main():
    args = parse_args()
    
    # Setup Output Directory
    output_dir = os.path.join('.', 'connectivity_analysis_outputs', args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    # Load Data
    num_users, user_items, bundle_items, user_train_bundles, user_test_bundles = load_data(args.data_root, args.dataset)
    
    # Compute Metrics for Each User
    print("Computing metrics for all users...")
    all_metrics = []
    
    # Need to iterate carefully.
    # We will assume user IDs are 0 to num_users-1
    for u_id in tqdm(range(num_users)):
        u_items = user_items.get(u_id, set())
        u_train = user_train_bundles.get(u_id, [])
        u_test = user_test_bundles.get(u_id, [])
        
        m = compute_user_metrics(u_id, u_items, u_train, u_test, bundle_items)
        all_metrics.append(m)
        
    # 1. Output Metrics CSV
    csv_path = os.path.join(output_dir, 'user_connectivity_metrics.csv')
    if all_metrics:
        keys = all_metrics[0].keys()
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_metrics)
        print(f"Saved metrics to {csv_path}")
    else:
        print("No metrics computed!")
    
    # 2. Summary Statistics
    summary_cols = [
        'train_connected_ratio',
        'train_disconnected_ratio',
        'user_ub_train_deg',
        'user_ui_deg',
        'avg_overlap_count_per_train_bundle',
        'avg_overlap_ratio_per_train_bundle',
        'user_test_avg_recover_ratio',
        'user_test_recover_zero_ratio'
    ]
    summary_stats = calculate_statistics(all_metrics, summary_cols)
    summary_json_path = os.path.join(output_dir, 'connectivity_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    print(f"Saved summary to {summary_json_path}")
    
    # 3. Bucket Analysis
    bucket_stats = analyze_buckets(all_metrics)
    buckets_json_path = os.path.join(output_dir, 'connectivity_buckets.json')
    with open(buckets_json_path, 'w') as f:
        json.dump(bucket_stats, f, indent=4)
    print(f"Saved buckets analysis to {buckets_json_path}")
    
    print("Done.")

if __name__ == "__main__":
    main()
