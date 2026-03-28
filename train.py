#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
import hashlib
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utility import Datasets
from models.MultiCBR import MultiCBR
from bi_diffusion_topk import build_bi_diffusion_module_from_sparse, resolve_keep_k


class BundleOnlyDataset(Dataset):
    def __init__(self, num_bundles: int):
        self.num_bundles = int(num_bundles)

    def __len__(self) -> int:
        return self.num_bundles

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(idx, dtype=torch.long)


def graph_stats(graph: sp.csr_matrix) -> dict:
    graph = graph.tocsr()
    row_nnz = np.diff(graph.indptr)
    col_nnz = np.asarray((graph > 0).sum(axis=0)).reshape(-1)
    total_possible = graph.shape[0] * graph.shape[1]
    return {
        "num_rows": int(graph.shape[0]),
        "num_cols": int(graph.shape[1]),
        "nnz": int(graph.nnz),
        "density": float(graph.nnz / max(total_possible, 1)),
        "avg_items_per_bundle": float(row_nnz.mean()) if len(row_nnz) > 0 else 0.0,
        "median_items_per_bundle": float(np.median(row_nnz)) if len(row_nnz) > 0 else 0.0,
        "p90_items_per_bundle": float(np.percentile(row_nnz, 90)) if len(row_nnz) > 0 else 0.0,
        "max_items_per_bundle": int(row_nnz.max()) if len(row_nnz) > 0 else 0,
        "min_items_per_bundle": int(row_nnz.min()) if len(row_nnz) > 0 else 0,
    }


def print_graph_stats(title: str, stats: dict) -> None:
    print("=" * 20 + f" {title} " + "=" * 20)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


def build_popularity_topk_graph(bi_graph: sp.csr_matrix, keep_ratio: float, min_keep: int) -> sp.csr_matrix:
    num_bundles, num_items = bi_graph.shape
    indptr = bi_graph.indptr
    indices = bi_graph.indices
    item_popularity = np.asarray((bi_graph > 0).sum(axis=0)).reshape(-1).astype(np.int64)

    rows, cols = [], []
    for b in range(num_bundles):
        items = indices[indptr[b]:indptr[b + 1]].copy()
        L = len(items)
        if L == 0:
            continue
        k = resolve_keep_k(L, keep_k=None, keep_ratio=keep_ratio, min_keep=min_keep)
        if L <= k:
            chosen = items
        else:
            pop = item_popularity[items]
            order = np.argsort(-pop, kind="stable")[:k]
            chosen = items[order]
        rows.extend([b] * len(chosen))
        cols.extend(chosen.tolist())
    vals = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((vals, (np.asarray(rows), np.asarray(cols))), shape=(num_bundles, num_items)).tocsr()


def prepare_bi_graph_for_training(conf: dict, dataset: Datasets, device: torch.device):
    """
    Preprocess the BI graph if bi_purifier is enabled.
    Returns:
        new_bi_graph: the original or purified sp.csr_matrix
        meta: a dictionary containing preprocessing logs/stats
    """
    bi_graph = dataset.graphs[2]
    purifier_conf = conf.get("bi_purifier", {})
    enabled = purifier_conf.get("enabled", False)

    if not enabled:
        print("Using original BI graph (bi_purifier.enabled=false)")
        return bi_graph, {"mode": "original"}

    mode = purifier_conf.get("mode", "diffusion")
    keep_ratio = purifier_conf.get("keep_ratio", 0.5)
    min_keep = purifier_conf.get("min_keep", 5)
    
    print(f"Using {mode} BI purifier (keep_ratio={keep_ratio}, min_keep={min_keep})")
    
    orig_stats = graph_stats(bi_graph)
    print_graph_stats("Original BI Graph Stats", orig_stats)

    cache_dir = purifier_conf.get("cache_dir", "./bi_purifier_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    if mode == "popularity":
        cache_name = f"{conf['dataset']}_popularity_ratio{keep_ratio}_min{min_keep}.npz"
        cache_path = os.path.join(cache_dir, cache_name)
        
        if purifier_conf.get("reuse_cached_graph", True) and os.path.isfile(cache_path):
            print(f"Loading popularity purified BI graph from cache: {cache_path}")
            purified_graph = sp.load_npz(cache_path)
        else:
            print("Building popularity purified BI graph...")
            purified_graph = build_popularity_topk_graph(bi_graph, keep_ratio, min_keep)
            if purifier_conf.get("save_purified_npz", True):
                sp.save_npz(cache_path, purified_graph)
                print(f"Saved purified graph to cache: {cache_path}")
                
    elif mode == "diffusion":
        cache_name = f"{conf['dataset']}_diffusion_ratio{keep_ratio}_min{min_keep}.npz"
        cache_path = os.path.join(cache_dir, cache_name)
        
        if purifier_conf.get("reuse_cached_graph", True) and os.path.isfile(cache_path):
            print(f"Loading diffusion purified BI graph from cache: {cache_path}")
            purified_graph = sp.load_npz(cache_path)
        else:
            print("Training diffusion BI purifier on the fly...")
            model = build_bi_diffusion_module_from_sparse(
                bi_graph=bi_graph,
                embedding_size=purifier_conf.get("embedding_size", 64),
                hidden_size=purifier_conf.get("hidden_size", 128),
                time_embed_size=purifier_conf.get("time_embed_size", 64),
                diffusion_steps=purifier_conf.get("diffusion_steps", 20),
                beta_start=purifier_conf.get("beta_start", 1e-4),
                beta_end=purifier_conf.get("beta_end", 2e-2),
                max_context_items=purifier_conf.get("max_context_items", 32),
                num_negative=purifier_conf.get("num_negative", 8),
                blcc_enabled=purifier_conf.get("blcc_enabled", False),
                blcc_lambda=purifier_conf.get("blcc_lambda", 0.1),
                dropout=purifier_conf.get("dropout", 0.1),
                use_layernorm=purifier_conf.get("use_layernorm", False),
                device=str(device)
            ).to(device)
            
            ckpt_path = purifier_conf.get("checkpoint_path", "")
            if ckpt_path and os.path.isfile(ckpt_path):
                print(f"Loading diffusion purifier weights from {ckpt_path}")
                payload = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(payload.get("model_state_dict", payload))
            else:
                bundle_dataset = BundleOnlyDataset(dataset.num_bundles)
                bundle_loader = DataLoader(bundle_dataset, batch_size=purifier_conf.get("batch_size", 256), shuffle=True)
                optimizer = optim.Adam(model.parameters(), lr=purifier_conf.get("lr", 1e-3), weight_decay=purifier_conf.get("weight_decay", 1e-5))
                
                epochs = purifier_conf.get("epochs", 20)
                model.train()
                for epoch in range(1, epochs + 1):
                    for batch_bundle_ids in bundle_loader:
                        batch_bundle_ids = batch_bundle_ids.to(device)
                        optimizer.zero_grad()
                        out = model(batch_bundle_ids)
                        out["total_loss"].backward()
                        optimizer.step()
                    print(f"Purifier Epoch {epoch}/{epochs} done.")
            
            print("Building purified BI graph with trained diffusion model...")
            model.eval()
            purified_graph = model.build_purified_bi_graph(keep_ratio=keep_ratio, min_keep=min_keep, keep_all_if_len_le_k=True)
            
            if purifier_conf.get("save_purified_npz", True):
                sp.save_npz(cache_path, purified_graph)
                print(f"Saved purified graph to cache: {cache_path}")
    else:
        raise ValueError(f"Unknown bi_purifier mode: {mode}")

    purified_stats = graph_stats(purified_graph)
    print_graph_stats(f"Purified BI Graph Stats ({mode})", purified_stats)
    
    return purified_graph, {
        "mode": mode,
        "keep_ratio": keep_ratio,
        "min_keep": min_keep,
        "orig_stats": orig_stats,
        "purified_stats": purified_stats
    }


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="NetEase", type=str, help="which dataset to use, options: NetEase, iFashion")
    parser.add_argument("-m", "--model", default="MultiCBR", type=str, help="which model to use, options: MultiCBR")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    args = parser.parse_args()

    return args


def main(args=None):
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    if args is None:
        paras = get_cmd().__dict__
    else:
        paras = args

    dataset_name = paras["dataset"]

    assert paras["model"] in ["MultiCBR"], "Pls select models from: MultiCBR"

    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    dataset = Datasets(conf)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    # Override config with any additional parameters passed in args
    for key, value in paras.items():
        if key not in ["dataset", "model", "gpu", "info"] and value is not None:
             # Only override if the key exists in the specific dataset config or create new one
             conf[key] = value

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)
    
    # Preprocess BI graph
    new_bi_graph, bi_meta = prepare_bi_graph_for_training(conf, dataset, device)
    dataset.graphs[2] = new_bi_graph

    for lr, l2_reg, UB_ratio, UI_ratio, BI_ratio, embedding_size, num_layers, c_lambda, c_temp in \
            product(conf['lrs'], conf['l2_regs'], conf['UB_ratios'], conf['UI_ratios'], conf['BI_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"]):
        log_path = "./log/%s/%s" % (conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            assert UB_ratio == 0 and UI_ratio == 0 and BI_ratio == 0

        settings += ["Neg_%d" % (conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg),
                     str(embedding_size)]

        conf["UB_ratio"] = UB_ratio
        conf["UI_ratio"] = UI_ratio
        conf["BI_ratio"] = BI_ratio
        conf["num_layers"] = num_layers
        settings += [str(UB_ratio), str(UI_ratio), str(BI_ratio), str(num_layers)]

        # Helper to format list to string without spaces and brackets
        def fmt_list(l):
            return str(l).replace(" ", "").replace("[", "").replace("]", "").replace(",", "-")

        settings += ["_".join([fmt_list(conf['fusion_weights']["modal_weight"]), fmt_list(conf['fusion_weights']["UB_layer"]),
                               fmt_list(conf['fusion_weights']["UI_layer"]), fmt_list(conf['fusion_weights']["BI_layer"])])]

        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        settings += [str(c_lambda), str(c_temp)]

        setting = "_".join(settings)

        # Windows path length limit fix: shorten setting string if too long
        if len(setting) > 100:
            hash_object = hashlib.md5(setting.encode())
            setting_hash = hash_object.hexdigest()[:8]
            # Truncate to keep it short and append hash for uniqueness
            setting = setting[:100] + "_" + setting_hash

        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        run = SummaryWriter(run_path)

        # model
        if conf['model'] == 'MultiCBR':
            model = MultiCBR(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" % (conf["model"]))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf['epochs']):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0:
                    ED_drop = True
                bpr_loss, c_loss = model(batch, ED_drop=ED_drop)
                loss = bpr_loss + conf["c_lambda"] * c_loss
                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {}
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" %(topk_))
    
    # Calculate sum of Recall@20, Recall@40, NDCG@20, NDCG@40
    def get_score(m):
        return m["recall"][20] + m["recall"][40] + m["ndcg"][20] + m["ndcg"][40]

    curr_score = get_score(metrics["val"])
    best_score = get_score(best_metrics["val"])

    if curr_score > best_score:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.get_multi_modal_representations(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1).cpu(), col_indice.view(-1).cpu()].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()