#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
import hashlib
import random
from datetime import datetime
from itertools import product

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.DWT import DWT, resolve_dwt_graph_config
from models.LatentDiffusionRebuilder import LatentDiffusionRebuilder
from models.MultiCBR import MultiCBR
from utility import Datasets, load_external_embedding_tensor, print_statistics


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="NetEase", type=str, help="which dataset to use, options: NetEase, iFashion")
    parser.add_argument("-m", "--model", default="MultiCBR", type=str, help="which model to use, options: MultiCBR")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    parser.add_argument("--ui_bundle_user_agg_beta", default=None, type=float, help="coefficient for user-side aggregation in UI view")
    parser.add_argument("--bi_user_bundle_agg_beta", default=None, type=float, help="coefficient for bundle-side aggregation in BI view")
    parser.add_argument("-e", "--epochs", default=None, type=int, help="number of epochs to train")
    parser.add_argument("--seed", default=None, type=int, help="random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed):
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Global random seed set to: {seed}")
    else:
        print("Random seed not set, running with random initialization.")


def resolve_train_style(conf):
    return conf.get("train_style", "cbr").strip().lower()


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def fmt_list(values):
    return str(values).replace(" ", "").replace("[", "").replace("]", "").replace(",", "-")


def summarize_sparse_graph(graph):
    nnz = int(graph.nnz)
    num_rows, num_cols = graph.shape
    avg_interactions = float(nnz / max(num_rows, 1))
    row_coverage = float((graph.getnnz(axis=1) > 0).sum() / max(num_rows, 1))
    col_coverage = float((graph.getnnz(axis=0) > 0).sum() / max(num_cols, 1))
    density = float(nnz / max(num_rows * num_cols, 1))
    return {
        "nnz": nnz,
        "avg_interactions": avg_interactions,
        "row_coverage": row_coverage,
        "col_coverage": col_coverage,
        "density": density,
    }


def report_latent_rebuild_epoch(run, log_path, epoch, latent_diffusion_loss, rebuilt_ub_graph, dataset_name, prefix):
    graph_stats = summarize_sparse_graph(rebuilt_ub_graph)
    message = (
        f"[{prefix}] epoch={epoch + 1} dataset={dataset_name} "
        f"loss={latent_diffusion_loss:.6f} rebuilt_edges={graph_stats['nnz']} "
        f"avg_user_edges={graph_stats['avg_interactions']:.6f} "
        f"user_coverage={graph_stats['row_coverage']:.6f} "
        f"bundle_coverage={graph_stats['col_coverage']:.6f} "
        f"density={graph_stats['density']:.10f}"
    )
    print(message)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

    run.add_scalar(f"{prefix}/loss_epoch", latent_diffusion_loss, epoch)
    run.add_scalar(f"{prefix}/rebuilt_edges", graph_stats["nnz"], epoch)
    run.add_scalar(f"{prefix}/avg_user_edges", graph_stats["avg_interactions"], epoch)
    run.add_scalar(f"{prefix}/user_coverage", graph_stats["row_coverage"], epoch)
    run.add_scalar(f"{prefix}/bundle_coverage", graph_stats["col_coverage"], epoch)
    run.add_scalar(f"{prefix}/density", graph_stats["density"], epoch)


def build_dwt_training_ub_graph(ub_graph, conf, device):
    adjacency_matrix = sp.bmat([
        [sp.csr_matrix((conf["num_users"], conf["num_users"])), ub_graph],
        [ub_graph.T, sp.csr_matrix((conf["num_bundles"], conf["num_bundles"]))],
    ])
    adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
    row_sum = np.array(adjacency_matrix.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    degree_matrix = sp.diags(d_inv)
    norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocoo()
    values = norm_adjacency.data
    indices = np.vstack((norm_adjacency.row, norm_adjacency.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(norm_adjacency.shape),
    ).to(device)


def build_observed_bundle_batch(dataset, user_indices, device):
    batch_users = np.asarray(user_indices, dtype=np.int64)
    batch_observed = [dataset.user_observed_bundles[int(user_id)] for user_id in batch_users]
    if len(batch_observed) == 0:
        raise ValueError("build_observed_bundle_batch received an empty user batch")
    max_len = max(len(observed) for observed in batch_observed)

    observed_indices = torch.full(
        (len(batch_users), max_len),
        -1,
        dtype=torch.long,
        device=device,
    )
    observed_mask = torch.zeros(
        (len(batch_users), max_len),
        dtype=torch.float32,
        device=device,
    )

    for row_idx, observed in enumerate(batch_observed):
        observed_tensor = torch.tensor(observed, dtype=torch.long, device=device)
        observed_indices[row_idx, :observed_tensor.shape[0]] = observed_tensor
        observed_mask[row_idx, :observed_tensor.shape[0]] = 1.0

    return {
        "user_indices": torch.tensor(batch_users, dtype=torch.long, device=device),
        "observed_indices": observed_indices,
        "observed_mask": observed_mask,
    }


def train_latent_diffusion_epoch(latent_diffusion_model, latent_diffusion_optimizer, dataset, rebuild_conf, device):
    active_users = np.asarray(dataset.train_active_user_indices, dtype=np.int64)
    if active_users.size == 0:
        return 0.0

    user_order = np.random.permutation(active_users)
    batch_size = rebuild_conf["latent_diffusion_batch_size"]
    step_num = (user_order.shape[0] + batch_size - 1) // batch_size
    pbar = tqdm(range(step_num), total=step_num, disable=True)
    total_loss = 0.0
    effective_steps = 0

    latent_diffusion_model.train(True)
    for step_idx in pbar:
        start = step_idx * batch_size
        end = min((step_idx + 1) * batch_size, dataset.num_users)
        batch = build_observed_bundle_batch(dataset, user_order[start:end], device)

        latent_diffusion_optimizer.zero_grad()
        loss = latent_diffusion_model.training_loss(batch)
        loss.backward()
        latent_diffusion_optimizer.step()

        total_loss += loss.detach().item()
        effective_steps += 1

    return total_loss / max(effective_steps, 1)


def rebuild_ub_graph_with_latent_diffusion(latent_diffusion_model, dataset, rebuild_conf, device):
    batch_size = rebuild_conf["latent_diffusion_infer_batch_size"]
    rebuild_k = rebuild_conf["rebuild_k"]
    user_indices = np.asarray(dataset.train_active_user_indices, dtype=np.int64)
    u_list = []
    b_list = []
    edge_list = []

    if user_indices.size == 0:
        return sp.coo_matrix(
            (dataset.num_users, dataset.num_bundles),
            dtype=np.float32,
        ).tocsr()

    latent_diffusion_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, user_indices.shape[0], batch_size), desc="LatentDiffusion Rebuild", disable=True):
            end = min(start + batch_size, user_indices.shape[0])
            batch = build_observed_bundle_batch(dataset, user_indices[start:end], device)
            selected_bundles, selected_valid = latent_diffusion_model.rebuild_topk(
                batch["user_indices"],
                batch["observed_indices"],
                rebuild_k=rebuild_k,
            )

            selected_bundles = selected_bundles.cpu().numpy()
            selected_valid = selected_valid.cpu().numpy()
            batch_users = batch["user_indices"].cpu().numpy()
            for row_idx, user_id in enumerate(batch_users):
                for col_idx in range(selected_bundles.shape[1]):
                    if not selected_valid[row_idx, col_idx]:
                        continue
                    u_list.append(int(user_id))
                    b_list.append(int(selected_bundles[row_idx, col_idx]))
                    edge_list.append(1.0)

    rebuilt_ub_graph = sp.coo_matrix(
        (
            np.array(edge_list, dtype=np.float32),
            (np.array(u_list, dtype=np.int32), np.array(b_list, dtype=np.int32)),
        ),
        shape=(dataset.num_users, dataset.num_bundles),
    ).tocsr()
    return rebuilt_ub_graph


def create_latent_diffusion_rebuilder(dwt_conf, user_embedding_tensor, bundle_embedding_tensor, device):
    latent_diffusion_model = LatentDiffusionRebuilder(
        dwt_conf,
        user_embedding_tensor.to(device),
        bundle_embedding_tensor.to(device),
    ).to(device)
    latent_diffusion_optimizer = torch.optim.Adam(
        latent_diffusion_model.parameters(),
        lr=dwt_conf["latent_diffusion_lr"],
        weight_decay=dwt_conf["latent_diffusion_weight_decay"],
    )
    return latent_diffusion_model, latent_diffusion_optimizer


def reset_latent_diffusion_rebuilder(dwt_conf, latent_diffusion_model, latent_diffusion_optimizer):
    latent_diffusion_model.reset_parameters()
    if dwt_conf.get("latent_diffusion_reset_optimizer", True):
        latent_diffusion_optimizer = torch.optim.Adam(
            latent_diffusion_model.parameters(),
            lr=dwt_conf["latent_diffusion_lr"],
            weight_decay=dwt_conf["latent_diffusion_weight_decay"],
        )
    return latent_diffusion_model, latent_diffusion_optimizer


def create_cbr_latent_rebuild_components(conf, dataset, device):
    rebuild_conf = conf.get("latent_rebuild", {})
    if not rebuild_conf.get("enabled", False):
        return None, None

    user_embedding_tensor = load_external_embedding_tensor(
        rebuild_conf["latent_diffusion_user_embedding_path"],
        dataset.num_users,
        "user",
        required_ids=dataset.train_active_user_indices,
        fill_missing_with_zeros=True,
    )
    bundle_embedding_tensor = load_external_embedding_tensor(
        rebuild_conf["latent_diffusion_bundle_embedding_path"],
        dataset.num_bundles,
        "bundle",
        required_ids=dataset.train_bundle_indices,
        fill_missing_with_zeros=True,
    )
    if user_embedding_tensor.shape[1] != bundle_embedding_tensor.shape[1]:
        raise ValueError("Latent diffusion user and bundle embeddings must share the same dimension")

    latent_diffusion_model, latent_diffusion_optimizer = create_latent_diffusion_rebuilder(
        rebuild_conf,
        user_embedding_tensor,
        bundle_embedding_tensor,
        device,
    )
    return latent_diffusion_model, latent_diffusion_optimizer


def run_cbr_training(conf, dataset, device):
    anchor_cl_conf = conf.get("anchor_cl", {})
    if anchor_cl_conf.get("enabled", False):
        print("=" * 20 + " Pre-fusion Anchor CL Enabled " + "=" * 20)
        print(f"lambda: {anchor_cl_conf.get('anchor_cl_lambda')}, temp: {anchor_cl_conf.get('temp')}")
        print(f"user_cl: {anchor_cl_conf.get('user_cl')}, bundle_cl: {anchor_cl_conf.get('bundle_cl')}")
        print(f"Weights -> u_ub_ui: {anchor_cl_conf.get('ub_ui_user_weight')}, u_ub_bi: {anchor_cl_conf.get('ub_bi_user_weight')}")
        print(f"Weights -> b_ub_ui: {anchor_cl_conf.get('ub_ui_bundle_weight')}, b_ub_bi: {anchor_cl_conf.get('ub_bi_bundle_weight')}")
        print("=" * 70)

    for lr, l2_reg, UB_ratio, UI_ratio, BI_ratio, embedding_size, num_layers, c_lambda, c_temp, ui_beta, bi_beta in product(
        conf["lrs"],
        conf["l2_regs"],
        conf["UB_ratios"],
        conf["UI_ratios"],
        conf["BI_ratios"],
        conf["embedding_sizes"],
        conf["num_layerss"],
        conf["c_lambdas"],
        conf["c_temps"],
        [conf["ui_bundle_user_agg_beta"]],
        [conf["bi_user_bundle_agg_beta"]],
    ):
        log_path = "./log/%s/%s" % (conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])
        ensure_dir(run_path)
        ensure_dir(log_path)
        ensure_dir(checkpoint_model_path)
        ensure_dir(checkpoint_conf_path)

        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size
        conf["UB_ratio"] = UB_ratio
        conf["UI_ratio"] = UI_ratio
        conf["BI_ratio"] = BI_ratio
        conf["num_layers"] = num_layers
        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        conf["ui_bundle_user_agg_beta"] = ui_beta
        conf["bi_user_bundle_agg_beta"] = bi_beta

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]
        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            assert UB_ratio == 0 and UI_ratio == 0 and BI_ratio == 0

        settings += [
            "Neg_%d" % conf["neg_num"],
            str(conf["batch_size_train"]),
            str(lr),
            str(l2_reg),
            str(embedding_size),
            str(UB_ratio),
            str(UI_ratio),
            str(BI_ratio),
            str(num_layers),
        ]
        settings += [
            "_".join(
                [
                    fmt_list(conf["fusion_weights"]["modal_weight"]),
                    fmt_list(conf["fusion_weights"]["UB_layer"]),
                    fmt_list(conf["fusion_weights"]["UI_layer"]),
                    fmt_list(conf["fusion_weights"]["BI_layer"]),
                ]
            )
        ]
        settings += [str(c_lambda), str(c_temp), str(ui_beta), str(bi_beta)]
        latent_rebuild_conf = conf.get("latent_rebuild", {})
        if latent_rebuild_conf.get("enabled", False):
            settings += [
                "LatentRebuild",
                f"k{latent_rebuild_conf['rebuild_k']}",
                f"step{latent_rebuild_conf['latent_diffusion_num_steps']}",
                f"ldlr{latent_rebuild_conf['latent_diffusion_lr']}",
                f"ldw{latent_rebuild_conf['latent_diffusion_set_loss_weight']}",
            ]

        setting = "_".join(settings)
        if len(setting) > 100:
            setting_hash = hashlib.md5(setting.encode()).hexdigest()[:8]
            setting = setting[:100] + "_" + setting_hash

        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        run = SummaryWriter(run_path)
        model = MultiCBR(conf, dataset.graphs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])
        latent_diffusion_model, latent_diffusion_optimizer = create_cbr_latent_rebuild_components(
            conf,
            dataset,
            device,
        )

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf["epochs"]):
            latent_rebuild_conf = conf.get("latent_rebuild", {})
            if latent_rebuild_conf.get("enabled", False):
                if (
                    latent_rebuild_conf.get("latent_diffusion_reset_after_epoch", -1) >= 0
                    and epoch == latent_rebuild_conf["latent_diffusion_reset_after_epoch"] + 1
                ):
                    latent_diffusion_model, latent_diffusion_optimizer = reset_latent_diffusion_rebuilder(
                        latent_rebuild_conf,
                        latent_diffusion_model,
                        latent_diffusion_optimizer,
                    )

                latent_diffusion_loss = train_latent_diffusion_epoch(
                    latent_diffusion_model,
                    latent_diffusion_optimizer,
                    dataset,
                    latent_rebuild_conf,
                    device,
                )
                rebuilt_ub_graph = rebuild_ub_graph_with_latent_diffusion(
                    latent_diffusion_model,
                    dataset,
                    latent_rebuild_conf,
                    device,
                )
                if epoch == 0:
                    print_statistics(rebuilt_ub_graph, "U-B statistics from Latent diffusion rebuild")
                report_latent_rebuild_epoch(
                    run,
                    log_path,
                    epoch,
                    latent_diffusion_loss,
                    rebuilt_ub_graph,
                    conf["dataset"],
                    "latent_rebuild",
                )
                model.set_ub_graph(rebuilt_ub_graph)

            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader), disable=True)

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0:
                    ED_drop = True

                bpr_loss, c_loss, anchor_cl_loss, anchor_cl_dict = model(batch, ED_drop=ED_drop)
                anchor_cl_lambda = conf.get("anchor_cl", {}).get("anchor_cl_lambda", 0.0)
                loss = bpr_loss + conf["c_lambda"] * c_loss + anchor_cl_lambda * anchor_cl_loss

                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                anchor_cl_loss_scalar = anchor_cl_loss.detach() if isinstance(anchor_cl_loss, torch.Tensor) else anchor_cl_loss

                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss_anchor_cl", anchor_cl_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                if conf.get("anchor_cl", {}).get("enabled", False):
                    for key, value in anchor_cl_dict.items():
                        run.add_scalar(f"loss_anchor_cl/{key}", value, batch_anchor)

                pbar.set_description(
                    "epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f, anchor_cl: %.4f"
                    % (epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar, anchor_cl_loss_scalar)
                )

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {
                        "val": test(model, dataset.val_loader, conf),
                        "test": test(model, dataset.test_loader, conf),
                    }
                    best_metrics, best_perform, best_epoch = log_metrics(
                        conf,
                        model,
                        metrics,
                        run,
                        log_path,
                        checkpoint_model_path,
                        checkpoint_conf_path,
                        epoch,
                        batch_anchor,
                        best_metrics,
                        best_perform,
                        best_epoch,
                    )


def run_dwt_training(conf, dataset, device):
    dwt_conf = conf.get("dwt", {})
    if not dwt_conf:
        raise ValueError("train_style=dwt requires a dwt config block.")
    use_latent_diffusion_rebuild = dwt_conf.get("use_latent_diffusion_rebuild", False)
    dwt_graph_conf = resolve_dwt_graph_config(conf)

    for lr, embedding_size, num_layers in product(
        conf["lrs"],
        conf["embedding_sizes"],
        conf["num_layerss"],
    ):
        log_path = "./log/%s/%s" % (conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])
        ensure_dir(run_path)
        ensure_dir(log_path)
        ensure_dir(checkpoint_model_path)
        ensure_dir(checkpoint_conf_path)

        conf["embedding_size"] = embedding_size
        conf["num_layers"] = num_layers
        conf["lr"] = lr
        conf["l2_reg"] = 0.0
        conf["c_lambda"] = 0.0
        conf["c_temp"] = dwt_conf["tau"]
        conf["ui_bundle_user_agg_beta"] = 0.0
        conf["bi_user_bundle_agg_beta"] = 0.0

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]
        settings += [
            "DWT",
            "LatentRebuild" if use_latent_diffusion_rebuild else "Noise",
            "Neg_%d" % conf["neg_num"],
            str(conf["batch_size_train"]),
            str(lr),
            str(embedding_size),
            str(num_layers),
            fmt_list([dwt_graph_conf["upsilon"]["UB"], dwt_graph_conf["upsilon"]["UI"], dwt_graph_conf["upsilon"]["BI"]]),
            fmt_list(dwt_graph_conf["layer_coefs"]["UB"]),
            fmt_list(dwt_graph_conf["layer_coefs"]["UI"]),
            fmt_list(dwt_graph_conf["layer_coefs"]["BI"]),
            str(dwt_graph_conf["omega"]),
            str(dwt_conf["gamma_1"]),
            str(dwt_conf["gamma_2"]),
            str(dwt_conf["tau"]),
            str(dwt_conf["lambda_1"]),
            str(dwt_conf["lambda_2"]),
        ]
        if use_latent_diffusion_rebuild:
            settings += [
                f"k{dwt_conf['rebuild_k']}",
                f"step{dwt_conf['latent_diffusion_num_steps']}",
                f"ldlr{dwt_conf['latent_diffusion_lr']}",
                f"ldw{dwt_conf['latent_diffusion_set_loss_weight']}",
            ]

        setting = "_".join(settings)
        if len(setting) > 100:
            setting_hash = hashlib.md5(setting.encode()).hexdigest()[:8]
            setting = setting[:100] + "_" + setting_hash

        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        run = SummaryWriter(run_path)
        model = DWT(conf, dataset.graphs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

        if use_latent_diffusion_rebuild:
            user_embedding_tensor = load_external_embedding_tensor(
                dwt_conf["latent_diffusion_user_embedding_path"],
                dataset.num_users,
                "user",
            )
            bundle_embedding_tensor = load_external_embedding_tensor(
                dwt_conf["latent_diffusion_bundle_embedding_path"],
                dataset.num_bundles,
                "bundle",
            )
            if user_embedding_tensor.shape[1] != bundle_embedding_tensor.shape[1]:
                raise ValueError("Latent diffusion user and bundle embeddings must share the same dimension")
            latent_diffusion_model, latent_diffusion_optimizer = create_latent_diffusion_rebuilder(
                dwt_conf,
                user_embedding_tensor,
                bundle_embedding_tensor,
                device,
            )
        else:
            latent_diffusion_model = None
            latent_diffusion_optimizer = None
            model.set_training_ub_graph(build_dwt_training_ub_graph(dataset.graphs[0], conf, device))

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        for epoch in range(conf["epochs"]):
            if (
                use_latent_diffusion_rebuild
                and dwt_conf.get("latent_diffusion_reset_after_epoch", -1) >= 0
                and epoch == dwt_conf["latent_diffusion_reset_after_epoch"] + 1
            ):
                latent_diffusion_model, latent_diffusion_optimizer = reset_latent_diffusion_rebuilder(
                    dwt_conf,
                    latent_diffusion_model,
                    latent_diffusion_optimizer,
                )

            if use_latent_diffusion_rebuild:
                latent_diffusion_loss = train_latent_diffusion_epoch(
                    latent_diffusion_model,
                    latent_diffusion_optimizer,
                    dataset,
                    dwt_conf,
                    device,
                )
                rebuilt_ub_graph = rebuild_ub_graph_with_latent_diffusion(
                    latent_diffusion_model,
                    dataset,
                    dwt_conf,
                    device,
                )
                if epoch == 0:
                    print_statistics(rebuilt_ub_graph, "U-B statistics from Latent diffusion rebuild")
                report_latent_rebuild_epoch(
                    run,
                    log_path,
                    epoch,
                    latent_diffusion_loss,
                    rebuilt_ub_graph,
                    conf["dataset"],
                    "latent_diffusion",
                )
                model.set_training_ub_graph(build_dwt_training_ub_graph(rebuilt_ub_graph, conf, device))

            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader), disable=True)

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                bpr_loss, dwt_cl_loss = model(batch)
                loss = bpr_loss + dwt_cl_loss

                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                dwt_cl_loss_scalar = dwt_cl_loss.detach()

                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_dwt_cl", dwt_cl_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                pbar.set_description(
                    "epoch: %d, loss: %.4f, bpr_loss: %.4f, dwt_cl: %.4f"
                    % (epoch, loss_scalar, bpr_loss_scalar, dwt_cl_loss_scalar)
                )

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {
                        "val": test(model, dataset.val_loader, conf),
                        "test": test(model, dataset.test_loader, conf),
                    }
                    best_metrics, best_perform, best_epoch = log_metrics(
                        conf,
                        model,
                        metrics,
                        run,
                        log_path,
                        checkpoint_model_path,
                        checkpoint_conf_path,
                        epoch,
                        batch_anchor,
                        best_metrics,
                        best_perform,
                        best_epoch,
                    )


def main(args=None):
    conf = yaml.safe_load(open("./config.yaml", encoding="utf-8"))
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
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    for key, value in paras.items():
        if key not in ["dataset", "model", "gpu", "info"] and value is not None:
            conf[key] = value

    if "ui_bundle_user_agg_beta" in paras and paras["ui_bundle_user_agg_beta"] is not None:
        conf["ui_bundle_user_agg_beta"] = paras["ui_bundle_user_agg_beta"]
    if "bi_user_bundle_agg_beta" in paras and paras["bi_user_bundle_agg_beta"] is not None:
        conf["bi_user_bundle_agg_beta"] = paras["bi_user_bundle_agg_beta"]
    if "epochs" in paras and paras["epochs"] is not None:
        conf["epochs"] = paras["epochs"]
    if "seed" in paras and paras["seed"] is not None:
        conf["seed"] = paras["seed"]

    train_style = resolve_train_style(conf)
    if train_style == "dwt":
        set_seed(conf.get("seed", None))

    dataset = Datasets(conf)
    if train_style != "dwt":
        set_seed(conf.get("seed", None))

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)

    print(f"train_style: {train_style}")

    if train_style == "cbr":
        run_cbr_training(conf, dataset, device)
    elif train_style == "dwt":
        run_dwt_training(conf, dataset, device)
    else:
        raise ValueError(f"Unsupported train_style: {train_style}")


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf["topk"]:
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

    for metric_name, val_score in val_scores.items():
        test_score = test_scores[metric_name]
        run.add_scalar("%s_%d/Val" % (metric_name, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" % (metric_name, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" % val_str)
    log.write("%s\n" % test_str)
    log.close()

    print(val_str)
    print(test_str)
    print("-" * 20)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" % topk_)

    def get_score(metric_dict):
        return metric_dict["recall"][20] + metric_dict["recall"][40] + metric_dict["ndcg"][20] + metric_dict["ndcg"][40]

    curr_score = get_score(metrics["val"])
    best_score = get_score(best_metrics["val"])

    if curr_score > best_score:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ui_beta = conf.get("ui_bundle_user_agg_beta", 0.0)
        bi_beta = conf.get("bi_user_bundle_agg_beta", 0.0)
        param_info = "褰撳墠缃戞牸鍙傛暟: %.2f + %.2f" % (
            ui_beta if ui_beta is not None else 0.0,
            bi_beta if bi_beta is not None else 0.0,
        )
        print(param_info)
        log.write(param_info + "\n")

        for topk in conf["topk"]:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" % (
                curr_time,
                best_epoch,
                topk,
                best_metrics["test"]["recall"][topk],
                best_metrics["test"]["ndcg"][topk],
            )
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" % (
                curr_time,
                best_epoch,
                topk,
                best_metrics["val"]["recall"][topk],
                best_metrics["val"]["ndcg"][topk],
            )
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, dataloader, conf):
    tmp_metrics = {}
    for metric_name in ["recall", "ndcg"]:
        tmp_metrics[metric_name] = {}
        for topk in conf["topk"]:
            tmp_metrics[metric_name][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.get_multi_modal_representations(test=True)
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device)
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for metric_name, topk_res in tmp_metrics.items():
        metrics[metric_name] = {}
        for topk, res in topk_res.items():
            metrics[metric_name][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1).cpu(), col_indice.view(-1).cpu()].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for metric_name, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for idx, value in enumerate(res):
                metrics[metric_name][topk][idx] += value

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

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
    IDCGs[0] = 1
    for idx in range(1, topk + 1):
        IDCGs[idx] = IDCG(idx, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg / idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
