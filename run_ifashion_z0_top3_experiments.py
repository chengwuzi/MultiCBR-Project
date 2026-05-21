#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import torch
import yaml

from train import Datasets, run_dwt_training, set_seed


DIFFUSION_TOP3_REFERENCE = {
    "recall": {
        10: 0.11445,
        20: 0.15668,
        40: 0.20674,
        80: 0.26436,
    },
    "ndcg": {
        10: 0.11454,
        20: 0.13148,
        40: 0.14908,
        80: 0.16629,
    },
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run iFashion origin and one or more z0-topK experiments, then write detail and summary txt files."
    )
    parser.add_argument("--gpu", default="0", type=str, help="GPU id passed to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--epochs", default=None, type=int, help="Optional epoch override for quick trial runs")
    parser.add_argument("--seed", default=None, type=int, help="Optional seed override")
    parser.add_argument(
        "--z0_ks",
        default="3",
        type=str,
        help="Comma-separated z0 TopK values to run, e.g. 6,9,12",
    )
    parser.add_argument(
        "--skip_origin",
        action="store_true",
        help="Only run z0-topK experiments and skip the origin baseline.",
    )
    parser.add_argument(
        "--output_dir",
        default="实验结果存档/模块一-z0-topk对照",
        type=str,
        help="Directory for the detail and summary txt files",
    )
    return parser.parse_args()


def parse_z0_ks(raw_value):
    z0_ks = []
    for value in raw_value.split(","):
        value = value.strip()
        if not value:
            continue
        k = int(value)
        if k <= 0:
            raise ValueError(f"z0 TopK must be positive, got {k}")
        z0_ks.append(k)
    if not z0_ks:
        raise ValueError("--z0_ks must contain at least one positive integer")
    return z0_ks


def make_experiment_conf(base_conf, experiment_name, rebuild_strategy, rebuild_k, args, timestamp):
    conf = copy.deepcopy(base_conf["iFashion"])
    conf["dataset"] = "iFashion"
    conf["model"] = "AnchorViewBundleNet"
    conf["gpu"] = args.gpu
    conf["info"] = f"{timestamp}_{experiment_name}"
    conf["experiment_name"] = experiment_name
    conf["train_style"] = "dwt"
    conf["show_progress"] = True

    if args.epochs is not None:
        conf["epochs"] = args.epochs
    if args.seed is not None:
        conf["seed"] = args.seed

    # Keep this comparison focused on module-1 edge purification only.
    conf["ui_bundle_user_agg_beta"] = 0.0
    conf["bi_user_bundle_agg_beta"] = 0.0
    conf.setdefault("anchor_cl", {})
    conf["anchor_cl"]["enabled"] = False

    conf.setdefault("dwt", {})
    conf["dwt"]["rebuild_strategy"] = rebuild_strategy
    conf["dwt"]["rebuild_k"] = rebuild_k
    conf["dwt"]["use_latent_diffusion_rebuild"] = rebuild_strategy == "diffusion"
    return conf


def serializable_conf(conf):
    payload = copy.deepcopy(conf)
    if "device" in payload:
        payload["device"] = str(payload["device"])
    return payload


def run_one_experiment(conf):
    print("\n" + "=" * 90)
    print(f"Starting experiment: {conf['experiment_name']}")
    print(f"Dataset: {conf['dataset']}")
    print(f"Rebuild strategy: {conf['dwt']['rebuild_strategy']}")
    print(f"Epochs: {conf['epochs']}")
    print(f"Seed: {conf.get('seed')}")
    print("=" * 90)

    set_seed(conf.get("seed", None))
    dataset = Datasets(conf)
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    print("[actual_run_config]")
    print(json.dumps(serializable_conf(conf), ensure_ascii=False, indent=2))
    print(f"Resolved device: {device}")

    results = run_dwt_training(conf, dataset, device)
    if not results:
        raise RuntimeError(f"Experiment {conf['experiment_name']} produced no result")
    result = results[0]
    result["experiment_name"] = conf["experiment_name"]
    result["rebuild_k"] = conf["dwt"].get("rebuild_k")
    return result


def format_metric_block(result):
    metrics = result["best_metrics"]["test"]
    rebuild_label = result["rebuild_strategy"]
    if result["rebuild_strategy"] == "z0_topk":
        rebuild_label = f"z0_topk@{result.get('rebuild_k')}"
    lines = [
        f"Experiment: {rebuild_label}",
        f"Best epoch: {result['best_epoch']}",
        f"Log path: {result['log_path']}",
    ]
    for topk in [10, 20, 40, 80]:
        recall = metrics["recall"][topk]
        ndcg = metrics["ndcg"][topk]
        lines.append(f"TOP {topk}: REC_T={recall:.5f}, NDCG_T={ndcg:.5f}")
    return "\n".join(lines)


def write_summary(summary_path, results, timestamp):
    origin = next((result for result in results if result["rebuild_strategy"] == "origin"), None)
    z0_results = [result for result in results if result["rebuild_strategy"] == "z0_topk"]

    lines = [
        "iFashion origin vs z0-topK experiment summary",
        f"Generated at: {timestamp}",
        "",
    ]
    for result in results:
        lines.append(format_metric_block(result))
        lines.append("")

    if origin is not None and z0_results:
        origin_metrics = origin["best_metrics"]["test"]
        for result in z0_results:
            lines.append(f"z0-top{result.get('rebuild_k')} improvement over origin")
            z0_metrics = result["best_metrics"]["test"]
            for topk in [10, 20, 40, 80]:
                recall_delta = z0_metrics["recall"][topk] - origin_metrics["recall"][topk]
                ndcg_delta = z0_metrics["ndcg"][topk] - origin_metrics["ndcg"][topk]
                lines.append(
                    f"TOP {topk}: ΔREC_T={recall_delta:+.5f}, ΔNDCG_T={ndcg_delta:+.5f}"
                )
            lines.append("")

    if z0_results:
        best_by_ndcg20 = max(z0_results, key=lambda result: result["best_metrics"]["test"]["ndcg"][20])
        best_metrics = best_by_ndcg20["best_metrics"]["test"]
        lines.append("Best z0-topK by NDCG@20 in this run")
        lines.append(
            f"z0-top{best_by_ndcg20.get('rebuild_k')}: "
            f"REC_T@20={best_metrics['recall'][20]:.5f}, "
            f"NDCG_T@20={best_metrics['ndcg'][20]:.5f}"
        )
        lines.append("")

    z0_top3 = next((result for result in z0_results if result.get("rebuild_k") == 3), None)
    if z0_top3 is not None:
        lines.append("Existing diffusion-top3 reference from experiment record")
        for topk in [10, 20, 40, 80]:
            lines.append(
                f"TOP {topk}: REC_T={DIFFUSION_TOP3_REFERENCE['recall'][topk]:.5f}, "
                f"NDCG_T={DIFFUSION_TOP3_REFERENCE['ndcg'][topk]:.5f}"
            )
        lines.append("")
        lines.append("z0-top3 difference against recorded diffusion-top3")
        z0_metrics = z0_top3["best_metrics"]["test"]
        for topk in [10, 20, 40, 80]:
            recall_delta = z0_metrics["recall"][topk] - DIFFUSION_TOP3_REFERENCE["recall"][topk]
            ndcg_delta = z0_metrics["ndcg"][topk] - DIFFUSION_TOP3_REFERENCE["ndcg"][topk]
            lines.append(
                f"TOP {topk}: ΔREC_T={recall_delta:+.5f}, ΔNDCG_T={ndcg_delta:+.5f}"
            )

        z0_ndcg20 = z0_metrics["ndcg"][20]
        diffusion_ndcg20 = DIFFUSION_TOP3_REFERENCE["ndcg"][20]
        if z0_ndcg20 > diffusion_ndcg20:
            conclusion = "z0-top3 is higher than recorded diffusion-top3 on NDCG@20."
        elif z0_ndcg20 < diffusion_ndcg20:
            conclusion = "z0-top3 is lower than recorded diffusion-top3 on NDCG@20."
        else:
            conclusion = "z0-top3 ties recorded diffusion-top3 on NDCG@20."
        lines.append("")
        lines.append(f"Primary NDCG@20 conclusion: {conclusion}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    z0_ks = parse_z0_ks(args.z0_ks)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z0_ks_label = "-".join(str(k) for k in z0_ks)
    origin_label = "z0" if args.skip_origin else "origin_z0"
    detail_path = output_dir / f"iFashion_{origin_label}_top{z0_ks_label}_detail_{timestamp}.txt"
    summary_path = output_dir / f"iFashion_{origin_label}_top{z0_ks_label}_summary_{timestamp}.txt"

    base_conf = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    experiments = []
    if not args.skip_origin:
        experiments.append(("origin", "origin", z0_ks[0]))
    for k in z0_ks:
        experiments.append((f"z0-top{k}", "z0_topk", k))

    results = []
    with detail_path.open("w", encoding="utf-8") as detail_file:
        tee_stdout = Tee(sys.stdout, detail_file)
        tee_stderr = Tee(sys.stderr, detail_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print("iFashion origin and z0-topK experiment runner")
            print(f"Detail output: {detail_path}")
            print(f"Summary output: {summary_path}")
            print(f"Started at: {timestamp}")
            print(f"z0 TopK values: {z0_ks}")
            print(f"Run origin: {not args.skip_origin}")

            for experiment_name, rebuild_strategy, rebuild_k in experiments:
                conf = make_experiment_conf(
                    base_conf,
                    experiment_name,
                    rebuild_strategy,
                    rebuild_k,
                    args,
                    timestamp,
                )
                result = run_one_experiment(conf)
                results.append(result)
                print("\n[experiment_result]")
                print(format_metric_block(result))

            print("\nAll experiments finished.")

    write_summary(summary_path, results, timestamp)
    print(f"\nDetail output written to: {detail_path}")
    print(f"Summary output written to: {summary_path}")


if __name__ == "__main__":
    main()
