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
        description="Run iFashion origin and z0-top3 experiments, then write detail and summary txt files."
    )
    parser.add_argument("--gpu", default="0", type=str, help="GPU id passed to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--epochs", default=None, type=int, help="Optional epoch override for quick trial runs")
    parser.add_argument("--seed", default=None, type=int, help="Optional seed override")
    parser.add_argument(
        "--output_dir",
        default="实验结果存档/模块一-z0-top3对照",
        type=str,
        help="Directory for the detail and summary txt files",
    )
    return parser.parse_args()


def make_experiment_conf(base_conf, experiment_name, rebuild_strategy, args, timestamp):
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
    conf["dwt"]["rebuild_k"] = 3
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
    return results[0]


def format_metric_block(result):
    metrics = result["best_metrics"]["test"]
    lines = [
        f"Experiment: {result['rebuild_strategy']}",
        f"Best epoch: {result['best_epoch']}",
        f"Log path: {result['log_path']}",
    ]
    for topk in [10, 20, 40, 80]:
        recall = metrics["recall"][topk]
        ndcg = metrics["ndcg"][topk]
        lines.append(f"TOP {topk}: REC_T={recall:.5f}, NDCG_T={ndcg:.5f}")
    return "\n".join(lines)


def write_summary(summary_path, results, timestamp):
    by_strategy = {result["rebuild_strategy"]: result for result in results}
    origin = by_strategy.get("origin")
    z0_topk = by_strategy.get("z0_topk")

    lines = [
        "iFashion origin vs z0-top3 experiment summary",
        f"Generated at: {timestamp}",
        "",
    ]
    for result in results:
        lines.append(format_metric_block(result))
        lines.append("")

    if origin is not None and z0_topk is not None:
        lines.append("z0-top3 improvement over origin")
        origin_metrics = origin["best_metrics"]["test"]
        z0_metrics = z0_topk["best_metrics"]["test"]
        for topk in [10, 20, 40, 80]:
            recall_delta = z0_metrics["recall"][topk] - origin_metrics["recall"][topk]
            ndcg_delta = z0_metrics["ndcg"][topk] - origin_metrics["ndcg"][topk]
            lines.append(
                f"TOP {topk}: ΔREC_T={recall_delta:+.5f}, ΔNDCG_T={ndcg_delta:+.5f}"
            )
        lines.append("")

    lines.append("Existing diffusion-top3 reference from experiment record")
    for topk in [10, 20, 40, 80]:
        lines.append(
            f"TOP {topk}: REC_T={DIFFUSION_TOP3_REFERENCE['recall'][topk]:.5f}, "
            f"NDCG_T={DIFFUSION_TOP3_REFERENCE['ndcg'][topk]:.5f}"
        )

    if z0_topk is not None:
        lines.append("")
        lines.append("z0-top3 difference against recorded diffusion-top3")
        z0_metrics = z0_topk["best_metrics"]["test"]
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / f"iFashion_origin_z0_top3_detail_{timestamp}.txt"
    summary_path = output_dir / f"iFashion_origin_z0_top3_summary_{timestamp}.txt"

    base_conf = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    experiments = [
        ("origin", "origin"),
        ("z0-top3", "z0_topk"),
    ]

    results = []
    with detail_path.open("w", encoding="utf-8") as detail_file:
        tee_stdout = Tee(sys.stdout, detail_file)
        tee_stderr = Tee(sys.stderr, detail_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print("iFashion origin and z0-top3 experiment runner")
            print(f"Detail output: {detail_path}")
            print(f"Summary output: {summary_path}")
            print(f"Started at: {timestamp}")

            for experiment_name, rebuild_strategy in experiments:
                conf = make_experiment_conf(
                    base_conf,
                    experiment_name,
                    rebuild_strategy,
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
