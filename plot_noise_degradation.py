#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create paper-ready relative-degradation plots from benchmark CSV results.

The plotted value for every model/dataset/noise condition is:
    (metric_at_noise / metric_at_zero - 1) * 100

By default this produces a 3 x 2 panel figure: rows are Youshu, iFashion,
and NetEase; columns are Recall@20 and NDCG@20.  It deliberately requires a
complete successful 54-task result set, so an incomplete figure cannot be
mistaken for a final paper figure.
"""

from __future__ import print_function

import argparse
import csv
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys


DEFAULT_DATASETS = ("Youshu", "iFashion", "NetEase")
DEFAULT_MODELS = ("CrossCBR", "MultiCBR", "ReCoBR")
DEFAULT_RATIOS = ("0", "0.05", "0.10", "0.15", "0.20", "0.25")
MODEL_STYLE = {
    "CrossCBR": {"color": "#1f77b4", "marker": "^", "label": "CrossCBR"},
    "MultiCBR": {"color": "#ff7f0e", "marker": "s", "label": "MultiCBR"},
    "ReCoBR": {"color": "#2ca02c", "marker": "o", "label": "ReCoBR"},
}


def ratio_value(value):
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("Invalid noise ratio: %s" % value)
    if parsed < 0:
        raise ValueError("Noise ratio cannot be negative: %s" % value)
    return parsed


def read_completed_results(csv_path, datasets, models, ratios, topk):
    """Read and validate one successful metric for each requested condition."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError("Result CSV was not found: %s" % csv_path)

    ratio_lookup = {ratio_value(ratio): ratio for ratio in ratios}
    required_metrics = ("recall@%d" % topk, "ndcg@%d" % topk)
    values = {}

    with open(str(csv_path), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"model", "dataset", "noise_ratio", "status"}.union(required_metrics)
        absent = sorted(required_columns.difference(reader.fieldnames or []))
        if absent:
            raise ValueError("Result CSV is missing required columns: %s" % ", ".join(absent))

        for row in reader:
            if row.get("model") not in models or row.get("dataset") not in datasets:
                continue
            try:
                normalized_ratio = ratio_value(row.get("noise_ratio", ""))
            except ValueError:
                continue
            if normalized_ratio not in ratio_lookup:
                continue

            key = (row["dataset"], row["model"], normalized_ratio)
            if row.get("status") != "succeeded":
                continue
            if key in values:
                raise ValueError(
                    "More than one completed result matches %s. Choose one seed/result CSV before plotting." %
                    (key,)
                )
            try:
                values[key] = {metric: float(row[metric]) for metric in required_metrics}
            except (TypeError, ValueError):
                raise ValueError("Invalid %s metrics for %s" % (topk, key))

    missing = []
    for dataset in datasets:
        for model in models:
            for ratio in ratio_lookup:
                if (dataset, model, ratio) not in values:
                    missing.append("%s/%s/noise=%s" % (dataset, model, ratio_lookup[ratio]))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            "The paper figure requires all completed conditions. Missing %d: %s%s" %
            (len(missing), preview, suffix)
        )
    return values, ratio_lookup


def relative_degradation(values, dataset, model, ratios, metric):
    zero = values[(dataset, model, Decimal("0"))][metric]
    if zero == 0:
        raise ValueError("Cannot calculate relative degradation from zero baseline: %s/%s" % (dataset, model))
    return [(values[(dataset, model, ratio)][metric] / zero - 1.0) * 100.0 for ratio in ratios]


def import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(
            "This plotting command requires matplotlib. Install it in the active environment with: "
            "python -m pip install matplotlib"
        )
    return plt


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create 3x2 relative Recall@K/NDCG@K noise-degradation plots."
    )
    parser.add_argument(
        "--results-csv",
        default="noise_benchmark_artifacts/results_summary.csv",
        help="benchmark result CSV (default: noise_benchmark_artifacts/results_summary.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="noise_benchmark_artifacts/figures",
        help="directory for PNG/PDF figures",
    )
    parser.add_argument("--topk", type=int, default=20, help="Top-K used for both metrics (default: 20)")
    return parser


def main():
    args = build_parser().parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")

    values, ratio_lookup = read_completed_results(
        args.results_csv, DEFAULT_DATASETS, DEFAULT_MODELS, DEFAULT_RATIOS, args.topk
    )
    plt = import_matplotlib()
    ratios = tuple(sorted(ratio_lookup))
    x_values = [float(ratio) for ratio in ratios]
    x_labels = [ratio_lookup[ratio] for ratio in ratios]
    metric_specs = (("recall@%d" % args.topk, "Relative Recall@%d (%%)" % args.topk),
                    ("ndcg@%d" % args.topk, "Relative NDCG@%d (%%)" % args.topk))

    all_relative_values = []
    for dataset in DEFAULT_DATASETS:
        for model in DEFAULT_MODELS:
            for metric, _ in metric_specs:
                all_relative_values.extend(relative_degradation(values, dataset, model, ratios, metric))
    lower_bound = min(-10.0, math.floor(min(all_relative_values) / 10.0) * 10.0 - 2.0)
    upper_bound = max(2.0, math.ceil(max(all_relative_values) / 5.0) * 5.0 + 1.0)

    figure, axes = plt.subplots(3, 2, figsize=(10.0, 11.0), sharex=True)
    panel_labels = ("(a)", "(b)", "(c)")
    for row_index, dataset in enumerate(DEFAULT_DATASETS):
        for column_index, (metric, ylabel) in enumerate(metric_specs):
            axis = axes[row_index][column_index]
            for model in DEFAULT_MODELS:
                style = MODEL_STYLE[model]
                y_values = relative_degradation(values, dataset, model, ratios, metric)
                axis.plot(
                    x_values, y_values, linestyle="--", linewidth=1.8, markersize=5,
                    color=style["color"], marker=style["marker"], label=style["label"],
                )
            axis.axhline(0, color="#555555", linewidth=0.8)
            axis.grid(True, alpha=0.35, linewidth=0.7)
            axis.set_ylim(lower_bound, upper_bound)
            axis.set_xticks(x_values)
            axis.set_xticklabels(x_labels)
            axis.set_ylabel(ylabel)
            axis.set_title("%s %s Dataset" % (panel_labels[row_index], dataset))
            axis.legend(loc="lower left", frameon=True, framealpha=0.9)
            if row_index == len(DEFAULT_DATASETS) - 1:
                axis.set_xlabel("Noise ratio")

    figure.suptitle("Performance degradation with respect to training UB interaction noise", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = "noise_degradation_top%d" % args.topk
    png_path = output_dir / (base_name + ".png")
    pdf_path = output_dir / (base_name + ".pdf")
    figure.savefig(str(png_path), dpi=300, bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)
    print("Created: %s" % png_path.resolve())
    print("Created: %s" % pdf_path.resolve())


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        sys.exit(2)
