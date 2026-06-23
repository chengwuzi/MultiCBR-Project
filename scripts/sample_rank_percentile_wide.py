#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import random
from pathlib import Path


DEFAULT_INPUT = Path("outputs/module1_ablation/edge_dynamics/selected_edges_pretrained_rank_percentile_wide.tsv")
DEFAULT_OUTPUT = Path("outputs/module1_ablation/edge_dynamics/selected_edges_pretrained_rank_percentile_wide_sample.tsv")
DEFAULT_SUMMARY = Path("outputs/module1_ablation/edge_dynamics/selected_edges_pretrained_rank_percentile_sample_summary.tsv")


def parse_args():
    parser = argparse.ArgumentParser(description="Sample wide-format rank percentile TSV for online violin plots.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def is_valid_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def reservoir_update(reservoir, value, count, sample_size, rng):
    if len(reservoir) < sample_size:
        reservoir.append(value)
        return
    replace_index = rng.randrange(count)
    if replace_index < sample_size:
        reservoir[replace_index] = value


def quantile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def stats(values):
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    sorted_values = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "median": quantile(sorted_values, 0.5),
        "q1": quantile(sorted_values, 0.25),
        "q3": quantile(sorted_values, 0.75),
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def main():
    args = parse_args()
    rng_by_column = {}
    reservoirs = {}
    all_values = {}
    counts = {}

    with args.input.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        columns = reader.fieldnames
        if not columns:
            raise ValueError(f"{args.input} has no header")

        for idx, column in enumerate(columns):
            rng_by_column[column] = random.Random(args.seed + idx)
            reservoirs[column] = []
            all_values[column] = []
            counts[column] = 0

        for row in reader:
            for column in columns:
                value = is_valid_number(row.get(column))
                if value is None:
                    continue
                counts[column] += 1
                all_values[column].append(value)
                reservoir_update(
                    reservoirs[column],
                    value,
                    counts[column],
                    args.sample_size,
                    rng_by_column[column],
                )

    max_len = max(len(values) for values in reservoirs.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(columns)
        for row_idx in range(max_len):
            writer.writerow(
                [
                    f"{reservoirs[column][row_idx]:.6f}" if row_idx < len(reservoirs[column]) else ""
                    for column in columns
                ]
            )

    fieldnames = [
        "position",
        "original_count",
        "sample_count",
        "original_mean",
        "sample_mean",
        "original_median",
        "sample_median",
        "original_q1",
        "sample_q1",
        "original_q3",
        "sample_q3",
        "original_min",
        "sample_min",
        "original_max",
        "sample_max",
    ]
    summary_rows = []
    for column in columns:
        original_stats = stats(all_values[column])
        sample_stats = stats(reservoirs[column])
        summary_rows.append(
            {
                "position": column,
                "original_count": counts[column],
                "sample_count": len(reservoirs[column]),
                "original_mean": original_stats["mean"],
                "sample_mean": sample_stats["mean"],
                "original_median": original_stats["median"],
                "sample_median": sample_stats["median"],
                "original_q1": original_stats["q1"],
                "sample_q1": sample_stats["q1"],
                "original_q3": original_stats["q3"],
                "sample_q3": sample_stats["q3"],
                "original_min": original_stats["min"],
                "sample_min": sample_stats["min"],
                "original_max": original_stats["max"],
                "sample_max": sample_stats["max"],
            }
        )

    with args.summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in summary_rows:
            formatted = dict(row)
            for key, value in formatted.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.6f}"
            writer.writerow(formatted)

    print(f"Saved sample to: {args.output}")
    print(f"Saved summary to: {args.summary}")
    for row in summary_rows:
        print(
            f"{row['position']}: original_count={row['original_count']}, "
            f"sample_count={row['sample_count']}, "
            f"original_mean={row['original_mean']:.6f}, "
            f"sample_mean={row['sample_mean']:.6f}, "
            f"original_median={row['original_median']:.6f}, "
            f"sample_median={row['sample_median']:.6f}"
        )


if __name__ == "__main__":
    main()
