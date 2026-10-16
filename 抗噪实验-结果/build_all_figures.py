#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the archived 108 conditions without changing any source results.

Run from any directory: python <path-to-this-file>
Creates 24 dataset figures, 8 overviews (PNG + SVG), provenance and a ZIP.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import itertools
import json
import math
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np


DATASETS = ("Youshu", "iFashion", "NetEase")
MODELS = ("CrossCBR", "MultiCBR", "ReCoBR")
MODES = ("replacement", "injection")
RATIOS = tuple(Decimal(s) for s in ("0", ".05", ".10", ".15", ".20", ".25"))
STYLES = {
    "CrossCBR": {"color": "#1f77b4", "marker": "^"},
    "MultiCBR": {"color": "#ff7f0e", "marker": "s"},
    "ReCoBR": {"color": "#2ca02c", "marker": "o"},
}
SOURCES = (
    ("替换式results_summary.csv", "replacement", 54),
    ("注入式(youshu+ifashion)results_summary (2).csv", "injection", 36),
    ("注入式-netease-results_summary (2).csv", "injection", 18),
)
BOTTOM = 0.30
TOP = 0.06
BREAK_HEIGHT = 0.012


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results(source_dir):
    records, provenance = {}, []
    for name, mode, expected_count in SOURCES:
        path = source_dir / name
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_count:
            raise ValueError("Unexpected record count: %s" % path)
        provenance.append({"file": name, "sha256": sha256(path), "rows": len(rows)})
        for row in rows:
            if row.get("noise_mode", mode) != mode or row["status"] != "succeeded":
                raise ValueError("Wrong mode or unfinished record: %s" % row["experiment_id"])
            key = (mode, row["dataset"], row["model"], Decimal(row["noise_ratio"]))
            if key in records:
                raise ValueError("Duplicate result: %s" % (key,))
            if (row["noise_seed"], row["train_seed"]) != ("20261014", "20261014"):
                raise ValueError("Unexpected seed: %s" % (key,))
            for k, metric in itertools.product((10, 20, 40, 80), ("recall", "ndcg")):
                value = float(row["%s@%d" % (metric, k)])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid metric: %s" % (key,))
            records[key] = row
    expected = set(itertools.product(MODES, DATASETS, MODELS, RATIOS))
    if set(records) != expected:
        raise ValueError("Results do not cover exactly the requested 108 conditions")
    for (mode, dataset, model, _), row in records.items():
        baseline = records[(mode, dataset, model, Decimal(0))]
        for k, metric in itertools.product((10, 20, 40, 80), ("recall", "ndcg")):
            field = "%s@%d" % (metric, k)
            zero = float(baseline[field])
            if zero <= 0:
                raise ValueError("Non-positive baseline: %s/%s" % (model, field))
            change = float(row[field]) / zero - 1
            if not math.isclose(float(row["%s_change_vs_zero@%d" % (metric, k)]),
                                change, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Stored relative change disagrees: %s/%s" % (model, field))
    return records, provenance


def values_for(records, mode, dataset, model, field, kind):
    values = np.array([float(records[(mode, dataset, model, ratio)][field]) for ratio in RATIOS])
    if kind == "relative":
        values = (values / values[0] - 1) * 100
    return values


def configure_y_axis(axis, series, dataset, kind):
    """Reserve 30% of visible axis height below the lowest observation.

    For iFashion absolute values, compress only the empty interval separating
    CrossCBR and the other models. The scales above and below remain identical.
    """
    observed = np.concatenate(list(series.values()))
    low, high = float(observed.min()), float(observed.max())
    span = max(high - low, abs(high) * 0.01, 0.001)
    gap = None
    if dataset == "iFashion" and kind == "absolute":
        cross_max = float(series["CrossCBR"].max())
        others_min = min(float(series[m].min()) for m in ("MultiCBR", "ReCoBR"))
        if others_min > cross_max:
            margin = (others_min - cross_max) * 0.06
            gap = (cross_max + margin, others_min - margin)
    if gap is None:
        height = span / (1 - BOTTOM - TOP)
        lower, upper = low - BOTTOM * height, high + TOP * height
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        forward = lambda y: np.asarray(y)
        inverse = forward
        break_info = None
    else:
        start, end = gap
        width = end - start
        assert not np.any((observed > start) & (observed < end))
        height = (high - low - width) / (1 - BOTTOM - TOP - BREAK_HEIGHT)
        compressed = height * BREAK_HEIGHT

        def forward(y):
            y = np.asarray(y)
            return np.where(y <= start, y, np.where(y >= end, y - width + compressed,
                            start + (y - start) * compressed / width))

        def inverse(y):
            y = np.asarray(y)
            return np.where(y <= start, y, np.where(y >= start + compressed,
                            y + width - compressed, start + (y - start) * width / compressed))

        lower_t = float(forward(low)) - BOTTOM * height
        upper_t = float(forward(high)) + TOP * height
        lower, upper = float(inverse(lower_t)), float(inverse(upper_t))
        axis.set_yscale("function", functions=(forward, inverse))
        axis.set_ylim(lower, upper)
        ticks = []
        for a, b in ((lower, start), (end, upper)):
            ticks.extend(t for t in MaxNLocator(nbins=3).tick_values(a, b) if a <= t <= b)
        axis.set_yticks(ticks)
        axis.axhspan(start, end, color="white", zorder=2)
        center = (float(forward((start + end) / 2)) - lower_t) / height
        for x in (0, 1):
            axis.plot([x - .012, x + .012], [center - .014, center + .004],
                      transform=axis.transAxes, color="#444444", linewidth=1.1,
                      clip_on=False, zorder=5)
            axis.plot([x - .012, x + .012], [center - .004, center + .014],
                      transform=axis.transAxes, color="#444444", linewidth=1.1,
                      clip_on=False, zorder=5)
        break_info = {"omitted_interval": [start, end], "visible_break_height": BREAK_HEIGHT}
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if kind == "absolute" else "%.1f"))
    actual_bottom = (float(forward(low)) - float(forward(lower))) / (
        float(forward(upper)) - float(forward(lower)))
    assert math.isclose(actual_bottom, BOTTOM, abs_tol=1e-10)
    probe = np.linspace(lower, upper, 301)
    assert np.allclose(inverse(forward(probe)), probe, rtol=1e-10, atol=1e-10)
    assert min(observed) >= lower and max(observed) <= upper
    return {"y_limits": [lower, upper], "bottom_fraction": actual_bottom, "break": break_info}


def draw_panel(axis, records, mode, dataset, k, metric, kind):
    field = "%s@%d" % (metric, k)
    series = {m: values_for(records, mode, dataset, m, field, kind) for m in MODELS}
    metadata = configure_y_axis(axis, series, dataset, kind)
    for model in MODELS:
        axis.plot([float(r) for r in RATIOS], series[model], label=model,
                  linestyle="--", linewidth=1.8, markersize=5.5, zorder=3, **STYLES[model])
    axis.set_xlim(-0.012, 0.262)
    axis.set_xticks([float(r) for r in RATIOS])
    axis.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.20", "0.25"])
    label = "%s@%d" % ("Recall" if metric == "recall" else "NDCG", k)
    axis.set_ylabel(label if kind == "absolute" else "Relative %s (%%)" % label)
    axis.set_xlabel("Noise ratio")
    axis.set_title("%s: %s" % (dataset, label), pad=9)
    axis.grid(color="#999999", alpha=.23, linewidth=.7, zorder=0)
    if kind == "relative":
        axis.axhline(0, color="#777777", linewidth=.7, zorder=1)
    axis.legend(loc="lower left", frameon=True, framealpha=.95,
                borderpad=.35, labelspacing=.25, handlelength=2.0, fontsize=10)
    metadata.update({"mode": mode, "dataset": dataset, "metric": field, "kind": kind,
                     "series": {m: y.tolist() for m, y in series.items()}})
    return metadata


def save_figure(figure, output_dir, basename):
    paths = []
    for extension in ("png", "svg"):
        path = output_dir / (basename + "." + extension)
        figure.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.10)
        paths.append(path)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = (args.output_dir or source_dir / "完整图表_20261016").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output-dir")
    records, provenance = load_results(source_dir)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.linewidth": 1.0, "svg.fonttype": "none"})
    output_dir.mkdir(parents=True, exist_ok=True)
    files, panels = [], []
    for mode, k, kind in itertools.product(MODES, (10, 20), ("absolute", "relative")):
        mode_dir = output_dir / mode
        mode_dir.mkdir(exist_ok=True)
        for dataset in DATASETS:
            figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
            for axis, metric in zip(axes, ("recall", "ndcg")):
                panels.append(draw_panel(axis, records, mode, dataset, k, metric, kind))
            figure.suptitle("%s noise | %s | K=%d" % (mode.capitalize(), dataset, k), fontsize=14)
            figure.tight_layout(rect=(0, 0, 1, .94), w_pad=2.3)
            name = "%s_%s_%s_top%d_bottom30" % (dataset.lower(), mode, kind, k)
            files.extend(save_figure(figure, mode_dir, name))
        figure, axes = plt.subplots(3, 2, figsize=(10.4, 11.8))
        for index, dataset in enumerate(DATASETS):
            for col, metric in enumerate(("recall", "ndcg")):
                panels.append(draw_panel(axes[index, col], records, mode, dataset, k, metric, kind))
        figure.suptitle("%s noise: %s metrics at K=%d" % (mode.capitalize(), kind, k),
                       fontsize=15, y=.997)
        figure.tight_layout(rect=(0, 0, 1, .98), h_pad=2.5, w_pad=2.5)
        name = "all_datasets_%s_%s_top%d_bottom30" % (mode, kind, k)
        files.extend(save_figure(figure, mode_dir, name))
        print("Created %s / K=%d / %s: 3 dataset figures + overview" % (mode, k, kind), flush=True)
    assert len(files) == 64 and len(panels) == 96
    for source in provenance:
        assert sha256(source_dir / source["file"]) == source["sha256"]
    manifest = {
        "sources": provenance, "condition_count": len(records),
        "relative_percent_definition": "(metric_at_noise / same_model_dataset_mode_metric_at_zero - 1) * 100",
        "noise_axis": "requested per-user noise ratio, not rounded actual aggregate ratio",
        "noise_ratios": [float(r) for r in RATIOS], "bottom_fraction": BOTTOM,
        "styles": STYLES, "png_dpi": 300,
        "axis_policy": "Independent labeled bounds per panel; iFashion absolute empty gaps compressed. "
                       "Equal units per height above/below each marked break. No breaks in relative charts.",
        "rerun_note": "ReCoBR/Youshu/injection/0.20 is read directly from archived CSV, "
                      "already updated to user-selected epoch 99 rerun. See engineering record item 23.",
        "outputs": [str(p.relative_to(output_dir)).replace('\\', '/') for p in files],
        "panels": panels,
    }
    # This generated manifest is an output of the reproducible plotting workflow.
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle_path = output_dir / "all_noise_figures_bottom30.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [manifest_path]:
            archive.write(path, path.relative_to(output_dir).as_posix())
    print("PASS: 108 conditions; all baseline changes validated; 32 PNG + 32 SVG; "
          "30% bottom space and reversible breaks verified; source CSVs unchanged.")
    print("Output: %s" % output_dir)


if __name__ == "__main__":
    main()
