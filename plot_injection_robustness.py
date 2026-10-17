"""Plot injection-noise robustness from the archived experimental CSV."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "图表数据" / "injection_top20_plot_data.csv"
OUTPUT_DIR = ROOT / "figures" / "injection_robustness"
DATASETS = ("Youshu", "iFashion", "NetEase")
RATIOS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
METRICS = {"recall@20": ("Recall@20", "recall_at_20"),
           "ndcg@20": ("NDCG@20", "ndcg_at_20")}
MODELS = {
    "ReCoBR": ("#1f77b4", "^"),
    "MultiCBR": ("#ff7f0e", "v"),
    "CrossCBR": ("#2ca02c", "o"),
}
# iFashion has a large gap between CrossCBR and the other two models.
# Each band uses actual metric values; the diagonal marks identify the omitted range.
IFASHION_BANDS = {
    "recall@20": ((0.143, 0.158), [0.144, 0.150, 0.156],
                  (0.100, 0.116), [0.102, 0.108, 0.114]),
    "ndcg@20": ((0.117, 0.134), [0.120, 0.126, 0.132],
                (0.075, 0.091), [0.078, 0.084, 0.090]),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.linewidth": 0.8,
    "axes.labelsize": 24, "xtick.labelsize": 20, "ytick.labelsize": 20,
    "legend.fontsize": 18, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_data() -> dict:
    data = {}
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["noise_mode"] != "injection":
                raise ValueError("Only injection-noise results are supported")
            key = (row["dataset"], row["model"], float(row["noise_ratio"]))
            if key in data:
                raise ValueError(f"Duplicate experiment condition: {key}")
            data[key] = {metric: float(row[metric]) for metric in METRICS}
    expected = {(dataset, model, ratio) for dataset in DATASETS
                for model in MODELS for ratio in RATIOS}
    if set(data) != expected:
        raise ValueError(f"Unexpected conditions: missing={expected - set(data)}, extra={set(data) - expected}")
    return data


def build_figure(data: dict, dataset: str, metric: str) -> plt.Figure:
    if dataset == "iFashion":
        return build_ifashion_figure(data, metric)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    all_values = []
    for model, (color, marker) in MODELS.items():
        values = [data[(dataset, model, ratio)][metric] for ratio in RATIOS]
        all_values.extend(values)
        ax.plot(RATIOS, values, label=model, color=color, marker=marker,
                linestyle="--", linewidth=3, markersize=10)

    low, high = min(all_values), max(all_values)
    span = high - low
    # Reserve space beneath the curves for a consistent legend position.
    ax.set_ylim(low - span * 0.48, high + span * 0.14)
    ax.set_xlim(-0.01, 0.26)
    ax.set_xticks(RATIOS, ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"])
    ax.set_xlabel("Noise Ratio")
    ax.set_ylabel(METRICS[metric][0])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, color="#b0b0b0", linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=3, fontsize=16, frameon=True,
              fancybox=True, shadow=True, framealpha=0.95,
              columnspacing=0.8, handlelength=1.8, handletextpad=0.5)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.17, top=0.97)
    return fig


def build_ifashion_figure(data: dict, metric: str) -> plt.Figure:
    top_range, top_ticks, bottom_range, bottom_ticks = IFASHION_BANDS[metric]
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 5), dpi=100,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06},
    )
    for ax in (top, bottom):
        for model, (color, marker) in MODELS.items():
            values = [data[("iFashion", model, ratio)][metric] for ratio in RATIOS]
            ax.plot(RATIOS, values, label=model, color=color, marker=marker,
                    linestyle="--", linewidth=3, markersize=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.grid(True, color="#b0b0b0", linewidth=0.8, alpha=1.0)
        ax.set_axisbelow(True)
    top.set_ylim(*top_range)
    top.set_yticks(top_ticks)
    bottom.set_ylim(*bottom_range)
    bottom.set_yticks(bottom_ticks)
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(labelbottom=False, bottom=False)
    bottom.set_xlim(-0.01, 0.26)
    bottom.set_xticks(RATIOS, ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"])
    bottom.set_xlabel("Noise Ratio")
    fig.text(0.025, 0.55, METRICS[metric][0], va="center", rotation="vertical", fontsize=24)

    marks = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=10,
                 linestyle="none", color="black", mec="black", mew=1.0, clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **marks)
    bottom.plot([0, 1], [1, 1], transform=bottom.transAxes, **marks)
    bottom.legend(loc="lower left", ncol=3, fontsize=16, frameon=True,
                  fancybox=True, shadow=True, framealpha=0.95,
                  columnspacing=0.8, handlelength=1.8, handletextpad=0.5)
    fig.subplots_adjust(left=0.21, right=0.975, bottom=0.17, top=0.97)
    return fig


def main() -> None:
    data = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for metric, (_, suffix) in METRICS.items():
            fig = build_figure(data, dataset, metric)
            stem = f"injection_{dataset.lower()}_{suffix}"
            fig.savefig(OUTPUT_DIR / f"{stem}_preview.png", dpi=100, facecolor="white")
            fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, facecolor="white")
            fig.savefig(OUTPUT_DIR / f"{stem}.pdf", facecolor="white")
            plt.close(fig)
            print(stem)


if __name__ == "__main__":
    main()
