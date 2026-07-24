"""Generate paper-style line charts for Module 1 ablation results.

The visual settings follow ``plot_line_demo.py`` as the frozen reference.
This script intentionally keeps its own data and output paths so the original
reference script remains untouched.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------------
# DATA: iFashion Module 1 UB graph reconstruction ablation.
# ---------------------------------------------------------------------------
X = np.array([1, 3, 5, 7, 9])
X_TICK_LABELS = ["1", "3", "5", "7", "9"]

METHODS = [
    {
        "label": "RawUB",
        "plot_label": "Original UB",
        "color": "#7f7f7f",
        "marker": None,
        "linestyle": "--",
    },
    {
        "label": "Random",
        "plot_label": "Random-κ",
        "color": "#ff7f0e",
        "marker": "v",
        "linestyle": "--",
    },
    {
        "label": "Popularity",
        "plot_label": "Popularity-κ",
        "color": "#d62728",
        "marker": "D",
        "linestyle": "--",
    },
    {
        "label": "Pretrain Sim",
        "plot_label": "Pretrain-Sim",
        "color": "#2ca02c",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "label": "Diffusion",
        "plot_label": "DiGO",
        "color": "#1f77b4",
        "marker": "^",
        "linestyle": "--",
    },
]

METRICS = {
    "recall_at_10": {
        "ylabel": "Recall@10",
        "output_stem": "module1_ablation_recall_at_10",
        "legend_loc": "lower right",
        "values": {
            "RawUB": [0.10600, 0.10600, 0.10600, 0.10600, 0.10600],
            "Diffusion": [0.11079, 0.11451, 0.11469, 0.11347, 0.11163],
            "Random": [0.10678, 0.10966, 0.10901, 0.10830, 0.10787],
            "Pretrain Sim": [0.09827, 0.10297, 0.10543, 0.10547, 0.10696],
            "Popularity": [0.10848, 0.11061, 0.11130, 0.11019, 0.10763],
        },
    },
    "ndcg_at_10": {
        "ylabel": "NDCG@10",
        "output_stem": "module1_ablation_ndcg_at_10",
        "legend_loc": "lower right",
        "values": {
            "RawUB": [0.10449, 0.10449, 0.10449, 0.10449, 0.10449],
            "Diffusion": [0.10899, 0.11461, 0.11509, 0.11328, 0.11120],
            "Random": [0.10365, 0.10762, 0.10735, 0.10652, 0.10613],
            "Pretrain Sim": [0.09545, 0.10043, 0.10315, 0.10378, 0.10492],
            "Popularity": [0.10534, 0.10867, 0.10941, 0.10724, 0.10385],
        },
    },
    "recall_at_20": {
        "ylabel": "Recall@20",
        "output_stem": "module1_ablation_recall_at_20",
        "legend_loc": "lower right",
        "values": {
            "RawUB": [0.15035, 0.15035, 0.15035, 0.15035, 0.15035],
            "Diffusion": [0.15600, 0.15691, 0.15716, 0.15651, 0.15528],
            "Random": [0.15091, 0.15311, 0.15306, 0.15234, 0.15305],
            "Pretrain Sim": [0.14217, 0.14662, 0.14897, 0.14897, 0.14963],
            "Popularity": [0.15140, 0.15138, 0.15174, 0.15210, 0.14987],
        },
    },
    "ndcg_at_20": {
        "ylabel": "NDCG@20",
        "output_stem": "module1_ablation_ndcg_at_20",
        "legend_loc": "lower right",
        "values": {
            "RawUB": [0.12243, 0.12243, 0.12243, 0.12243, 0.12243],
            "Diffusion": [0.12727, 0.13160, 0.13213, 0.13056, 0.12878],
            "Random": [0.12161, 0.12524, 0.12512, 0.12431, 0.12441],
            "Pretrain Sim": [0.11333, 0.11812, 0.12080, 0.12142, 0.12220],
            "Popularity": [0.05785, 0.06017, 0.06076, 0.05925, 0.05599],
        },
    },
}


# ---------------------------------------------------------------------------
# STYLE: calibrated to the frozen line chart reference.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.8,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def validate_metric(metric_conf: dict) -> None:
    for method in METHODS:
        label = method["label"]
        values = metric_conf["values"][label]
        if len(values) != len(X):
            raise ValueError(
                f"{label!r} has {len(values)} values, "
                f"but X contains {len(X)} coordinates."
            )


def metric_ylim(metric_conf: dict) -> tuple[float, float]:
    values = np.array(
        [metric_conf["values"][method["label"]] for method in METHODS],
        dtype=float,
    )
    ymin = float(values.min())
    ymax = float(values.max())
    pad = max((ymax - ymin) * 0.18, 0.001)
    return ymin - pad, ymax + pad


def build_figure(metric_conf: dict) -> tuple[plt.Figure, plt.Axes]:
    validate_metric(metric_conf)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    for method in METHODS:
        label = method["label"]
        ax.plot(
            X,
            np.array(metric_conf["values"][label]),
            color=method["color"],
            linestyle=method["linestyle"],
            linewidth=3,
            marker=method["marker"],
            markersize=9,
            label=method.get("plot_label", label),
        )

    ax.set_xticks(X)
    ax.set_xticklabels(X_TICK_LABELS)
    ax.set_xlabel("Retention Number κ")
    ax.set_ylabel(metric_conf["ylabel"])
    ax.set_ylim(*metric_ylim(metric_conf))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.grid(True, color="#b0b0b0", linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax.legend(
        loc=metric_conf["legend_loc"],
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.19, right=0.978, bottom=0.16, top=0.97)
    return fig, ax


def save_metric_figure(metric_conf: dict, output_dir: Path) -> None:
    fig, _ = build_figure(metric_conf)
    output_stem = metric_conf["output_stem"]

    preview_path = output_dir / f"{output_stem}_preview.png"
    paper_png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"

    fig.savefig(preview_path, dpi=100, facecolor="white")
    fig.savefig(paper_png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"Preview PNG: {preview_path}")
    print(f"600-DPI PNG: {paper_png_path}")
    print(f"PDF: {pdf_path}")


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures" / "module1_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_conf in METRICS.values():
        save_metric_figure(metric_conf, output_dir)


if __name__ == "__main__":
    main()
