"""Generate sparse-group style charts for Module 2 results.

The chart layout follows the sparse reference figures: grouped bars for
Baseline/Module2 on the left axis and a relative-improvement line on the right
axis.  The frozen ``plot_line_demo.py`` reference is not modified.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------------
# DATA: Module 2 user sparsity group results.
# ---------------------------------------------------------------------------
GROUPS = ["G1", "G2", "G3", "G4"]
X = np.arange(len(GROUPS))

CHARTS = {
    "recall_at_20": {
        "ylabel": "Recall@20",
        "output_stem": "module2_sparse_recall_at_20",
        "baseline": np.array([0.097661, 0.095531, 0.086572, 0.080803]),
        "module2": np.array([0.110004, 0.096718, 0.090008, 0.083627]),
        "improvement": np.array([12.64, 1.23, 3.97, 3.50]),
        "left_ylim": (0.075, 0.115),
        "left_yticks": [0.08, 0.09, 0.10, 0.11],
        "right_ylim": (0.0, 20.0),
        "right_yticks": [0, 5, 10, 15, 20],
        "label_x": [0.0, 1.0, 2.0, 3.0],
        "label_offsets": [0.25, 0.25, 0.25, 0.25],
    },
    "ndcg_at_20": {
        "ylabel": "NDCG@20",
        "output_stem": "module2_sparse_ndcg_at_20",
        "baseline": np.array([0.043600, 0.045538, 0.046702, 0.060717]),
        "module2": np.array([0.047664, 0.045780, 0.048415, 0.061845]),
        "improvement": np.array([9.32, 0.53, 3.67, 1.86]),
        "left_ylim": (0.040, 0.065),
        "left_yticks": [0.04, 0.05, 0.06],
        "right_ylim": (0.0, 14.0),
        "right_yticks": [0, 2, 4, 6, 8, 10, 12, 14],
        "label_x": [0.0, 1.0, 2.0, 3.0],
        "label_offsets": [0.25, 0.25, 0.25, 0.25],
    },
}


# ---------------------------------------------------------------------------
# STYLE: sparse reference uses large labels, white background, and twin axes.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.8,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


BAR_WIDTH = 0.34
BASELINE_COLOR = "#06f08b"
MODULE2_COLOR = "#4c87b9"
IMPROVEMENT_COLOR = "#ff7f50"


def build_figure(chart_conf: dict) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9.89, 5.89), dpi=100)
    ax2 = ax.twinx()

    bars_baseline = ax.bar(
        X - BAR_WIDTH / 2,
        chart_conf["baseline"],
        width=BAR_WIDTH,
        color=BASELINE_COLOR,
        label="MultiCBR",
        zorder=3,
    )
    bars_module2 = ax.bar(
        X + BAR_WIDTH / 2,
        chart_conf["module2"],
        width=BAR_WIDTH,
        color=MODULE2_COLOR,
        label="DiGO",
        zorder=3,
    )

    line = ax2.plot(
        X,
        chart_conf["improvement"],
        color=IMPROVEMENT_COLOR,
        marker="o",
        markersize=8,
        linewidth=4,
        label="Improvement",
        zorder=4,
    )[0]

    ax.set_xticks(X)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel(chart_conf["ylabel"])
    ax.set_ylim(*chart_conf["left_ylim"])
    ax.set_yticks(chart_conf["left_yticks"])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax2.set_ylabel("Percentage Difference (%)")
    ax2.set_ylim(*chart_conf["right_ylim"])
    ax2.set_yticks(chart_conf["right_yticks"])
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.grid(False)
    ax2.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    for idx, improvement in enumerate(chart_conf["improvement"]):
        ax2.text(
            chart_conf["label_x"][idx],
            improvement + chart_conf["label_offsets"][idx],
            f"{improvement:.2f}%",
            ha="center",
            va="bottom",
            fontsize=24,
            color="black",
        )

    legend1 = ax.legend(
        handles=[bars_baseline, bars_module2],
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )
    ax.add_artist(legend1)
    ax2.legend(
        handles=[line],
        loc="upper center",
        bbox_to_anchor=(0.59, 1.0),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )

    fig.subplots_adjust(left=0.16, right=0.86, bottom=0.12, top=0.955)
    return fig, ax


def save_chart(chart_conf: dict, output_dir: Path) -> None:
    fig, _ = build_figure(chart_conf)
    output_stem = chart_conf["output_stem"]

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
    output_dir = Path(__file__).resolve().parent / "figures" / "module2_sparse"
    output_dir.mkdir(parents=True, exist_ok=True)

    for chart_conf in CHARTS.values():
        save_chart(chart_conf, output_dir)


if __name__ == "__main__":
    main()
