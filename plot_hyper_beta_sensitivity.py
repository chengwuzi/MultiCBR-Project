"""Generate the diffusion noise strength sensitivity chart for Recall@20.

The visual settings follow ``plot_line_demo.py`` as the frozen reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------------
# DATA: diffusion noise strength sensitivity, Recall@20.
# ---------------------------------------------------------------------------
X = np.arange(6)
X_TICK_LABELS = ["0.005", "0.010", "0.020", "0.040", "0.060", "0.080"]

SERIES = [
    {
        "label": "Youshu",
        "values": np.array([0.28568, 0.28609, 0.28926, 0.28833, 0.28631, 0.28684]),
        "color": "#1f77b4",
        "marker": "^",
    },
    {
        "label": "iFashion",
        "values": np.array([0.15647, 0.15650, 0.15657, 0.15648, 0.15640, 0.15629]),
        "color": "#2ca02c",
        "marker": "o",
    },
    {
        "label": "NetEase",
        "values": np.array([0.09401, 0.09423, 0.09455, 0.09537, 0.09482, 0.09438]),
        "color": "#ff7f0e",
        "marker": "v",
    },
]

Y_LABEL = "Recall@20"
X_LABEL = r"Diffusion Noise Strength $\beta_T$"
OUTPUT_STEM = "hyper_beta_end_recall_at_20"

Y_AXIS_BANDS = [
    {
        "ylim": (0.2853, 0.2897),
        "yticks": [0.2860, 0.2880],
    },
    {
        "ylim": (0.1562, 0.1567),
        "yticks": [0.1563, 0.1565, 0.1567],
    },
    {
        "ylim": (0.0938, 0.0956),
        "yticks": [0.0940, 0.0950],
    },
]


# ---------------------------------------------------------------------------
# STYLE: calibrated to the frozen line chart reference.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.8,
        "axes.labelsize": 23,
        "xtick.labelsize": 18,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def validate_data() -> None:
    for item in SERIES:
        if len(item["values"]) != len(X):
            raise ValueError(
                f"{item['label']!r} has {len(item['values'])} values, "
                f"but X contains {len(X)} coordinates."
            )


def plot_series(ax: plt.Axes) -> None:
    for item in SERIES:
        ax.plot(
            X,
            item["values"],
            color=item["color"],
            linestyle="--",
            linewidth=3,
            marker=item["marker"],
            markersize=9,
            label=item["label"],
        )


def add_break_marks(ax_upper: plt.Axes, ax_lower: plt.Axes) -> None:
    diagonal_kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=10,
        linestyle="none",
        color="black",
        mec="black",
        mew=1.0,
        clip_on=False,
    )
    ax_upper.plot([0, 1], [0, 0], transform=ax_upper.transAxes, **diagonal_kwargs)
    ax_lower.plot([0, 1], [1, 1], transform=ax_lower.transAxes, **diagonal_kwargs)


def build_figure() -> tuple[plt.Figure, list[plt.Axes]]:
    validate_data()

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(8, 5),
        dpi=100,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.15], "hspace": 0.06},
    )

    for ax, band in zip(axes, Y_AXIS_BANDS):
        plot_series(ax)
        ax.set_ylim(*band["ylim"])
        ax.set_yticks(band["yticks"])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        ax.grid(True, color="#b0b0b0", linewidth=0.8, alpha=1.0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)
    axes[2].spines["top"].set_visible(False)

    axes[0].tick_params(labelbottom=False, bottom=False)
    axes[1].tick_params(labelbottom=False, bottom=False)
    axes[2].set_xticks(X)
    axes[2].set_xticklabels(X_TICK_LABELS)
    axes[2].set_xlabel(X_LABEL)

    add_break_marks(axes[0], axes[1])
    add_break_marks(axes[1], axes[2])

    fig.text(0.035, 0.54, Y_LABEL, va="center", rotation="vertical", fontsize=24)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.215, right=0.978, bottom=0.16, top=0.82)
    return fig, list(axes)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures" / "hyperparameter_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = build_figure()
    preview_path = output_dir / f"{OUTPUT_STEM}_preview.png"
    paper_png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"

    fig.savefig(preview_path, dpi=100, facecolor="white")
    fig.savefig(paper_png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"Preview PNG: {preview_path}")
    print(f"600-DPI PNG: {paper_png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
