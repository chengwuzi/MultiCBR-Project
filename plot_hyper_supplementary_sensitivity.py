"""Generate supplementary hyperparameter sensitivity charts for Recall@20.

The two charts complement Figure 3 and use the same broken-y-axis visual
language: each horizontal band displays one dataset on its natural scale.
All values are the test-set Recall@20 results in ``图表数据/超参数实验数据.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------------
# DATA: ReCoBR supplementary hyperparameter sensitivity experiments.
# ---------------------------------------------------------------------------
CHARTS = [
    {
        "output_stem": "hyper_set_loss_lambda_recall_at_20",
        "x_tick_labels": ["0", "0.005", "0.010", "0.020", "0.040", "0.050", "0.100", "0.200"],
        "x_label": r"SetLoss Weight $\lambda_{\mathrm{set}}$",
        "series": [
            {
                "label": "Youshu",
                "values": [0.28309, 0.28499, 0.28809, 0.28849, 0.28811, 0.28855, 0.28950, 0.28733],
                "color": "#1f77b4",
                "marker": "^",
            },
            {
                "label": "iFashion",
                "values": [0.13818, 0.13839, 0.13820, 0.13841, 0.13819, 0.13804, 0.13786, 0.13781],
                "color": "#2ca02c",
                "marker": "o",
            },
            {
                "label": "NetEase",
                "values": [0.09459, 0.09509, 0.09484, 0.09497, 0.09474, 0.09496, 0.09514, 0.09472],
                "color": "#ff7f0e",
                "marker": "v",
            },
        ],
        "y_axis_bands": [
            {"ylim": (0.2815, 0.2900), "yticks": [0.2820, 0.2860, 0.2900]},
            {"ylim": (0.1371, 0.1386), "yticks": [0.1372, 0.1378, 0.1384]},
            {"ylim": (0.0942, 0.0953), "yticks": [0.0944, 0.0948, 0.0952]},
        ],
        "height_ratios": [1.15, 1.25, 1.0],
    },
    {
        "output_stem": "hyper_post_fusion_cl_lambda_recall_at_20",
        "x_tick_labels": ["0", "0.010", "0.020", "0.040", "0.050", "0.100", "0.200", "0.400"],
        "x_label": r"Post-fusion CL Weight $\lambda_c$",
        "series": [
            {
                "label": "Youshu",
                "values": [0.25734, 0.27648, 0.28468, 0.28465, 0.28163, 0.25263, 0.21051, 0.14373],
                "color": "#1f77b4",
                "marker": "^",
            },
            {
                "label": "iFashion",
                "values": [0.13876, 0.13987, 0.13985, 0.13912, 0.13899, 0.13797, 0.13628, 0.13249],
                "color": "#2ca02c",
                "marker": "o",
            },
            {
                "label": "NetEase",
                "values": [0.06645, 0.07901, 0.08500, 0.09572, 0.09468, 0.08947, 0.08293, 0.07045],
                "color": "#ff7f0e",
                "marker": "v",
            },
        ],
        "y_axis_bands": [
            {"ylim": (0.135, 0.292), "yticks": [0.150, 0.200, 0.250, 0.290]},
            {"ylim": (0.131, 0.141), "yticks": [0.132, 0.136, 0.140]},
            {"ylim": (0.064, 0.098), "yticks": [0.070, 0.080, 0.090]},
        ],
        "height_ratios": [1.35, 1.0, 1.0],
    },
]


# ---------------------------------------------------------------------------
# STYLE: calibrated to the original Figure 3 plotting scripts.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.8,
        "axes.labelsize": 23,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def validate_chart(chart: dict) -> None:
    point_count = len(chart["x_tick_labels"])
    if len(chart["series"]) != len(chart["y_axis_bands"]):
        raise ValueError("Each dataset requires one y-axis band.")
    for item in chart["series"]:
        if len(item["values"]) != point_count:
            raise ValueError(
                f"{item['label']!r} has {len(item['values'])} values, "
                f"but the chart has {point_count} x-axis positions."
            )


def plot_series(ax: plt.Axes, item: dict, x: np.ndarray) -> None:
    """Draw only the series represented by this y-axis band.

    The bands have independent value ranges.  Plotting one dataset per band
    prevents an outlying value from appearing in a neighbouring dataset's
    band, which can happen when value ranges overlap.
    """
    ax.plot(
        x,
        item["values"],
        color=item["color"],
        linestyle="--",
        linewidth=3,
        marker=item["marker"],
        markersize=8,
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


def build_figure(chart: dict) -> tuple[plt.Figure, list[plt.Axes]]:
    validate_chart(chart)
    x = np.arange(len(chart["x_tick_labels"]))

    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(8, 5),
        dpi=100,
        gridspec_kw={"height_ratios": chart["height_ratios"], "hspace": 0.06},
    )

    for ax, band, item in zip(axes, chart["y_axis_bands"], chart["series"]):
        plot_series(ax, item, x)
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

    axes[2].set_xlim(-0.20, len(x) - 0.75)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(chart["x_tick_labels"])
    axes[2].set_xlabel(chart["x_label"], labelpad=8)

    add_break_marks(axes[0], axes[1])
    add_break_marks(axes[1], axes[2])
    fig.text(0.035, 0.54, "Recall@20", va="center", rotation="vertical", fontsize=24)

    handles = [
        Line2D(
            [],
            [],
            color=item["color"],
            linestyle="--",
            linewidth=3,
            marker=item["marker"],
            markersize=8,
            label=item["label"],
        )
        for item in chart["series"]
    ]
    labels = [item["label"] for item in chart["series"]]
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


def save_chart(chart: dict, output_dir: Path) -> None:
    fig, _ = build_figure(chart)
    output_stem = chart["output_stem"]
    fig.savefig(output_dir / f"{output_stem}_preview.png", dpi=100, facecolor="white")
    fig.savefig(output_dir / f"{output_stem}.png", dpi=600, facecolor="white")
    fig.savefig(output_dir / f"{output_stem}.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures" / "hyperparameter_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    for chart in CHARTS:
        save_chart(chart, output_dir)
        print(f"Created: {chart['output_stem']}")


if __name__ == "__main__":
    main()
