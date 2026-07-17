"""Generate a paper-style line chart using replaceable mock data.

The visual settings are calibrated against the line-chart reference images in
``工作说明/参考图片``.  Replace only the arrays in the DATA section when real
experimental results are available.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------------
# DATA: replace these arrays with real experimental results later.
# ---------------------------------------------------------------------------
X = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])

SERIES = [
    {
        "label": "Ours",
        "values": np.array([0.0, -3.1, -6.0, -8.7, -11.9, -14.6]),
        "color": "#1f77b4",
        "marker": "^",
    },
    {
        "label": "DirectAU",
        "values": np.array([0.0, -4.3, -8.4, -12.7, -18.8, -25.9]),
        "color": "#ff7f0e",
        "marker": "v",
    },
    {
        "label": "LightGCN",
        "values": np.array([0.0, -11.8, -16.2, -19.5, -25.4, -30.9]),
        "color": "#2ca02c",
        "marker": "o",
    },
]

Y_LABEL = "Relative NDCG@20 (%)"
OUTPUT_STEM = "line_demo_noise_sensitivity"


# ---------------------------------------------------------------------------
# STYLE: keep this section stable so figures use one consistent visual system.
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
        "legend.fontsize": 20,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def validate_data() -> None:
    """Fail early if a replacement series does not match the x coordinates."""
    for item in SERIES:
        if len(item["values"]) != len(X):
            raise ValueError(
                f"{item['label']!r} has {len(item['values'])} values, "
                f"but X contains {len(X)} coordinates."
            )


def build_figure() -> tuple[plt.Figure, plt.Axes]:
    validate_data()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    for item in SERIES:
        ax.plot(
            X,
            item["values"],
            color=item["color"],
            linestyle="--",
            linewidth=3,
            marker=item["marker"],
            markersize=11,
            label=item["label"],
        )

    ax.set_xticks(X)
    ax.set_xticklabels(["0.0", "0.05", "0.1", "0.15", "0.2", "0.25"])
    ax.set_ylim(-50, 3)
    ax.set_yticks([0, -10, -20, -30, -40, -50])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(Y_LABEL)

    ax.grid(True, color="#b0b0b0", linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax.legend(
        loc="lower left",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.95,
    )

    # Fixed margins keep the preview at exactly 800 x 500 pixels while leaving
    # room for the large y-axis label used by the reference figures.
    fig.subplots_adjust(left=0.165, right=0.978, bottom=0.105, top=0.97)
    return fig, ax


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = build_figure()
    preview_path = output_dir / f"{OUTPUT_STEM}_preview.png"
    paper_png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"

    # The preview matches the 800 x 500 reference-image canvas.  The 600-DPI
    # PNG and vector PDF are the publication-ready exports.
    fig.savefig(preview_path, dpi=100, facecolor="white")
    fig.savefig(paper_png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"Preview PNG: {preview_path}")
    print(f"600-DPI PNG: {paper_png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
