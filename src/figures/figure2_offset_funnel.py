"""Figure 2 right panel: text-vs-score-gap funnel.

Post-rebuttal score-increase rate by Gemini offset bucket
(under / match / over), combined ICLR 2024+2025, with per-year
markers overlaid to show cross-year replication.

Run:
    python src/figures/figure2_offset_funnel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "data" / "results" / "overestimation_analysis" / "tables" / "funnel_stats.csv"
OUT_DIR = REPO_ROOT / "data" / "results" / "gemini_prediction" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKETS = [
    ("Under", "under", "#5B86B5"),
    ("Match", "match", "#7A7A7A"),
    ("Over", "over", "#D88C5A"),
]
YEAR_MARKERS = {2024: ("o", "#56B4E9"), 2025: ("s", "#009E73")}


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    _setup_rc()
    df = pd.read_csv(SRC)
    df = df[df["year"].isin([2024, 2025])].copy()

    combined_n = {b: int(df[f"{key}_n"].sum()) for _, key, _ in BUCKETS for b in [key]}
    combined_inc = {
        b: int(df[f"{key}_increased"].sum()) for _, key, _ in BUCKETS for b in [key]
    }
    combined_rate = {b: 100.0 * combined_inc[b] / combined_n[b] for b in combined_n}

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    x = list(range(len(BUCKETS)))
    bar_h = [combined_rate[key] for _, key, _ in BUCKETS]
    bar_colors = [color for _, _, color in BUCKETS]
    bars = ax.bar(
        x,
        bar_h,
        width=0.62,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    for i, (label, key, _) in enumerate(BUCKETS):
        ax.text(
            i,
            combined_rate[key] + 1.0,
            f"{combined_rate[key]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#4F4F4F",
        )
        ax.text(
            i,
            -3.6,
            f"n={combined_n[key]:,}",
            ha="center",
            va="top",
            fontsize=8,
            color="#7A7A7A",
        )

    for year, (marker, color) in YEAR_MARKERS.items():
        row = df[df["year"] == year].iloc[0]
        per_year = [row[f"{key}_increase_rate_pct"] for _, key, _ in BUCKETS]
        ax.plot(
            x,
            per_year,
            marker=marker,
            markersize=6,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"ICLR {year}",
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in BUCKETS], fontsize=10)
    ax.set_ylabel("Post-rebuttal score-increase rate (%)", fontsize=10)
    ax.set_xlabel("Gemini text-vs-score offset bucket", fontsize=10)
    ax.set_ylim(0, max(bar_h) * 1.25)
    ax.set_title(
        "Increase rate rises monotonically with offset",
        fontsize=11,
        fontweight="bold",
        color="#4F4F4F",
        pad=14,
    )
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=8,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        title="Per-year replication",
        title_fontsize=8,
    )
    legend.get_title().set_color("#4F4F4F")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"funnel_increase_rate.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
