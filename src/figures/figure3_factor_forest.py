"""Figure 3: 44-factor codebook validation forest.

Test 1 (in-sample, ICLR 2025, n=1,000) and Test 2 (held-out, ICLR 2024,
n=952) odds ratios side-by-side per factor, with the 16 doubly-Bonferroni-
significant factors highlighted as the conservative core.

Run:
    python src/figures/figure3_factor_forest.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "data" / "results" / "factor_validation"
OUT_DIR = SRC / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_BONF = 0.05 / 44

T1_COLOR = "#56B4E9"
T2_COLOR = "#D88C5A"
GRID = "#E5E5E5"
DIM = "#999999"
DARK = "#4F4F4F"


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
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


def _load() -> pd.DataFrame:
    t1 = pd.read_csv(SRC / "test1_same1k_factor_tests.csv")
    t2 = pd.read_csv(SRC / "test2_full2024_factor_tests.csv")
    keep = ["factor_id", "polarity", "label", "odds_ratio", "or_ci_lo", "or_ci_hi", "perm_p"]
    df = t1[keep].merge(
        t2[["factor_id", "odds_ratio", "or_ci_lo", "or_ci_hi", "perm_p"]],
        on="factor_id",
        suffixes=("_t1", "_t2"),
    )
    df["bonf_t1"] = df["perm_p_t1"] < ALPHA_BONF
    df["bonf_t2"] = df["perm_p_t2"] < ALPHA_BONF
    df["doubly_sig"] = df["bonf_t1"] & df["bonf_t2"]

    def _label(row: pd.Series) -> str:
        text = row["label"].rstrip(".").strip()
        if len(text) > 60:
            text = text[:57].rstrip() + "..."
        return f"{row['factor_id']}  {text}"

    df["display_label"] = df.apply(_label, axis=1)
    return df


def main() -> None:
    _setup_rc()
    df = _load()

    pos = df[df["polarity"] == "positive"].sort_values("odds_ratio_t1", ascending=True)
    neg = df[df["polarity"] == "negative"].sort_values("odds_ratio_t1", ascending=False)

    n_pos, n_neg = len(pos), len(neg)
    spacer = 1.5
    pos_y = np.arange(n_pos, dtype=float) + n_neg + spacer
    neg_y = np.arange(n_neg, dtype=float)
    df_pos = pos.assign(y=pos_y)
    df_neg = neg.assign(y=neg_y)
    plot_df = pd.concat([df_neg, df_pos])

    fig, ax = plt.subplots(figsize=(7.6, 10.8))

    or_min, or_max = 0.05, 30.0
    ax.axvline(1.0, color=DIM, linewidth=0.8, linestyle="-", zorder=1)
    ax.axvspan(or_min, 1.0, color="#FAF1E6", alpha=0.6, zorder=0)
    ax.axvspan(1.0, or_max, color="#EAF2F8", alpha=0.6, zorder=0)

    offset = 0.18
    for _, row in plot_df.iterrows():
        for which, color, marker, dy in (
            ("t1", T1_COLOR, "o", offset),
            ("t2", T2_COLOR, "s", -offset),
        ):
            yi = row["y"] + dy
            mfc = color if row["doubly_sig"] else "white"
            mec = color
            ax.errorbar(
                row[f"odds_ratio_{which}"],
                yi,
                xerr=[
                    [row[f"odds_ratio_{which}"] - row[f"or_ci_lo_{which}"]],
                    [row[f"or_ci_hi_{which}"] - row[f"odds_ratio_{which}"]],
                ],
                fmt=marker,
                markersize=4.6,
                markerfacecolor=mfc,
                markeredgecolor=mec,
                markeredgewidth=0.9,
                color=color,
                elinewidth=0.7,
                capsize=0,
                alpha=0.95 if row["doubly_sig"] else 0.55,
                zorder=3,
            )

    ax.set_yticks(plot_df["y"].tolist())
    labels = []
    weights = []
    for _, row in plot_df.iterrows():
        labels.append(row["display_label"])
        weights.append("bold" if row["doubly_sig"] else "normal")
    ax.set_yticklabels(labels)
    for tick, w in zip(ax.get_yticklabels(), weights):
        tick.set_fontweight(w)
        tick.set_color(DARK if w == "bold" else "#777777")
        tick.set_fontsize(7.2)

    ax.set_xscale("log")
    ax.set_xlim(or_min, or_max)
    xticks = [0.1, 0.25, 0.5, 1, 2, 4, 10, 25]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=8)
    ax.set_xlabel("Odds ratio (presence of factor in increase vs same/decrease)", fontsize=9)

    ymid = n_neg + spacer / 2
    ax.axhline(ymid, color=DIM, linewidth=0.5, linestyle=":")
    ax.text(
        or_min * 1.5,
        n_neg + spacer + n_pos - 0.4,
        "Positive polarity",
        fontsize=9,
        fontweight="bold",
        color="#3F6796",
    )
    ax.text(
        or_min * 1.5,
        n_neg - 0.4,
        "Negative polarity",
        fontsize=9,
        fontweight="bold",
        color="#A05A2C",
    )

    ax.grid(axis="x", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.0, n_neg + spacer + n_pos)

    test1_handle = plt.Line2D(
        [0], [0], marker="o", linestyle="", markersize=6,
        markerfacecolor=T1_COLOR, markeredgecolor=T1_COLOR, markeredgewidth=0.9,
        label="Test 1: in-sample ICLR 2025 (n=1,000)",
    )
    test2_handle = plt.Line2D(
        [0], [0], marker="s", linestyle="", markersize=6,
        markerfacecolor=T2_COLOR, markeredgecolor=T2_COLOR, markeredgewidth=0.9,
        label="Test 2: held-out ICLR 2024 (n=5,730)",
    )
    n_doubly = int(df["doubly_sig"].sum())
    bonf_handle = plt.Line2D(
        [0], [0], marker="o", linestyle="", markersize=6,
        markerfacecolor=DARK, markeredgecolor=DARK, markeredgewidth=0.9,
        label=f"Bonferroni-significant in BOTH tests ($\\alpha = 0.05/44$); {n_doubly} of 44",
    )
    legend = ax.legend(
        handles=[test1_handle, test2_handle, bonf_handle],
        loc="lower right",
        fontsize=7.6,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        ncol=1,
    )
    legend.get_frame().set_linewidth(0.5)

    ax.set_title(
        "44-factor codebook: cross-model and cross-year validation",
        fontsize=10.5,
        fontweight="bold",
        color=DARK,
        loc="left",
        pad=10,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"factor_validation_forest.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
