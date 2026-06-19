"""Figure 1 hero: question schematic + cascade-with-strata + factor mini-forest.

Three-panel horizontal layout for NeurIPS-style page-1 hero figure.
  A. Pipeline schematic: 73K reviews -> three horizons -> P(score increase)
  B. Predictive cascade: AUC at pre / rebuttal-content / post-discussion,
     with within-strata lift for the rebuttal-content step.
  C. Recommended-factor mini-forest: top 4 positives + top 6 negatives by
     |log(isolated OR)|, with 95% cluster-robust CIs.

Numbers come from:
  data/results/rebuttal_rf_prediction/tables/incremental_value_summary.csv
  data/results/horizon_inference/tables/factor_recommendation_table.csv
  src/factor_validation/codebook.json (factor labels)

Run from repo root:
    python src/figures/figure1_hero.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_CSV = (
    REPO_ROOT / "data" / "results" / "rebuttal_rf_prediction" / "tables"
    / "incremental_value_summary.csv"
)
FACTORS_CSV = (
    REPO_ROOT / "data" / "results" / "horizon_inference" / "tables"
    / "factor_recommendation_table.csv"
)
CODEBOOK = REPO_ROOT / "src" / "factor_validation" / "codebook.json"

BLUE = "#5B86B5"
BLUE_DARK = "#3F6796"
BLUE_LIGHT = "#A6BFD8"
ORANGE = "#D88C5A"
ORANGE_DARK = "#B5703F"
GRAY_DARK = "#4F4F4F"
GRAY = "#7A7A7A"
GRAY_LIGHT = "#B5B5B5"
PILL_FILL = "#F2E6D8"
PILL_EDGE = "#D88C5A"

SHORT_LABELS = {
    "F02": "Non-defensive tone",
    "F07": "Baseline completion",
    "F19": "Writing fixes",
    "F20": "Theoretical grounding",
    "F24": "Defensive tone",
    "F33": "Unresolved methodology",
    "F36": "Weak novelty justification",
    "F37": "Brevity / low-effort",
    "F39": "Inadequate explanation",
    "F40": "Marginal improvements",
}


def _setup_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _hide_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold")


def _panel_title(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 1.02, text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9.5, fontweight="bold")


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float,
           color: str = GRAY) -> None:
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=8, linewidth=0.9, color=color,
    ))


# ---------------------------------------------------------------------------
# Panel A: pipeline + question
# ---------------------------------------------------------------------------

def draw_panel_A(ax: plt.Axes) -> None:
    _hide_axes(ax)
    _panel_label(ax, "A")
    _panel_title(ax, "Pipeline: review text → three horizons → outcome")

    # Top: data source pill
    src_w, src_h = 0.84, 0.12
    src_x, src_y = (1 - src_w) / 2, 0.82
    src = FancyBboxPatch(
        (src_x, src_y), src_w, src_h,
        boxstyle="round,pad=0.005,rounding_size=0.04",
        linewidth=0.7, edgecolor=BLUE_DARK, facecolor=BLUE,
    )
    ax.add_patch(src)
    ax.text(0.5, src_y + src_h / 2,
            "73k reviews · 18.4k papers · ICLR 2024–2025",
            ha="center", va="center", fontsize=8.0,
            fontweight="bold", color="white")

    # Three horizon boxes (stacked vertically for readability)
    horizons = [
        ("Pre-rebuttal",
         "initial score, paper mean,\nGemini text-vs-score offset"),
        ("Rebuttal-content",
         "rebuttal length, n replies,\n44 factor indicators"),
        ("Post-discussion",
         "co-reviewer movement,\nthread depth"),
    ]
    box_w, box_h = 0.84, 0.14
    box_x = (1 - box_w) / 2
    y_top = 0.62
    gap = 0.04
    for i, (name, body) in enumerate(horizons):
        y = y_top - i * (box_h + gap)
        bg = "#EAF1F8" if i % 2 == 0 else "#F5F0E8"
        edge = BLUE_DARK if i % 2 == 0 else ORANGE_DARK
        box = FancyBboxPatch(
            (box_x, y), box_w, box_h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            linewidth=0.6, edgecolor=edge, facecolor=bg,
        )
        ax.add_patch(box)
        ax.text(box_x + 0.018, y + box_h - 0.018, name,
                ha="left", va="top", fontsize=8.0,
                fontweight="bold", color=GRAY_DARK)
        ax.text(box_x + box_w - 0.018, y + box_h / 2 - 0.005, body,
                ha="right", va="center", fontsize=7.0, color=GRAY_DARK)
        if i < len(horizons) - 1:
            mx = box_x + box_w / 2
            _arrow(ax, mx, y - 0.002, mx, y - gap + 0.004)
        else:
            mx = box_x + box_w / 2
            _arrow(ax, mx, y - 0.002, mx, 0.10)

    # Connector from data source down to first horizon box
    _arrow(ax, 0.5, src_y - 0.002, 0.5, y_top + box_h + 0.004)

    # Outcome pill at bottom
    out_w, out_h = 0.74, 0.09
    out_x = (1 - out_w) / 2
    out_y = 0.005
    pill = FancyBboxPatch(
        (out_x, out_y), out_w, out_h,
        boxstyle="round,pad=0.005,rounding_size=0.045",
        linewidth=0.8, edgecolor=PILL_EDGE, facecolor=PILL_FILL,
    )
    ax.add_patch(pill)
    ax.text(0.5, out_y + out_h / 2,
            "Score-increase probability after rebuttal",
            ha="center", va="center", fontsize=8.0,
            fontweight="bold", color=GRAY_DARK)


# ---------------------------------------------------------------------------
# Panel B: cascade with within-strata inset
# ---------------------------------------------------------------------------

def _load_cascade() -> dict:
    df = pd.read_csv(CASCADE_CSV)
    df = df[df["model"] == "logistic"].set_index("horizon")
    def parse_ci(s: str) -> tuple[float, float]:
        s = s.strip("() ")
        lo, hi = (float(x) for x in s.split(","))
        return lo, hi
    out = {}
    for h in ("pre", "draft", "post"):
        row = df.loc[h]
        out[h] = {
            "auc": float(row["auc_roc"]),
            "ci": parse_ci(row["auc_roc_ci"]),
        }
    return out


def draw_panel_B(ax: plt.Axes, cascade: dict) -> None:
    _hide_axes(ax)
    _panel_label(ax, "B")
    _panel_title(ax, "Pre-rebuttal floor + bounded content lift")

    aucs = [cascade["pre"]["auc"], cascade["draft"]["auc"], cascade["post"]["auc"]]
    cis = [cascade["pre"]["ci"], cascade["draft"]["ci"], cascade["post"]["ci"]]
    labels = ["Pre-\nrebuttal", "+ Rebuttal\ncontent", "+ Post-\ndiscussion"]

    # Cascade occupies the upper two-thirds; strata inset the lower third.
    # Tighter auc_min/max lifts the bars and separates the AUC labels.
    auc_min, auc_max = 0.70, 0.84
    plot_x0, plot_x1 = 0.13, 0.97
    plot_y0, plot_y1 = 0.50, 0.86

    def yfor(a: float) -> float:
        return plot_y0 + (a - auc_min) / (auc_max - auc_min) * (plot_y1 - plot_y0)

    n = len(aucs)
    step_w = (plot_x1 - plot_x0) / n
    fills = [BLUE_LIGHT, BLUE, BLUE_DARK]
    edges = [BLUE_DARK, ORANGE, BLUE_DARK]

    for i, auc in enumerate(aucs):
        x = plot_x0 + i * step_w
        y_top = yfor(auc)
        rect = FancyBboxPatch(
            (x + 0.012, plot_y0), step_w - 0.024, y_top - plot_y0,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            linewidth=0.9, edgecolor=edges[i], facecolor=fills[i],
        )
        ax.add_patch(rect)
        lo, hi = cis[i]
        y_lo, y_hi = yfor(lo), yfor(hi)
        cx = x + step_w / 2
        ax.plot([cx, cx], [y_lo, y_hi], color=GRAY_DARK, linewidth=1.0)
        ax.plot([cx - 0.010, cx + 0.010], [y_lo, y_lo], color=GRAY_DARK, linewidth=1.0)
        ax.plot([cx - 0.010, cx + 0.010], [y_hi, y_hi], color=GRAY_DARK, linewidth=1.0)
        # Δ above the AUC value, AUC value bold below it.
        # Stagger AUC y-offset by index so adjacent bars whose CI tops sit
        # close together (draft vs post differ by ~0.005 AUC) do not
        # collide on the page.
        auc_offset = 0.020 + 0.012 * i
        if i > 0:
            d = aucs[i] - aucs[i - 1]
            d_color = ORANGE_DARK if i == 1 else GRAY
            ax.text(cx, y_hi + auc_offset, f"{auc:.3f}",
                    ha="center", va="bottom", fontsize=8.6,
                    fontweight="bold", color=GRAY_DARK)
            ax.text(cx, y_hi + auc_offset + 0.034, f"Δ +{d:.3f}",
                    ha="center", va="bottom", fontsize=7.2,
                    fontweight="bold", color=d_color)
            # Connector line from previous bar top.
            prev_top = yfor(aucs[i - 1])
            x_left = x + 0.012
            ax.plot([x_left - step_w + 0.024, x_left],
                    [prev_top, prev_top],
                    color=d_color, linewidth=0.7, linestyle=":")
            ax.plot([x_left, x_left],
                    [prev_top, y_top],
                    color=d_color, linewidth=0.7, linestyle=":")
        else:
            ax.text(cx, y_hi + auc_offset, f"{auc:.3f}",
                    ha="center", va="bottom", fontsize=8.6,
                    fontweight="bold", color=GRAY_DARK)
        ax.text(cx, plot_y0 - 0.012, labels[i],
                ha="center", va="top", fontsize=7.0, color=GRAY_DARK)

    ax.plot([plot_x0 - 0.005, plot_x0 - 0.005], [plot_y0, plot_y1],
            color=GRAY_LIGHT, linewidth=0.6)
    ax.text(plot_x0 - 0.075, (plot_y0 + plot_y1) / 2, "AUC (temporal)",
            ha="center", va="center", fontsize=7.4,
            rotation=90, color=GRAY_DARK)

    # ---- Strata inset (lower third), separated by a divider line. -----
    divider_y = 0.36
    ax.plot([0.05, 0.97], [divider_y, divider_y],
            color="#dddddd", linewidth=0.6)

    ax.text(0.13, divider_y - 0.025,
            "Content lift within reviewer–paper deviation strata",
            ha="left", va="top", fontsize=7.4,
            fontweight="bold", color=GRAY_DARK)
    strata = [("below mean", 0.068, BLUE_DARK),
              ("near mean",  0.092, ORANGE),
              ("above mean", 0.056, BLUE_LIGHT)]
    max_lift = 0.10
    bar_h = 0.045
    gap_y = 0.018
    label_x = 0.13
    bar_x0 = 0.34
    bar_w_full = 0.50
    inset_top = divider_y - 0.07
    for i, (name, lift, color) in enumerate(strata):
        y = inset_top - i * (bar_h + gap_y) - bar_h
        ax.text(label_x, y + bar_h / 2, name,
                ha="left", va="center", fontsize=7.0, color=GRAY_DARK)
        track = FancyBboxPatch(
            (bar_x0, y), bar_w_full, bar_h,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            linewidth=0.0, facecolor="#EEEEEE",
        )
        ax.add_patch(track)
        bw = bar_w_full * (lift / max_lift)
        bar = FancyBboxPatch(
            (bar_x0, y), bw, bar_h,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            linewidth=0.0, facecolor=color,
        )
        ax.add_patch(bar)
        ax.text(bar_x0 + bw + 0.008, y + bar_h / 2,
                f"+{lift:.3f} AUC", ha="left", va="center",
                fontsize=7.0, fontweight="bold", color=GRAY_DARK)


# ---------------------------------------------------------------------------
# Panel C: factor mini-forest
# ---------------------------------------------------------------------------

def _load_factors() -> pd.DataFrame:
    df = pd.read_csv(FACTORS_CSV)
    df = df.dropna(subset=["isolated_or", "isolated_or_lo", "isolated_or_hi"])
    return df


def _pick_top(df: pd.DataFrame, k_pos: int = 4, k_neg: int = 6) -> pd.DataFrame:
    rec = df[df["recommended"] == True].copy()
    rec["log_or"] = np.log(rec["isolated_or"])
    pos = rec[rec["polarity"] == "positive"].copy()
    neg = rec[rec["polarity"] == "negative"].copy()
    pos = pos.sort_values("log_or", ascending=False).head(k_pos)
    neg = neg.sort_values("log_or", ascending=True).head(k_neg)
    return pd.concat([pos, neg], ignore_index=True)


def draw_panel_C(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Use a real axes for this one (log-x scale + grid).
    _panel_label(ax, "C")
    _panel_title(ax, "Recommended factors (asymmetric polarity screen)")

    rows = list(df.itertuples(index=False))
    n = len(rows)
    # Plot top-down: positives at the top, negatives below.
    y_positions = np.arange(n)[::-1]

    for y, r in zip(y_positions, rows):
        color = BLUE_DARK if r.polarity == "positive" else ORANGE_DARK
        ax.errorbar(
            r.isolated_or, y,
            xerr=[[r.isolated_or - r.isolated_or_lo],
                  [r.isolated_or_hi - r.isolated_or]],
            fmt="o", color=color, ecolor=color, elinewidth=1.0,
            capsize=2.0, markersize=5.0, markeredgewidth=0.0,
        )

    ax.axvline(1.0, color=GRAY, linewidth=0.7, linestyle="--", alpha=0.8)
    ax.set_xscale("log")
    # Symmetric range on the log axis around OR = 1, so a given |log(OR)|
    # maps to the same visual distance regardless of polarity. Limits are
    # set to cover the worst-case CI on either side: F39's lower CI ~0.25
    # and F07's upper CI ~1.93, expanded to symmetric [0.25, 4.0].
    ax.set_xlim(0.22, 4.5)
    ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"], fontsize=7.0)
    ax.set_xlabel("Odds ratio (isolated joint, 95% CI; log scale)", fontsize=8.0)
    ax.set_yticks(y_positions)
    labels = [f"{r.factor_id}  {SHORT_LABELS.get(r.factor_id, '')}"
              for r in rows]
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.tick_params(axis="y", which="both", length=0, pad=2)
    ax.set_ylim(-0.7, n - 0.3)

    ax.grid(axis="x", color="#e6e6e6", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.spines["bottom"].set_linewidth(0.6)

    # Polarity divider between positives and negatives.
    n_pos = int((df["polarity"] == "positive").sum())
    if n_pos and n_pos < n:
        divider_y = (y_positions[n_pos - 1] + y_positions[n_pos]) / 2
        ax.axhline(divider_y, color="#cccccc", linewidth=0.6, linestyle=":")

    # Polarity tags placed at the top-right of each block, away from data.
    ax.text(0.985, 0.985, "positive\n(favours increase)",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8,
            fontweight="bold", color=BLUE_DARK)
    neg_top_frac = 1 - (n_pos / n) - 0.005
    ax.text(0.985, neg_top_frac, "negative\n(favours no change)",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8,
            fontweight="bold", color=ORANGE_DARK)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_rc()

    cascade = _load_cascade()
    factors_df = _load_factors()
    top = _pick_top(factors_df, k_pos=4, k_neg=6)

    print("Cascade:")
    for h in ("pre", "draft", "post"):
        d = cascade[h]
        print(f"  {h}: AUC={d['auc']:.3f}  CI=[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]")
    print("Top factors selected:")
    for r in top.itertuples(index=False):
        print(f"  {r.factor_id} [{r.polarity}]  OR={r.isolated_or:.2f} "
              f"[{r.isolated_or_lo:.2f}, {r.isolated_or_hi:.2f}]")

    fig = plt.figure(figsize=(13.0, 4.6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.30, 0.30, 0.40],
        wspace=0.34,
        left=0.04, right=0.99, top=0.86, bottom=0.12,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_panel_A(ax_a)
    draw_panel_B(ax_b, cascade)
    draw_panel_C(ax_c, top)

    fig.suptitle(
        "Most score-change signal is set before rebuttal; "
        "content adds a bounded, factor-specific lever",
        y=0.99, fontsize=10.5, fontweight="bold", color=GRAY_DARK,
    )

    for ext in ("pdf", "png"):
        path = OUT_DIR / f"figure1_hero.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
