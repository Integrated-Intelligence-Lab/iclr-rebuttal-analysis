"""
Gemini score prediction accuracy vs ground truth.

Compares Gemini's predicted reviewer scores against actual ratings,
using both the final rating and (where available) the initial pre-rebuttal
rating from external ground truth.

Produces:
  1. Confusion matrix heatmaps (reviewer-level)
  2. Paper-level scatter plots (mean predicted vs mean actual)
  3. Per-score accuracy breakdown
  4. Error / residual distribution
  5. Accuracy by decision category
  6. Summary statistics table (Pearson, Spearman, MAE, exact match %)

Usage:
    python src/analyze/gemini_prediction_accuracy.py               # all years
    python src/analyze/gemini_prediction_accuracy.py --year 2025   # single year
    python src/analyze/gemini_prediction_accuracy.py --no-save     # display only
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from core.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from core.plot_theme import YEAR_COLORS, setup_publication_theme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
FIGURES_DIR = RESULTS_DIR / "gemini_prediction" / "figures"
TABLES_DIR = RESULTS_DIR / "gemini_prediction" / "tables"

ICLR_SCALE = [1, 3, 5, 6, 8, 10]
SCORE_TO_POS = {s: i for i, s in enumerate(ICLR_SCALE)}


def score_to_step_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Convert raw score errors to category-step errors.

    E.g. actual=5, predicted=3 -> -1 step; actual=5, predicted=1 -> -2 steps.
    """
    pos_a = np.array([SCORE_TO_POS.get(int(a), np.nan) for a in actual])
    pos_p = np.array([SCORE_TO_POS.get(int(p), np.nan) for p in predicted])
    return pos_p - pos_a


def save_figure(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{name}.png"
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {png.name}, {pdf.name}")
    return png, pdf


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_year(year: int) -> pd.DataFrame | None:
    path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "gemini_pred_score" not in df.columns:
        return None
    df = df[df["gemini_pred_score"].notna() & df["rating"].notna()].copy()
    if len(df) == 0:
        return None
    df["gemini_pred_score"] = df["gemini_pred_score"].astype(int)
    df["rating"] = df["rating"].astype(int)
    df["year"] = year
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute suite of prediction accuracy metrics.

    Uses category steps (not raw point differences) for step-based metrics.
    """
    pearson_r, pearson_p = stats.pearsonr(actual, predicted)
    spearman_r, spearman_p = stats.spearmanr(actual, predicted)
    step_errors = score_to_step_error(actual, predicted)
    valid = ~np.isnan(step_errors)
    step_errors = step_errors[valid]
    return {
        "n": len(actual),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_r,
        "spearman_p": spearman_p,
        "mae_steps": np.mean(np.abs(step_errors)),
        "bias_steps": np.mean(step_errors),
        "exact_match_pct": np.mean(step_errors == 0) * 100,
        "within_1_step_pct": np.mean(np.abs(step_errors) <= 1) * 100,
        "within_2_steps_pct": np.mean(np.abs(step_errors) <= 2) * 100,
    }


def print_metrics(label: str, m: dict):
    print(f"\n  {label}:")
    print(f"    n={m['n']:,}  |  Pearson r={m['pearson_r']:.3f} (p={m['pearson_p']:.1e})"
          f"  |  Spearman rho={m['spearman_rho']:.3f} (p={m['spearman_p']:.1e})")
    print(f"    MAE={m['mae_steps']:.2f} steps  |  Bias={m['bias_steps']:+.2f} steps")
    print(f"    Exact match: {m['exact_match_pct']:.1f}%  |  Within 1 step: {m['within_1_step_pct']:.1f}%"
          f"  |  Within 2 steps: {m['within_2_steps_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _cmap():
    return LinearSegmentedColormap.from_list(
        "white_blue", ["#ffffff", "#d0e4f7", "#56B4E9", "#1a6b99"]
    )


def plot_confusion_heatmap(
    actual: np.ndarray, predicted: np.ndarray,
    title: str, ax: plt.Axes | None = None,
    normalize: str = "row",
) -> plt.Axes:
    """Confusion matrix heatmap on the ICLR scale."""
    scores = sorted(set(ICLR_SCALE) & (set(actual) | set(predicted)))

    # Build matrix
    mat = np.zeros((len(scores), len(scores)))
    score_idx = {s: i for i, s in enumerate(scores)}
    for a, p in zip(actual, predicted):
        if a in score_idx and p in score_idx:
            mat[score_idx[a], score_idx[p]] += 1

    # Normalize
    if normalize == "row":
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat_pct = mat / row_sums * 100
    else:
        mat_pct = mat / mat.sum() * 100

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))

    im = ax.imshow(mat_pct, cmap=_cmap(), vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(scores)
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(scores)
    ax.set_xlabel("Gemini predicted score")
    ax.set_ylabel("Actual score")
    ax.set_title(title)

    for i in range(len(scores)):
        for j in range(len(scores)):
            val = mat_pct[i, j]
            count = int(mat[i, j])
            if val >= 0.05:
                color = "white" if val > 50 else "black"
                if normalize == "row":
                    text = f"{val:.1f}%\n({count:,})"
                else:
                    text = f"{val:.1f}%"
                ax.text(j, i, text, ha="center", va="center",
                        color=color, fontsize=6.8, linespacing=0.95)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% of actual score" if normalize == "row" else "% of total", fontsize=9)
    return ax


def plot_paper_scatter(
    df: pd.DataFrame, actual_col: str, title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of paper-level mean predicted vs mean actual."""
    paper = df.groupby("paper_key").agg(
        mean_pred=("gemini_pred_score", "mean"),
        mean_actual=(actual_col, "mean"),
        decision=("decision", "first"),
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Color by decision
    decision_colors = {
        "Oral": "#1a6b3c",
        "Spotlight": "#2d9e5e",
        "Poster": "#6cc88a",
        "Rejected": "#d9534f",
        "Withdrawn": "#999999",
    }

    for dec in ["Rejected", "Poster", "Spotlight", "Oral", "Withdrawn"]:
        mask = paper["decision"] == dec
        if mask.any():
            ax.scatter(
                paper.loc[mask, "mean_actual"],
                paper.loc[mask, "mean_pred"],
                c=decision_colors.get(dec, "#666"),
                label=dec, alpha=0.4, s=12, edgecolors="none",
            )

    # Diagonal
    lims = [0, 11]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    # Regression line
    r, p = stats.pearsonr(paper["mean_actual"], paper["mean_pred"])
    slope, intercept = np.polyfit(paper["mean_actual"], paper["mean_pred"], 1)
    xs = np.linspace(paper["mean_actual"].min(), paper["mean_actual"].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#E69F00", linewidth=1.5, alpha=0.8)

    ax.set_xlabel(f"Actual mean score ({actual_col})")
    ax.set_ylabel("Gemini predicted mean score")
    ax.set_title(f"{title}\nr={r:.3f}, n={len(paper):,} papers")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_aspect("equal")

    return ax


def plot_error_distribution(
    step_errors: np.ndarray, title: str, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of prediction errors in category steps."""
    step_errors = step_errors[~np.isnan(step_errors)].astype(int)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    # Count per step value
    from collections import Counter
    counts_map = Counter(step_errors)
    steps = sorted(counts_map.keys())
    counts = [counts_map[s] for s in steps]

    colors = []
    for s in steps:
        if s < 0:
            colors.append("#d9534f")
        elif s == 0:
            colors.append("#56B4E9")
        else:
            colors.append("#E69F00")

    bars = ax.bar(range(len(steps)), counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{s:+d}" for s in steps])

    # Annotate percentages
    total = len(step_errors)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        if pct >= 0.1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Prediction error (category steps: Gemini - Actual)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    return ax


def plot_accuracy_by_decision(df: pd.DataFrame, actual_col: str, title: str) -> plt.Figure:
    """Bar chart: MAE and exact match % grouped by decision."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    order = ["Oral", "Spotlight", "Poster", "Rejected"]
    decision_colors = {
        "Oral": "#1a6b3c", "Spotlight": "#2d9e5e",
        "Poster": "#6cc88a", "Rejected": "#d9534f",
    }

    records = []
    for dec in order:
        sub = df[df["decision"] == dec]
        if len(sub) < 10:
            continue
        m = compute_metrics(sub[actual_col].values, sub["gemini_pred_score"].values)
        m["decision"] = dec
        m["color"] = decision_colors.get(dec, "#666")
        records.append(m)

    if not records:
        return fig

    decs = [r["decision"] for r in records]
    x = range(len(decs))

    # MAE (in steps)
    maes = [r["mae_steps"] for r in records]
    colors = [r["color"] for r in records]
    bars1 = ax1.bar(x, maes, color=colors, edgecolor="white")
    for bar, val in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(decs)
    ax1.set_ylabel("MAE (category steps)")
    ax1.set_title(f"{title} -- MAE by decision")

    # Exact match %
    exact = [r["exact_match_pct"] for r in records]
    bars2 = ax2.bar(x, exact, color=colors, edgecolor="white")
    for bar, val in zip(bars2, exact):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(decs)
    ax2.set_ylabel("Exact match %")
    ax2.set_title(f"{title} -- Exact match by decision")

    fig.tight_layout()
    return fig


def plot_per_score_breakdown(
    actual: np.ndarray, predicted: np.ndarray, title: str,
) -> plt.Figure:
    """For each actual score, show distribution of predictions as stacked bars."""
    scores = sorted(set(ICLR_SCALE) & set(actual))
    pred_scores = sorted(set(ICLR_SCALE) & set(predicted))
    pred_colors = {
        1: "#d9534f", 3: "#E69F00", 5: "#F0E442",
        6: "#56B4E9", 8: "#009E73", 10: "#CC79A7",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scores))

    bottom = np.zeros(len(scores))
    for ps in pred_scores:
        heights = []
        for i, s in enumerate(scores):
            mask = actual == s
            total = mask.sum()
            match = ((actual == s) & (predicted == ps)).sum()
            heights.append(match / total * 100 if total > 0 else 0)
        ax.bar(x, heights, bottom=bottom, label=f"Pred={ps}",
               color=pred_colors.get(ps, "#666"), edgecolor="white", linewidth=0.5)
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels(scores)
    ax.set_xlabel("Actual score")
    ax.set_ylabel("% of reviews")
    ax.set_title(title)
    ax.legend(title="Predicted", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-year comparison
# ---------------------------------------------------------------------------
def plot_year_metrics_comparison(all_metrics: dict[int, dict]) -> plt.Figure:
    """Side-by-side comparison of key metrics across years."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    years = sorted(all_metrics.keys())
    x = range(len(years))
    colors = [YEAR_COLORS.get(y, "#666") for y in years]

    fields = [
        ("pearson_r", "Pearson r", "{:.3f}"),
        ("spearman_rho", "Spearman rho", "{:.3f}"),
        ("mae_steps", "MAE (steps)", "{:.2f}"),
        ("exact_match_pct", "Exact match %", "{:.1f}%"),
    ]

    for ax, (field, label, fmt) in zip(axes, fields):
        vals = [all_metrics[y][field] for y in years]
        bars = ax.bar(x, vals, color=colors, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"ICLR {y}" for y in years], fontsize=9)
        ax.set_title(label)

    fig.suptitle("Gemini prediction accuracy by year", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemini prediction accuracy analysis")
    parser.add_argument("--year", type=int, help="Single year to analyze")
    parser.add_argument("--no-save", action="store_true", help="Display only")
    args = parser.parse_args()

    setup_publication_theme()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    years = [args.year] if args.year else [2024, 2025]
    dfs = {}
    for y in years:
        df = load_year(y)
        if df is not None:
            dfs[y] = df
            print(f"ICLR {y}: {len(df):,} reviews with predictions")
        else:
            print(f"ICLR {y}: no prediction data, skipping")

    if not dfs:
        print("No data to analyze.")
        return

    all_metrics = {}
    all_metrics_initial = {}
    init_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    summary_rows = []

    for year, df in sorted(dfs.items()):
        print(f"\n{'='*60}")
        print(f"ICLR {year}")
        print(f"{'='*60}")

        actual = df["rating"].values
        predicted = df["gemini_pred_score"].values

        # --- Metrics vs final rating ---
        m = compute_metrics(actual, predicted)
        all_metrics[year] = m
        print_metrics("vs final rating", m)
        summary_rows.append({"year": year, "ground_truth": "final_rating", **m})

        # --- Metrics vs initial rating (if available) ---
        has_initial = "ext_rating_before" in df.columns and df["ext_rating_before"].notna().any()
        if has_initial:
            df_init = df[df["ext_rating_before"].notna()].copy()
            df_init["ext_rating_before"] = df_init["ext_rating_before"].astype(int)
            actual_init = df_init["ext_rating_before"].values
            pred_init = df_init["gemini_pred_score"].values
            m_init = compute_metrics(actual_init, pred_init)
            all_metrics_initial[year] = m_init
            init_arrays[year] = (actual_init, pred_init)
            print_metrics("vs initial rating (pre-rebuttal)", m_init)
            summary_rows.append({"year": year, "ground_truth": "initial_rating", **m_init})

        # --- 1. Confusion heatmap (vs final) ---
        fig, ax = plt.subplots(figsize=(7, 5.5))
        plot_confusion_heatmap(actual, predicted,
                               f"ICLR {year} -- Gemini vs actual (final, n={len(df):,})", ax)
        if not args.no_save:
            save_figure(fig, f"gemini_pred_{year}_confusion_final")
        plt.close(fig)

        # Confusion heatmap (vs initial)
        if has_initial:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            plot_confusion_heatmap(actual_init, pred_init,
                                   f"ICLR {year} -- Gemini vs actual (initial, n={len(df_init):,})", ax)
            if not args.no_save:
                save_figure(fig, f"gemini_pred_{year}_confusion_initial")
            plt.close(fig)

        # --- 2. Paper-level scatter ---
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_paper_scatter(df, "rating", f"ICLR {year} -- Paper-level (final)", ax)
        if not args.no_save:
            save_figure(fig, f"gemini_pred_{year}_scatter_final")
        plt.close(fig)

        if has_initial:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_paper_scatter(df_init, "ext_rating_before",
                               f"ICLR {year} -- Paper-level (initial)", ax)
            if not args.no_save:
                save_figure(fig, f"gemini_pred_{year}_scatter_initial")
            plt.close(fig)

        # --- 3. Error distribution (in category steps) ---
        step_errors = score_to_step_error(actual, predicted)
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_error_distribution(step_errors, f"ICLR {year} -- Prediction errors (vs final, category steps)", ax)
        if not args.no_save:
            save_figure(fig, f"gemini_pred_{year}_errors")
        plt.close(fig)

        # --- 4. Per-score breakdown ---
        fig = plot_per_score_breakdown(actual, predicted,
                                       f"ICLR {year} -- Prediction distribution by actual score")
        if not args.no_save:
            save_figure(fig, f"gemini_pred_{year}_per_score")
        plt.close(fig)

        # --- 5. Accuracy by decision ---
        fig = plot_accuracy_by_decision(df, "rating", f"ICLR {year}")
        if not args.no_save:
            save_figure(fig, f"gemini_pred_{year}_by_decision")
        plt.close(fig)

    # --- Multi-year comparison ---
    if len(all_metrics) > 1:
        fig = plot_year_metrics_comparison(all_metrics)
        if not args.no_save:
            save_figure(fig, "gemini_pred_year_comparison_final")
        plt.close(fig)

        if all_metrics_initial:
            fig = plot_year_metrics_comparison(all_metrics_initial)
            if not args.no_save:
                save_figure(fig, "gemini_pred_year_comparison_initial")
            plt.close(fig)

    # --- Combined-years confusion matrix (vs initial pre-rebuttal score) ---
    if len(init_arrays) > 1:
        years_sorted = sorted(init_arrays)
        combined_actual = np.concatenate([init_arrays[y][0] for y in years_sorted])
        combined_pred = np.concatenate([init_arrays[y][1] for y in years_sorted])
        m_combined = compute_metrics(combined_actual, combined_pred)
        years_label = "+".join(str(y) for y in years_sorted)
        print(f"\n{'='*60}\nCombined ICLR {years_label}\n{'='*60}")
        print_metrics(f"vs initial rating (combined {years_label})", m_combined)
        summary_rows.append(
            {"year": f"combined_{years_label}", "ground_truth": "initial_rating", **m_combined}
        )

        fig, ax = plt.subplots(figsize=(7, 5.5))
        plot_confusion_heatmap(
            combined_actual,
            combined_pred,
            f"ICLR {years_label} -- Gemini vs initial score (n={len(combined_actual):,})",
            ax,
        )
        if not args.no_save:
            save_figure(fig, "gemini_pred_combined_confusion_initial")
        plt.close(fig)

    # --- Summary table ---
    if not args.no_save:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(TABLES_DIR / "prediction_accuracy_summary.csv", index=False)
        print(f"\n  Summary table saved to {TABLES_DIR / 'prediction_accuracy_summary.csv'}")

    print(f"\nFigures: {FIGURES_DIR}")
    print(f"Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
