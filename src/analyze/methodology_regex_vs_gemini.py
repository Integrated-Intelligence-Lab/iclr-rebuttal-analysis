"""
Paper: Appendix (app:score-change) -- method validation for score-change detection.

Regex vs Gemini verdict accuracy against external ground truth.

Compares the two score-change detection methods (regex extraction and
Gemini enrichment) against the actual score direction from external
ground truth (papercopilot before/after ratings).

Produces:
  1. Classification reports (P/R/F1) for each method per year
  2. Confusion matrices (3x3: increase/same/decrease)
  3. Combined summary table (CSV)
  4. Coverage analysis (how many actual changes each method detects)

Paper reference: Section 3.4 (Score Change Detection validation)
Depends on:     Parquet files with regex_verdict, gemini_verdict, ext_direction
Produces:       data/results/validation/regex_vs_gemini/

API required:   No
Estimated time: Under 30 seconds

Usage:
    python src/analyze/regex_vs_gemini_accuracy.py
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

from core.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from core.plot_theme import setup_publication_theme, YEAR_COLORS

setup_publication_theme()

PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
OUT_DIR = RESULTS_DIR / "methodology" / "score_change_validation" / "regex_vs_gemini"
FIGURES_DIR = OUT_DIR / "figures"
TABLES_DIR = OUT_DIR / "tables"

LABELS = ["increase", "same", "decrease"]
YEARS = [2024, 2025]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_verdict_data(year: int) -> pd.DataFrame | None:
    path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    mask = (
        df["regex_verdict"].notna()
        & df["gemini_verdict"].notna()
        & df["ext_direction"].notna()
    )
    return df[mask].copy() if mask.any() else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_method_metrics(y_true, y_pred, method_name: str) -> dict:
    """Compute accuracy, kappa, and per-class P/R/F1."""
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0,
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=LABELS)
    return {
        "method": method_name,
        "n": len(y_true),
        "accuracy": report["accuracy"],
        "kappa": kappa,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        **{
            f"{label}_{metric}": report[label][metric]
            for label in LABELS
            for metric in ["precision", "recall", "f1-score", "support"]
        },
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrices(y_true, regex_pred, gemini_pred, title: str):
    """Side-by-side confusion matrices for regex and Gemini."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pred, name in [(ax1, regex_pred, "Regex"), (ax2, gemini_pred, "Gemini")]:
        cm = confusion_matrix(y_true, pred, labels=LABELS)
        # Row-normalize for percentages
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"{name}")

        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                val = cm_pct[i, j]
                count = cm[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}%\n({count:,})",
                        ha="center", va="center", color=color, fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.8, label="% of GT class")

    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


def plot_coverage_analysis(dfs: dict[int, pd.DataFrame]):
    """Bar chart showing how many actual changes each method detects vs misses."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(6 * len(dfs), 5))
    if len(dfs) == 1:
        axes = [axes]

    for ax, (year, df_full) in zip(axes, sorted(dfs.items())):
        # Load full parquet to get all reviews with ext_direction
        full = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")
        has_gt = full["ext_direction"].notna()
        gt_all = full[has_gt]

        actual_changes = gt_all[gt_all["ext_direction"] != "same"]
        n_actual = len(actual_changes)

        # How many actual changes were covered by each method?
        regex_covered = actual_changes["regex_verdict"].notna().sum()
        gemini_covered = actual_changes["gemini_verdict"].notna().sum()

        # Of those covered, how many got the direction right?
        regex_correct = (
            (actual_changes["regex_verdict"] == actual_changes["ext_direction"])
            & actual_changes["regex_verdict"].notna()
        ).sum()
        gemini_correct = (
            (actual_changes["gemini_verdict"] == actual_changes["ext_direction"])
            & actual_changes["gemini_verdict"].notna()
        ).sum()

        x = np.arange(2)
        width = 0.3
        ax.bar(x - width / 2, [regex_covered, gemini_covered], width,
               label="Covered (has verdict)", color="#56B4E9", edgecolor="white")
        ax.bar(x + width / 2, [regex_correct, gemini_correct], width,
               label="Correct direction", color="#009E73", edgecolor="white")
        ax.axhline(n_actual, color="#d9534f", linestyle="--", linewidth=1.5,
                   label=f"Actual changes ({n_actual:,})")

        ax.set_xticks(x)
        ax.set_xticklabels(["Regex", "Gemini"])
        ax.set_ylabel("Number of reviews")
        ax.set_title(f"ICLR {year}", fontweight="bold")
        ax.legend(fontsize=8)

        for i, (cov, cor) in enumerate(
            [(regex_covered, regex_correct), (gemini_covered, gemini_correct)]
        ):
            ax.text(i - width / 2, cov + 50, f"{cov:,}", ha="center", fontsize=8)
            ax.text(i + width / 2, cor + 50, f"{cor:,}", ha="center", fontsize=8)

    fig.suptitle("Score Change Detection Coverage", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for y in YEARS:
        df = load_verdict_data(y)
        if df is not None:
            dfs[y] = df
            print(f"ICLR {y}: {len(df):,} reviews with all three signals")
        else:
            print(f"ICLR {y}: insufficient data, skipping")

    if not dfs:
        print("No data to analyze.")
        return

    summary_rows = []

    for year, df in sorted(dfs.items()):
        y_true = df["ext_direction"].values
        y_regex = df["regex_verdict"].values
        y_gemini = df["gemini_verdict"].values

        print(f"\n{'='*60}")
        print(f"  ICLR {year} (n={len(df):,})")
        print(f"{'='*60}")

        # Classification reports
        for method, pred in [("regex", y_regex), ("gemini", y_gemini)]:
            metrics = compute_method_metrics(y_true, pred, method)
            metrics["year"] = year
            summary_rows.append(metrics)

            print(f"\n  {method.upper()}:")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    Cohen's kappa: {metrics['kappa']:.3f}")
            print(f"    Macro F1: {metrics['macro_f1']:.3f}")
            for label in LABELS:
                p = metrics[f"{label}_precision"]
                r = metrics[f"{label}_recall"]
                f1 = metrics[f"{label}_f1-score"]
                n = int(metrics[f"{label}_support"])
                print(f"    {label:>10s}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (n={n:,})")

        # Confusion matrices
        fig = plot_confusion_matrices(
            y_true, y_regex, y_gemini,
            f"ICLR {year}: Verdict Accuracy vs Ground Truth (n={len(df):,})",
        )
        png = FIGURES_DIR / f"confusion_{year}.png"
        pdf = FIGURES_DIR / f"confusion_{year}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        print(f"\n  Saved: {png.name}, {pdf.name}")
        plt.close(fig)

    # Combined (all years)
    if len(dfs) > 1:
        combined = pd.concat(dfs.values(), ignore_index=True)
        y_true = combined["ext_direction"].values
        y_regex = combined["regex_verdict"].values
        y_gemini = combined["gemini_verdict"].values

        print(f"\n{'='*60}")
        print(f"  Combined (n={len(combined):,})")
        print(f"{'='*60}")

        for method, pred in [("regex", y_regex), ("gemini", y_gemini)]:
            metrics = compute_method_metrics(y_true, pred, method)
            metrics["year"] = "combined"
            summary_rows.append(metrics)
            print(f"\n  {method.upper()}:")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
            print(f"    Cohen's kappa: {metrics['kappa']:.3f}")
            print(f"    Macro F1: {metrics['macro_f1']:.3f}")

        fig = plot_confusion_matrices(
            y_true, y_regex, y_gemini,
            f"Combined ICLR 2024+2025: Verdict Accuracy (n={len(combined):,})",
        )
        png = FIGURES_DIR / "confusion_combined.png"
        pdf = FIGURES_DIR / "confusion_combined.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        print(f"\n  Saved: {png.name}, {pdf.name}")
        plt.close(fig)

    # Coverage analysis
    fig = plot_coverage_analysis(dfs)
    png = FIGURES_DIR / "coverage_analysis.png"
    pdf = FIGURES_DIR / "coverage_analysis.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved: {png.name}, {pdf.name}")
    plt.close(fig)

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = TABLES_DIR / "regex_vs_gemini_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Summary table: {csv_path.name}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
