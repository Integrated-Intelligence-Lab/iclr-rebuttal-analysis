"""
Paper: Appendix (app:reviewer-agree) -- independent measure of review-scoring noise.

Inter-reviewer agreement benchmark for contextualizing LLM prediction accuracy.

Computes pairwise agreement between human reviewers on the same paper,
using initial (pre-rebuttal) scores from the external ground truth. This
provides the "ceiling" for Gemini's score prediction: if human reviewers
only agree with each other X% of the time, expecting Gemini to exceed X%
is unreasonable.

Paper reference: RQ1 -- Instrument validation
Depends on:     data/processed/parquet/iclr_{2024,2025}_reviews.parquet
Produces:       data/results/reviewer_agreement/

API required:   No
Estimated time: Under 1 minute

Usage:
    python src/analyze/reviewer_agreement.py
    python src/analyze/reviewer_agreement.py --year 2024
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from core.plot_theme import YEAR_COLORS, setup_publication_theme

setup_publication_theme()

PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
OUT_DIR = RESULTS_DIR / "methodology" / "reviewer_agreement"
FIGURES_DIR = OUT_DIR / "figures"
TABLES_DIR = OUT_DIR / "tables"

ICLR_SCALE = [1, 3, 5, 6, 8, 10]
SCORE_TO_POS = {s: i for i, s in enumerate(ICLR_SCALE)}


def save_figure(fig: plt.Figure, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print(f"  Saved: {name}.png/pdf")


def load_parquet(year: int) -> pd.DataFrame:
    path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df[df["ext_rating_before"].notna()].copy()


def compute_pairwise_agreement(df: pd.DataFrame) -> dict:
    """Compute pairwise agreement statistics across all papers."""
    exact_matches = 0
    within_one_matches = 0
    total_pairs = 0
    step_diffs = []

    for forum, group in df.groupby("forum"):
        scores = group["ext_rating_before"].values
        if len(scores) < 2:
            continue
        for s1, s2 in combinations(scores, 2):
            total_pairs += 1
            if s1 == s2:
                exact_matches += 1
            p1 = SCORE_TO_POS.get(int(s1))
            p2 = SCORE_TO_POS.get(int(s2))
            if p1 is not None and p2 is not None:
                diff = abs(p1 - p2)
                step_diffs.append(diff)
                if diff <= 1:
                    within_one_matches += 1

    if total_pairs == 0:
        return {}

    step_diffs = np.array(step_diffs)
    return {
        "total_pairs": total_pairs,
        "exact_match_pct": 100 * exact_matches / total_pairs,
        "within_1_step_pct": 100 * within_one_matches / total_pairs,
        "mean_step_diff": float(step_diffs.mean()),
        "median_step_diff": float(np.median(step_diffs)),
        "n_papers": df["forum"].nunique(),
        "n_reviews": len(df),
    }


def compute_per_score_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """For each initial score, compute how often co-reviewers give the same score."""
    rows = []
    for forum, group in df.groupby("forum"):
        scores = group["ext_rating_before"].values
        if len(scores) < 2:
            continue
        for i, s1 in enumerate(scores):
            others = np.concatenate([scores[:i], scores[i+1:]])
            exact = (others == s1).mean()
            diffs = [abs(SCORE_TO_POS.get(int(s1), 0) - SCORE_TO_POS.get(int(s2), 0))
                     for s2 in others if int(s1) in SCORE_TO_POS and int(s2) in SCORE_TO_POS]
            within_1 = np.mean([d <= 1 for d in diffs]) if diffs else np.nan
            rows.append({
                "score": int(s1),
                "exact_match_with_peers": exact,
                "within_1_step_with_peers": within_1,
            })
    return pd.DataFrame(rows)


def plot_comparison(results: dict, gemini_results: dict):
    """Bar chart comparing human-human vs Gemini-human agreement."""
    years = sorted(results.keys())
    labels = [f"Human\nICLR {y}" for y in years] + ["Gemini\n(combined)"]
    exact_vals = [results[y]["exact_match_pct"] for y in years] + [gemini_results["exact"]]
    within1_vals = [results[y]["within_1_step_pct"] for y in years] + [gemini_results["within_1"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, exact_vals, width, label="Exact match",
                   color="#56B4E9", alpha=0.85)
    bars2 = ax.bar(x + width/2, within1_vals, width, label="Within 1 step",
                   color="#009E73", alpha=0.85)

    ax.set_ylabel("Agreement (%)")
    ax.set_title("Inter-Reviewer Agreement vs. Gemini Prediction Accuracy",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def plot_per_score(per_score_df: pd.DataFrame, year: int):
    """Box plot of agreement rates by initial score."""
    fig, ax = plt.subplots(figsize=(8, 5))

    scores = sorted(per_score_df["score"].unique())
    data_exact = [per_score_df[per_score_df["score"] == s]["exact_match_with_peers"].values
                  for s in scores]

    bp = ax.boxplot(data_exact, positions=range(len(scores)),
                    tick_labels=[str(s) for s in scores],
                    patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_alpha(0.7)

    ax.set_xlabel("Initial reviewer score")
    ax.set_ylabel("Exact match rate with co-reviewers")
    ax.set_title(f"ICLR {year}: Per-Score Inter-Reviewer Agreement", fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Inter-reviewer agreement benchmark")
    parser.add_argument("--year", type=int, nargs="+", help="Year(s) to analyze")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    candidate_years = args.year or [2024, 2025]
    results = {}
    all_per_score = []

    for year in candidate_years:
        df = load_parquet(year)
        if df.empty:
            print(f"ICLR {year}: no data with ext_rating_before, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  ICLR {year}: Inter-Reviewer Agreement")
        print(f"{'='*60}")

        stats = compute_pairwise_agreement(df)
        if not stats:
            continue

        results[year] = stats
        print(f"  Papers: {stats['n_papers']:,}")
        print(f"  Reviews: {stats['n_reviews']:,}")
        print(f"  Reviewer pairs: {stats['total_pairs']:,}")
        print(f"  Exact match: {stats['exact_match_pct']:.1f}%")
        print(f"  Within 1 step: {stats['within_1_step_pct']:.1f}%")
        print(f"  Mean step difference: {stats['mean_step_diff']:.2f}")

        per_score = compute_per_score_agreement(df)
        per_score["year"] = year
        all_per_score.append(per_score)

        # Per-score agreement plot
        fig = plot_per_score(per_score, year)
        save_figure(fig, f"per_score_agreement_{year}")
        plt.close(fig)

        # Per-score summary table
        score_summary = per_score.groupby("score").agg(
            n=("exact_match_with_peers", "count"),
            mean_exact=("exact_match_with_peers", "mean"),
            mean_within1=("within_1_step_with_peers", "mean"),
        ).round(3)
        score_summary.to_csv(TABLES_DIR / f"per_score_agreement_{year}.csv")
        print(f"\n  Per-score agreement:")
        for score, row in score_summary.iterrows():
            print(f"    Score {score}: exact={row['mean_exact']:.1%}, "
                  f"within-1={row['mean_within1']:.1%} (n={int(row['n'])})")

    if not results:
        print("No data found.")
        return

    # Combined statistics
    if len(results) > 1:
        all_df = pd.concat([load_parquet(y) for y in results.keys()])
        combined = compute_pairwise_agreement(all_df)
        int_years = sorted(y for y in results.keys() if isinstance(y, int))
        results["combined"] = combined
        print(f"\n{'='*60}")
        print(f"  Combined ({'+'.join(str(y) for y in int_years)})")
        print(f"{'='*60}")
        print(f"  Reviewer pairs: {combined['total_pairs']:,}")
        print(f"  Exact match: {combined['exact_match_pct']:.1f}%")
        print(f"  Within 1 step: {combined['within_1_step_pct']:.1f}%")

    # Comparison figure (human vs Gemini)
    gemini_combined = {"exact": 40.7, "within_1": 87.6}
    fig = plot_comparison(
        {y: r for y, r in results.items() if isinstance(y, int)},
        gemini_combined,
    )
    save_figure(fig, "human_vs_gemini_agreement")
    plt.close(fig)

    # Summary CSV
    summary_rows = []
    for key, stats in results.items():
        row = {"year": key}
        row.update(stats)
        summary_rows.append(row)
    # Add Gemini row for comparison
    summary_rows.append({
        "year": "gemini_combined",
        "exact_match_pct": 40.7,
        "within_1_step_pct": 87.6,
        "n_reviews": 73000,
        "total_pairs": None,
        "mean_step_diff": None,
    })
    summary_df = pd.DataFrame(summary_rows)
    csv_path = TABLES_DIR / "agreement_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Summary table: {csv_path}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
