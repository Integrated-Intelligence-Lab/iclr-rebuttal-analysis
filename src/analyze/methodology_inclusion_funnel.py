"""
Paper: Appendix (app:dataset) -- inclusion funnel for the analysis cohort.

Inclusion funnel: how many observations survive each analysis filter.

Shows the sample selection at each stage and compares retained vs excluded
subsets on key variables (initial score, confidence, decision, year).
Critical for transparency: readers must see who enters each analysis.

Paper reference: Methodology / Appendix
Depends on:     data/processed/parquet/iclr_{2024,2025}_reviews.parquet
                data/results/score_change/gemini_verdicts_v2_*.json
Produces:       data/results/inclusion_funnel/

API required:   No
Estimated time: Under 1 minute

Usage:
    python src/analyze/inclusion_funnel.py
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from core.plot_theme import setup_publication_theme

setup_publication_theme()

PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
VERDICT_DIR = RESULTS_DIR / "score_change"
OUT_DIR = RESULTS_DIR / "methodology" / "inclusion_funnel"
FIGURES_DIR = OUT_DIR / "figures"
TABLES_DIR = OUT_DIR / "tables"


def load_all_reviews() -> pd.DataFrame:
    """Load and combine parquet files for 2024 and 2025."""
    dfs = []
    for year in [2024, 2025]:
        path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["year"] = year
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def count_verdict_entries() -> dict[int, int]:
    """Count entries in verdict v2 files per year."""
    counts = {}
    for year, filename in [(2024, "gemini_verdicts_v2_full.json"),
                           (2025, "gemini_verdicts_v2_iclr2025_full.json")]:
        path = VERDICT_DIR / filename
        if path.exists():
            data = json.load(open(path))
            counts[year] = len(data)
    return counts


def describe_subset(df: pd.DataFrame, label: str) -> dict:
    """Compute summary statistics for a subset."""
    stats = {
        "label": label,
        "n_reviews": len(df),
        "n_papers": df["forum"].nunique() if "forum" in df.columns else 0,
    }
    if "rating" in df.columns:
        stats["mean_rating"] = df["rating"].mean()
        stats["median_rating"] = df["rating"].median()
    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        stats["mean_confidence"] = conf.mean()
    if "decision" in df.columns:
        accepted = df["decision"].isin(["Oral", "Spotlight", "Poster"])
        stats["acceptance_rate"] = accepted.mean()
    if "year" in df.columns:
        for y in sorted(df["year"].unique()):
            stats[f"pct_year_{y}"] = (df["year"] == y).mean()
    if "ext_rating_before" in df.columns:
        stats["pct_has_ext_before"] = df["ext_rating_before"].notna().mean()
    return stats


def build_funnel(df: pd.DataFrame) -> list[dict]:
    """Build the inclusion funnel with progressive filters."""
    funnel = []

    # Stage 0: All reviews
    funnel.append(describe_subset(df, "All reviews (2024+2025)"))

    # Stage 1: With Gemini predictions
    has_pred = df[df["gemini_pred_score"].notna()]
    funnel.append(describe_subset(has_pred, "With Gemini predictions"))

    # Stage 2: With external ground truth (before/after)
    has_ext = df[df["ext_rating_before"].notna() & df["ext_rating_after"].notna()]
    funnel.append(describe_subset(has_ext, "With external GT (before+after)"))

    # Stage 3: With Gemini predictions AND external GT
    has_both = has_ext[has_ext["gemini_pred_score"].notna()]
    funnel.append(describe_subset(has_both, "With predictions + external GT"))

    # Stage 4: With Gemini verdict (had follow-up discussion)
    has_verdict = df[df["gemini_verdict"].notna()]
    funnel.append(describe_subset(has_verdict, "With Gemini verdict (follow-up)"))

    # Stage 5: With verdict AND external GT
    has_verdict_ext = has_verdict[
        has_verdict["ext_rating_before"].notna() & has_verdict["ext_rating_after"].notna()
    ]
    funnel.append(describe_subset(has_verdict_ext, "Verdict + external GT"))

    return funnel


def compare_retained_excluded(df: pd.DataFrame, mask: pd.Series, stage_name: str) -> pd.DataFrame:
    """Compare retained vs excluded reviews on key variables."""
    retained = df[mask]
    excluded = df[~mask]

    rows = []
    for var, col in [("Mean rating", "rating"), ("Mean confidence", "confidence"),
                     ("Acceptance rate", "decision"), ("Pct 2025", "year")]:
        if col not in df.columns:
            continue
        if col == "decision":
            val_ret = retained["decision"].isin(["Oral", "Spotlight", "Poster"]).mean()
            val_exc = excluded["decision"].isin(["Oral", "Spotlight", "Poster"]).mean()
        elif col == "year":
            val_ret = (retained["year"] == 2025).mean()
            val_exc = (excluded["year"] == 2025).mean()
        else:
            val_ret = pd.to_numeric(retained[col], errors="coerce").mean()
            val_exc = pd.to_numeric(excluded[col], errors="coerce").mean()
        rows.append({
            "variable": var,
            "retained": val_ret,
            "excluded": val_exc,
            "difference": val_ret - val_exc,
        })
    return pd.DataFrame(rows)


def plot_funnel(funnel: list[dict]):
    """Horizontal bar chart showing sample attrition."""
    labels = [f["label"] for f in funnel]
    counts = [f["n_reviews"] for f in funnel]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(labels))
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(labels)))

    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Inclusion Funnel: Sample Selection at Each Analysis Stage",
                 fontweight="bold")
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        pct = count / counts[0] * 100
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.0f}%)", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_reviews()
    if df.empty:
        print("No data found.")
        return

    print(f"Loaded {len(df):,} reviews across {df['forum'].nunique():,} papers")

    # Build funnel
    funnel = build_funnel(df)

    print(f"\n{'='*70}")
    print("  INCLUSION FUNNEL")
    print(f"{'='*70}")
    for stage in funnel:
        n = stage["n_reviews"]
        pct = n / funnel[0]["n_reviews"] * 100
        print(f"  {stage['label']:<45s} {n:>8,} ({pct:5.1f}%)")

    # Verdict v2 counts (separate from parquet)
    verdict_counts = count_verdict_entries()
    if verdict_counts:
        total_verdicts = sum(verdict_counts.values())
        print(f"\n  Verdict v2 entries (enriched thread analysis):")
        for year, count in sorted(verdict_counts.items()):
            print(f"    ICLR {year}: {count:,}")
        print(f"    Total: {total_verdicts:,}")

    # Compare retained vs excluded at key stage
    mask_ext = df["ext_rating_before"].notna() & df["ext_rating_after"].notna()
    comparison = compare_retained_excluded(df, mask_ext, "External GT available")
    print(f"\n  Balance check: reviews WITH vs WITHOUT external GT:")
    for _, row in comparison.iterrows():
        print(f"    {row['variable']:<20s}: retained={row['retained']:.3f}, "
              f"excluded={row['excluded']:.3f}, diff={row['difference']:+.3f}")

    # Save
    funnel_df = pd.DataFrame(funnel)
    funnel_df.to_csv(TABLES_DIR / "inclusion_funnel.csv", index=False)
    comparison.to_csv(TABLES_DIR / "balance_check_ext_gt.csv", index=False)

    fig = plot_funnel(funnel)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"inclusion_funnel.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: inclusion_funnel.png/pdf")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
