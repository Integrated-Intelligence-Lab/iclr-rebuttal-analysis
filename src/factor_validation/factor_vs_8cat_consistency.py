"""
Phase A4 — sanity-check F01..F44 against the existing 8-category coding.

Joins interaction_factor_labels_{year}.csv to coded_interactions_{year}.csv on
(year, paper_key, reviewer_id). For every (factor, 8-category) pair, computes
the phi coefficient (binary correlation). Saves the full 44 x 8 matrix and
flags the strongest associations to validate that the codebook is coherent.

Expected sanity signs (from the codebook):
  F06 (Concrete empirical evidence) ~ cat_new_experiments
  F11 (Ablation studies)            ~ cat_evaluation_improvements
  F08 (Clarification of methodology)~ cat_clarification

Output: data/results/32_factor_taxonomy/factor_vs_8cat_consistency.csv

Usage:
    python -m src.factor_validation.factor_vs_8cat_consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd

from core.paths import RESULTS_DIR
from factor_validation.prompt import VALID_FACTOR_IDS

OUT_DIR = RESULTS_DIR / "32_factor_taxonomy"
FACTOR_COLS = sorted(VALID_FACTOR_IDS)
CAT_COLS = [
    "cat_new_experiments",
    "cat_clarification",
    "cat_novelty_reframing",
    "cat_writing_fixes",
    "cat_theoretical_additions",
    "cat_evaluation_improvements",
    "cat_reproducibility",
    "cat_scope_narrowing",
]


def _phi(a: pd.Series, b: pd.Series) -> float:
    """Phi coefficient for two binary vectors."""
    a = a.astype(int).to_numpy()
    b = b.astype(int).to_numpy()
    n = len(a)
    if n == 0:
        return float("nan")
    mean_a, mean_b = a.mean(), b.mean()
    if mean_a in (0.0, 1.0) or mean_b in (0.0, 1.0):
        return float("nan")
    cov = ((a - mean_a) * (b - mean_b)).sum() / n
    sd_a = np.sqrt(mean_a * (1 - mean_a))
    sd_b = np.sqrt(mean_b * (1 - mean_b))
    return cov / (sd_a * sd_b)


def main() -> None:
    rows = []
    for year in (2024, 2025):
        f_path = OUT_DIR / f"interaction_factor_labels_{year}.csv"
        c_path = RESULTS_DIR / "rebuttal_content_coding" / "tables" / f"coded_interactions_{year}.csv"
        if not f_path.exists() or not c_path.exists():
            print(f"[{year}] missing input — skip")
            continue
        f_df = pd.read_csv(f_path)
        c_df = pd.read_csv(c_path)
        merged = f_df.merge(
            c_df[["year", "paper_key", "reviewer_id"] + CAT_COLS],
            on=["year", "paper_key", "reviewer_id"],
            how="inner",
        )
        print(f"[{year}] joined: {len(merged)} rows  (factor_csv={len(f_df)}, 8cat_csv={len(c_df)})")
        if len(merged) == 0:
            continue
        for fid in FACTOR_COLS:
            for cat in CAT_COLS:
                rows.append({
                    "year": year,
                    "factor_id": fid,
                    "category": cat,
                    "phi": _phi(merged[fid], merged[cat]),
                    "n": len(merged),
                    "factor_prevalence": float(merged[fid].mean()),
                    "category_prevalence": float(merged[cat].mean()),
                })

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "factor_vs_8cat_consistency.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} (year, factor, category) rows -> {out_path.name}")

    print("\nTop 10 positive (factor, category) associations per year:")
    for year in (2024, 2025):
        sub = out_df[(out_df["year"] == year) & out_df["phi"].notna()].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("phi", ascending=False).head(10)
        print(f"\n[{year}]")
        for _, r in sub.iterrows():
            print(f"  {r['factor_id']} x {r['category']:<28}  phi={r['phi']:+.3f}  "
                  f"(f_prev={r['factor_prevalence']:.2f}, c_prev={r['category_prevalence']:.2f})")


if __name__ == "__main__":
    main()
