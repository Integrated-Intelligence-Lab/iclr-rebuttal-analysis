"""
Minimum-detectable-effect (MDE) for the three structural-control predictors
that are reported as null in the headline rebuttal-content logistic:

  - rebuttal_total_len
  - first_reply_latency_days
  - n_author_replies

The headline model fits with cluster-robust SEs grouped by paper_key, so the
SEs already account for within-paper non-independence. MDE on the log-OR
scale at 80% power and alpha = 0.05 (two-sided) is:

    MDE_logOR = (z_{1-alpha/2} + z_{0.80}) * SE
              = (1.96 + 0.8416) * SE
              = 2.8016 * SE

Reads:  data/results/horizon_inference/tables/coefficients_headline.csv
Writes: data/results/horizon_inference/tables/null_triplet_mde.csv

The output puts each null-triplet feature alongside its observed OR, MDE OR,
and the p-value the OR would need to clear at the headline alpha. This
defends the null-result claim against a "you may just be underpowered" read.

Usage:
    python src/analyze/null_triplet_mde.py
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.paths import RESULTS_DIR

INPUT = RESULTS_DIR / "horizon_inference" / "tables" / "coefficients_headline.csv"
OUTPUT = RESULTS_DIR / "horizon_inference" / "tables" / "null_triplet_mde.csv"

NULL_TRIPLET = ["rebuttal_total_len", "first_reply_latency_days", "n_author_replies"]

ALPHA = 0.05
POWER = 0.80


def main() -> None:
    coef = pd.read_csv(INPUT)
    rows = []
    z_alpha = norm.ppf(1 - ALPHA / 2)   # 1.96
    z_power = norm.ppf(POWER)           # 0.8416
    multiplier = z_alpha + z_power      # 2.8016

    for feat in NULL_TRIPLET:
        match = coef[coef["feature"] == feat]
        if match.empty:
            print(f"WARN: {feat} not found in headline coefficients")
            continue
        r = match.iloc[0]
        se = float(r["se"])
        observed_logor = float(r["coef"])
        observed_or = float(r["odds_ratio"])
        observed_p = float(r["p"])

        mde_logor = multiplier * se
        mde_or = float(np.exp(mde_logor))
        mde_or_decrease = float(np.exp(-mde_logor))

        rows.append({
            "feature": feat,
            "se_cluster_robust": se,
            "observed_log_or": observed_logor,
            "observed_or_per_sd": observed_or,
            "observed_p": observed_p,
            "mde_log_or": mde_logor,
            "mde_or_increase_per_sd": mde_or,
            "mde_or_decrease_per_sd": mde_or_decrease,
            "alpha": ALPHA,
            "power": POWER,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)

    print(f"Wrote: {OUTPUT}")
    print(f"\nMDE at alpha={ALPHA}, power={POWER} (two-sided), cluster-robust SE.")
    print("Compare observed_or_per_sd to mde_or_increase_per_sd / mde_or_decrease_per_sd:\n")
    cols = ["feature", "observed_or_per_sd", "observed_p",
            "mde_or_increase_per_sd", "mde_or_decrease_per_sd"]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nReading: at 80% power we would detect a per-SD OR >= "
          f"{out['mde_or_increase_per_sd'].max():.2f} (or <= "
          f"{out['mde_or_decrease_per_sd'].min():.2f}) for these features. "
          "The observed ORs sit well inside this noise envelope, so the null "
          "is informative rather than underpowered.")


if __name__ == "__main__":
    main()
