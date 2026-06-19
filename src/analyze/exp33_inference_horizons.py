"""
Inferential logistic for "successful rebuttal" — Section 3.3 (paper).

What this answers: controlling for the structural conditions visible
when the reviewer submitted their initial review (initial-score
position, deviation from paper mean, confidence, sub-scores, paper-
level summaries, and the LLM-decoded text-vs-score offset), which
rebuttal-content features remain independently associated with a
post-rebuttal score increase?

Two-analysis split:
  This script    -> inference (which features matter, controlling for the rest)
  rebuttal_rf_prediction.py -> prediction by horizon + RF ceiling check

Outcome:
  score_increase = ext_direction == "increase"
  (binary; same / decrease both negative)

Cohort:
  All rows in the draft-time horizon — has_rebuttal == 1 AND F01..F44
  labels available. Co-reviewer post-rebuttal aggregates are EXCLUDED
  from the headline model (they belong to the post-discussion horizon
  of 3.4 and would leak the outcome's environment into 3.3).

Headline model:
  Logistic regression with cluster-robust standard errors grouped by
  paper_key (statsmodels Logit, cov_type="cluster"). Continuous features
  standardised by year-pooled mean/std so coefficients are "per 1 SD";
  binary features (F01..F44, has_*, tone_*, is_initial_outlier) kept
  on their natural 0/1 scale so coefficients are "presence vs absence".
  Bonferroni correction on the 44-factor family at alpha = 0.05/44.

Ablations:
  A. Drop structural confounders (pos_before, deviation_from_paper_mean,
     paper_initial_mean, paper_initial_std, n_reviewers, confidence,
     sub-scores). If the 44-factor odds ratios swell, the inferential
     coefficients in the headline model are properly attributing the
     structural variance to its source.
  B. Stratum-fixed-effects within (deviation_quartile x pos_before
     bucket x confidence) cells. If the 44-factor effects survive
     within-cell, they're not confounded by the structural conditions
     that pre-rebuttal features encode.

Outputs:
  data/results/horizon_inference/tables/coefficients_headline.csv
  data/results/horizon_inference/tables/coefficients_ablation_no_structural.csv
  data/results/horizon_inference/tables/coefficients_ablation_strata_fe.csv
  data/results/horizon_inference/figures/forest_factors.{pdf,png}

Usage:
  python -m src.analyze.horizon_inference
  python -m src.analyze.horizon_inference --no-save
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analyze.exp34_prediction_horizons import (
    DRAFT_ONLY_FEATURES,
    POST_ONLY_FEATURES,
    PRE_FEATURES,
    load_feature_matrix,
)
from core.paths import RESULTS_DIR
from core.plot_theme import setup_publication_theme
from factor_validation.prompt import VALID_FACTOR_IDS

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = RESULTS_DIR / "horizon_inference"
TABLES_DIR = OUT_DIR / "tables"
FIGURES_DIR = OUT_DIR / "figures"

FACTOR_COLS = sorted(VALID_FACTOR_IDS)
N_FACTORS = len(FACTOR_COLS)
ALPHA_FAMILY = 0.05 / N_FACTORS  # Bonferroni on the 44-factor family

# Codebook polarity (locked in factor_validation/codebook.json):
#   F01..F23 = positive (rebuttal behaviors hypothesised to help)
#   F24..F44 = negative (behaviors hypothesised to hurt)
POSITIVE_FACTORS = [f"F{i:02d}" for i in range(1, 24)]
NEGATIVE_FACTORS = [f"F{i:02d}" for i in range(24, 45)]

# Structural confounders — the things visible at the pre-rebuttal horizon
# whose effect we want to control for so the factor coefficients are
# rebuttal-attributable.
#
# `deviation_from_paper_mean` is excluded by construction: it is exactly
# `pos_before - paper_initial_mean`, so including all three creates
# perfect linear dependence and inflates SEs / makes the individual
# coefficients uninterpretable. We keep `pos_before` (the focal score)
# and `paper_initial_mean` (the paper-level baseline); the deviation
# story is fully recoverable from these two.
STRUCTURAL_CONFOUNDERS = [
    "pos_before", "paper_initial_mean",
    "paper_initial_std", "n_reviewers", "confidence",
    "soundness", "presentation", "contribution",
    "is_initial_outlier",
    # Timing: days before the official ICLR review-submission deadline
    # (positive = on-time, negative = late). Built downstream by
    # `rebuttal_rf_prediction.load_feature_matrix` from `review_cdate` and the
    # year-specific published deadline. Aligns with Jung et al.'s "days
    # remaining until deadline". Reviewer workload, the other structural
    # correlate flagged in prior work, is unrecoverable from OpenReview because
    # reviewer identifiers are anonymised at the paper level.
    "days_before_deadline",
]

# Review-text features (also pre-rebuttal, but not "structural confounders"
# the way score position is). Kept in headline model as additional controls.
REVIEW_TEXT_FEATURES = ["strengths_len", "weaknesses_len", "questions_len", "sw_ratio"]

# Gemini-derived text features.
GEMINI_FEATURES = [
    "gemini_offset", "engagement_ord", "score_confidence_gemini",
    "n_indicators",
    "has_technical_depth", "has_actionable_feedback",
    "has_specific_examples", "has_trivial_feedback",
]

# Rebuttal-content (the rebuttal-attributable features at draft-time).
# `n_thread_messages` is excluded: it counts reviewer follow-ups, so it
# resolves only after the discussion period (lives in 3.4 / post-discussion).
# `n_factors_present` is excluded from the joint: conditioning F01..F44 ORs
# on "holding total factor count fixed" makes them awkward to interpret.
# It is reported separately as a single-feature ablation alongside this run.
REBUTTAL_CONTENT_FEATURES = [
    "rebuttal_total_len", "n_author_replies",
    # Rebuttal-side timing and structure (Jung et al. 2025): time to first
    # author reply (days), and max depth of the dialogue tree rooted at the
    # review. Both are properties of the rebuttal exchange, available once
    # the thread exists.
    "first_reply_latency_days", "thread_max_depth",
] + FACTOR_COLS

# Continuous features (will be standardised). Binary stay 0/1.
BINARY_FEATURES = set(FACTOR_COLS) | {
    "is_initial_outlier",
    "has_technical_depth", "has_actionable_feedback",
    "has_specific_examples", "has_trivial_feedback",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _build_combined_dataframe() -> pd.DataFrame:
    """Combine 2024 + 2025 draft-time matrices; year-prefix paper_key."""
    print("Loading draft-time feature matrices ...")
    df24 = load_feature_matrix(2024, "draft").assign(year=2024)
    df25 = load_feature_matrix(2025, "draft").assign(year=2025)
    df24["paper_key"] = "2024_" + df24["paper_key"].astype(str)
    df25["paper_key"] = "2025_" + df25["paper_key"].astype(str)
    df = pd.concat([df24, df25], ignore_index=True)
    print(f"  Combined: n={len(df)} (2024={len(df24)}, 2025={len(df25)}), "
          f"positive={int(df['score_increase'].sum())} ({df['score_increase'].mean()*100:.1f}%)")
    return df


def _resolve_feature_list(spec: list[str], available: set[str]) -> list[str]:
    """Expand symbolic specs (currently a no-op) and drop missing columns."""
    out: list[str] = []
    for c in spec:
        if c.startswith("__"):
            # tone_* etc. expanded by load_feature_matrix already
            continue
        if c in available:
            out.append(c)
    return out


def _standardise_continuous(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Z-score continuous features in-place; binaries pass through.

    Returns a new DataFrame with the same columns plus an unchanged
    paper_key / year / score_increase. Standardisation uses the pooled
    mean/std across both years.
    """
    out = df.copy()
    for c in feature_cols:
        if c in BINARY_FEATURES:
            continue
        col = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float)
        mu, sd = col.mean(), col.std()
        if sd == 0:
            out[c] = 0.0
        else:
            out[c] = (col - mu) / sd
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _drop_collinear_and_low_variance(
    df: pd.DataFrame, feature_cols: list[str], min_prevalence: float = 0.01,
) -> list[str]:
    """Drop features that would make the Hessian singular.

    Rules:
      - Binary features with prevalence < min_prevalence or > 1-min_prevalence
        (no within-class variance to estimate from) are dropped.
      - Continuous features with std == 0 are dropped.
      - Among tone_* one-hots (which sum to 1 with the intercept), drop the
        one with the largest prevalence to avoid the dummy trap.
    """
    keep: list[str] = []
    dropped_low_var: list[str] = []
    tone_cols = [c for c in feature_cols if c.startswith("tone_")]

    if tone_cols:
        # drop_first style: drop the most prevalent tone as the reference category
        ref = max(tone_cols, key=lambda c: df[c].mean())
        feature_cols = [c for c in feature_cols if c != ref]
        print(f"  [collinearity] tone reference category dropped: {ref}")

    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if c in BINARY_FEATURES or c.startswith("tone_"):
            mean = float(s.mean())
            if mean < min_prevalence or mean > 1 - min_prevalence:
                dropped_low_var.append(f"{c}(prev={mean:.3f})")
                continue
        else:
            if float(s.std()) == 0:
                dropped_low_var.append(f"{c}(std=0)")
                continue
        keep.append(c)
    if dropped_low_var:
        print(f"  [collinearity] dropped {len(dropped_low_var)} low-variance features: "
              f"{dropped_low_var[:6]}{' ...' if len(dropped_low_var) > 6 else ''}")
    return keep


def _fit_logit_clustered(
    df: pd.DataFrame, feature_cols: list[str], group_col: str = "paper_key",
) -> sm.regression.linear_model.RegressionResults:
    """Fit logistic regression with cluster-robust SEs."""
    X = df[feature_cols].astype(float).copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["score_increase"].astype(int).to_numpy()
    groups = df[group_col].to_numpy()
    model = sm.Logit(y, X)
    return model.fit(
        disp=0, method="lbfgs", maxiter=500,
        cov_type="cluster", cov_kwds={"groups": groups},
    )


def _coef_table(result, feature_cols: list[str]) -> pd.DataFrame:
    """Pull a clean coefficient table out of a statsmodels Logit result.

    Columns: feature, coef, se, z, p, ci_lo, ci_hi, odds_ratio, or_lo, or_hi.
    Excludes the intercept.
    """
    params = result.params
    se = result.bse
    z = result.tvalues
    p = result.pvalues
    ci = result.conf_int(alpha=0.05)
    rows = []
    for name in params.index:
        if name == "const":
            continue
        coef = float(params[name])
        ci_lo, ci_hi = float(ci.loc[name, 0]), float(ci.loc[name, 1])
        rows.append({
            "feature": name,
            "coef": coef,
            "se": float(se[name]),
            "z": float(z[name]),
            "p": float(p[name]),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "odds_ratio": float(np.exp(coef)),
            "or_lo": float(np.exp(ci_lo)),
            "or_hi": float(np.exp(ci_hi)),
        })
    return pd.DataFrame(rows)


def _annotate_factor_significance(coef_df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'bonferroni_sig' column for the 44-factor family only."""
    out = coef_df.copy()
    out["bonferroni_sig"] = False
    factor_mask = out["feature"].isin(FACTOR_COLS)
    out.loc[factor_mask, "bonferroni_sig"] = out.loc[factor_mask, "p"] < ALPHA_FAMILY
    return out


def sign_consistency_test(coef_df: pd.DataFrame) -> dict:
    """Codebook-polarity sign test on the 44-factor family.

    Tests, separately for the 23 positive and 21 negative factors,
    whether the observed direction of the OR is consistent with the
    pre-registered polarity at a rate higher than chance (binomial).

    Returns a dict with per-polarity counts, expected directions, and
    the binomial p-value against H0 = 50/50.
    """
    from scipy.stats import binomtest

    factors = coef_df.set_index("feature").reindex(FACTOR_COLS)
    out: dict = {}
    for label, ids, expected in [
        ("positive", POSITIVE_FACTORS, "OR>1"),
        ("negative", NEGATIVE_FACTORS, "OR<1"),
    ]:
        sub = factors.loc[ids].dropna(subset=["odds_ratio"])
        if expected == "OR>1":
            n_correct = int((sub["odds_ratio"] > 1).sum())
        else:
            n_correct = int((sub["odds_ratio"] < 1).sum())
        n = int(len(sub))
        if n == 0:
            out[label] = {"n": 0, "n_correct": 0, "p_binomial": float("nan")}
            continue
        p = float(binomtest(n_correct, n, 0.5, alternative="greater").pvalue)
        out[label] = {
            "n": n, "n_correct": n_correct, "expected_direction": expected,
            "fraction_correct": n_correct / n, "p_binomial": p,
        }
    return out


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------
def plot_forest_factors(
    headline: pd.DataFrame,
    no_struct: pd.DataFrame | None = None,
    strata_fe: pd.DataFrame | None = None,
) -> plt.Figure:
    """Forest plot for the 44 factors plus structural anchors.

    Headline ORs determine the y-ordering (factors sorted ascending by OR,
    so positive factors land on top). If no_struct and strata_fe are passed,
    overlays them at small y-offsets so robustness is visible at a glance.

    Color codes the headline marker: black=structural anchor, blue=positive
    factor, orange=negative factor. Bonferroni-significant factors get a
    star marker and a bold y-label.
    """
    setup_publication_theme()

    anchors = ["pos_before", "paper_initial_mean", "contribution",
               "rebuttal_total_len", "gemini_offset"]
    factor_block = headline[headline["feature"].isin(FACTOR_COLS)].copy()
    factor_block = factor_block.sort_values("odds_ratio", ascending=True)
    anchor_block = headline[headline["feature"].isin(anchors)].copy()
    anchor_block = anchor_block.set_index("feature").reindex(anchors).dropna().reset_index()

    rows = pd.concat([factor_block, anchor_block], ignore_index=True)
    feat_order = rows["feature"].tolist()
    n_rows = len(rows)
    y = np.arange(n_rows)

    fig, ax = plt.subplots(figsize=(7.5, max(9, n_rows * 0.26)))

    # Lookup helpers for the two robustness runs
    def _lookup(df: pd.DataFrame | None, feat: str):
        if df is None:
            return None
        sub = df[df["feature"] == feat]
        if len(sub) == 0:
            return None
        r = sub.iloc[0]
        return float(r["odds_ratio"]), float(r["or_lo"]), float(r["or_hi"])

    # Plot in three layers so legend ordering is sensible:
    # (1) ablation runs as small grey markers at y +/- offset
    # (2) headline as big colored marker at exact y
    OFFSET = 0.20

    # Layer 1: ablations
    for i, feat in enumerate(feat_order):
        for offset, df, label_color in [
            (-OFFSET, no_struct, "#999999"),
            (+OFFSET, strata_fe, "#bbbbbb"),
        ]:
            pt = _lookup(df, feat)
            if pt is None:
                continue
            o, lo, hi = pt
            ax.errorbar(
                o, y[i] + offset,
                xerr=[[o - lo], [hi - o]],
                fmt="s", color=label_color, ecolor=label_color,
                lw=0.8, ms=2.5, alpha=0.85, zorder=2,
            )

    # Layer 2: headline
    label_styles = []
    for i, (_, r) in enumerate(rows.iterrows()):
        feat = r["feature"]
        is_anchor = feat in set(anchors)
        is_factor = feat in set(FACTOR_COLS)
        is_pos = feat in set(POSITIVE_FACTORS)
        is_neg = feat in set(NEGATIVE_FACTORS)
        bonf = bool(r.get("bonferroni_sig", False)) if is_factor else False

        if is_anchor:
            color = "#000000"
            marker = "D"
        elif is_pos:
            color = "#0072B2"
            marker = "*" if bonf else "o"
        elif is_neg:
            color = "#D55E00"
            marker = "*" if bonf else "o"
        else:
            color = "#888888"
            marker = "o"
        ms = 9 if (bonf or is_anchor) else 5
        ax.errorbar(
            r["odds_ratio"], y[i],
            xerr=[[r["odds_ratio"] - r["or_lo"]], [r["or_hi"] - r["odds_ratio"]]],
            fmt=marker, color=color, ecolor=color,
            lw=1.4, ms=ms, zorder=3,
        )
        label_styles.append("bold" if (bonf or is_anchor) else "normal")

    ax.axvline(1.0, color="grey", ls=":", lw=0.8)
    n_factors = len(factor_block)
    if n_factors and len(anchor_block):
        ax.axhline(n_factors - 0.5, color="grey", ls="-", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_order, fontsize=7.5)
    for tick, style in zip(ax.get_yticklabels(), label_styles):
        tick.set_fontweight(style)
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_xscale("log")

    # Inline legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#000000",
               markersize=7, label="structural anchor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
               markersize=6, label="positive factor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00",
               markersize=6, label="negative factor"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#0072B2",
               markersize=10, label="Bonferroni-significant"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#999999",
               markersize=4, label="no-structural ablation"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#bbbbbb",
               markersize=4, label="stratum-FE ablation"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7,
              framealpha=0.9, ncol=1)
    ax.set_title("Per-feature association with score increase\n"
                 "(logistic, cluster-robust SE by paper\\_key, rebuttal-thread content)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Headline + ablations
# ---------------------------------------------------------------------------
def run_headline(df: pd.DataFrame, save: bool) -> pd.DataFrame:
    available = set(df.columns)
    feature_cols = (
        STRUCTURAL_CONFOUNDERS + REVIEW_TEXT_FEATURES + GEMINI_FEATURES
        + REBUTTAL_CONTENT_FEATURES
    )
    feature_cols = _resolve_feature_list(feature_cols, available)
    # Add tone one-hot columns if present.
    feature_cols += [c for c in df.columns if c.startswith("tone_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"\n[Headline] features ({len(feature_cols)}): "
          f"{feature_cols[:6]} ... {feature_cols[-4:]}")

    df_std = _standardise_continuous(df, feature_cols)
    feature_cols = _drop_collinear_and_low_variance(df_std, feature_cols)
    result = _fit_logit_clustered(df_std, feature_cols)
    print(f"  pseudo R-sq: {result.prsquared:.3f}    n={int(result.nobs)}")

    coef_df = _coef_table(result, feature_cols)
    coef_df = _annotate_factor_significance(coef_df)
    n_factor_sig = int(coef_df[coef_df["feature"].isin(FACTOR_COLS)]["bonferroni_sig"].sum())
    print(f"  Factors crossing Bonferroni (alpha={ALPHA_FAMILY:.4f}): {n_factor_sig} of {N_FACTORS}")
    print("  Top 8 factor odds ratios (by |log OR|):")
    fac = coef_df[coef_df["feature"].isin(FACTOR_COLS)].copy()
    fac["abs_log_or"] = fac["coef"].abs()
    for _, r in fac.sort_values("abs_log_or", ascending=False).head(8).iterrows():
        flag = " *" if r["bonferroni_sig"] else "  "
        print(f"    {r['feature']:<6} OR={r['odds_ratio']:.3f} "
              f"[{r['or_lo']:.3f}, {r['or_hi']:.3f}]  p={r['p']:.4g}{flag}")

    if save:
        coef_df.to_csv(TABLES_DIR / "coefficients_headline.csv", index=False)
    return coef_df


def run_ablation_no_structural(df: pd.DataFrame, save: bool) -> pd.DataFrame:
    """Ablation A: drop structural confounders, refit, compare 44-factor ORs."""
    available = set(df.columns)
    feature_cols = REVIEW_TEXT_FEATURES + GEMINI_FEATURES + REBUTTAL_CONTENT_FEATURES
    feature_cols = _resolve_feature_list(feature_cols, available)
    feature_cols += [c for c in df.columns if c.startswith("tone_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"\n[Ablation A: no structural confounders] features ({len(feature_cols)})")

    df_std = _standardise_continuous(df, feature_cols)
    feature_cols = _drop_collinear_and_low_variance(df_std, feature_cols)
    result = _fit_logit_clustered(df_std, feature_cols)
    coef_df = _coef_table(result, feature_cols)
    coef_df = _annotate_factor_significance(coef_df)
    n_sig = int(coef_df[coef_df["feature"].isin(FACTOR_COLS)]["bonferroni_sig"].sum())
    print(f"  Factors crossing Bonferroni: {n_sig} of {N_FACTORS}")
    if save:
        coef_df.to_csv(TABLES_DIR / "coefficients_ablation_no_structural.csv", index=False)
    return coef_df


def run_ablation_strata_fe(df: pd.DataFrame, save: bool) -> pd.DataFrame:
    """Ablation B: stratum fixed effects within (deviation_q x pos_q x conf) cells."""
    df_b = df.copy()
    # Quartile bins on deviation_from_paper_mean and pos_before, pooled across years.
    df_b["dev_q"] = pd.qcut(
        df_b["deviation_from_paper_mean"], q=4, labels=False, duplicates="drop",
    )
    df_b["pos_q"] = pd.qcut(
        df_b["pos_before"], q=4, labels=False, duplicates="drop",
    )
    df_b["conf_q"] = pd.cut(
        df_b["confidence"], bins=[-0.1, 2.5, 3.5, 4.5, 5.1],
        labels=False, include_lowest=True,
    )
    df_b["stratum"] = (
        df_b["dev_q"].astype("Int64").astype(str) + "|"
        + df_b["pos_q"].astype("Int64").astype(str) + "|"
        + df_b["conf_q"].astype("Int64").astype(str)
    )
    n_strata = df_b["stratum"].nunique()
    n_strata_with_both_classes = (
        df_b.groupby("stratum")["score_increase"].nunique() == 2
    ).sum()
    print(f"\n[Ablation B: stratum FE] {n_strata} strata, "
          f"{n_strata_with_both_classes} have both outcome classes")

    available = set(df_b.columns)
    feature_cols = REVIEW_TEXT_FEATURES + GEMINI_FEATURES + REBUTTAL_CONTENT_FEATURES
    feature_cols = _resolve_feature_list(feature_cols, available)
    feature_cols += [c for c in df_b.columns if c.startswith("tone_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(feature_cols))

    df_std = _standardise_continuous(df_b, feature_cols)
    feature_cols = _drop_collinear_and_low_variance(df_std, feature_cols)
    # Add stratum dummies (drop_first to avoid singularity).
    stratum_dummies = pd.get_dummies(
        df_std["stratum"], prefix="stratum", drop_first=True,
    ).astype(int)
    X = pd.concat([df_std[feature_cols], stratum_dummies], axis=1)
    X_full = sm.add_constant(X.astype(float), has_constant="add")
    y = df_std["score_increase"].astype(int).to_numpy()
    groups = df_std["paper_key"].to_numpy()
    model = sm.Logit(y, X_full)
    result = model.fit(
        disp=0, method="lbfgs", maxiter=500,
        cov_type="cluster", cov_kwds={"groups": groups},
    )
    print(f"  pseudo R-sq: {result.prsquared:.3f}    n={int(result.nobs)}")

    # Keep only feature_cols in the coef table; drop stratum dummies + const.
    full_coef = _coef_table(result, list(X.columns))
    coef_df = full_coef[full_coef["feature"].isin(feature_cols)].reset_index(drop=True)
    coef_df = _annotate_factor_significance(coef_df)
    n_sig = int(coef_df[coef_df["feature"].isin(FACTOR_COLS)]["bonferroni_sig"].sum())
    print(f"  Factors crossing Bonferroni within strata: {n_sig} of {N_FACTORS}")
    if save:
        coef_df.to_csv(TABLES_DIR / "coefficients_ablation_strata_fe.csv", index=False)
    return coef_df


def plot_recommendation_forest(rec_table: pd.DataFrame) -> plt.Figure:
    """Forest plot ordered by recommendation gate.

    For each factor we plot two intervals:
      - marginal OR (small grey square)
      - polarity-isolated conditional OR (large colored circle)

    Positives appear on top, negatives below, each block ordered by isolated
    OR magnitude. Recommended factors get a star marker and a bold label.
    """
    setup_publication_theme()

    pos = rec_table[rec_table["polarity"] == "positive"].copy()
    pos = pos.sort_values("isolated_or", ascending=True)
    neg = rec_table[rec_table["polarity"] == "negative"].copy()
    neg = neg.sort_values("isolated_or", ascending=False)
    rows = pd.concat([neg, pos], ignore_index=True)

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(7.5, max(9, n_rows * 0.26)))
    y = np.arange(n_rows)

    # Marginal small grey markers
    for i, (_, r) in enumerate(rows.iterrows()):
        if not np.isnan(r["marginal_or"]):
            ax.errorbar(
                r["marginal_or"], y[i] - 0.18,
                xerr=[[r["marginal_or"] - r["marginal_or_lo"]],
                      [r["marginal_or_hi"] - r["marginal_or"]]],
                fmt="s", color="#888", ecolor="#aaa", lw=0.7, ms=3, alpha=0.85,
                zorder=2,
            )

    # Isolated big colored markers
    label_styles = []
    for i, (_, r) in enumerate(rows.iterrows()):
        is_pos = r["polarity"] == "positive"
        rec = bool(r["recommended"])
        color = "#0072B2" if is_pos else "#D55E00"
        marker = "*" if rec else "o"
        ms = 11 if rec else 5
        if not np.isnan(r["isolated_or"]):
            ax.errorbar(
                r["isolated_or"], y[i],
                xerr=[[r["isolated_or"] - r["isolated_or_lo"]],
                      [r["isolated_or_hi"] - r["isolated_or"]]],
                fmt=marker, color=color, ecolor=color, lw=1.4, ms=ms,
                zorder=3,
            )
        label_styles.append("bold" if rec else "normal")

    ax.axvline(1.0, color="grey", ls=":", lw=0.8)
    ax.axhline(len(neg) - 0.5, color="grey", ls="-", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["factor_id"].tolist(), fontsize=7.5)
    for tick, style in zip(ax.get_yticklabels(), label_styles):
        tick.set_fontweight(style)
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_xscale("log")
    ax.set_title("Per-factor association with score increase\n"
                 "(recommendation = isolated OR + marginal OR both reject codebook null)")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
               markersize=7, label="positive (codebook)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00",
               markersize=7, label="negative (codebook)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#444",
               markersize=11, label="recommended"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
               markersize=4, label="marginal OR"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7,
              framealpha=0.9)
    fig.tight_layout()
    return fig


def fit_marginal_per_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Univariate logistic per factor, paper-clustered SE.

    Tests whether each factor *alone* (no other factors, no structural
    confounders) is associated with score increase. Disambiguates whether
    the conditional null on negative factors in the headline model comes
    from multicollinearity (negatives recover OR<1 here) or from structural
    confounding (negatives stay near 1 even alone).
    """
    rows = []
    y = df["score_increase"].astype(int).to_numpy()
    groups = df["paper_key"].to_numpy()
    for fid in FACTOR_COLS:
        s = pd.to_numeric(df[fid], errors="coerce")
        prev = float(s.mean())
        if prev < 0.01 or prev > 0.99:
            rows.append({"feature": fid, "n": int(s.notna().sum()),
                         "prev": prev, "odds_ratio": np.nan,
                         "or_lo": np.nan, "or_hi": np.nan, "p": np.nan,
                         "skipped_low_var": True})
            continue
        X = sm.add_constant(s.fillna(0).astype(float).to_frame(fid),
                            has_constant="add")
        try:
            res = sm.Logit(y, X).fit(
                disp=0, method="lbfgs", maxiter=500,
                cov_type="cluster", cov_kwds={"groups": groups},
            )
            coef = float(res.params[fid])
            ci = res.conf_int(alpha=0.05)
            ci_lo, ci_hi = float(ci.loc[fid, 0]), float(ci.loc[fid, 1])
            p = float(res.pvalues[fid])
            rows.append({"feature": fid, "n": int(res.nobs), "prev": prev,
                         "odds_ratio": float(np.exp(coef)),
                         "or_lo": float(np.exp(ci_lo)),
                         "or_hi": float(np.exp(ci_hi)),
                         "p": p, "skipped_low_var": False})
        except Exception as e:
            rows.append({"feature": fid, "n": 0, "prev": prev,
                         "odds_ratio": np.nan, "or_lo": np.nan,
                         "or_hi": np.nan, "p": np.nan,
                         "skipped_low_var": False,
                         "error": str(e)[:80]})
    return pd.DataFrame(rows)


def run_negatives_only(df: pd.DataFrame, save: bool) -> pd.DataFrame:
    """Refit without positive factors: negatives + structural + Gemini + tone.

    If negative-factor ORs recover the codebook direction (OR<1) here but
    not in the headline model, the conditional null on negatives is being
    driven by multicollinearity with the positive-factor block, not by
    structural confounding.
    """
    available = set(df.columns)
    feature_cols = (
        STRUCTURAL_CONFOUNDERS + REVIEW_TEXT_FEATURES + GEMINI_FEATURES
        + ["rebuttal_total_len", "n_author_replies"]
        + NEGATIVE_FACTORS  # only negatives, no positives
    )
    feature_cols = _resolve_feature_list(feature_cols, available)
    feature_cols += [c for c in df.columns if c.startswith("tone_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"\n[Negatives-only] features ({len(feature_cols)})")

    df_std = _standardise_continuous(df, feature_cols)
    feature_cols = _drop_collinear_and_low_variance(df_std, feature_cols)
    result = _fit_logit_clustered(df_std, feature_cols)
    coef_df = _coef_table(result, feature_cols)
    coef_df = _annotate_factor_significance(coef_df)
    sig_neg = coef_df[coef_df["feature"].isin(NEGATIVE_FACTORS)]
    n_or_lt_1 = int((sig_neg["odds_ratio"] < 1).sum())
    print(f"  Negative factors with OR<1: {n_or_lt_1} of {len(sig_neg)}")
    if save:
        coef_df.to_csv(TABLES_DIR / "coefficients_negatives_only.csv", index=False)
    return coef_df


def build_recommendation_table(
    headline: pd.DataFrame, marginal: pd.DataFrame,
    negatives_only: pd.DataFrame, save: bool,
) -> pd.DataFrame:
    """Per-factor: marginal + polarity-isolated conditional ORs + recommendation gate.

    Positives and negatives anti-correlate within rebuttals, so a single
    joint model crowds the negatives out (we verified this with the
    `negatives-only` refit). The recommendation gate is therefore asymmetric:

      positive factor "recommended" iff
        marginal OR > 1 AND headline-conditional OR > 1 AND both CIs exclude 1
      negative factor "recommended" iff
        marginal OR < 1 AND negatives-only-conditional OR < 1 AND both CIs exclude 1

    Both columns are reported in the table so the reader can see the joint
    headline OR for every factor too.
    """
    h = headline.set_index("feature")
    n = negatives_only.set_index("feature")
    m = marginal.set_index("feature")

    rows = []
    for fid in FACTOR_COLS:
        polarity = "positive" if fid in POSITIVE_FACTORS else "negative"
        target = (lambda x: x > 1) if polarity == "positive" else (lambda x: x < 1)

        # headline (always): the joint draft-time model's OR for this factor
        h_or = float(h.loc[fid, "odds_ratio"]) if fid in h.index else np.nan
        h_lo = float(h.loc[fid, "or_lo"]) if fid in h.index else np.nan
        h_hi = float(h.loc[fid, "or_hi"]) if fid in h.index else np.nan

        # marginal (always): univariate
        m_or = float(m.loc[fid, "odds_ratio"]) if fid in m.index else np.nan
        m_lo = float(m.loc[fid, "or_lo"]) if fid in m.index else np.nan
        m_hi = float(m.loc[fid, "or_hi"]) if fid in m.index else np.nan

        # polarity-isolated: which conditional model to gate on
        if polarity == "positive":
            iso = h
        else:
            iso = n
        iso_or = float(iso.loc[fid, "odds_ratio"]) if fid in iso.index else np.nan
        iso_lo = float(iso.loc[fid, "or_lo"]) if fid in iso.index else np.nan
        iso_hi = float(iso.loc[fid, "or_hi"]) if fid in iso.index else np.nan

        marginal_dir = bool(target(m_or)) if not np.isnan(m_or) else False
        iso_dir = bool(target(iso_or)) if not np.isnan(iso_or) else False
        marginal_ci_excludes_1 = (
            (m_lo > 1 or m_hi < 1) if not (np.isnan(m_lo) or np.isnan(m_hi)) else False
        )
        iso_ci_excludes_1 = (
            (iso_lo > 1 or iso_hi < 1) if not (np.isnan(iso_lo) or np.isnan(iso_hi)) else False
        )
        recommended = (
            marginal_dir and iso_dir
            and marginal_ci_excludes_1 and iso_ci_excludes_1
        )
        rows.append({
            "factor_id": fid,
            "polarity": polarity,
            "marginal_or": m_or, "marginal_or_lo": m_lo, "marginal_or_hi": m_hi,
            "conditional_or_headline": h_or,
            "conditional_or_headline_lo": h_lo,
            "conditional_or_headline_hi": h_hi,
            "isolated_or": iso_or,
            "isolated_or_lo": iso_lo,
            "isolated_or_hi": iso_hi,
            "isolated_model": "headline" if polarity == "positive" else "negatives_only",
            "marginal_dir_matches": marginal_dir,
            "isolated_dir_matches": iso_dir,
            "marginal_ci_excludes_1": marginal_ci_excludes_1,
            "isolated_ci_excludes_1": iso_ci_excludes_1,
            "recommended": recommended,
        })
    out = pd.DataFrame(rows)
    if save:
        out.to_csv(TABLES_DIR / "factor_recommendation_table.csv", index=False)

    n_pos_rec = int(((out["polarity"] == "positive") & out["recommended"]).sum())
    n_neg_rec = int(((out["polarity"] == "negative") & out["recommended"]).sum())
    print(f"\n[Recommendation gate]  "
          f"positive: {n_pos_rec} of 23  |  negative: {n_neg_rec} of 21")
    print("  Top 10 recommended (by |log isolated_or|):")
    rec = out[out["recommended"]].copy()
    rec["abs_log_or"] = np.log(rec["isolated_or"]).abs()
    for _, r in rec.sort_values("abs_log_or", ascending=False).head(10).iterrows():
        print(f"    {r['factor_id']:<6}  pol={r['polarity']:<8}  "
              f"marg OR={r['marginal_or']:.2f}  iso OR={r['isolated_or']:.2f} "
              f"[{r['isolated_or_lo']:.2f}, {r['isolated_or_hi']:.2f}] ({r['isolated_model']})")
    return out


def _build_predictive_design(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, dict]]:
    """Same feature set + standardisation as the headline model, but returns
    the design matrix and the per-feature standardisation params so we can
    transform synthetic rows for scenarios.
    """
    available = set(df.columns)
    feature_cols = (
        STRUCTURAL_CONFOUNDERS + REVIEW_TEXT_FEATURES + GEMINI_FEATURES
        + REBUTTAL_CONTENT_FEATURES
    )
    feature_cols = _resolve_feature_list(feature_cols, available)
    feature_cols += [c for c in df.columns if c.startswith("tone_") and c not in feature_cols]
    feature_cols = list(dict.fromkeys(feature_cols))

    # Track standardisation parameters per feature so we can scale scenario rows.
    std_params: dict[str, dict] = {}
    out = df.copy()
    for c in feature_cols:
        col = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float)
        if c in BINARY_FEATURES or c.startswith("tone_"):
            std_params[c] = {"mean": 0.0, "std": 1.0, "binary": True,
                             "raw_mean": float(col.mean())}
            out[c] = col
        else:
            mu, sd = float(col.mean()), float(col.std())
            std_params[c] = {"mean": mu, "std": sd if sd != 0 else 1.0,
                             "binary": False, "raw_mean": mu}
            out[c] = (col - mu) / (sd if sd != 0 else 1.0)
    feature_cols = _drop_collinear_and_low_variance(out, feature_cols)
    return out, feature_cols, std_params


def fit_predictive_model_on_2024(df: pd.DataFrame) -> tuple[
    sm.regression.linear_model.RegressionResults, list[str], dict[str, dict]
]:
    """Refit the draft-time logistic on 2024 only for honest 2025 calibration."""
    df_2024 = df[df["year"] == 2024].copy()
    df_std, feature_cols, std_params = _build_predictive_design(df_2024)
    print(f"\n[Predictive] fitting on 2024 only (n={len(df_std)}); "
          f"{len(feature_cols)} features")
    result = _fit_logit_clustered(df_std, feature_cols)
    return result, feature_cols, std_params


def _predict_proba(
    result: sm.regression.linear_model.RegressionResults,
    df: pd.DataFrame, feature_cols: list[str], std_params: dict[str, dict],
) -> np.ndarray:
    """Apply the same standardisation that train data saw, then predict."""
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
        p = std_params[c]
        if p["binary"]:
            X[c] = col
        else:
            X[c] = (col - p["mean"]) / (p["std"] if p["std"] != 0 else 1.0)
    X = sm.add_constant(X.astype(float), has_constant="add")
    return np.asarray(result.predict(X))


def calibration_on_temporal_test(
    result, df: pd.DataFrame, feature_cols: list[str],
    std_params: dict[str, dict], save: bool,
) -> pd.DataFrame:
    """Calibration: bin predicted P, plot observed increase rate per bin.

    Train-on-2024, evaluate-on-2025: honest out-of-time calibration.
    """
    df_2025 = df[df["year"] == 2025].copy()
    if len(df_2025) == 0:
        print("[Calibration] no 2025 rows — skipping")
        return pd.DataFrame()
    p_hat = _predict_proba(result, df_2025, feature_cols, std_params)
    y = df_2025["score_increase"].to_numpy()

    bins = np.linspace(0, 1, 11)
    df_cal = pd.DataFrame({"p_hat": p_hat, "y": y})
    df_cal["bin"] = pd.cut(df_cal["p_hat"], bins=bins, include_lowest=True, labels=False)
    grouped = df_cal.groupby("bin").agg(
        bin_low=("p_hat", "min"),
        bin_high=("p_hat", "max"),
        mean_pred=("p_hat", "mean"),
        observed_rate=("y", "mean"),
        n=("y", "size"),
    ).reset_index()
    print("\n[Calibration on 2025 (train 2024)]")
    print(grouped.to_string(index=False))
    if save:
        grouped.to_csv(TABLES_DIR / "calibration_temporal.csv", index=False)
        setup_publication_theme()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.8, label="perfect")
        ax.scatter(grouped["mean_pred"], grouped["observed_rate"],
                   s=grouped["n"].clip(lower=10), color="#0072B2",
                   edgecolor="white", linewidths=0.8, zorder=3,
                   label="2025 (out-of-time)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(score increase)")
        ax.set_ylabel("Observed score-increase rate")
        ax.set_title("Calibration of the rebuttal-thread-content pipeline\n"
                     "(train 2024, evaluate 2025; bubble area $\\propto$ n in bin)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "calibration_temporal.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "calibration_temporal.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return grouped


def scenario_demo(
    df: pd.DataFrame, result, feature_cols: list[str],
    std_params: dict[str, dict], rec_table: pd.DataFrame, save: bool,
) -> pd.DataFrame:
    """Predict P(increase) under a series of factor configurations.

    Baseline: a 'median rebuttal' on the 2024 cohort (median continuous
    features, median binary indicators, median tone). Scenarios then
    add or remove specific recommended factors and read off the
    pipeline's predicted probability.
    """
    df_2024 = df[df["year"] == 2024].copy()

    # Build the median rebuttal row in raw (pre-standardisation) units.
    base_row: dict[str, float] = {}
    for c in feature_cols:
        col = pd.to_numeric(df_2024[c], errors="coerce").fillna(0)
        if std_params[c]["binary"]:
            base_row[c] = float((col.mean() >= 0.5))
        else:
            base_row[c] = float(col.median())

    rec_pos = rec_table[(rec_table["polarity"] == "positive")
                        & rec_table["recommended"]].copy()
    rec_pos["log_or"] = np.log(rec_pos["isolated_or"])
    top_pos = rec_pos.sort_values("log_or", ascending=False)["factor_id"].tolist()

    rec_neg = rec_table[(rec_table["polarity"] == "negative")
                        & rec_table["recommended"]].copy()
    rec_neg["log_or"] = np.log(rec_neg["isolated_or"])
    top_neg = rec_neg.sort_values("log_or", ascending=True)["factor_id"].tolist()

    def _row_with(overrides: dict[str, float]) -> pd.DataFrame:
        r = dict(base_row)
        r.update(overrides)
        return pd.DataFrame([r])

    scenarios = [
        ("Median rebuttal (no recommended factors set)",
         {fid: 0 for fid in FACTOR_COLS if fid in feature_cols}),
        ("+ add F07 (baseline completion) only",
         {**{fid: 0 for fid in FACTOR_COLS if fid in feature_cols},
          **({"F07": 1} if "F07" in feature_cols else {})}),
        ("+ add top-3 recommended positives",
         {**{fid: 0 for fid in FACTOR_COLS if fid in feature_cols},
          **{fid: 1 for fid in top_pos[:3] if fid in feature_cols}}),
        ("+ add top-5 recommended positives",
         {**{fid: 0 for fid in FACTOR_COLS if fid in feature_cols},
          **{fid: 1 for fid in top_pos[:5] if fid in feature_cols}}),
        ("+ top-5 positives AND top-3 negatives also present",
         {**{fid: 0 for fid in FACTOR_COLS if fid in feature_cols},
          **{fid: 1 for fid in top_pos[:5] if fid in feature_cols},
          **{fid: 1 for fid in top_neg[:3] if fid in feature_cols}}),
        ("+ top-3 negatives only (and zero positives)",
         {**{fid: 0 for fid in FACTOR_COLS if fid in feature_cols},
          **{fid: 1 for fid in top_neg[:3] if fid in feature_cols}}),
    ]

    rows = []
    for label, overrides in scenarios:
        row_df = _row_with(overrides)
        # Set n_factors_present consistent with the overrides
        n_present = int(sum(int(row_df.iloc[0].get(fid, 0))
                            for fid in FACTOR_COLS if fid in feature_cols))
        if "n_factors_present" in feature_cols:
            row_df["n_factors_present"] = n_present
        p = float(_predict_proba(result, row_df, feature_cols, std_params)[0])
        rows.append({"scenario": label, "n_factors_present": n_present,
                     "p_increase": p})

    out = pd.DataFrame(rows)
    print("\n[Scenario demo on a 2024-median rebuttal profile]")
    for _, r in out.iterrows():
        print(f"  P(increase)={r['p_increase']:.3f}  ({r['scenario']})")
    if save:
        out.to_csv(TABLES_DIR / "scenario_predictions.csv", index=False)
    return out


def run_movability_strata(df: pd.DataFrame, save: bool) -> dict[str, pd.DataFrame]:
    """Stratify by deviation_from_paper_mean and refit headline within each.

    Splits the cohort into three movability strata using tertile cuts on
    deviation_from_paper_mean (below paper mean, near, above), then refits
    the same headline logistic within each stratum and reports the per-factor
    odds ratios. The motivation is presentational: deviation_from_paper_mean
    is by far the largest single predictor of post-rebuttal increase, so a
    reader who sees only the pooled fit might conclude that "rebuttals don't
    matter, only initial movability does". The strata view tests the
    complementary question: within a fixed movability bucket, do the
    rebuttal-content factors still associate with score increase?

    Returns a dict mapping stratum-name -> coefficient DataFrame, plus a
    "summary" entry with one row per (stratum, factor).
    """
    cuts = df["deviation_from_paper_mean"].quantile([0.0, 1.0/3, 2.0/3, 1.0]).values
    # Force the middle cut points to be deterministic across runs:
    lo_hi = (-0.5, 0.5)  # tertile cut points found empirically; round numbers
    df_b = df.copy()
    df_b["movability"] = pd.cut(
        df_b["deviation_from_paper_mean"],
        bins=[-np.inf, lo_hi[0], lo_hi[1], np.inf],
        labels=["below_paper_mean", "near_paper_mean", "above_paper_mean"],
        include_lowest=True,
    )
    print("\n[Movability strata] cohort sizes:")
    for label, sub in df_b.groupby("movability", observed=True):
        rate = sub["score_increase"].mean()
        print(f"  {label:<22}  n={len(sub):>4}  base_rate={rate:.3f}")

    strata_results: dict[str, pd.DataFrame] = {}
    for stratum, sub in df_b.groupby("movability", observed=True):
        if len(sub) < 200:
            print(f"  [skip] stratum {stratum} too small (n={len(sub)})")
            continue
        available = set(sub.columns)
        feature_cols = (
            STRUCTURAL_CONFOUNDERS + REVIEW_TEXT_FEATURES + GEMINI_FEATURES
            + REBUTTAL_CONTENT_FEATURES
        )
        feature_cols = _resolve_feature_list(feature_cols, available)
        feature_cols += [c for c in sub.columns
                         if c.startswith("tone_") and c not in feature_cols]
        feature_cols = list(dict.fromkeys(feature_cols))
        df_std = _standardise_continuous(sub, feature_cols)
        feature_cols = _drop_collinear_and_low_variance(df_std, feature_cols)
        try:
            result = _fit_logit_clustered(df_std, feature_cols)
        except Exception as e:
            print(f"  [warn] stratum {stratum} fit failed: {e}")
            continue
        coef = _coef_table(result, feature_cols)
        coef = _annotate_factor_significance(coef)
        coef["stratum"] = str(stratum)
        coef["n_stratum"] = len(sub)
        coef["base_rate_stratum"] = float(sub["score_increase"].mean())
        strata_results[str(stratum)] = coef
        n_sig = int(coef[coef["feature"].isin(FACTOR_COLS)]["bonferroni_sig"].sum())
        print(f"  {stratum:<22}  pseudo R-sq={result.prsquared:.3f}  "
              f"factors crossing Bonferroni={n_sig}")

    if save and strata_results:
        out_long = pd.concat(strata_results.values(), ignore_index=True)
        out_long.to_csv(TABLES_DIR / "coefficients_movability_strata.csv",
                        index=False)
    return strata_results


def plot_movability_strata(strata_results: dict[str, pd.DataFrame]) -> plt.Figure:
    """Compare per-stratum factor ORs as a 1xN forest panel.

    Y-axis is the union of factors that pass the recommendation gate in any
    stratum (so the panels are comparable). Each panel shows the odds ratio
    with cluster-robust 95% CI for one stratum, with markers coloured by
    polarity (blue = positive, orange = negative).
    """
    setup_publication_theme()
    if not strata_results:
        return plt.figure()

    # Pick the union of "interesting" factors: any factor with |log(OR)| > 0.3
    # in any stratum, plus the structural anchors so the reader sees scale.
    factors_to_show: list[str] = []
    for s, df in strata_results.items():
        f = df[df["feature"].isin(FACTOR_COLS) &
               (df["odds_ratio"].apply(np.log).abs() > 0.3)]
        factors_to_show.extend(f["feature"].tolist())
    factors_to_show = list(dict.fromkeys(factors_to_show))
    if not factors_to_show:
        return plt.figure()

    # Sort by mean log(OR) across strata for stable y-ordering.
    mean_log_or = {}
    for f in factors_to_show:
        vals = []
        for df in strata_results.values():
            sub = df[df["feature"] == f]
            if len(sub):
                vals.append(np.log(float(sub["odds_ratio"].iloc[0])))
        mean_log_or[f] = float(np.mean(vals)) if vals else 0.0
    factors_to_show = sorted(factors_to_show, key=lambda f: mean_log_or[f])

    n_strata = len(strata_results)
    fig, axes = plt.subplots(
        1, n_strata, figsize=(4.5 * n_strata, max(5, len(factors_to_show) * 0.28)),
        sharey=True,
    )
    if n_strata == 1:
        axes = [axes]
    y = np.arange(len(factors_to_show))

    for ax, (stratum, df) in zip(axes, strata_results.items()):
        for i, f in enumerate(factors_to_show):
            sub = df[df["feature"] == f]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            is_pos = f in set(POSITIVE_FACTORS)
            color = "#0072B2" if is_pos else "#D55E00"
            bonf = bool(r.get("bonferroni_sig", False))
            ms = 8 if bonf else 5
            marker = "*" if bonf else "o"
            ax.errorbar(
                r["odds_ratio"], y[i],
                xerr=[[r["odds_ratio"] - r["or_lo"]],
                      [r["or_hi"] - r["odds_ratio"]]],
                fmt=marker, color=color, ecolor=color, lw=1.2, ms=ms,
            )
        ax.axvline(1.0, color="grey", ls=":", lw=0.8)
        ax.set_xscale("log")
        n = int(df["n_stratum"].iloc[0])
        rate = float(df["base_rate_stratum"].iloc[0])
        ax.set_title(f"{stratum.replace('_',' ')}\nn={n}, base rate={rate:.2f}",
                     fontsize=10)
        ax.set_xlabel("Odds ratio (log scale)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(factors_to_show, fontsize=8)
    fig.suptitle("Within-movability factor ORs (deviation tertiles)", y=1.02)
    fig.tight_layout()
    return fig


def compare_runs(headline: pd.DataFrame, no_struct: pd.DataFrame,
                 strata: pd.DataFrame, save: bool) -> None:
    """Side-by-side OR comparison for the 44 factors across runs."""
    def _factor_or(df: pd.DataFrame) -> pd.Series:
        return df.set_index("feature").reindex(FACTOR_COLS)["odds_ratio"]

    cmp = pd.DataFrame({
        "factor_id": FACTOR_COLS,
        "or_headline": _factor_or(headline).values,
        "or_no_structural": _factor_or(no_struct).values,
        "or_strata_fe": _factor_or(strata).values,
    })
    cmp["delta_no_struct_pct"] = (
        (cmp["or_no_structural"] - cmp["or_headline"]) / cmp["or_headline"] * 100
    )
    cmp["delta_strata_pct"] = (
        (cmp["or_strata_fe"] - cmp["or_headline"]) / cmp["or_headline"] * 100
    )
    print("\n[Comparison] 44-factor OR shifts (median absolute % change vs headline):")
    print(f"  no_structural : {cmp['delta_no_struct_pct'].abs().median():.1f}%")
    print(f"  strata_fe     : {cmp['delta_strata_pct'].abs().median():.1f}%")
    if save:
        cmp.to_csv(TABLES_DIR / "factor_or_comparison.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(save: bool = True) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = _build_combined_dataframe()
    headline = run_headline(df, save=save)
    print("\n[Marginal univariate per-factor refit] ...")
    marginal = fit_marginal_per_factor(df)
    if save:
        marginal.to_csv(TABLES_DIR / "coefficients_marginal_per_factor.csv", index=False)
    n_neg_marg = int(((marginal["odds_ratio"] < 1)
                      & marginal["feature"].isin(NEGATIVE_FACTORS)).sum())
    n_pos_marg = int(((marginal["odds_ratio"] > 1)
                      & marginal["feature"].isin(POSITIVE_FACTORS)).sum())
    print(f"  positive factors with marginal OR>1: {n_pos_marg} of {len(POSITIVE_FACTORS)}")
    print(f"  negative factors with marginal OR<1: {n_neg_marg} of {len(NEGATIVE_FACTORS)}")

    no_struct = run_ablation_no_structural(df, save=save)
    negatives_only = run_negatives_only(df, save=save)
    strata = run_ablation_strata_fe(df, save=save)
    movability = run_movability_strata(df, save=save)
    if save and movability:
        fig_mov = plot_movability_strata(movability)
        fig_mov.savefig(FIGURES_DIR / "forest_movability_strata.pdf",
                        bbox_inches="tight")
        fig_mov.savefig(FIGURES_DIR / "forest_movability_strata.png", dpi=300,
                        bbox_inches="tight")
        plt.close(fig_mov)
    compare_runs(headline, no_struct, strata, save=save)
    rec_table = build_recommendation_table(headline, marginal, negatives_only, save=save)
    if save:
        fig = plot_recommendation_forest(rec_table)
        fig.savefig(FIGURES_DIR / "forest_recommendation.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "forest_recommendation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Predictive use of the same model class: calibration + scenarios -----
    pred_result, pred_features, std_params = fit_predictive_model_on_2024(df)
    calibration_on_temporal_test(pred_result, df, pred_features, std_params, save=save)
    scenario_demo(df, pred_result, pred_features, std_params, rec_table, save=save)

    print("\n=== Sign-consistency test (codebook polarity vs. observed OR) ===")
    for run_name, df_coef in [
        ("headline", headline),
        ("no_structural", no_struct),
        ("strata_fe", strata),
    ]:
        result = sign_consistency_test(df_coef)
        print(f"  {run_name}:")
        for pol, stats in result.items():
            print(f"    {pol:<8}  {stats['n_correct']}/{stats['n']} {stats['expected_direction']}  "
                  f"p_binomial={stats['p_binomial']:.3g}")

    if save:
        sign_rows = []
        for run_name, df_coef in [
            ("headline", headline),
            ("no_structural", no_struct),
            ("strata_fe", strata),
        ]:
            result = sign_consistency_test(df_coef)
            for pol, stats in result.items():
                sign_rows.append({"run": run_name, "polarity": pol, **stats})
        pd.DataFrame(sign_rows).to_csv(
            TABLES_DIR / "sign_consistency_test.csv", index=False,
        )

        fig = plot_forest_factors(headline, no_struct=no_struct, strata_fe=strata)
        fig.savefig(FIGURES_DIR / "forest_factors.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "forest_factors.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    main(save=not args.no_save)
