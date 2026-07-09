"""
Paper: Section 3.1 (LLM as instrument) -- text-vs-score gap and offset funnel.

Gemini over-estimation analysis: why does Gemini predict higher than actual
reviewer scores, and what initial review features predict score increases?

Part A: Funnel/prevalence (total reviews -> predictions -> external -> changes)
Part B: Characterizing over-estimation (engagement, tone, confidence, stance)
Part C: Predicting over-estimation vs score increase (dual logistic models)
Part D: Mediation analysis (is Gemini offset mediated by engagement quality?)

Usage:
    python src/analyze/gemini_overestimation_analysis.py
    python src/analyze/gemini_overestimation_analysis.py --year 2025
    python src/analyze/gemini_overestimation_analysis.py --no-save
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import gc
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import average_precision_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from core.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from core.plot_theme import YEAR_COLORS, setup_publication_theme

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
FIGURES_DIR = RESULTS_DIR / "31_text_score_gap" / "figures"
TABLES_DIR = RESULTS_DIR / "31_text_score_gap" / "tables"
BASE_DIR = Path(__file__).resolve().parent.parent.parent

ICLR_SCALE = [1, 3, 5, 6, 8, 10]
SCORE_TO_POS = {s: i for i, s in enumerate(ICLR_SCALE)}

GEMINI_FILES = {
    2024: "gemini_flash_iclr2024_full.json",
    2025: "gemini_flash_iclr2025_full.json",
}

ENGAGEMENT_LEVELS = ["superficial", "moderate", "thorough", "exceptional"]
ENGAGEMENT_TO_ORD = {e: i for i, e in enumerate(ENGAGEMENT_LEVELS)}

CANONICAL_INDICATORS = [
    "technical_depth", "actionable_feedback",
    "specific_examples", "trivial_feedback",
]

TONE_CATEGORIES = [
    "constructive", "critical", "enthusiastic", "neutral",
    "cautious", "dismissive", "supportive",
]

INTENSITY_MAP = {"null": 0, None: 0, "low": 1, "medium": 2, "high": 3}
LEVEL_MAP = {"none": 0, "low": 1, "minor": 1, "moderate": 2, "high": 3, "serious": 3}

OFFSET_COLORS = {"under": "#d9534f", "match": "#999999", "over": "#56B4E9"}
OFFSET_ORDER = ["under", "match", "over"]
OFFSET_LABELS = {"under": "Under-estimate", "match": "Match", "over": "Over-estimate"}


def save_figure(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{name}.png"
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {png.name}, {pdf.name}")
    return png, pdf


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------
def cramers_v(ct: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum()
    r, c = ct.shape
    return np.sqrt(chi2 / (n * min(r - 1, c - 1))) if n > 0 else 0.0


def epsilon_squared(H: float, n: int, k: int) -> float:
    return (H - k + 1) / (n - k) if n > k else 0.0


def rank_biserial(U: float, n1: int, n2: int) -> float:
    return 1 - 2 * U / (n1 * n2) if n1 * n2 > 0 else 0.0


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - spread), min(1, center + spread))


def bonferroni_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_gemini_features(year: int) -> pd.DataFrame:
    """Read per-reviewer + paper-level Gemini features from the parquet.

    Per-reviewer fields are stored raw in the parquet (e.g. tone primary
    as the LLM emitted it) and encoded here via ENGAGEMENT_TO_ORD,
    INTENSITY_MAP, LEVEL_MAP.
    """
    df = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")
    needed = {
        "paper_key", "review_order",
        "gemini_tone_primary", "gemini_tone_secondary",
        "gemini_engagement_quality", "gemini_score_confidence_llm",
        "gemini_indicators_json",
        "gemini_sentiment_score", "gemini_sentiment_label",
        "gemini_stance_excitement_intensity",
        "gemini_stance_frustration_intensity",
        "gemini_stance_disappointment_intensity",
        "gemini_stance_appreciation_intensity",
        "gemini_stance_confidence_in_contribution",
        "gemini_stance_concern_about_validity",
    }
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Parquet missing Gemini columns {missing}. "
            f"Run scripts/build_fulltext_parquet.py to rebuild."
        )
    sub = df[list(needed)].copy()
    sub = sub[sub["gemini_tone_primary"].notna()]
    indicators_lists = sub["gemini_indicators_json"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s else []
    )

    rows = []
    for (_, r), inds in zip(sub.iterrows(), indicators_lists):
        tone_secondary = r["gemini_tone_secondary"]
        if tone_secondary in (None, "null", ""):
            tone_secondary = "none"
        engagement = r["gemini_engagement_quality"]
        score_conf = r["gemini_score_confidence_llm"]

        row = {
            "paper_key": str(r["paper_key"]),
            "review_order": int(r["review_order"]),
            "tone_primary": r["gemini_tone_primary"] or "unknown",
            "tone_secondary": tone_secondary,
            "engagement_quality": engagement if engagement else "unknown",
            "engagement_ord": ENGAGEMENT_TO_ORD.get(engagement, -1),
            "score_confidence": float(score_conf) if pd.notna(score_conf) else np.nan,
            "n_indicators": len(inds),
            "sentiment_score": (
                float(r["gemini_sentiment_score"])
                if pd.notna(r["gemini_sentiment_score"]) else None
            ),
            "sentiment_label": r["gemini_sentiment_label"],
            "excitement_int": INTENSITY_MAP.get(r["gemini_stance_excitement_intensity"], 0),
            "frustration_int": INTENSITY_MAP.get(r["gemini_stance_frustration_intensity"], 0),
            "disappointment_int": INTENSITY_MAP.get(r["gemini_stance_disappointment_intensity"], 0),
            "appreciation_int": INTENSITY_MAP.get(r["gemini_stance_appreciation_intensity"], 0),
            "confidence_contrib": (
                LEVEL_MAP.get(r["gemini_stance_confidence_in_contribution"], 0)
                if r["gemini_stance_confidence_in_contribution"] else 0
            ),
            "concern_validity": (
                LEVEL_MAP.get(r["gemini_stance_concern_about_validity"], 0)
                if r["gemini_stance_concern_about_validity"] else 0
            ),
        }
        for ind in CANONICAL_INDICATORS:
            row[f"has_{ind}"] = 1 if ind in inds else 0
        rows.append(row)

    print(f"  Loaded Gemini features for {len(rows):,} reviewer entries")
    return pd.DataFrame(rows)


def extract_text_features(year: int) -> pd.DataFrame:
    """Read review text length features from the per-year parquet."""
    df = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")
    needed = {"paper_key", "reviewer_suffix", "review_order",
              "strengths_len", "weaknesses_len", "questions_len"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Parquet missing text columns {missing}. "
            f"Run scripts/build_fulltext_parquet.py to rebuild."
        )
    out = df[list(needed)].copy()
    out["sw_ratio"] = out["strengths_len"] / out["weaknesses_len"].clip(lower=1)
    return out

def build_feature_matrix(year: int, require_external: bool = True) -> pd.DataFrame:
    """Combine parquet + Gemini features + text features."""
    print(f"\nBuilding feature matrix for ICLR {year}...")

    df = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")

    # Filter to reviews with Gemini predictions
    mask = df["gemini_pred_score"].notna()
    if require_external:
        mask = mask & df["ext_rating_before"].notna() & df["ext_rating_after"].notna()
    df = df[mask].copy()

    df["gemini_pred_score"] = df["gemini_pred_score"].astype(int)
    if "ext_rating_before" in df.columns and df["ext_rating_before"].notna().any():
        df["ext_rating_before"] = pd.to_numeric(df["ext_rating_before"], errors="coerce")
        df["ext_rating_after"] = pd.to_numeric(df["ext_rating_after"], errors="coerce")
        df.loc[df["ext_rating_before"].notna(), "ext_rating_before"] = (
            df.loc[df["ext_rating_before"].notna(), "ext_rating_before"].astype(int)
        )
        df.loc[df["ext_rating_after"].notna(), "ext_rating_after"] = (
            df.loc[df["ext_rating_after"].notna(), "ext_rating_after"].astype(int)
        )

    # Core computed features
    df["pos_pred"] = df["gemini_pred_score"].map(SCORE_TO_POS)
    if require_external:
        df["pos_before"] = df["ext_rating_before"].map(SCORE_TO_POS)
        df["gemini_offset"] = df["pos_pred"] - df["pos_before"]
        df["offset_cat"] = pd.cut(
            df["gemini_offset"],
            bins=[-999, -0.5, 0.5, 999],
            labels=["under", "match", "over"],
        )
        df["score_increased"] = (df["ext_rating_after"] > df["ext_rating_before"]).astype(int)
        df["is_overestimate"] = (df["gemini_offset"] > 0).astype(int)

    # Paper-level stats
    score_col = "ext_rating_before" if require_external else "rating"
    paper_stats = df.groupby("paper_key").agg(
        n_reviewers=(score_col, "count"),
        score_std=(score_col, "std"),
        score_mean=(score_col, "mean"),
    ).reset_index()
    paper_stats["score_std"] = paper_stats["score_std"].fillna(0)
    df = df.merge(paper_stats, on="paper_key", how="left")

    for col in ["soundness", "presentation", "contribution"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Review order for matching
    df = df.sort_values(["paper_key", "reviewer_id"])
    df["review_order"] = df.groupby("paper_key").cumcount()

    # Text features
    text_df = extract_text_features(year)
    df = df.merge(
        text_df, on=["paper_key", "reviewer_suffix"], how="left",
        suffixes=("", "_text"),
    )
    if "review_order_text" in df.columns:
        df["review_order"] = df["review_order_text"].fillna(df["review_order"]).astype(int)
        df.drop(columns=["review_order_text"], inplace=True)

    # Gemini features (extended)
    gemini_df = extract_gemini_features(year)
    if len(gemini_df) > 0:
        df = df.merge(gemini_df, on=["paper_key", "review_order"], how="left")

    print(f"  Feature matrix: {len(df):,} rows, {len(df.columns)} columns")
    if "score_increased" in df.columns:
        print(f"  Target balance: {df['score_increased'].mean()*100:.1f}% increased")

    return df


# ===========================================================================
# PART A: Funnel / Prevalence
# ===========================================================================
def compute_funnel_stats(year: int) -> dict:
    """Compute funnel counts from raw parquet data."""
    df = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")
    total = len(df)
    has_gemini = int(df["gemini_pred_score"].notna().sum())
    has_ext = int(
        (df["ext_rating_before"].notna() & df["ext_rating_after"].notna()).sum()
    )
    has_both = int(
        (
            df["gemini_pred_score"].notna()
            & df["ext_rating_before"].notna()
            & df["ext_rating_after"].notna()
        ).sum()
    )

    # Score changes in the "both" subset
    both_df = df[
        df["gemini_pred_score"].notna()
        & df["ext_rating_before"].notna()
        & df["ext_rating_after"].notna()
    ].copy()
    both_df["ext_rating_before"] = both_df["ext_rating_before"].astype(int)
    both_df["ext_rating_after"] = both_df["ext_rating_after"].astype(int)
    both_df["gemini_pred_score"] = both_df["gemini_pred_score"].astype(int)

    increased = int((both_df["ext_rating_after"] > both_df["ext_rating_before"]).sum())
    decreased = int((both_df["ext_rating_after"] < both_df["ext_rating_before"]).sum())
    same = int((both_df["ext_rating_after"] == both_df["ext_rating_before"]).sum())

    # By offset group
    both_df["pos_pred"] = both_df["gemini_pred_score"].map(SCORE_TO_POS)
    both_df["pos_before"] = both_df["ext_rating_before"].map(SCORE_TO_POS)
    both_df["offset"] = both_df["pos_pred"] - both_df["pos_before"]
    both_df["inc"] = (both_df["ext_rating_after"] > both_df["ext_rating_before"]).astype(int)

    offset_stats = {}
    for cat, mask in [
        ("under", both_df["offset"] < 0),
        ("match", both_df["offset"] == 0),
        ("over", both_df["offset"] > 0),
    ]:
        sub = both_df[mask]
        offset_stats[cat] = {
            "n": len(sub),
            "increased": int(sub["inc"].sum()),
            "rate": sub["inc"].mean() if len(sub) > 0 else 0,
        }

    return {
        "year": year,
        "total": total,
        "has_gemini": has_gemini,
        "has_ext": has_ext,
        "has_both": has_both,
        "increased": increased,
        "decreased": decreased,
        "same": same,
        "offset": offset_stats,
    }


def print_funnel(funnel_stats: list[dict]):
    """Print funnel table to console."""
    print("\n" + "=" * 70)
    print("PART A: Funnel / Prevalence")
    print("=" * 70)

    for s in funnel_stats:
        y = s["year"]
        print(f"\n  ICLR {y}:")
        print(f"    Total reviews:          {s['total']:>8,}")
        print(f"    With Gemini predictions: {s['has_gemini']:>8,}  ({s['has_gemini']/s['total']*100:.1f}%)")
        print(f"    With external data:      {s['has_ext']:>8,}  ({s['has_ext']/s['total']*100:.1f}%)")
        print(f"    With both:               {s['has_both']:>8,}  ({s['has_both']/s['total']*100:.1f}%)")
        print(f"    Score increased:          {s['increased']:>8,}  ({s['increased']/max(s['has_both'],1)*100:.1f}% of both)")
        print(f"    Score decreased:          {s['decreased']:>8,}  ({s['decreased']/max(s['has_both'],1)*100:.1f}%)")
        print(f"    Score same:               {s['same']:>8,}  ({s['same']/max(s['has_both'],1)*100:.1f}%)")
        print(f"    By offset group:")
        for cat in OFFSET_ORDER:
            o = s["offset"][cat]
            pct = o["n"] / max(s["has_both"], 1) * 100
            print(f"      {OFFSET_LABELS[cat]:>16s}: n={o['n']:>6,} ({pct:>5.1f}%)  increase rate={o['rate']*100:.1f}%")


def plot_funnel_chart(funnel_stats: list[dict]) -> plt.Figure:
    """Horizontal bar chart showing the funnel stages."""
    stages = ["Total reviews", "With Gemini pred.", "With external data", "With both", "Score increased"]
    fig, ax = plt.subplots(figsize=(9, 4))

    n_years = len(funnel_stats)
    bar_h = 0.35
    y_pos = np.arange(len(stages))

    for i, s in enumerate(funnel_stats):
        counts = [s["total"], s["has_gemini"], s["has_ext"], s["has_both"], s["increased"]]
        offset = (i - n_years / 2 + 0.5) * bar_h
        bars = ax.barh(
            y_pos + offset, counts, bar_h,
            label=f"ICLR {s['year']}",
            color=YEAR_COLORS.get(s["year"], "#666"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, count, total in zip(bars, counts, [s["total"]] * 5):
            pct = count / total * 100
            ax.text(
                bar.get_width() + max(s["total"] * 0.01, 100),
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.0f}%)",
                va="center", fontsize=7,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.set_xlabel("Number of reviews")
    ax.set_title("Data availability funnel")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_offset_breakdown(funnel_stats: list[dict]) -> plt.Figure:
    """Grouped bar chart: offset group counts + increase rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_years = len(funnel_stats)
    x = np.arange(len(OFFSET_ORDER))
    bar_w = 0.35

    for i, s in enumerate(funnel_stats):
        offset_data = s["offset"]
        counts = [offset_data[c]["n"] for c in OFFSET_ORDER]
        rates = [offset_data[c]["rate"] * 100 for c in OFFSET_ORDER]
        off = (i - n_years / 2 + 0.5) * bar_w
        year_color = YEAR_COLORS.get(s["year"], "#666")

        # Counts
        bars1 = ax1.bar(
            x + off, counts, bar_w,
            label=f"ICLR {s['year']}", color=year_color,
            edgecolor="white", linewidth=0.5,
        )
        for bar, count in zip(bars1, counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center", va="bottom", fontsize=7,
            )

        # Rates
        bars2 = ax2.bar(
            x + off, rates, bar_w,
            label=f"ICLR {s['year']}", color=year_color,
            edgecolor="white", linewidth=0.5,
        )
        for bar, rate, count in zip(bars2, rates, counts):
            lo, hi = wilson_ci(rate / 100, count)
            ax2.errorbar(
                bar.get_x() + bar.get_width() / 2,
                rate, yerr=[[rate - lo * 100], [hi * 100 - rate]],
                color="black", capsize=3, linewidth=0.8,
            )
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (hi * 100 - rate) + 1,
                f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([OFFSET_LABELS[c] for c in OFFSET_ORDER])
    ax1.set_ylabel("Number of reviews")
    ax1.set_title("Reviews by Gemini offset group")
    ax1.legend()

    ax2.set_xticks(x)
    ax2.set_xticklabels([OFFSET_LABELS[c] for c in OFFSET_ORDER])
    ax2.set_ylabel("% reviews with score increase")
    ax2.set_title("Score increase rate by offset group")
    ax2.legend()

    fig.suptitle("Gemini offset groups: prevalence and rebuttal success", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def save_funnel_table(funnel_stats: list[dict]):
    """Save funnel data as CSV."""
    rows = []
    for s in funnel_stats:
        base = {
            "year": s["year"],
            "total_reviews": s["total"],
            "with_gemini": s["has_gemini"],
            "with_external": s["has_ext"],
            "with_both": s["has_both"],
            "score_increased": s["increased"],
            "score_decreased": s["decreased"],
            "score_same": s["same"],
            "increase_rate_pct": s["increased"] / max(s["has_both"], 1) * 100,
        }
        for cat in OFFSET_ORDER:
            o = s["offset"][cat]
            base[f"{cat}_n"] = o["n"]
            base[f"{cat}_increased"] = o["increased"]
            base[f"{cat}_increase_rate_pct"] = o["rate"] * 100
        rows.append(base)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "funnel_stats.csv", index=False)
    print(f"  Saved: funnel_stats.csv")


# ===========================================================================
# PART B: Characterizing over-estimation
# ===========================================================================
def compare_offset_groups(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Statistical comparison of features across offset groups."""
    print(f"\n  Comparing offset groups ({label}):")
    results = []
    groups = {cat: df[df["offset_cat"] == cat] for cat in OFFSET_ORDER}
    group_sizes = {cat: len(g) for cat, g in groups.items()}

    # Ordinal / continuous features
    ordinal_features = [
        "engagement_ord", "score_confidence", "n_indicators",
        "excitement_int", "frustration_int", "disappointment_int",
        "appreciation_int", "confidence_contrib", "concern_validity",
        "strengths_len", "weaknesses_len", "sw_ratio", "questions_len",
        "confidence", "soundness", "presentation", "contribution",
        "sentiment_score",
    ]

    for feat in ordinal_features:
        if feat not in df.columns:
            continue
        vals = [pd.to_numeric(groups[c][feat], errors="coerce").dropna() for c in OFFSET_ORDER]
        if any(len(v) < 5 for v in vals):
            continue

        H, p = stats.kruskal(*vals)
        n_total = sum(len(v) for v in vals)
        eps2 = epsilon_squared(H, n_total, 3)

        # Pairwise: over vs under
        u_stat, p_pair = stats.mannwhitneyu(vals[2], vals[0], alternative="two-sided")
        rb = rank_biserial(u_stat, len(vals[2]), len(vals[0]))

        results.append({
            "feature": feat,
            "type": "ordinal",
            "test": "Kruskal-Wallis",
            "statistic": H,
            "p_raw": p,
            "effect_size": eps2,
            "effect_name": "epsilon_sq",
            "pairwise_stat": u_stat,
            "pairwise_p": p_pair,
            "pairwise_effect": rb,
            "pairwise_effect_name": "rank_biserial",
            "mean_under": vals[0].mean(),
            "mean_match": vals[1].mean(),
            "mean_over": vals[2].mean(),
        })

    # Boolean indicator features
    for ind in CANONICAL_INDICATORS:
        col = f"has_{ind}"
        if col not in df.columns:
            continue
        ct = pd.crosstab(df["offset_cat"], df[col].fillna(0).astype(int))
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        # Reindex to ensure consistent order
        ct = ct.reindex(OFFSET_ORDER, fill_value=0)

        chi2, p, _, _ = stats.chi2_contingency(ct.values)
        cv = cramers_v(ct.values)

        results.append({
            "feature": col,
            "type": "boolean",
            "test": "chi-squared",
            "statistic": chi2,
            "p_raw": p,
            "effect_size": cv,
            "effect_name": "cramers_v",
            "pairwise_stat": np.nan,
            "pairwise_p": np.nan,
            "pairwise_effect": np.nan,
            "pairwise_effect_name": "",
            "mean_under": groups["under"][col].mean(),
            "mean_match": groups["match"][col].mean(),
            "mean_over": groups["over"][col].mean(),
        })

    # Categorical: engagement_quality, tone_primary
    for feat in ["engagement_quality", "tone_primary", "tone_secondary"]:
        if feat not in df.columns:
            continue
        ct = pd.crosstab(df["offset_cat"], df[feat])
        ct = ct.reindex(OFFSET_ORDER, fill_value=0)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        chi2, p, _, _ = stats.chi2_contingency(ct.values)
        cv = cramers_v(ct.values)

        results.append({
            "feature": feat,
            "type": "categorical",
            "test": "chi-squared",
            "statistic": chi2,
            "p_raw": p,
            "effect_size": cv,
            "effect_name": "cramers_v",
            "pairwise_stat": np.nan,
            "pairwise_p": np.nan,
            "pairwise_effect": np.nan,
            "pairwise_effect_name": "",
            "mean_under": np.nan,
            "mean_match": np.nan,
            "mean_over": np.nan,
        })

    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        res_df["p_adjusted"] = bonferroni_correct(res_df["p_raw"].tolist())
        res_df["significant"] = res_df["p_adjusted"] < 0.05

    # Print summary
    print(f"    {'Feature':>25s} | {'Test':>15s} | {'Stat':>10s} | {'p_adj':>10s} | {'Effect':>10s} | {'Under':>8s} | {'Match':>8s} | {'Over':>8s}")
    print("    " + "-" * 115)
    for _, r in res_df.iterrows():
        sig = " ***" if r.get("significant") else ""
        mean_u = f"{r['mean_under']:.3f}" if pd.notna(r["mean_under"]) else "---"
        mean_m = f"{r['mean_match']:.3f}" if pd.notna(r["mean_match"]) else "---"
        mean_o = f"{r['mean_over']:.3f}" if pd.notna(r["mean_over"]) else "---"
        print(
            f"    {r['feature']:>25s} | {r['test']:>15s} | {r['statistic']:>10.2f} | "
            f"{r['p_adjusted']:>10.2e} | {r['effect_size']:>10.4f} | "
            f"{mean_u:>8s} | {mean_m:>8s} | {mean_o:>8s}{sig}"
        )

    return res_df


def plot_engagement_by_offset(df: pd.DataFrame, label: str) -> plt.Figure:
    """Stacked bar chart: engagement quality distribution by offset group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ct = pd.crosstab(df["offset_cat"], df["engagement_quality"], normalize="index")
    ct = ct.reindex(index=OFFSET_ORDER, columns=ENGAGEMENT_LEVELS, fill_value=0)

    eng_colors = ["#f0f0f0", "#b0d0e8", "#56B4E9", "#1a6b99"]
    bottom = np.zeros(len(OFFSET_ORDER))
    x = np.arange(len(OFFSET_ORDER))

    for j, level in enumerate(ENGAGEMENT_LEVELS):
        vals = ct[level].values * 100
        ax.bar(
            x, vals, bottom=bottom, label=level.capitalize(),
            color=eng_colors[j], edgecolor="white", linewidth=0.5,
        )
        for k, v in enumerate(vals):
            if v > 3:
                ax.text(k, bottom[k] + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([OFFSET_LABELS[c] for c in OFFSET_ORDER])
    ax.set_ylabel("% of reviews")
    ax.set_title(f"{label} -- Engagement quality by Gemini offset group")
    ax.legend(title="Engagement", loc="upper right")
    fig.tight_layout()
    return fig


def plot_indicators_by_offset(df: pd.DataFrame, label: str) -> plt.Figure:
    """Grouped bar chart: indicator prevalence by offset group, with 95% CI."""
    fig, ax = plt.subplots(figsize=(9, 5))

    n_ind = len(CANONICAL_INDICATORS)
    n_cats = len(OFFSET_ORDER)
    bar_w = 0.25
    x = np.arange(n_ind)

    for i, cat in enumerate(OFFSET_ORDER):
        sub = df[df["offset_cat"] == cat]
        n_sub = len(sub)
        off = (i - n_cats / 2 + 0.5) * bar_w

        rates = []
        ci_lo = []
        ci_hi = []
        for ind in CANONICAL_INDICATORS:
            col = f"has_{ind}"
            p_hat = sub[col].mean() if col in sub.columns else 0
            lo, hi = wilson_ci(p_hat, n_sub)
            rates.append(p_hat * 100)
            ci_lo.append((p_hat - lo) * 100)
            ci_hi.append((hi - p_hat) * 100)

        bars = ax.bar(
            x + off, rates, bar_w,
            label=OFFSET_LABELS[cat], color=OFFSET_COLORS[cat],
            edgecolor="white", linewidth=0.5,
        )
        ax.errorbar(
            x + off, rates,
            yerr=[ci_lo, ci_hi],
            fmt="none", color="black", capsize=2, linewidth=0.7,
        )

    short_names = [ind.replace("_", "\n") for ind in CANONICAL_INDICATORS]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("% of reviews with indicator")
    ax.set_title(f"{label} -- Engagement indicators by offset group")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_tone_heatmap(df: pd.DataFrame, label: str) -> plt.Figure:
    """Heatmap: tone distribution across offset groups."""
    ct = pd.crosstab(df["tone_primary"], df["offset_cat"], normalize="columns")
    tones_present = [t for t in TONE_CATEGORIES if t in ct.index]
    ct = ct.reindex(index=tones_present, columns=OFFSET_ORDER, fill_value=0) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto", vmin=0)

    ax.set_xticks(range(len(OFFSET_ORDER)))
    ax.set_xticklabels([OFFSET_LABELS[c] for c in OFFSET_ORDER])
    ax.set_yticks(range(len(tones_present)))
    ax.set_yticklabels([t.capitalize() for t in tones_present])

    for i in range(len(tones_present)):
        for j in range(len(OFFSET_ORDER)):
            val = ct.values[i, j]
            color = "white" if val > 30 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% within offset group", fontsize=9)
    ax.set_title(f"{label} -- Primary tone by offset group")
    fig.tight_layout()
    return fig


def plot_stance_heatmap(df: pd.DataFrame, label: str) -> plt.Figure:
    """Heatmap: mean stance intensity across offset groups."""
    stance_cols = [
        "excitement_int", "frustration_int", "disappointment_int",
        "appreciation_int", "confidence_contrib", "concern_validity",
    ]
    stance_labels = [
        "Excitement", "Frustration", "Disappointment",
        "Appreciation", "Confidence\nin contrib.", "Concern\nabout validity",
    ]

    available = [c for c in stance_cols if c in df.columns]
    if not available:
        return plt.figure()

    data = []
    for cat in OFFSET_ORDER:
        sub = df[df["offset_cat"] == cat]
        means = [pd.to_numeric(sub[c], errors="coerce").mean() for c in available]
        data.append(means)
    mat = np.array(data).T  # rows = stance, cols = offset

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(len(OFFSET_ORDER)))
    ax.set_xticklabels([OFFSET_LABELS[c] for c in OFFSET_ORDER])
    avail_labels = [stance_labels[stance_cols.index(c)] for c in available]
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(avail_labels, fontsize=9)

    for i in range(len(available)):
        for j in range(len(OFFSET_ORDER)):
            val = mat[i, j]
            color = "white" if val > 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean intensity (0-3)", fontsize=9)
    ax.set_title(f"{label} -- Reviewer stance by offset group")
    fig.tight_layout()
    return fig


# ===========================================================================
# PART C: Predicting over-estimation vs score increase
# ===========================================================================
SHARED_FEATURES = [
    "pos_before", "confidence", "soundness", "presentation", "contribution",
    "strengths_len", "weaknesses_len", "sw_ratio", "questions_len",
    "n_reviewers", "score_std", "score_mean", "sentiment_score",
    "engagement_ord", "score_confidence", "n_indicators",
    "excitement_int", "frustration_int", "disappointment_int",
    "appreciation_int", "confidence_contrib", "concern_validity",
]


def prepare_model_features(
    df: pd.DataFrame, target_col: str, include_offset: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare feature matrix for a model."""
    feature_names = []
    arrays = []

    cols = list(SHARED_FEATURES)
    if include_offset:
        cols = ["gemini_offset"] + cols

    for col in cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
            arrays.append(vals.reshape(-1, 1))
            feature_names.append(col)

    # Indicator booleans
    for ind in CANONICAL_INDICATORS:
        col = f"has_{ind}"
        if col in df.columns:
            vals = df[col].fillna(0).astype(float).values
            arrays.append(vals.reshape(-1, 1))
            feature_names.append(col)

    # Tone one-hot
    if "tone_primary" in df.columns:
        for tone in TONE_CATEGORIES:
            vals = (df["tone_primary"] == tone).astype(float).values
            arrays.append(vals.reshape(-1, 1))
            feature_names.append(f"tone_{tone}")

    X = np.hstack(arrays)
    y = df[target_col].values
    return X, y, feature_names


def run_dual_models(df: pd.DataFrame, label: str) -> dict:
    """Run two logistic regressions: predicting over-estimation and score increase."""
    groups = df["paper_key"].values
    cv = GroupKFold(n_splits=5)
    results = {}

    for model_name, target, include_offset, desc in [
        ("overestimation", "is_overestimate", False, "Predicting Gemini over-estimation"),
        ("score_increase", "score_increased", True, "Predicting score increase"),
    ]:
        print(f"\n  Model: {desc} ({label})")
        X, y, feat_names = prepare_model_features(df, target, include_offset)
        print(f"    Features: {len(feat_names)}, samples: {len(y)}, positive rate: {y.mean()*100:.1f}%")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Grouped out-of-fold predictions (no same-paper leakage)
        y_prob = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in cv.split(X_scaled, y, groups):
            lr_fold = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            lr_fold.fit(X_scaled[train_idx], y[train_idx])
            y_prob[test_idx] = lr_fold.predict_proba(X_scaled[test_idx])[:, 1]

        auc = roc_auc_score(y, y_prob)
        auprc = average_precision_score(y, y_prob)
        print(f"    AUC-ROC: {auc:.4f}  AUPRC: {auprc:.4f}")

        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        lr.fit(X_scaled, y)
        coef_df = pd.DataFrame({
            "feature": feat_names,
            "coefficient": lr.coef_[0],
            "odds_ratio": np.exp(lr.coef_[0]),
        }).sort_values("coefficient", ascending=False)

        print(f"    Top features (by |coefficient|):")
        for _, r in coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index).head(10).iterrows():
            print(f"      {r['feature']:>25s}: coef={r['coefficient']:+.4f}  OR={r['odds_ratio']:.3f}")

        results[model_name] = {
            "auc": auc, "auprc": auprc, "y_prob": y_prob, "coef_df": coef_df,
            "model": lr, "scaler": scaler,
        }

    # Coefficient correlation
    m1_coefs = results["overestimation"]["coef_df"].set_index("feature")["coefficient"]
    m2_coefs = results["score_increase"]["coef_df"].set_index("feature")["coefficient"]
    shared = m1_coefs.index.intersection(m2_coefs.index)
    if len(shared) > 3:
        rho, p_corr = stats.spearmanr(m1_coefs[shared], m2_coefs[shared])
        print(f"\n  Coefficient similarity (Spearman): rho={rho:.3f}, p={p_corr:.2e}")
        results["coef_correlation"] = {"rho": rho, "p": p_corr, "n_shared": len(shared)}

    return results


def plot_dual_coefficients(results: dict, label: str) -> plt.Figure:
    """Side-by-side coefficient comparison."""
    c1 = results["overestimation"]["coef_df"].set_index("feature")["coefficient"]
    c2 = results["score_increase"]["coef_df"].set_index("feature")["coefficient"]
    shared = c1.index.intersection(c2.index)

    # Sort by absolute average importance
    avg_abs = ((c1[shared].abs() + c2[shared].abs()) / 2).sort_values()
    top_features = avg_abs.tail(20).index

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(top_features) * 0.3)), sharey=True)

    y_pos = range(len(top_features))

    colors1 = ["#d9534f" if c < 0 else "#56B4E9" for c in c1[top_features]]
    ax1.barh(y_pos, c1[top_features], color=colors1, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features, fontsize=8)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("Standardized coefficient")
    ax1.set_title("Predicting over-estimation")

    colors2 = ["#d9534f" if c < 0 else "#E69F00" for c in c2[top_features]]
    ax2.barh(y_pos, c2[top_features], color=colors2, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Standardized coefficient")
    ax2.set_title("Predicting score increase")

    corr_info = results.get("coef_correlation", {})
    if corr_info:
        fig.suptitle(
            f"{label} -- Dual model coefficients "
            f"(Spearman rho={corr_info['rho']:.3f}, p={corr_info['p']:.2e})",
            fontsize=12, fontweight="bold",
        )
    else:
        fig.suptitle(f"{label} -- Dual model coefficients", fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_dual_roc(results: dict, df: pd.DataFrame, label: str) -> plt.Figure:
    """ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, key, color in [
        ("Over-estimation", "overestimation", "#56B4E9"),
        ("Score increase", "score_increase", "#E69F00"),
    ]:
        r = results[key]
        target_col = "is_overestimate" if key == "overestimation" else "score_increased"
        y_true = df[target_col].values
        RocCurveDisplay.from_predictions(
            y_true, r["y_prob"],
            name=f"{name} (AUC={r['auc']:.3f})",
            ax=ax, color=color,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title(f"{label} -- ROC curves (dual models)")
    return fig


# ===========================================================================
# PART D: Mediation analysis
# ===========================================================================
def run_mediation(df: pd.DataFrame, label: str, n_boot: int = 1000) -> dict:
    """Baron-Kenny mediation: offset -> engagement -> score increase."""
    print(f"\n  Mediation analysis ({label}):")

    # Prepare data
    cols_needed = ["gemini_offset", "engagement_ord", "score_increased"]
    sub = df[cols_needed].dropna().copy()
    X_offset = sub["gemini_offset"].values.reshape(-1, 1)
    M_engage = sub["engagement_ord"].values
    Y_increase = sub["score_increased"].values

    # Path c: offset -> increase (total effect)
    lr_c = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_c.fit(StandardScaler().fit_transform(X_offset), Y_increase)
    c_coef = lr_c.coef_[0, 0]
    print(f"    Path c (total effect): coef={c_coef:.4f}, OR={np.exp(c_coef):.3f}")

    # Path a: offset -> engagement (OLS)
    ols_a = LinearRegression()
    ols_a.fit(X_offset, M_engage)
    a_coef = ols_a.coef_[0]
    a_pred = ols_a.predict(X_offset)
    ss_res = np.sum((M_engage - a_pred) ** 2)
    se_a = np.sqrt(ss_res / (len(M_engage) - 2) / np.sum((X_offset - X_offset.mean()) ** 2))
    print(f"    Path a (offset->engagement): coef={a_coef:.4f}, SE={se_a:.4f}")

    # Paths b + c': offset + engagement -> increase
    X_bc = np.column_stack([X_offset.ravel(), M_engage])
    scaler_bc = StandardScaler()
    X_bc_scaled = scaler_bc.fit_transform(X_bc)
    lr_bc = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_bc.fit(X_bc_scaled, Y_increase)
    c_prime = lr_bc.coef_[0, 0]
    b_coef = lr_bc.coef_[0, 1]
    print(f"    Path b (engagement->increase|offset): coef={b_coef:.4f}")
    print(f"    Path c' (direct effect): coef={c_prime:.4f}, OR={np.exp(c_prime):.3f}")

    # Bootstrap indirect effect
    print(f"    Bootstrapping indirect effect ({n_boot} iterations)...")
    rng = np.random.RandomState(42)
    indirect_effects = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        X_b = X_offset[idx]
        M_b = M_engage[idx]
        Y_b = Y_increase[idx]

        # Path a
        ols_b = LinearRegression()
        ols_b.fit(X_b, M_b)
        a_b = ols_b.coef_[0]

        # Path b
        X_bc_b = np.column_stack([X_b.ravel(), M_b])
        lr_b = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        try:
            lr_b.fit(StandardScaler().fit_transform(X_bc_b), Y_b)
            b_b = lr_b.coef_[0, 1]
            indirect_effects.append(a_b * b_b)
        except Exception:
            continue

    indirect_effects = np.array(indirect_effects)
    ci_lo = np.percentile(indirect_effects, 2.5)
    ci_hi = np.percentile(indirect_effects, 97.5)
    indirect_mean = np.mean(indirect_effects)
    significant = (ci_lo > 0) or (ci_hi < 0)

    # Proportion mediated
    prop_mediated = (c_coef - c_prime) / c_coef if abs(c_coef) > 1e-6 else 0

    print(f"    Indirect effect (a*b): {indirect_mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    Significant: {'Yes' if significant else 'No'} (95% CI excludes zero)")
    print(f"    Proportion mediated: {prop_mediated:.1%}")

    return {
        "c": c_coef, "a": a_coef, "b": b_coef, "c_prime": c_prime,
        "indirect_mean": indirect_mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "significant": significant, "prop_mediated": prop_mediated,
        "n": n, "n_boot": n_boot,
    }


def plot_mediation_diagram(med: dict, label: str) -> plt.Figure:
    """Path diagram for mediation analysis."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Boxes
    box_style = dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#333")
    ax.text(1, 1.5, "Gemini\nOffset", ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=box_style)
    ax.text(5, 4.5, "Engagement\nQuality", ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=box_style)
    ax.text(9, 1.5, "Score\nIncreased", ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=box_style)

    # Path a: offset -> engagement
    ax.annotate("", xy=(4, 4.2), xytext=(2, 2.2),
                arrowprops=dict(arrowstyle="->", color="#56B4E9", lw=2))
    ax.text(2.5, 3.5, f"a = {med['a']:.3f}", fontsize=10, color="#56B4E9", fontweight="bold")

    # Path b: engagement -> increase
    ax.annotate("", xy=(8, 2.2), xytext=(6, 4.2),
                arrowprops=dict(arrowstyle="->", color="#E69F00", lw=2))
    ax.text(7.5, 3.5, f"b = {med['b']:.3f}", fontsize=10, color="#E69F00", fontweight="bold")

    # Path c / c': offset -> increase (direct)
    ax.annotate("", xy=(7.8, 1.5), xytext=(2.2, 1.5),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.text(5, 0.8,
            f"c = {med['c']:.3f}  (total)\nc' = {med['c_prime']:.3f}  (direct)",
            fontsize=10, ha="center", fontweight="bold")

    # Indirect effect box
    sig_str = "CI excludes 0" if med["significant"] else "CI includes 0"
    ax.text(5, 5.5,
            f"Indirect effect: {med['indirect_mean']:.4f} "
            f"[{med['ci_lo']:.4f}, {med['ci_hi']:.4f}]\n"
            f"{sig_str}  |  Proportion mediated: {med['prop_mediated']:.1%}",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffdd", edgecolor="#ccc"))

    ax.set_title(f"{label} -- Mediation: Offset -> Engagement -> Score increase",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def save_mediation_table(med: dict, slug: str):
    df = pd.DataFrame([{
        "path": "c (total)",
        "coefficient": med["c"],
        "interpretation": "Gemini offset -> score increase",
    }, {
        "path": "a",
        "coefficient": med["a"],
        "interpretation": "Gemini offset -> engagement quality",
    }, {
        "path": "b",
        "coefficient": med["b"],
        "interpretation": "Engagement -> score increase (controlling offset)",
    }, {
        "path": "c' (direct)",
        "coefficient": med["c_prime"],
        "interpretation": "Gemini offset -> score increase (controlling engagement)",
    }, {
        "path": "indirect (a*b)",
        "coefficient": med["indirect_mean"],
        "interpretation": f"95% CI: [{med['ci_lo']:.4f}, {med['ci_hi']:.4f}]",
    }, {
        "path": "proportion_mediated",
        "coefficient": med["prop_mediated"],
        "interpretation": f"n={med['n']:,}, bootstrap={med['n_boot']}",
    }])
    df.to_csv(TABLES_DIR / f"{slug}_mediation.csv", index=False)


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Gemini over-estimation analysis")
    parser.add_argument("--year", type=int, help="Single year (default: both 2024+2025)")
    parser.add_argument("--no-save", action="store_true", help="Display only")
    args = parser.parse_args()

    setup_publication_theme()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    years = [args.year] if args.year else [2024, 2025]

    # =====================================================================
    # PART A: Funnel
    # =====================================================================
    funnel_stats = [compute_funnel_stats(y) for y in years]
    print_funnel(funnel_stats)

    if not args.no_save:
        save_funnel_table(funnel_stats)
        fig = plot_funnel_chart(funnel_stats)
        save_figure(fig, "funnel_chart")
        plt.close(fig)
        fig = plot_offset_breakdown(funnel_stats)
        save_figure(fig, "offset_breakdown")
        plt.close(fig)

    # =====================================================================
    # PARTS B-D: Per year + combined
    # =====================================================================
    all_dfs = []
    for year in years:
        df = build_feature_matrix(year, require_external=True)
        df["year"] = year
        all_dfs.append(df)

    if not all_dfs:
        print("No data.")
        return

    datasets = [(f"ICLR {y}", d) for y, d in zip(years, all_dfs)]
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        datasets.append(("Combined 2024+2025", combined))

    for label, df in datasets:
        slug = label.lower().replace(" ", "_").replace("+", "")
        print(f"\n{'='*70}")
        print(f"{label} -- {len(df):,} reviews")
        print(f"{'='*70}")

        # Part B
        print(f"\n--- Part B: Characterizing offset groups ---")
        stats_df = compare_offset_groups(df, label)

        if not args.no_save and len(stats_df) > 0:
            stats_df.to_csv(TABLES_DIR / f"{slug}_feature_comparison.csv", index=False)

            fig = plot_engagement_by_offset(df, label)
            save_figure(fig, f"{slug}_engagement_by_offset")
            plt.close(fig)

            fig = plot_indicators_by_offset(df, label)
            save_figure(fig, f"{slug}_indicators_by_offset")
            plt.close(fig)

            fig = plot_tone_heatmap(df, label)
            save_figure(fig, f"{slug}_tone_heatmap")
            plt.close(fig)

            fig = plot_stance_heatmap(df, label)
            save_figure(fig, f"{slug}_stance_heatmap")
            plt.close(fig)

        # Part C
        print(f"\n--- Part C: Dual predictive models ---")
        model_results = run_dual_models(df, label)

        if not args.no_save:
            fig = plot_dual_coefficients(model_results, label)
            save_figure(fig, f"{slug}_dual_coefficients")
            plt.close(fig)

            fig = plot_dual_roc(model_results, df, label)
            save_figure(fig, f"{slug}_dual_roc")
            plt.close(fig)

            for key in ["overestimation", "score_increase"]:
                model_results[key]["coef_df"].to_csv(
                    TABLES_DIR / f"{slug}_{key}_coefficients.csv", index=False)

        # Part D
        print(f"\n--- Part D: Mediation analysis ---")
        if "engagement_ord" in df.columns and df["engagement_ord"].notna().sum() > 100:
            med = run_mediation(df, label)

            if not args.no_save:
                fig = plot_mediation_diagram(med, label)
                save_figure(fig, f"{slug}_mediation")
                plt.close(fig)
                save_mediation_table(med, slug)
        else:
            print("  Skipped: insufficient engagement data")

    print(f"\nFigures: {FIGURES_DIR}")
    print(f"Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
