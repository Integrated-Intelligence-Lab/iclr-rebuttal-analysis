"""
What triggers score changes for low- vs high-confidence reviewers?

Groups (below-mean outlier reviewers):
  1a) Low confidence  + no increase
  1b) Low confidence  + increase
  2a) High confidence + no increase
  2b) High confidence + increase

Potential triggers:
  A. Peer pressure: deviation from mean, co-reviewer movement, paper decision
  B. Rebuttal effort: author reply length/count to this reviewer
  C. Review characteristics: sub-scores, text lengths, questions
  D. Engagement: Gemini engagement quality, actionable feedback

Usage:
    python src/analyze/confidence_triggers.py
    python src/analyze/confidence_triggers.py --no-save
    python src/analyze/confidence_triggers.py --threshold 2   # sensitivity check
    python src/analyze/confidence_triggers.py --sensitivity    # run all thresholds (2,3,4)
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
import pyarrow.parquet as pq
from scipy import stats

from core.paths import PROCESSED_DATA_DIR, RESULTS_DIR
from core.plot_theme import setup_publication_theme

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
FIGURES_DIR = RESULTS_DIR / "confidence_triggers" / "figures"
TABLES_DIR = RESULTS_DIR / "confidence_triggers" / "tables"
BASE_DIR = Path(__file__).resolve().parent.parent.parent

ICLR_SCALE = [1, 3, 5, 6, 8, 10]
SCORE_TO_POS = {s: i for i, s in enumerate(ICLR_SCALE)}

GEMINI_FILES = {
    2024: "gemini_3flash_iclr2024_full.json",
    2025: "gemini_31flash_iclr2025_full.json",
}
ENGAGEMENT_TO_ORD = {"superficial": 0, "moderate": 1, "thorough": 2, "exceptional": 3}

GROUP_COLORS = {
    "Low conf\nno increase": "#f4a582",
    "Low conf\nincrease": "#d6604d",
    "High conf\nno increase": "#92c5de",
    "High conf\nincrease": "#2166ac",
}
GROUP_ORDER = list(GROUP_COLORS.keys())


def save_figure(fig: plt.Figure, name: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{name}.png"
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {png.name}, {pdf.name}")
    return png, pdf


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def extract_rebuttal_features(year: int) -> pd.DataFrame:
    """Per-reviewer rebuttal + review-text-length features.

    Public-release path: read the bundled analysis-ready parquet, which already
    carries every rebuttal aggregate as a column. Only fall back to the heavy raw
    normalized JSON in the developer tree, where the parquet may be absent. This
    keeps sec33/sec34 and the leakage test runnable from a plain clone, since the
    normalized JSON is not shipped (it lives on the Zenodo record / HF mirror).
    """
    parquet_path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    reb_cols = [
        "paper_key", "reviewer_suffix", "review_order",
        "has_rebuttal", "n_author_replies", "rebuttal_total_len",
        "n_thread_messages", "first_reply_latency_days", "thread_max_depth",
        "strengths_len", "weaknesses_len", "questions_len",
    ]
    if parquet_path.exists() and set(reb_cols).issubset(pq.read_schema(parquet_path).names):
        print(f"  Loading rebuttal features from parquet ({year})...")
        df = pd.read_parquet(parquet_path, columns=reb_cols)
        df["paper_key"] = df["paper_key"].astype(str)
        # sw_ratio is the one derived column not stored in the parquet.
        df["sw_ratio"] = df["strengths_len"] / df["weaknesses_len"].clip(lower=1)
        return df

    # Developer-tree fallback: rebuild the features from the raw normalized JSON.
    norm_path = BASE_DIR / "data" / "processed" / "normalized" / f"ICLR_{year}_normalized.json"
    print(f"  Loading normalized data for rebuttal features ({year})...")
    with open(norm_path) as f:
        norm = json.load(f)

    rows = []
    for paper_key, paper in norm["papers"].items():
        reviews = paper.get("reviews", [])
        comments = paper.get("comments", [])

        # Author comments
        author_comments = [c for c in comments if c.get("participant_type") == "author"]
        # All reviewer/AC comments (for discussion depth)
        all_discussion = [c for c in comments]

        # Map review_id -> rebuttal stats
        for i, rev in enumerate(reviews):
            rev_id = rev.get("id")
            reviewer_id = rev.get("reviewer_id", "")
            suffix = (
                reviewer_id.split("Reviewer_")[-1]
                if "Reviewer_" in reviewer_id
                else reviewer_id.split("/")[-1]
            )

            # Direct author replies to this review
            direct_replies = [c for c in author_comments if c.get("reply_to") == rev_id]
            n_direct = len(direct_replies)
            total_reply_len = sum(len(c.get("text") or "") for c in direct_replies)

            # Threaded replies (replies to the direct replies)
            direct_ids = {c.get("id") for c in direct_replies}
            threaded = [c for c in all_discussion if c.get("reply_to") in direct_ids]
            n_thread_total = n_direct + len(threaded)

            # Did author reply at all?
            has_rebuttal = 1 if n_direct > 0 else 0

            first_reply_latency_days = float("nan")
            review_cdate = rev.get("cdate")
            if has_rebuttal and isinstance(review_cdate, (int, float)):
                reply_cdates = [
                    c.get("cdate")
                    for c in direct_replies
                    if isinstance(c.get("cdate"), (int, float))
                ]
                if reply_cdates:
                    first_reply_latency_days = (min(reply_cdates) - review_cdate) / 86_400_000.0

            thread_max_depth = 1
            if has_rebuttal:
                children = {}
                for c in all_discussion:
                    parent = c.get("reply_to")
                    if parent is not None:
                        children.setdefault(parent, []).append(c)
                stack = [(rev_id, 1)]
                while stack:
                    nid, depth = stack.pop()
                    thread_max_depth = max(thread_max_depth, depth)
                    for child in children.get(nid, []):
                        stack.append((child.get("id"), depth + 1))

            # Review text features
            strengths = rev.get("strengths", "") or ""
            weaknesses = rev.get("weaknesses", "") or ""
            sw_combined = rev.get("strengths_and_weaknesses", "") or ""
            questions = rev.get("questions", "") or ""

            s_len = len(strengths) + len(sw_combined) // 2
            w_len = len(weaknesses) + len(sw_combined) // 2

            rows.append({
                "paper_key": str(paper_key),
                "reviewer_suffix": suffix,
                "review_order": i,
                # Rebuttal features
                "has_rebuttal": has_rebuttal,
                "n_author_replies": n_direct,
                "rebuttal_total_len": total_reply_len,
                "n_thread_messages": n_thread_total,
                "first_reply_latency_days": first_reply_latency_days,
                "thread_max_depth": thread_max_depth,
                # Text features
                "strengths_len": s_len,
                "weaknesses_len": w_len,
                "questions_len": len(questions),
                "sw_ratio": s_len / max(w_len, 1),
            })

    del norm
    gc.collect()
    return pd.DataFrame(rows)


def extract_gemini_engagement(year: int) -> pd.DataFrame:
    """Extract per-reviewer engagement features from Gemini."""
    fname = GEMINI_FILES.get(year)
    if not fname:
        return pd.DataFrame()
    pred_path = BASE_DIR / "data" / "batches" / "output" / fname
    if not pred_path.exists():
        return pd.DataFrame()

    print(f"  Loading Gemini engagement features ({year})...")
    with open(pred_path) as f:
        preds = json.load(f)

    rows = []
    for p in preds:
        paper_key = str(p["custom_id"])
        parsed = p.get("parsed")
        if not parsed or not isinstance(parsed, dict):
            continue
        per_rev = parsed.get("per_reviewer", [])
        if not isinstance(per_rev, list):
            continue
        for i, rev in enumerate(per_rev):
            engagement = rev.get("engagement_quality", "unknown")
            indicators = rev.get("indicators_of_engagement", [])
            if not isinstance(indicators, list):
                indicators = []
            rows.append({
                "paper_key": paper_key,
                "review_order": i,
                "engagement_ord": ENGAGEMENT_TO_ORD.get(engagement, -1),
                "has_actionable_feedback": 1 if "actionable_feedback" in indicators else 0,
                "has_technical_depth": 1 if "technical_depth" in indicators else 0,
            })

    del preds
    gc.collect()
    return pd.DataFrame(rows)


def build_dataset(year: int, conf_threshold: int = 3) -> pd.DataFrame:
    """Build the full dataset with all trigger features."""
    print(f"\nBuilding dataset for ICLR {year}...")
    df = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")

    # Filter to reviews with external before/after
    mask = df["ext_rating_before"].notna() & df["ext_rating_after"].notna()
    df = df[mask].copy()
    df["ext_rating_before"] = df["ext_rating_before"].astype(int)
    df["ext_rating_after"] = df["ext_rating_after"].astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    # Core
    df["increased"] = (df["ext_rating_after"] > df["ext_rating_before"]).astype(int)
    for col in ["soundness", "presentation", "contribution"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Paper-level stats
    paper_stats = df.groupby("paper_key").agg(
        n_reviewers=("ext_rating_before", "count"),
        score_std=("ext_rating_before", "std"),
        score_mean=("ext_rating_before", "mean"),
    ).reset_index()
    paper_stats["score_std"] = paper_stats["score_std"].fillna(0)
    df = df.merge(paper_stats, on="paper_key", how="left")
    df["deviation_from_mean"] = df["ext_rating_before"].astype(float) - df["score_mean"]

    # Co-reviewer movement: did OTHER reviewers on same paper also increase?
    paper_inc = df.groupby("paper_key")["increased"].agg(["sum", "count"]).reset_index()
    paper_inc.columns = ["paper_key", "total_increases", "total_reviewers"]
    df = df.merge(paper_inc, on="paper_key", how="left")
    # Exclude self: how many co-reviewers increased?
    df["coreviewer_increases"] = df["total_increases"] - df["increased"]
    df["coreviewer_increase_rate"] = df["coreviewer_increases"] / (df["total_reviewers"] - 1).clip(lower=1)

    # Decision
    df["accepted"] = df["decision"].isin(["Oral", "Spotlight", "Poster"]).astype(int)

    # Review order for merging
    df = df.sort_values(["paper_key", "reviewer_id"])
    df["review_order"] = df.groupby("paper_key").cumcount()

    # Rebuttal features
    rebuttal_df = extract_rebuttal_features(year)
    df = df.merge(
        rebuttal_df, on=["paper_key", "reviewer_suffix"], how="left",
        suffixes=("", "_reb"),
    )
    if "review_order_reb" in df.columns:
        df["review_order"] = df["review_order_reb"].fillna(df["review_order"]).astype(int)
        df.drop(columns=["review_order_reb"], inplace=True)

    # Gemini engagement
    gemini_df = extract_gemini_engagement(year)
    if len(gemini_df) > 0:
        df = df.merge(gemini_df, on=["paper_key", "review_order"], how="left")

    # Group assignment
    df["conf_group"] = np.where(df["confidence"] <= conf_threshold, "low", "high")
    df["group"] = np.where(
        (df["conf_group"] == "low") & (df["increased"] == 0), "Low conf\nno increase",
        np.where(
            (df["conf_group"] == "low") & (df["increased"] == 1), "Low conf\nincrease",
            np.where(
                (df["conf_group"] == "high") & (df["increased"] == 0), "High conf\nno increase",
                "High conf\nincrease",
            ),
        ),
    )

    df["year"] = year
    print(f"  Dataset: {len(df):,} reviews")
    for g in GROUP_ORDER:
        n = (df["group"] == g).sum()
        print(f"    {g.replace(chr(10), ' '):>25s}: {n:>7,} ({n/len(df)*100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
TRIGGER_FEATURES = {
    # Peer pressure
    "deviation_from_mean": "Deviation from paper mean",
    "score_std": "Score std across reviewers",
    "coreviewer_increase_rate": "Co-reviewer increase rate",
    "accepted": "Paper accepted",
    "n_reviewers": "Number of reviewers",
    # Rebuttal
    "has_rebuttal": "Author replied to review",
    "rebuttal_total_len": "Total rebuttal length (chars)",
    "n_author_replies": "Number of author replies",
    "n_thread_messages": "Thread messages (incl. follow-up)",
    # Review characteristics
    "soundness": "Soundness sub-score",
    "contribution": "Contribution sub-score",
    "presentation": "Presentation sub-score",
    "strengths_len": "Strengths text length",
    "weaknesses_len": "Weaknesses text length",
    "sw_ratio": "Strengths/weaknesses ratio",
    "questions_len": "Questions text length",
    # Engagement
    "engagement_ord": "Engagement quality (ordinal)",
    "has_actionable_feedback": "Has actionable feedback",
    "has_technical_depth": "Has technical depth",
}


def compare_groups(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compare trigger features across the 4 groups."""
    print(f"\n  Comparing groups ({label}):")
    results = []

    for feat, feat_label in TRIGGER_FEATURES.items():
        if feat not in df.columns:
            continue

        vals_by_group = {}
        for g in GROUP_ORDER:
            v = pd.to_numeric(df[df["group"] == g][feat], errors="coerce").dropna()
            vals_by_group[g] = v

        if any(len(v) < 10 for v in vals_by_group.values()):
            continue

        # Kruskal-Wallis across all 4 groups
        H, p_all = stats.kruskal(*vals_by_group.values())

        # Key pairwise: within low-conf (increase vs not) and within high-conf
        _, p_low = stats.mannwhitneyu(
            vals_by_group["Low conf\nincrease"],
            vals_by_group["Low conf\nno increase"],
            alternative="two-sided",
        )
        _, p_high = stats.mannwhitneyu(
            vals_by_group["High conf\nincrease"],
            vals_by_group["High conf\nno increase"],
            alternative="two-sided",
        )

        results.append({
            "feature": feat,
            "label": feat_label,
            "mean_low_noinc": vals_by_group["Low conf\nno increase"].mean(),
            "mean_low_inc": vals_by_group["Low conf\nincrease"].mean(),
            "mean_high_noinc": vals_by_group["High conf\nno increase"].mean(),
            "mean_high_inc": vals_by_group["High conf\nincrease"].mean(),
            "H_stat": H,
            "p_all": p_all,
            "p_low": p_low,
            "p_high": p_high,
        })

    res_df = pd.DataFrame(results)
    n_tests = len(res_df)
    res_df["p_all_adj"] = (res_df["p_all"] * n_tests).clip(upper=1)
    res_df["p_low_adj"] = (res_df["p_low"] * n_tests).clip(upper=1)
    res_df["p_high_adj"] = (res_df["p_high"] * n_tests).clip(upper=1)

    # Print
    print(f"\n    {'Feature':<30s} | {'Low:NoInc':>9s} | {'Low:Inc':>9s} | {'p(low)':>10s} | {'Hi:NoInc':>9s} | {'Hi:Inc':>9s} | {'p(high)':>10s}")
    print("    " + "-" * 105)
    for _, r in res_df.iterrows():
        sig_low = " *" if r["p_low_adj"] < 0.05 else ""
        sig_high = " *" if r["p_high_adj"] < 0.05 else ""
        print(
            f"    {r['label']:<30s} | {r['mean_low_noinc']:>9.2f} | {r['mean_low_inc']:>9.2f} | "
            f"{r['p_low_adj']:>10.2e}{sig_low:2s} | {r['mean_high_noinc']:>9.2f} | {r['mean_high_inc']:>9.2f} | "
            f"{r['p_high_adj']:>10.2e}{sig_high:2s}"
        )

    return res_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_trigger_comparison(res_df: pd.DataFrame, label: str) -> plt.Figure:
    """Grouped bar chart of key triggers across the 4 groups."""
    # Select features with largest effect (difference between inc and no-inc within a confidence group)
    res_df = res_df.copy()
    res_df["effect_low"] = abs(res_df["mean_low_inc"] - res_df["mean_low_noinc"])
    res_df["effect_high"] = abs(res_df["mean_high_inc"] - res_df["mean_high_noinc"])
    res_df["max_effect"] = res_df[["effect_low", "effect_high"]].max(axis=1)

    # Normalize by mean to get relative effect
    res_df["mean_overall"] = (
        res_df["mean_low_noinc"] + res_df["mean_low_inc"]
        + res_df["mean_high_noinc"] + res_df["mean_high_inc"]
    ) / 4
    res_df["rel_effect"] = res_df["max_effect"] / res_df["mean_overall"].clip(lower=0.01)

    # Pick top features by relative effect, exclude those with tiny absolute values
    top = res_df[res_df["p_all_adj"] < 0.05].nlargest(10, "rel_effect")

    if len(top) == 0:
        return plt.figure()

    fig, axes = plt.subplots(2, min(5, len(top)), figsize=(min(5, len(top)) * 3.5, 8))
    if len(top) < 5:
        axes = axes.reshape(2, -1)

    for idx, (_, row) in enumerate(top.iterrows()):
        if idx >= 10:
            break

        ax_row = idx // 5
        ax_col = idx % 5
        ax = axes[ax_row, ax_col] if len(top) > 5 else axes[ax_row, ax_col] if axes.ndim > 1 else axes[idx]

        means = [
            row["mean_low_noinc"], row["mean_low_inc"],
            row["mean_high_noinc"], row["mean_high_inc"],
        ]
        colors = list(GROUP_COLORS.values())
        bars = ax.bar(range(4), means, color=colors, edgecolor="white")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["L:no", "L:yes", "H:no", "H:yes"], fontsize=7)
        ax.set_title(row["label"], fontsize=8, fontweight="bold")

        # Annotate
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"{label} -- Key triggers by confidence x increase group",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_rebuttal_effect(df: pd.DataFrame, label: str) -> plt.Figure:
    """Focus plot: rebuttal length effect within confidence groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, conf_label, conf_mask in [
        (ax1, "Low confidence (1-3)", df["conf_group"] == "low"),
        (ax2, "High confidence (4-5)", df["conf_group"] == "high"),
    ]:
        sub = df[conf_mask & df["rebuttal_total_len"].notna()].copy()
        # Only bin reviews that have a rebuttal
        sub = sub[sub["rebuttal_total_len"] > 0].copy()
        try:
            sub["reb_bin"] = pd.qcut(
                sub["rebuttal_total_len"],
                q=5, duplicates="drop",
            )
            # Rename bins to labels
            bin_labels = {b: l for b, l in zip(
                sorted(sub["reb_bin"].dropna().unique()),
                ["Very short", "Short", "Medium", "Long", "Very long"][:sub["reb_bin"].nunique()],
            )}
            sub["reb_bin"] = sub["reb_bin"].map(bin_labels)
        except ValueError:
            sub["reb_bin"] = pd.qcut(
                sub["rebuttal_total_len"].rank(method="first"),
                q=4, labels=["Short", "Medium", "Long", "Very long"],
            )

        grouped = sub.groupby("reb_bin").agg(
            n=("increased", "count"),
            rate=("increased", "mean"),
        )

        x = range(len(grouped))
        rates = grouped["rate"].values * 100
        counts = grouped["n"].values
        colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]

        bars = ax.bar(x, rates, color=colors[:len(x)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("% with score increase")
        ax.set_title(conf_label)

        for bar, rate, n in zip(bars, rates, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"{label} -- Score increase rate by rebuttal length",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_peer_pressure(df: pd.DataFrame, label: str) -> plt.Figure:
    """Focus plot: co-reviewer movement effect within confidence groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, conf_label, conf_mask in [
        (ax1, "Low confidence (1-3)", df["conf_group"] == "low"),
        (ax2, "High confidence (4-5)", df["conf_group"] == "high"),
    ]:
        sub = df[conf_mask].copy()

        # Bin co-reviewer increase rate
        sub["peer_bin"] = pd.cut(
            sub["coreviewer_increase_rate"],
            bins=[-0.01, 0.01, 0.25, 0.5, 0.75, 1.01],
            labels=["None", "1-25%", "26-50%", "51-75%", "76-100%"],
        )

        grouped = sub.groupby("peer_bin", observed=True).agg(
            n=("increased", "count"),
            rate=("increased", "mean"),
        )

        x = range(len(grouped))
        rates = grouped["rate"].values * 100
        counts = grouped["n"].values
        colors = ["#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]

        bars = ax.bar(x, rates, color=colors[:len(x)], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, fontsize=8)
        ax.set_ylabel("% with score increase")
        ax.set_xlabel("Co-reviewers who also increased")
        ax.set_title(conf_label)

        for bar, rate, n in zip(bars, rates, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{rate:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"{label} -- Score increase rate by co-reviewer movement",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_deviation_x_confidence(df: pd.DataFrame, label: str) -> plt.Figure:
    """Heatmap: increase rate by deviation x confidence."""
    df = df.copy()
    df["dev_bin"] = pd.cut(
        df["deviation_from_mean"],
        bins=[-999, -2.5, -0.5, 0.5, 2.5, 999],
        labels=["Much below", "Below", "At mean", "Above", "Much above"],
    )
    df["conf_bin"] = pd.cut(
        df["confidence"],
        bins=[0, 2.5, 3.5, 4.5, 5.5],
        labels=["Low (1-2)", "Medium (3)", "High (4)", "Very high (5)"],
    )

    ct = df.groupby(["dev_bin", "conf_bin"], observed=True).agg(
        n=("increased", "count"),
        rate=("increased", "mean"),
    ).reset_index()

    rate_pivot = ct.pivot(index="dev_bin", columns="conf_bin", values="rate") * 100
    n_pivot = ct.pivot(index="dev_bin", columns="conf_bin", values="n")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rate_pivot.values, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=60)

    ax.set_xticks(range(len(rate_pivot.columns)))
    ax.set_xticklabels(rate_pivot.columns, fontsize=9)
    ax.set_yticks(range(len(rate_pivot.index)))
    ax.set_yticklabels(rate_pivot.index, fontsize=9)
    ax.set_xlabel("Reviewer confidence")
    ax.set_ylabel("Deviation from paper mean")

    for i in range(len(rate_pivot.index)):
        for j in range(len(rate_pivot.columns)):
            rate = rate_pivot.values[i, j]
            n = n_pivot.values[i, j]
            if np.isnan(rate):
                continue
            color = "white" if rate > 35 else "black"
            ax.text(j, i, f"{rate:.1f}%\n(n={int(n):,})",
                    ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("% with score increase", fontsize=9)
    ax.set_title(f"{label} -- Score increase rate: deviation x confidence", fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(conf_threshold: int = 3, save: bool = True):
    """Run the full trigger analysis with a given confidence threshold."""
    suffix = f"_t{conf_threshold}" if conf_threshold != 3 else ""
    print(f"\n{'#'*70}")
    print(f"  Confidence threshold: <= {conf_threshold} = low, > {conf_threshold} = high")
    print(f"{'#'*70}")

    dfs = []
    for year in [2024, 2025]:
        df = build_dataset(year, conf_threshold=conf_threshold)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined):,} reviews")

    datasets = [("Combined 2024+2025", combined)]
    for year, df in zip([2024, 2025], dfs):
        datasets.append((f"ICLR {year}", df))

    for label, df in datasets:
        slug = label.lower().replace(" ", "_").replace("+", "")
        print(f"\n{'='*70}")
        print(f"{label} -- {len(df):,} reviews")
        print(f"{'='*70}")

        res_df = compare_groups(df, label)

        if save and len(res_df) > 0:
            res_df.to_csv(TABLES_DIR / f"{slug}_trigger_comparison{suffix}.csv", index=False)

            fig = plot_trigger_comparison(res_df, label)
            save_figure(fig, f"{slug}_triggers{suffix}")
            plt.close(fig)

            fig = plot_rebuttal_effect(df, label)
            save_figure(fig, f"{slug}_rebuttal_effect{suffix}")
            plt.close(fig)

            fig = plot_peer_pressure(df, label)
            save_figure(fig, f"{slug}_peer_pressure{suffix}")
            plt.close(fig)

            fig = plot_deviation_x_confidence(df, label)
            save_figure(fig, f"{slug}_deviation_x_confidence{suffix}")
            plt.close(fig)

    return combined


def run_sensitivity(save: bool = True):
    """Run analysis across multiple thresholds and compare key metrics."""
    thresholds = [2, 3, 4]
    summary_rows = []

    for t in thresholds:
        combined = run_analysis(conf_threshold=t, save=save)
        low_n = (combined["conf_group"] == "low").sum()
        high_n = (combined["conf_group"] == "high").sum()
        low_inc_rate = combined[combined["conf_group"] == "low"]["increased"].mean()
        high_inc_rate = combined[combined["conf_group"] == "high"]["increased"].mean()
        summary_rows.append({
            "threshold": t,
            "low_n": low_n,
            "high_n": high_n,
            "low_increase_rate": low_inc_rate,
            "high_increase_rate": high_inc_rate,
            "rate_difference": low_inc_rate - high_inc_rate,
        })

    summary_df = pd.DataFrame(summary_rows)
    if save:
        summary_df.to_csv(TABLES_DIR / "sensitivity_summary.csv", index=False)
    print(f"\n{'='*70}")
    print("  Sensitivity Analysis Summary")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Confidence-based trigger analysis")
    parser.add_argument("--no-save", action="store_true", help="Display only")
    parser.add_argument("--threshold", type=int, default=3,
                        help="Confidence threshold for low/high split (default: 3)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis across thresholds 2, 3, 4")
    args = parser.parse_args()

    setup_publication_theme()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.sensitivity:
        run_sensitivity(save=not args.no_save)
    else:
        run_analysis(conf_threshold=args.threshold, save=not args.no_save)

    print(f"\nFigures: {FIGURES_DIR}")
    print(f"Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
