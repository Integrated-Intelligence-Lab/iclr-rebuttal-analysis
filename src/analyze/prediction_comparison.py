"""
LLM-based vs metadata-only prediction of reviewer score increases.

Compares progressively richer feature sets to quantify how much an LLM
adds beyond structured review metadata already available on OpenReview.

Models (ablation):
  M1  Initial score position only
  M2  Sub-scores only (soundness, contribution, presentation)
  M3  Sub-scores + relative position + confidence
  M4  All metadata (M3 + text lengths + paper stats)
  M5  Gemini offset only
  M6  Gemini full (offset + engagement + tone + score_confidence)
  M7  Metadata + Gemini offset  (M4 + offset)
  M8  Metadata + all Gemini     (M4 + M6 features)

Evaluation:
  - 5-fold grouped CV within each year (split by paper_key to prevent
    same-paper leakage; reviews within a paper are not independent)
  - Cross-year temporal validation (train 2024 -> test 2025, and reverse)
  - Reports both AUC-ROC and AUPRC (score increase ~20% base rate)

Usage:
    python src/analyze/prediction_comparison.py
    python src/analyze/prediction_comparison.py --no-save
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import ast
import gc
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
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
FIGURES_DIR = RESULTS_DIR / "prediction_comparison" / "figures"
TABLES_DIR = RESULTS_DIR / "prediction_comparison" / "tables"
BASE_DIR = Path(__file__).resolve().parent.parent.parent

ICLR_SCALE = [1, 3, 5, 6, 8, 10]
SCORE_TO_POS = {s: i for i, s in enumerate(ICLR_SCALE)}

GEMINI_FILES = {
    2024: "gemini_flash_iclr2024_full.json",
    2025: "gemini_flash_iclr2025_full.json",
}

ENGAGEMENT_TO_ORD = {"superficial": 0, "moderate": 1, "thorough": 2, "exceptional": 3}

TONE_CATEGORIES = [
    "constructive", "critical", "enthusiastic", "neutral",
    "cautious", "dismissive", "supportive",
]

CANONICAL_INDICATORS = [
    "technical_depth", "actionable_feedback",
    "specific_examples", "trivial_feedback",
]


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
# Names of parquet columns containing review-text lengths and Gemini features.
# The shipped public_release parquet (data/processed/parquet/) bundles these
# directly so no normalized JSON or batch output is required.
_PARQUET_TEXT_COLS = ["strengths_len", "weaknesses_len", "questions_len"]
_PARQUET_GEMINI_COLS = [
    "gemini_tone_primary", "gemini_engagement_quality",
    "gemini_score_confidence_llm", "gemini_indicators_json",
]


def _parse_indicators(value) -> list[str]:
    """Coerce the parquet-stored indicators field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
        except (TypeError, ValueError):
            try:
                obj = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return []
        if isinstance(obj, list):
            return [str(v) for v in obj]
    return []


def extract_text_features(year: int) -> pd.DataFrame:
    """Extract review text length features.

    Reads from the shipped parquet when the columns are present; otherwise
    falls back to the normalized JSON (developer-tree path). This lets the
    public_release tree run without `data/processed/normalized/`.
    """
    parquet_path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(
            parquet_path,
            columns=["paper_key", "reviewer_suffix"] + _PARQUET_TEXT_COLS,
        )
        if all(c in df.columns for c in _PARQUET_TEXT_COLS):
            df = df.copy()
            df["paper_key"] = df["paper_key"].astype(str)
            for c in _PARQUET_TEXT_COLS:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df["sw_ratio"] = df["strengths_len"] / df["weaknesses_len"].clip(lower=1)
            return df[["paper_key", "reviewer_suffix"]
                      + _PARQUET_TEXT_COLS + ["sw_ratio"]]

    # Fallback: normalized JSON (developer tree only).
    norm_path = BASE_DIR / "data" / "processed" / "normalized" / f"ICLR_{year}_normalized.json"
    print(f"  Loading normalized data for text features...")
    with open(norm_path) as f:
        norm = json.load(f)

    rows = []
    for paper_key, paper in norm["papers"].items():
        for i, rev in enumerate(paper.get("reviews", [])):
            reviewer_id = rev.get("reviewer_id", "")
            suffix = (
                reviewer_id.split("Reviewer_")[-1]
                if "Reviewer_" in reviewer_id
                else reviewer_id.split("/")[-1]
            )
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
                "strengths_len": s_len,
                "weaknesses_len": w_len,
                "questions_len": len(questions),
                "sw_ratio": s_len / max(w_len, 1),
            })

    del norm
    gc.collect()
    return pd.DataFrame(rows)


def extract_gemini_features(year: int) -> pd.DataFrame:
    """Extract per-reviewer Gemini features.

    Reads from the shipped parquet when the columns are present; otherwise
    falls back to the raw batch-output JSON (developer-tree path).
    """
    parquet_path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(
            parquet_path,
            columns=["paper_key", "review_order"] + _PARQUET_GEMINI_COLS,
        )
        if all(c in df.columns for c in _PARQUET_GEMINI_COLS):
            df = df.copy()
            df["paper_key"] = df["paper_key"].astype(str)
            df["review_order"] = pd.to_numeric(df["review_order"],
                                               errors="coerce").fillna(0).astype(int)
            df["tone_primary"] = df["gemini_tone_primary"].fillna("unknown").astype(str)
            df["engagement_ord"] = (df["gemini_engagement_quality"]
                                    .map(ENGAGEMENT_TO_ORD).fillna(-1).astype(int))
            df["score_confidence_gemini"] = pd.to_numeric(
                df["gemini_score_confidence_llm"], errors="coerce"
            )
            indicator_lists = df["gemini_indicators_json"].map(_parse_indicators)
            df["n_indicators"] = indicator_lists.map(len).astype(int)
            for ind in CANONICAL_INDICATORS:
                df[f"has_{ind}"] = indicator_lists.map(
                    lambda lst, _ind=ind: int(_ind in lst)
                )
            return df[["paper_key", "review_order",
                       "tone_primary", "engagement_ord",
                       "score_confidence_gemini", "n_indicators"]
                      + [f"has_{ind}" for ind in CANONICAL_INDICATORS]]

    # Fallback: developer-tree batch output JSON.
    fname = GEMINI_FILES.get(year)
    if not fname:
        return pd.DataFrame()
    pred_path = BASE_DIR / "data" / "batches" / "output" / fname
    if not pred_path.exists():
        return pd.DataFrame()

    print(f"  Loading Gemini predictions...")
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
            per_rev = []

        for i, rev in enumerate(per_rev):
            tone = rev.get("tone", {})
            if isinstance(tone, dict):
                tone_primary = tone.get("primary", "unknown")
            elif isinstance(tone, str):
                tone_primary = tone
            else:
                tone_primary = "unknown"

            engagement = rev.get("engagement_quality", "unknown")
            score_conf = rev.get("score_confidence")
            if not isinstance(score_conf, (int, float)):
                score_conf = np.nan

            indicators = rev.get("indicators_of_engagement", [])
            if not isinstance(indicators, list):
                indicators = []

            row = {
                "paper_key": paper_key,
                "review_order": i,
                "tone_primary": tone_primary,
                "engagement_ord": ENGAGEMENT_TO_ORD.get(engagement, -1),
                "score_confidence_gemini": score_conf,
                "n_indicators": len(indicators),
            }
            for ind in CANONICAL_INDICATORS:
                row[f"has_{ind}"] = 1 if ind in indicators else 0

            rows.append(row)

    del preds
    gc.collect()
    return pd.DataFrame(rows)


def build_dataset(year: int) -> pd.DataFrame:
    """Build full feature matrix for a year."""
    print(f"\nBuilding dataset for ICLR {year}...")

    df = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")
    mask = (
        df["gemini_pred_score"].notna()
        & df["ext_rating_before"].notna()
        & df["ext_rating_after"].notna()
    )
    df = df[mask].copy()

    df["gemini_pred_score"] = df["gemini_pred_score"].astype(int)
    df["ext_rating_before"] = df["ext_rating_before"].astype(int)
    df["ext_rating_after"] = df["ext_rating_after"].astype(int)

    # Core
    df["pos_pred"] = df["gemini_pred_score"].map(SCORE_TO_POS)
    df["pos_before"] = df["ext_rating_before"].map(SCORE_TO_POS)
    df["gemini_offset"] = df["pos_pred"] - df["pos_before"]
    df["score_increased"] = (df["ext_rating_after"] > df["ext_rating_before"]).astype(int)

    # Sub-scores
    for col in ["soundness", "presentation", "contribution", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Paper-level
    paper_stats = df.groupby("paper_key").agg(
        n_reviewers=("ext_rating_before", "count"),
        score_std=("ext_rating_before", "std"),
        score_mean=("ext_rating_before", "mean"),
    ).reset_index()
    paper_stats["score_std"] = paper_stats["score_std"].fillna(0)
    df = df.merge(paper_stats, on="paper_key", how="left")

    # Deviation from paper mean
    df["deviation_from_mean"] = df["ext_rating_before"].astype(float) - df["score_mean"]

    # Review order
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

    # Gemini features. The shipped public_release parquet already attaches
    # the per-reviewer Gemini columns row-by-row, so we read them inline
    # rather than going back through (paper_key, review_order) — that join
    # path is fragile because review_order was reassigned a few lines above
    # via cumcount(reviewer_id sort) and need not match the parquet's own
    # review_order. We fall back to the JSON extractor only if the parquet
    # does not carry these columns (developer tree).
    parquet_gemini_cols = [
        "gemini_tone_primary", "gemini_engagement_quality",
        "gemini_score_confidence_llm", "gemini_indicators_json",
    ]
    if all(c in df.columns for c in parquet_gemini_cols):
        df["tone_primary"] = df["gemini_tone_primary"].fillna("unknown").astype(str)
        df["engagement_ord"] = (
            df["gemini_engagement_quality"].map(ENGAGEMENT_TO_ORD).fillna(-1).astype(int)
        )
        df["score_confidence_gemini"] = pd.to_numeric(
            df["gemini_score_confidence_llm"], errors="coerce",
        )
        ind_lists = df["gemini_indicators_json"].map(_parse_indicators)
        df["n_indicators"] = ind_lists.map(len).astype(int)
        for ind in CANONICAL_INDICATORS:
            df[f"has_{ind}"] = ind_lists.map(
                lambda lst, _ind=ind: int(_ind in lst)
            )
    else:
        gemini_df = extract_gemini_features(year)
        if len(gemini_df) > 0:
            df = df.merge(gemini_df, on=["paper_key", "review_order"], how="left")

    print(f"  Dataset: {len(df):,} reviews, {df['score_increased'].mean()*100:.1f}% increased")
    return df


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "M1: Score position": ["pos_before"],
    "M2: Sub-scores": ["soundness", "contribution", "presentation"],
    "M3: Sub-scores+": [
        "soundness", "contribution", "presentation",
        "pos_before", "deviation_from_mean", "confidence",
    ],
    "M4: All metadata": [
        "soundness", "contribution", "presentation",
        "pos_before", "deviation_from_mean", "confidence",
        "strengths_len", "weaknesses_len", "sw_ratio", "questions_len",
        "n_reviewers", "score_std", "score_mean",
    ],
    "M5: Gemini offset": ["gemini_offset"],
    "M6: Gemini full": [
        "gemini_offset", "engagement_ord", "score_confidence_gemini",
        "n_indicators",
        "has_technical_depth", "has_actionable_feedback",
        "has_specific_examples", "has_trivial_feedback",
        # tone one-hot added dynamically
        "__tone__",
    ],
    "M7: Metadata + offset": [
        "soundness", "contribution", "presentation",
        "pos_before", "deviation_from_mean", "confidence",
        "strengths_len", "weaknesses_len", "sw_ratio", "questions_len",
        "n_reviewers", "score_std", "score_mean",
        "gemini_offset",
    ],
    "M8: Metadata + Gemini": [
        "soundness", "contribution", "presentation",
        "pos_before", "deviation_from_mean", "confidence",
        "strengths_len", "weaknesses_len", "sw_ratio", "questions_len",
        "n_reviewers", "score_std", "score_mean",
        "gemini_offset", "engagement_ord", "score_confidence_gemini",
        "n_indicators",
        "has_technical_depth", "has_actionable_feedback",
        "has_specific_examples", "has_trivial_feedback",
        "__tone__",
    ],
}

# Colors for each model (progressing from light to dark)
MODEL_COLORS = {
    "M1: Score position": "#cccccc",
    "M2: Sub-scores": "#E69F00",
    "M3: Sub-scores+": "#D55E00",
    "M4: All metadata": "#CC79A7",
    "M5: Gemini offset": "#56B4E9",
    "M6: Gemini full": "#0072B2",
    "M7: Metadata + offset": "#009E73",
    "M8: Metadata + Gemini": "#000000",
}


def prepare_X(df: pd.DataFrame, feature_spec: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from specification."""
    arrays = []
    names = []
    for col in feature_spec:
        if col == "__tone__":
            if "tone_primary" in df.columns:
                for tone in TONE_CATEGORIES:
                    vals = (df["tone_primary"] == tone).astype(float).values
                    arrays.append(vals.reshape(-1, 1))
                    names.append(f"tone_{tone}")
            continue
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
            arrays.append(vals.reshape(-1, 1))
            names.append(col)
    X = np.hstack(arrays) if arrays else np.zeros((len(df), 1))
    return X, names


def run_cv(df: pd.DataFrame, label: str) -> dict:
    """Run all models with 5-fold grouped CV (split by paper_key), return results."""
    y = df["score_increased"].values
    groups = df["paper_key"].values
    cv = GroupKFold(n_splits=5)
    results = {}

    for model_name, feat_spec in MODEL_SPECS.items():
        X, feat_names = prepare_X(df, feat_spec)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

        # Grouped out-of-fold predictions (no same-paper leakage)
        y_prob = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in cv.split(X_scaled, y, groups):
            lr_fold = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            lr_fold.fit(X_scaled[train_idx], y[train_idx])
            y_prob[test_idx] = lr_fold.predict_proba(X_scaled[test_idx])[:, 1]

        auc = roc_auc_score(y, y_prob)
        auprc = average_precision_score(y, y_prob)

        # Full fit for coefficients
        lr.fit(X_scaled, y)
        coef_df = pd.DataFrame({
            "feature": feat_names,
            "coefficient": lr.coef_[0],
            "odds_ratio": np.exp(lr.coef_[0]),
        }).sort_values("coefficient", key=abs, ascending=False)

        results[model_name] = {
            "auc": auc,
            "auprc": auprc,
            "y_prob": y_prob,
            "coef_df": coef_df,
            "model": lr,
            "scaler": scaler,
            "feat_names": feat_names,
            "n_features": len(feat_names),
        }

    return results


def bootstrap_block_deltas(
    df: pd.DataFrame, model_pairs: list[tuple[str, str]],
    n_boot: int = 1000, seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap CIs for AUC differences between model pairs, clustered by paper.

    Resamples papers (not individual reviews) to respect clustering.
    Returns a DataFrame with columns: pair, delta, ci_lo, ci_hi.
    """
    rng = np.random.RandomState(seed)
    y = df["score_increased"].values
    papers = df["paper_key"].values
    unique_papers = np.unique(papers)

    # Pre-compute OOF probabilities for all models we need
    needed_models = set()
    for m1, m2 in model_pairs:
        needed_models.add(m1)
        needed_models.add(m2)

    model_probs = {}
    cv = GroupKFold(n_splits=5)
    for model_name in needed_models:
        feat_spec = MODEL_SPECS[model_name]
        X, _ = prepare_X(df, feat_spec)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_prob = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in cv.split(X_scaled, y, papers):
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
            lr.fit(X_scaled[train_idx], y[train_idx])
            y_prob[test_idx] = lr.predict_proba(X_scaled[test_idx])[:, 1]
        model_probs[model_name] = y_prob

    # Build paper -> review index mapping once
    paper_to_idx = {}
    for i, p in enumerate(papers):
        paper_to_idx.setdefault(p, []).append(i)
    paper_to_idx_arr = {p: np.array(idxs) for p, idxs in paper_to_idx.items()}

    # Bootstrap: resample papers with replacement
    results = []
    for m1_name, m2_name in model_pairs:
        deltas = []
        p1 = model_probs[m1_name]
        p2 = model_probs[m2_name]

        for _ in range(n_boot):
            sampled_papers = rng.choice(unique_papers, size=len(unique_papers), replace=True)
            idx = np.concatenate([paper_to_idx_arr[p] for p in sampled_papers])
            y_boot = y[idx]
            if len(np.unique(y_boot)) < 2:
                continue
            auc1 = roc_auc_score(y_boot, p1[idx])
            auc2 = roc_auc_score(y_boot, p2[idx])
            deltas.append(auc2 - auc1)

        deltas = np.array(deltas)
        results.append({
            "pair": f"{m2_name} vs {m1_name}",
            "delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(deltas, 2.5)),
            "ci_hi": float(np.percentile(deltas, 97.5)),
            "significant": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
        })

    return pd.DataFrame(results)


def run_temporal_validation(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Train on one year, test on another."""
    y_train = df_train["score_increased"].values
    y_test = df_test["score_increased"].values
    results = {}

    for model_name, feat_spec in MODEL_SPECS.items():
        X_train, feat_names = prepare_X(df_train, feat_spec)
        X_test, _ = prepare_X(df_test, feat_spec)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        lr.fit(X_train_s, y_train)
        y_prob = lr.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)

        results[model_name] = {"auc": auc, "auprc": auprc, "y_prob": y_prob}

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_results(results: dict, label: str):
    print(f"\n  {'Model':<30s} | {'Features':>8s} | {'AUC-ROC':>8s} | {'AUPRC':>8s}")
    print("  " + "-" * 70)

    metadata_auc = results.get("M4: All metadata", {}).get("auc", 0)
    for name, r in results.items():
        delta = r["auc"] - metadata_auc
        delta_str = f"({delta:+.3f})" if name != "M4: All metadata" else "  (ref)"
        auprc = r.get("auprc", float("nan"))
        print(f"  {name:<30s} | {r.get('n_features', '?'):>8} | {r['auc']:>7.4f} {delta_str} | {auprc:>7.4f}")


def print_temporal(results: dict, train_year: int, test_year: int):
    print(f"\n  Train on {train_year}, test on {test_year}:")
    for name, r in results.items():
        auprc = r.get("auprc", float("nan"))
        print(f"    {name:<30s} | AUC={r['auc']:.4f} | AUPRC={auprc:.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc_comparison(y: np.ndarray, results: dict, label: str) -> plt.Figure:
    """All models on one ROC plot."""
    fig, ax = plt.subplots(figsize=(7, 7))

    for model_name in MODEL_SPECS:
        r = results[model_name]
        color = MODEL_COLORS[model_name]
        RocCurveDisplay.from_predictions(
            y, r["y_prob"],
            name=f"{model_name} ({r['auc']:.3f})",
            ax=ax, color=color,
            linewidth=2 if "M4" in model_name or "M8" in model_name else 1.2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title(f"{label} -- ROC comparison: metadata vs LLM features")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_auc_summary(cv_results: dict, temporal_results: dict, label: str) -> plt.Figure:
    """Bar chart comparing AUCs across evaluation strategies."""
    model_names = list(MODEL_SPECS.keys())
    n = len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    bar_w = 0.25

    # CV AUCs
    cv_aucs = [cv_results[m]["auc"] for m in model_names]
    ax.bar(x - bar_w, cv_aucs, bar_w, label="5-fold CV (combined)",
           color=[MODEL_COLORS[m] for m in model_names], edgecolor="white")

    # Temporal: train 2024 -> test 2025
    if "2024->2025" in temporal_results:
        t1_aucs = [temporal_results["2024->2025"][m]["auc"] for m in model_names]
        ax.bar(x, t1_aucs, bar_w, label="Train 2024, test 2025",
               color=[MODEL_COLORS[m] for m in model_names], edgecolor="white",
               alpha=0.6, hatch="//")

    # Temporal: train 2025 -> test 2024
    if "2025->2024" in temporal_results:
        t2_aucs = [temporal_results["2025->2024"][m]["auc"] for m in model_names]
        ax.bar(x + bar_w, t2_aucs, bar_w, label="Train 2025, test 2024",
               color=[MODEL_COLORS[m] for m in model_names], edgecolor="white",
               alpha=0.4, hatch="\\\\")

    ax.set_xticks(x)
    ax.set_xticklabels([m.split(":")[0] for m in model_names], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"{label} -- Model comparison across evaluation strategies")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 0.85)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")

    # Annotate AUC values on CV bars
    for i, auc in enumerate(cv_aucs):
        ax.text(i - bar_w, auc + 0.003, f"{auc:.3f}", ha="center", fontsize=7, rotation=90)

    fig.tight_layout()
    return fig


def plot_coefficient_comparison(cv_results: dict, label: str) -> plt.Figure:
    """Side-by-side coefficients: M4 (metadata) vs M8 (metadata + Gemini)."""
    c4 = cv_results["M4: All metadata"]["coef_df"].set_index("feature")["coefficient"]
    c8 = cv_results["M8: Metadata + Gemini"]["coef_df"].set_index("feature")["coefficient"]

    # Get top features from M8
    top = c8.abs().sort_values(ascending=False).head(20).index
    # Also include M4-only features
    m4_feats = set(c4.index)
    all_feats = list(dict.fromkeys(list(top) + [f for f in c4.abs().sort_values(ascending=False).head(10).index]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(all_feats) * 0.28)), sharey=True)
    y_pos = range(len(all_feats))

    # M4
    vals1 = [c4.get(f, 0) for f in all_feats]
    colors1 = ["#d9534f" if v < 0 else "#CC79A7" for v in vals1]
    ax1.barh(y_pos, vals1, color=colors1, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(all_feats, fontsize=8)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("Standardized coefficient")
    ax1.set_title(f"M4: All metadata (AUC={cv_results['M4: All metadata']['auc']:.3f})")

    # M8
    vals2 = [c8.get(f, 0) for f in all_feats]
    colors2 = ["#d9534f" if v < 0 else "#000000" for v in vals2]
    ax2.barh(y_pos, vals2, color=colors2, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Standardized coefficient")
    ax2.set_title(f"M8: Metadata + Gemini (AUC={cv_results['M8: Metadata + Gemini']['auc']:.3f})")

    fig.suptitle(f"{label} -- What the LLM adds to metadata", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_incremental_gain(cv_results: dict, label: str) -> plt.Figure:
    """Waterfall showing incremental AUC gain from each feature group."""
    steps = [
        ("Baseline\n(pos_before)", "M1: Score position"),
        ("+ Sub-scores", "M2: Sub-scores"),
        ("+ Relative pos.\n+ confidence", "M3: Sub-scores+"),
        ("+ Text lengths\n+ paper stats", "M4: All metadata"),
        ("+ Gemini offset", "M7: Metadata + offset"),
        ("+ Gemini full\n(engagement, tone)", "M8: Metadata + Gemini"),
    ]

    aucs = [cv_results[m]["auc"] for _, m in steps]
    deltas = [aucs[0] - 0.5] + [aucs[i] - aucs[i - 1] for i in range(1, len(aucs))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = 0.5
    x = range(len(steps))

    colors = ["#cccccc", "#E69F00", "#D55E00", "#CC79A7", "#009E73", "#000000"]

    for i, (step_label, _) in enumerate(steps):
        ax.bar(i, deltas[i], bottom=bottom, color=colors[i],
               edgecolor="white", linewidth=0.5)
        # Annotate delta
        if deltas[i] > 0.002:
            ax.text(i, bottom + deltas[i] / 2, f"+{deltas[i]:.3f}",
                    ha="center", va="center", fontsize=9, fontweight="bold")
        # Annotate cumulative AUC
        ax.text(i, bottom + deltas[i] + 0.005, f"{aucs[i]:.3f}",
                ha="center", va="bottom", fontsize=8, color="#333")
        bottom += deltas[i]

    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in steps], fontsize=8)
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"{label} -- Incremental AUC gain by feature group")
    ax.set_ylim(0.45, max(aucs) + 0.05)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--", label="Random")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_calibration_table(df: pd.DataFrame, label: str) -> plt.Figure:
    """Empirical offset calibration: P(increase) by Gemini offset."""
    grouped = df.groupby("gemini_offset").agg(
        n=("score_increased", "count"),
        rate=("score_increased", "mean"),
    ).reset_index()
    grouped = grouped[grouped["n"] >= 20]

    fig, ax = plt.subplots(figsize=(9, 5))
    offsets = grouped["gemini_offset"].values
    rates = grouped["rate"].values * 100
    counts = grouped["n"].values
    colors = ["#d9534f" if o < 0 else ("#999999" if o == 0 else "#56B4E9") for o in offsets]

    bars = ax.bar(range(len(offsets)), rates, color=colors, edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([f"{o:+d}" for o in offsets])
    ax.set_xlabel("Gemini offset (predicted - actual, category steps)")
    ax.set_ylabel("% reviews with score increase")
    ax.set_title(f"{label} -- Empirical P(increase) by Gemini offset")

    for bar, rate, count in zip(bars, rates, counts):
        se = np.sqrt(rate / 100 * (1 - rate / 100) / count) * 100
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2, rate,
            yerr=1.96 * se, color="black", capsize=3, linewidth=0.8,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2, rate + 1.96 * se + 1,
            f"{rate:.1f}%\n(n={count:,})",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Table saving
# ---------------------------------------------------------------------------
def save_auc_table(cv_results: dict, temporal_results: dict, label: str):
    """Save comprehensive AUC comparison table."""
    rows = []
    for model_name in MODEL_SPECS:
        row = {
            "model": model_name,
            "n_features": cv_results[model_name].get("n_features", 0),
            "auc_cv": cv_results[model_name]["auc"],
            "auprc_cv": cv_results[model_name].get("auprc", float("nan")),
        }
        for key, tr in temporal_results.items():
            row[f"auc_{key}"] = tr[model_name]["auc"]
            row[f"auprc_{key}"] = tr[model_name].get("auprc", float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Add delta from M4
    m4_auc = cv_results["M4: All metadata"]["auc"]
    df["delta_vs_metadata"] = df["auc_cv"] - m4_auc
    df.to_csv(TABLES_DIR / "auc_comparison.csv", index=False)
    print(f"  Saved: auc_comparison.csv")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM vs metadata prediction comparison")
    parser.add_argument("--no-save", action="store_true", help="Display only")
    args = parser.parse_args()

    setup_publication_theme()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    df_2024 = build_dataset(2024)
    df_2025 = build_dataset(2025)
    combined = pd.concat([df_2024, df_2025], ignore_index=True)

    # =====================================================================
    # Within-year CV
    # =====================================================================
    print("\n" + "=" * 70)
    print("5-fold Cross-Validation (combined 2024+2025)")
    print("=" * 70)
    cv_results = run_cv(combined, "Combined")
    print_results(cv_results, "Combined")

    print("\n" + "=" * 70)
    print("5-fold Cross-Validation (ICLR 2024 only)")
    print("=" * 70)
    cv_2024 = run_cv(df_2024, "ICLR 2024")
    print_results(cv_2024, "ICLR 2024")

    print("\n" + "=" * 70)
    print("5-fold Cross-Validation (ICLR 2025 only)")
    print("=" * 70)
    cv_2025 = run_cv(df_2025, "ICLR 2025")
    print_results(cv_2025, "ICLR 2025")

    # =====================================================================
    # Temporal validation
    # =====================================================================
    print("\n" + "=" * 70)
    print("Temporal Validation")
    print("=" * 70)

    t_2024_2025 = run_temporal_validation(df_2024, df_2025)
    print_temporal(t_2024_2025, 2024, 2025)

    t_2025_2024 = run_temporal_validation(df_2025, df_2024)
    print_temporal(t_2025_2024, 2025, 2024)

    temporal_results = {"2024->2025": t_2024_2025, "2025->2024": t_2025_2024}

    # =====================================================================
    # Key result summary
    # =====================================================================
    m4_cv = cv_results["M4: All metadata"]["auc"]
    m7_cv = cv_results["M7: Metadata + offset"]["auc"]
    m8_cv = cv_results["M8: Metadata + Gemini"]["auc"]
    m5_cv = cv_results["M5: Gemini offset"]["auc"]

    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    print(f"  Metadata only (M4):           AUC = {m4_cv:.4f}")
    print(f"  Gemini offset only (M5):      AUC = {m5_cv:.4f}")
    print(f"  Metadata + offset (M7):       AUC = {m7_cv:.4f}  (delta = {m7_cv - m4_cv:+.4f})")
    print(f"  Metadata + all Gemini (M8):   AUC = {m8_cv:.4f}  (delta = {m8_cv - m4_cv:+.4f})")
    print(f"\n  LLM contribution beyond metadata: +{m8_cv - m4_cv:.4f} AUC")

    # Top features in M4
    print("\n  Top features (M4: metadata only):")
    for _, r in cv_results["M4: All metadata"]["coef_df"].head(8).iterrows():
        print(f"    {r['feature']:>25s}: coef={r['coefficient']:+.4f}  OR={r['odds_ratio']:.3f}")

    # =====================================================================
    # Bootstrap block deltas
    # =====================================================================
    print("\n" + "=" * 70)
    print("Bootstrap Block Deltas (1000 paper-clustered resamples)")
    print("=" * 70)
    boot_pairs = [
        ("M1: Score position", "M3: Sub-scores+"),
        ("M3: Sub-scores+", "M4: All metadata"),
        ("M4: All metadata", "M7: Metadata + offset"),
        ("M4: All metadata", "M8: Metadata + Gemini"),
    ]
    boot_df = bootstrap_block_deltas(combined, boot_pairs, n_boot=500)
    for _, row in boot_df.iterrows():
        sig = "*" if row["significant"] else ""
        print(f"  {row['pair']:<50s}: delta={row['delta']:+.4f}  "
              f"95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] {sig}")

    # =====================================================================
    # Save outputs
    # =====================================================================
    if not args.no_save:
        y_combined = combined["score_increased"].values

        # ROC comparison
        fig = plot_roc_comparison(y_combined, cv_results, "Combined 2024+2025")
        save_figure(fig, "roc_comparison_combined")
        plt.close(fig)

        # AUC summary bars
        fig = plot_auc_summary(cv_results, temporal_results, "Combined 2024+2025")
        save_figure(fig, "auc_summary")
        plt.close(fig)

        # Incremental gain waterfall
        fig = plot_incremental_gain(cv_results, "Combined 2024+2025")
        save_figure(fig, "incremental_gain")
        plt.close(fig)

        # Coefficient comparison M4 vs M8
        fig = plot_coefficient_comparison(cv_results, "Combined 2024+2025")
        save_figure(fig, "coefficient_comparison")
        plt.close(fig)

        # Calibration table / offset chart
        fig = plot_calibration_table(combined, "Combined 2024+2025")
        save_figure(fig, "offset_calibration")
        plt.close(fig)

        # Per-year ROC
        for year_label, year_results, year_df in [
            ("ICLR 2024", cv_2024, df_2024),
            ("ICLR 2025", cv_2025, df_2025),
        ]:
            slug = year_label.lower().replace(" ", "_")
            y_year = year_df["score_increased"].values
            fig = plot_roc_comparison(y_year, year_results, year_label)
            save_figure(fig, f"roc_comparison_{slug}")
            plt.close(fig)

        # Tables
        auc_df = save_auc_table(cv_results, temporal_results, "Combined")

        # Save per-year AUC tables
        for year_label, year_results in [("iclr_2024", cv_2024), ("iclr_2025", cv_2025)]:
            rows = []
            for mn in MODEL_SPECS:
                rows.append({"model": mn, "auc_cv": year_results[mn]["auc"]})
            pd.DataFrame(rows).to_csv(TABLES_DIR / f"auc_{year_label}.csv", index=False)

        # Coefficient tables
        for key in ["M4: All metadata", "M8: Metadata + Gemini"]:
            slug = key.split(":")[0].lower().strip()
            cv_results[key]["coef_df"].to_csv(
                TABLES_DIR / f"coefficients_{slug}_combined.csv", index=False)

        # Bootstrap deltas table
        boot_df.to_csv(TABLES_DIR / "bootstrap_block_deltas.csv", index=False)
        print(f"  Saved: bootstrap_block_deltas.csv")

        print(f"\nFigures: {FIGURES_DIR}")
        print(f"Tables:  {TABLES_DIR}")


if __name__ == "__main__":
    main()
