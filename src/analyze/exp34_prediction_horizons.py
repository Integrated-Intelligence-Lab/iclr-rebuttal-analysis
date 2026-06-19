"""
Three-horizon prediction of successful rebuttal.

For each authoring stage we ask: how much of the post-rebuttal score
movement is recoverable from features available at that stage?

  pre   — features available before the author writes anything
          (reviewer-supplied scores, sub-scores, deviation from paper
          initial mean, Gemini text-derived offset and indicators).
  draft — pre + rebuttal-content features (length, replies, F01..F44).
  post  — draft + co-reviewer movement features (only visible after
          the discussion period; framed descriptively, not actionable).

For each horizon we fit two models:
  - Logistic regression (interpretable baseline)
  - Random forest      (predictive ceiling)

Outcome:
  score_increase = (ext_rating_after > ext_rating_before)
  i.e., "same" and "decrease" collapsed to the negative class.

Evaluation:
  - Temporal split: train ICLR 2024 -> test ICLR 2025 (headline).
  - GroupKFold(5) by paper_key on combined data (robustness).
  - Bootstrap (1000 resamples) 95% CIs.
  - Permutation importance per horizon on the temporal test set.

Usage:
    python src/analyze/rebuttal_rf_prediction.py
    python src/analyze/rebuttal_rf_prediction.py --horizons pre draft post
    python src/analyze/rebuttal_rf_prediction.py --no-save
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analyze.confidence_triggers import extract_rebuttal_features
from analyze.prediction_comparison import TONE_CATEGORIES, build_dataset
from core.paths import RESULTS_DIR
from core.plot_theme import setup_publication_theme
from factor_validation.prompt import VALID_FACTOR_IDS

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
FIGURES_DIR = RESULTS_DIR / "rebuttal_rf_prediction" / "figures"
TABLES_DIR = RESULTS_DIR / "rebuttal_rf_prediction" / "tables"
LABELS_DIR = RESULTS_DIR / "32_factor_taxonomy"
RANDOM_STATE = 50  # project convention (see CLAUDE.md sampling)

FACTOR_COLS = sorted(VALID_FACTOR_IDS)

# Feature buckets per horizon. Columns are added cumulatively:
# draft = pre + draft-only; post = draft + post-only.

PRE_FEATURES = [
    # Reviewer-supplied at submission time
    "pos_before", "confidence", "soundness", "presentation", "contribution",
    "strengths_len", "weaknesses_len", "questions_len", "sw_ratio",
    # Paper-level computed from initial scores ONLY (no co-reviewer movement)
    "paper_initial_mean", "paper_initial_std", "n_reviewers",
    "deviation_from_paper_mean", "is_initial_outlier",
    # Timing: days before the official ICLR review-submission deadline for the
    # year (positive = on-time, negative = late). Replaces the older within-year
    # percentile to align with Jung et al.'s "days remaining until deadline".
    "days_before_deadline",
    # Gemini text-derived
    "gemini_offset", "engagement_ord", "score_confidence_gemini",
    "n_indicators",
    "has_technical_depth", "has_actionable_feedback",
    "has_specific_examples", "has_trivial_feedback",
    # tone one-hot expanded dynamically
    "__tone__",
]

# Ablation horizon: pre-rebuttal *without* the focal reviewer's initial score.
# Tests whether pre-rebuttal predictability is a tautology of pos_before, or
# whether the paper-context envelope (paper_initial_mean, deviation, etc.)
# carries independent signal. Reported alongside the full pre-rebuttal model.
PRE_NO_SCORE_FEATURES = [c for c in PRE_FEATURES if c != "pos_before"]

DRAFT_ONLY_FEATURES = [
    "rebuttal_total_len", "n_author_replies",
    # Rebuttal-side timing and structure (Jung et al. 2025): mean / first
    # author-reply latency in days, and max thread depth from this review.
    "first_reply_latency_days", "thread_max_depth",
    "n_factors_present",
] + FACTOR_COLS

POST_ONLY_FEATURES = [
    # n_thread_messages counts every thread message, including reviewer
    # follow-ups; it only resolves after the discussion period, so it
    # belongs here rather than at draft-time.
    "n_thread_messages",
    "co_reviewer_post_change_mean",
    "co_reviewer_max_increase",
    "co_reviewer_n_increased",
]


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------
def _expand_tone(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot expand tone_primary into columns; drop the categorical."""
    df = df.copy()
    if "tone_primary" in df.columns:
        for tone in TONE_CATEGORIES:
            df[f"tone_{tone}"] = (df["tone_primary"] == tone).astype(int)
    return df


def _add_paper_initial_features(df: pd.DataFrame) -> pd.DataFrame:
    """paper_initial_mean/std + outlier flag, all from ext_rating_before only."""
    df = df.copy()
    paper_stats = df.groupby("paper_key")["ext_rating_before"].agg(
        paper_initial_mean="mean", paper_initial_std="std",
    ).reset_index()
    paper_stats["paper_initial_std"] = paper_stats["paper_initial_std"].fillna(0)
    df = df.merge(paper_stats, on="paper_key", how="left")
    df["deviation_from_paper_mean"] = (
        df["ext_rating_before"].astype(float) - df["paper_initial_mean"]
    )
    df["is_initial_outlier"] = (
        (df["deviation_from_paper_mean"].abs() >= 1.0)
        & (df["confidence"] >= 4)
    ).astype(int)
    return df


def _add_co_reviewer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Post-discussion only: aggregate OTHER reviewers' movement per paper.

    For each row (paper, reviewer), compute aggregates over the OTHER reviewers
    of the same paper using ext_rating_before/after. Excludes self by construction.
    """
    df = df.copy()
    df["_delta"] = df["ext_rating_after"].astype(float) - df["ext_rating_before"].astype(float)
    df["_inc"] = (df["_delta"] > 0).astype(int)

    paper_sum_delta = df.groupby("paper_key")["_delta"].transform("sum")
    paper_sum_inc = df.groupby("paper_key")["_inc"].transform("sum")
    paper_max_delta = df.groupby("paper_key")["_delta"].transform("max")
    paper_n_at_max = df.groupby("paper_key")["_delta"].transform(
        lambda s: (s == s.max()).sum()
    )
    paper_second_max = df.groupby("paper_key")["_delta"].transform(
        lambda s: s.nlargest(2).iloc[-1] if len(s) >= 2 else float("nan")
    )
    paper_n = df.groupby("paper_key")["_delta"].transform("count")

    other_n = (paper_n - 1).clip(lower=1)
    df["co_reviewer_post_change_mean"] = (paper_sum_delta - df["_delta"]) / other_n
    df["co_reviewer_n_increased"] = (paper_sum_inc - df["_inc"]).astype(int)

    # Max increase among co-reviewers: if self uniquely holds the paper max,
    # use the second-max; otherwise paper_max already includes a co-reviewer.
    is_self_unique_max = (
        (df["_delta"] >= paper_max_delta - 1e-9) & (paper_n_at_max == 1)
    )
    df["co_reviewer_max_increase"] = np.where(
        is_self_unique_max, paper_second_max, paper_max_delta
    )
    df["co_reviewer_max_increase"] = (
        df["co_reviewer_max_increase"].astype(float).fillna(0.0)
    )

    df = df.drop(columns=["_delta", "_inc"])
    return df


def _load_factor_labels(year: int) -> pd.DataFrame:
    """Load F01..F44 labels for the year."""
    p = LABELS_DIR / f"interaction_factor_labels_{year}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing factor labels: {p}")
    return pd.read_csv(p)


def load_feature_matrix(year: int, horizon: str) -> pd.DataFrame:
    """Build the per-review feature matrix for a given year and horizon.

    horizon: "pre" | "draft" | "post"

    All horizons are filtered to the same subset (rows that have a rebuttal
    AND a factor-label row), so that AUCs are comparable across horizons.
    """
    if horizon not in {"pre", "pre_no_score", "draft", "post"}:
        raise ValueError(f"Unknown horizon: {horizon}")

    base = build_dataset(year)
    base = base.rename(columns={
        "score_increased": "score_increase",
        "score_confidence_gemini": "score_confidence_gemini",
    })
    base = _add_paper_initial_features(base)

    # Timing: days before the official ICLR review-submission deadline for
    # this cohort. Positive = on-time, negative = late submitter. Aligns with
    # Jung et al.'s "days remaining until deadline" feature. The published
    # ICLR review deadlines are 2023-11-10 23:59 UTC (ICLR 2024) and
    # 2024-11-12 23:59 UTC (ICLR 2025); 99.0% / 99.7% of recorded review_cdates
    # fall on or before these times respectively, confirming the anchor.
    DEADLINE_CDATE_MS = {2024: 1699660799000, 2025: 1731455999000}
    if "review_cdate" in base.columns and year in DEADLINE_CDATE_MS:
        cd = pd.to_numeric(base["review_cdate"], errors="coerce")
        deadline_ms = DEADLINE_CDATE_MS[year]
        base["days_before_deadline"] = (deadline_ms - cd) / 86_400_000.0
        base["days_before_deadline"] = base["days_before_deadline"].fillna(
            base["days_before_deadline"].median()
        )
    else:
        base["days_before_deadline"] = 0.0

    # Rebuttal features (has_rebuttal + thread aggregates + Jung-style timing/depth).
    rebuttal = extract_rebuttal_features(year)
    rebuttal_cols = [
        "has_rebuttal", "n_author_replies", "rebuttal_total_len",
        "n_thread_messages", "first_reply_latency_days", "thread_max_depth",
    ]
    base = base.drop(columns=[c for c in rebuttal_cols if c in base.columns])
    base = base.merge(
        rebuttal[["paper_key", "reviewer_suffix"] + rebuttal_cols],
        on=["paper_key", "reviewer_suffix"], how="left",
    )

    # Factor labels (F01..F44 + n_factors_present).
    labels = _load_factor_labels(year)
    label_keep = ["paper_key", "reviewer_id", "n_factors_present"] + FACTOR_COLS
    base["paper_key"] = base["paper_key"].astype(str)
    labels["paper_key"] = labels["paper_key"].astype(str)
    base = base.merge(labels[label_keep], on=["paper_key", "reviewer_id"], how="left")

    # Same denominator across horizons: keep rows with rebuttal AND factor labels.
    has_factor_labels = base["n_factors_present"].notna()
    has_rebuttal = (base.get("has_rebuttal", 0).fillna(0).astype(int) == 1)
    base = base[has_factor_labels & has_rebuttal].copy()

    base = _add_co_reviewer_features(base)
    base = _expand_tone(base)

    if horizon == "pre":
        feature_spec = PRE_FEATURES
    elif horizon == "pre_no_score":
        feature_spec = PRE_NO_SCORE_FEATURES
    elif horizon == "draft":
        feature_spec = PRE_FEATURES + DRAFT_ONLY_FEATURES
    else:
        feature_spec = PRE_FEATURES + DRAFT_ONLY_FEATURES + POST_ONLY_FEATURES

    cols: list[str] = ["paper_key", "score_increase"]
    for c in feature_spec:
        if c == "__tone__":
            cols.extend([f"tone_{t}" for t in TONE_CATEGORIES if f"tone_{t}" in base.columns])
        elif c in base.columns:
            cols.append(c)
    cols = list(dict.fromkeys(cols))  # de-dup, preserve order

    out = base[cols].copy()
    # Coerce all features to numeric; impute remaining NaN with 0.
    for c in cols:
        if c in {"paper_key"}:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(0)
    out["score_increase"] = out["score_increase"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def fit_rf(X: pd.DataFrame, y: np.ndarray) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X.values, y)
    return rf


def fit_logistic(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    """Linear baseline (matches M8 of prediction_comparison.py)."""
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    lr.fit(X_scaled, y)
    lr._scaler = scaler  # stash for predict
    return lr


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, LogisticRegression):
        return model.predict_proba(model._scaler.transform(X.values))[:, 1]
    return model.predict_proba(X.values)[:, 1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def bootstrap_ci(
    y_true: np.ndarray, y_score: np.ndarray, metric, n: int = 1000, seed: int = RANDOM_STATE
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = metric(y_true, y_score)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        boots.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def evaluate_temporal_split(
    df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list[str]
) -> dict:
    """Train on 2024, test on 2025. Returns metrics for RF and logistic.

    Also stashes y_true and y_score arrays under "_curves" so the caller can
    plot ROC / PR overlays without refitting.
    """
    X_train, y_train = df_train[feature_cols], df_train["score_increase"].to_numpy()
    X_test, y_test = df_test[feature_cols], df_test["score_increase"].to_numpy()

    out = {"n_train": len(X_train), "n_test": len(X_test), "base_rate_test": float(y_test.mean())}
    curves = {"y_test": y_test}
    for name, fitter in [("rf", fit_rf), ("logistic", fit_logistic)]:
        model = fitter(X_train, y_train)
        scores = predict_proba(model, X_test)
        curves[name] = scores
        auc, auc_lo, auc_hi = bootstrap_ci(y_test, scores, roc_auc_score)
        ap, ap_lo, ap_hi = bootstrap_ci(y_test, scores, average_precision_score)
        out[name] = {
            "auc_roc": auc, "auc_roc_ci": (auc_lo, auc_hi),
            "auprc": ap, "auprc_ci": (ap_lo, ap_hi),
        }
    out["_curves"] = curves
    return out


def evaluate_grouped_cv(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5) -> dict:
    """Paper-clustered cross-validation. Returns mean +/- std of metrics."""
    X = df[feature_cols]
    y = df["score_increase"].to_numpy()
    groups = df["paper_key"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    metrics = {"rf": [], "logistic": []}
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        for name, fitter in [("rf", fit_rf), ("logistic", fit_logistic)]:
            model = fitter(X_tr, y_tr)
            scores = predict_proba(model, X_te)
            metrics[name].append({
                "auc_roc": roc_auc_score(y_te, scores),
                "auprc": average_precision_score(y_te, scores),
            })

    summary = {}
    for name, fold_metrics in metrics.items():
        summary[name] = {
            "auc_roc_mean": float(np.mean([m["auc_roc"] for m in fold_metrics])),
            "auc_roc_std": float(np.std([m["auc_roc"] for m in fold_metrics])),
            "auprc_mean": float(np.mean([m["auprc"] for m in fold_metrics])),
            "auprc_std": float(np.std([m["auprc"] for m in fold_metrics])),
        }
    return summary


def compute_permutation_importance(
    model: RandomForestClassifier, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10
) -> pd.DataFrame:
    result = permutation_importance(
        model, X.values, y, n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1, scoring="average_precision",
    )
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
HORIZON_TITLES = {
    "pre": "Pre-rebuttal",
    "pre_no_score": "Pre-rebuttal (no initial score)",
    "draft": "Rebuttal-thread content",
    "post": "Post-discussion",
}


def plot_horizons(temporal_by_horizon: dict[str, dict]) -> plt.Figure:
    """3-panel figure: ROC + PR overlays per horizon, RF and logistic both."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    setup_publication_theme()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex="row", sharey="row")

    horizons = ["pre", "draft", "post"]
    for col, h in enumerate(horizons):
        if h not in temporal_by_horizon:
            for r in (0, 1):
                axes[r, col].set_visible(False)
            continue
        m = temporal_by_horizon[h]
        curves = m["_curves"]
        y = curves["y_test"]
        for name, color, marker in [("rf", "#0072B2", "-"), ("logistic", "#D55E00", "--")]:
            scores = curves[name]
            fpr, tpr, _ = roc_curve(y, scores)
            prec, rec, _ = precision_recall_curve(y, scores)
            axes[0, col].plot(
                fpr, tpr, marker, color=color,
                label=f"{name.upper()} (AUC={m[name]['auc_roc']:.3f})",
            )
            axes[1, col].plot(
                rec, prec, marker, color=color,
                label=f"{name.upper()} (AP={m[name]['auprc']:.3f})",
            )
        axes[0, col].plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
        axes[1, col].axhline(m["base_rate_test"], ls=":", color="grey", lw=0.8,
                             label=f"base rate={m['base_rate_test']:.2f}")
        axes[0, col].set_title(f"{HORIZON_TITLES[h]}\nn_train={m['n_train']}, n_test={m['n_test']}")
        axes[0, col].set_xlabel("False positive rate")
        axes[0, col].set_ylabel("True positive rate" if col == 0 else "")
        axes[1, col].set_xlabel("Recall")
        axes[1, col].set_ylabel("Precision" if col == 0 else "")
        axes[0, col].legend(loc="lower right", fontsize=8)
        axes[1, col].legend(loc="upper right", fontsize=8)

    fig.suptitle("Predicting successful rebuttal at three observation horizons", y=1.02)
    fig.tight_layout()
    return fig


def plot_permutation_importance(importance_df: pd.DataFrame, top_k: int = 15,
                                title: str = "RF permutation importance") -> plt.Figure:
    setup_publication_theme()
    fig, ax = plt.subplots(figsize=(7, max(3, top_k * 0.28)))
    top = importance_df.head(top_k).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Permutation importance (delta AUPRC)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _serialize_metrics(metrics: dict) -> dict:
    """Drop _curves (numpy arrays) so dict can be JSON-dumped."""
    return {k: v for k, v in metrics.items() if k != "_curves"}


def run_horizon(horizon: str, save: bool) -> dict:
    """Build matrices for one horizon and run all three evaluations."""
    print(f"\n{'#' * 66}\n# Horizon: {horizon}\n{'#' * 66}")
    df_2024 = load_feature_matrix(2024, horizon)
    df_2025 = load_feature_matrix(2025, horizon)
    feature_cols = [c for c in df_2024.columns
                    if c not in {"paper_key", "score_increase"}]
    print(f"Features ({len(feature_cols)}): {feature_cols[:6]} ... {feature_cols[-6:]}")
    print(f"n_2024={len(df_2024)} (positive={int(df_2024['score_increase'].sum())}), "
          f"n_2025={len(df_2025)} (positive={int(df_2025['score_increase'].sum())})")

    print(f"\n--- [{horizon}] Temporal split (train 2024 -> test 2025) ---")
    temporal = evaluate_temporal_split(df_2024, df_2025, feature_cols)
    print(json.dumps(_serialize_metrics(temporal), indent=2, default=str))

    print(f"\n--- [{horizon}] GroupKFold(5) on 2024+2025 ---")
    # Year-prefix paper_key so 2024_pk0 != 2025_pk0 in GroupKFold.
    df_2024_g = df_2024.assign(paper_key="2024_" + df_2024["paper_key"].astype(str))
    df_2025_g = df_2025.assign(paper_key="2025_" + df_2025["paper_key"].astype(str))
    df_combined = pd.concat([df_2024_g, df_2025_g], ignore_index=True)
    cv = evaluate_grouped_cv(df_combined, feature_cols)
    print(json.dumps(cv, indent=2))

    print(f"\n--- [{horizon}] RF permutation importance on temporal test ---")
    rf = fit_rf(df_2024[feature_cols], df_2024["score_increase"].to_numpy())
    importance_df = compute_permutation_importance(
        rf, df_2025[feature_cols], df_2025["score_increase"].to_numpy(),
    )
    print(importance_df.head(15).to_string(index=False))

    if save:
        pd.DataFrame([_serialize_metrics(temporal)]).to_csv(
            TABLES_DIR / f"temporal_split_metrics_{horizon}.csv", index=False,
        )
        pd.DataFrame(cv).to_csv(TABLES_DIR / f"grouped_cv_metrics_{horizon}.csv")
        importance_df.to_csv(
            TABLES_DIR / f"permutation_importance_{horizon}.csv", index=False,
        )
        fig_imp = plot_permutation_importance(
            importance_df, title=f"RF permutation importance — {HORIZON_TITLES[horizon]}",
        )
        fig_imp.savefig(FIGURES_DIR / f"permutation_importance_{horizon}.pdf")
        fig_imp.savefig(FIGURES_DIR / f"permutation_importance_{horizon}.png", dpi=300)
        plt.close(fig_imp)

    return {"horizon": horizon, "temporal": temporal, "cv": cv,
            "n_train": len(df_2024), "n_test": len(df_2025),
            "n_features": len(feature_cols)}


def main(horizons: list[str], save: bool = True) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    summary_rows = []
    for h in horizons:
        res = run_horizon(h, save=save)
        results[h] = res["temporal"]
        for model in ("rf", "logistic"):
            m = res["temporal"][model]
            cv_m = res["cv"][model]
            summary_rows.append({
                "horizon": h, "model": model,
                "n_features": res["n_features"],
                "n_train": res["n_train"], "n_test": res["n_test"],
                "auc_roc": m["auc_roc"], "auc_roc_ci": str(m["auc_roc_ci"]),
                "auprc": m["auprc"], "auprc_ci": str(m["auprc_ci"]),
                "auc_roc_cv_mean": cv_m["auc_roc_mean"],
                "auc_roc_cv_std": cv_m["auc_roc_std"],
                "auprc_cv_mean": cv_m["auprc_mean"],
                "auprc_cv_std": cv_m["auprc_std"],
            })

    if save:
        pd.DataFrame(summary_rows).to_csv(
            TABLES_DIR / "incremental_value_summary.csv", index=False,
        )
        if len(results) >= 1:
            fig = plot_horizons(results)
            fig.savefig(FIGURES_DIR / "figure_horizons.pdf", bbox_inches="tight")
            fig.savefig(FIGURES_DIR / "figure_horizons.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print(f"\nOutputs saved to: {RESULTS_DIR / 'rebuttal_rf_prediction'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", nargs="+", default=["pre", "draft", "post"],
                        choices=["pre", "pre_no_score", "draft", "post"])
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    main(horizons=args.horizons, save=not args.no_save)
