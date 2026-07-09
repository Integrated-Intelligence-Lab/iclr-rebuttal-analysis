"""
Leakage audit for the three-horizon feature matrices.

The pre-rebuttal bucket must contain ZERO information that only becomes
available after the rebuttal or after the discussion period. We assert
this by construction (column whitelist) and on observed data:

  test_pre_has_no_post_columns
    The pre-bucket DataFrame contains none of the draft- or post-only
    column names. Tripwire: someone copies the wrong feature list.

  test_paper_initial_mean_is_initial_only
    Synthetic case: a single co-reviewer changes their post-rebuttal
    score. paper_initial_mean (the pre-bucket paper-level summary) must
    NOT change. Detects accidental use of ext_rating_after.

  test_outcome_not_in_features
    score_increase must not have a perfect correlate among the feature
    columns at any horizon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from analyze.exp34_prediction_horizons import (  # noqa: E402
    DRAFT_ONLY_FEATURES,
    POST_ONLY_FEATURES,
    _add_co_reviewer_features,
    _add_paper_initial_features,
    load_feature_matrix,
)


@pytest.fixture(scope="module")
def pre_2024() -> pd.DataFrame:
    return load_feature_matrix(2024, "pre")


@pytest.fixture(scope="module")
def draft_2024() -> pd.DataFrame:
    return load_feature_matrix(2024, "draft")


@pytest.fixture(scope="module")
def post_2024() -> pd.DataFrame:
    return load_feature_matrix(2024, "post")


def _draft_only_concrete() -> set[str]:
    """Expand the symbolic feature spec to concrete column names."""
    return {c for c in DRAFT_ONLY_FEATURES if not c.startswith("__")}


def _post_only_concrete() -> set[str]:
    return {c for c in POST_ONLY_FEATURES if not c.startswith("__")}


def test_pre_has_no_draft_or_post_columns(pre_2024):
    forbidden = _draft_only_concrete() | _post_only_concrete()
    leaked = forbidden & set(pre_2024.columns)
    assert not leaked, f"pre-bucket leaked draft/post columns: {sorted(leaked)}"


def test_pre_columns_are_well_formed(pre_2024):
    """Sanity: pre-bucket has the expected metadata columns."""
    expected = {
        "paper_key", "score_increase",
        "pos_before", "deviation_from_paper_mean",
        "paper_initial_mean", "paper_initial_std",
    }
    missing = expected - set(pre_2024.columns)
    assert not missing, f"pre-bucket missing expected columns: {sorted(missing)}"


def test_paper_initial_mean_uses_initial_only():
    """paper_initial_mean must NOT shift if a co-reviewer changes ext_rating_after."""
    base = pd.DataFrame({
        "paper_key": ["P1"] * 3,
        "ext_rating_before": [5, 6, 8],
        "ext_rating_after": [5, 6, 8],  # baseline: no movement
        "confidence": [4, 4, 4],
    })
    enriched = _add_paper_initial_features(base)
    mean_a = enriched["paper_initial_mean"].iloc[0]

    perturbed = base.copy()
    perturbed.loc[1, "ext_rating_after"] = 10  # co-reviewer raised score
    enriched_b = _add_paper_initial_features(perturbed)
    mean_b = enriched_b["paper_initial_mean"].iloc[0]

    assert mean_a == pytest.approx(mean_b), (
        "paper_initial_mean changed when ext_rating_after changed — "
        "bucket is leaking post-discussion information."
    )


def test_co_reviewer_features_exclude_self():
    """Synthetic 3-reviewer paper: row 0's co_reviewer_post_change_mean must be
    the average of rows 1 and 2 only."""
    df = pd.DataFrame({
        "paper_key": ["P1"] * 3,
        "ext_rating_before": [5, 5, 5],
        "ext_rating_after": [5, 6, 7],  # deltas: 0, 1, 2
        "confidence": [4, 4, 4],
    })
    out = _add_co_reviewer_features(df)
    # Row 0: mean of others' deltas = (1+2)/2 = 1.5
    assert out["co_reviewer_post_change_mean"].iloc[0] == pytest.approx(1.5)
    # Row 0: max of others' deltas = 2
    assert out["co_reviewer_max_increase"].iloc[0] == pytest.approx(2.0)
    # Row 0: count of others increased = 2
    assert out["co_reviewer_n_increased"].iloc[0] == 2

    # Row 2 holds the unique max — should report second-largest = 1
    assert out["co_reviewer_max_increase"].iloc[2] == pytest.approx(1.0)


def test_outcome_not_perfectly_correlated_with_any_feature(pre_2024, draft_2024, post_2024):
    """No single feature should give away the binary outcome."""
    for label, df in [("pre", pre_2024), ("draft", draft_2024), ("post", post_2024)]:
        y = df["score_increase"].astype(float).to_numpy()
        for col in df.columns:
            if col in {"paper_key", "score_increase"}:
                continue
            x = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isnan(x).any():
                continue
            if x.std() == 0 or y.std() == 0:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            assert abs(r) < 0.99, (
                f"[{label}] feature '{col}' has |r|={abs(r):.3f} with the outcome — "
                "looks like outcome leakage."
            )
