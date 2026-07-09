"""
Build the per-year "full-text" parquet that the analysis scripts read from.

Joins three sources into one analysis-ready table per year:

  1. The slim review parquet at data/processed/parquet/iclr_{year}_reviews.parquet
     (paper metadata + reviewer scores + external GT + slim Gemini fields).
  2. The normalized JSON at data/processed/normalized/ICLR_{year}_normalized.json
     (full review text: summary, strengths, weaknesses, questions).
  3. The Gemini full prediction JSON in data/batches/output/
     (per-reviewer tone, engagement_quality, score_confidence,
     indicators_of_engagement).

The output lands at data/processed/parquet/iclr_{year}_reviews.parquet and
**replaces** the slim version (it is a strict superset of those columns).

Why this exists
---------------
The public release ships only the parquet, not the heavy normalized JSON
(~846 MB across 4 years) or the Gemini full predictions (~273 MB across 2
years). This script lets a power user regenerate the parquet from scratch
once they have the raw + normalized + Gemini outputs.

For paper readers who just want to re-run the analyses, the rebuilt parquet
is already shipped under data/processed/parquet/.

Usage
-----
    python scripts/build_fulltext_parquet.py --year 2024
    python scripts/build_fulltext_parquet.py --year 2024 --year 2025
    python scripts/build_fulltext_parquet.py            # all years with sources

Inputs are looked up by name; missing years are skipped with a warning.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import pandas as pd

from core.paths import PROCESSED_DATA_DIR, BATCH_OUTPUT_DIR

PARQUET_DIR = PROCESSED_DATA_DIR / "parquet"
NORMALIZED_DIR = PROCESSED_DATA_DIR / "normalized"

GEMINI_FILES = {
    2023: "gemini_flash_iclr2023_full.json",
    2024: "gemini_flash_iclr2024_full.json",
    2025: "gemini_flash_iclr2025_full.json",
    2026: "gemini_flash_iclr2026_full.json",
}

NEW_COLS = [
    "review_summary",
    "review_strengths",
    "review_weaknesses",
    "review_strengths_and_weaknesses",
    "review_questions",
    "summary_len",
    "strengths_len",
    "weaknesses_len",
    "questions_len",
    "gemini_tone_primary",
    "gemini_tone_secondary",
    "gemini_engagement_quality",
    "gemini_score_confidence_llm",
    "gemini_indicators_json",
    "gemini_sentiment_score",
    "gemini_sentiment_label",
    "gemini_stance_excitement_intensity",
    "gemini_stance_frustration_intensity",
    "gemini_stance_disappointment_intensity",
    "gemini_stance_appreciation_intensity",
    "gemini_stance_confidence_in_contribution",
    "gemini_stance_concern_about_validity",
    "has_rebuttal",
    "n_author_replies",
    "rebuttal_total_len",
    "n_thread_messages",
    "first_reply_latency_days",
    "thread_max_depth",
    "discussion_thread_json",
    "review_order",
]


_INTENSITY_DIMS = ["excitement", "frustration", "disappointment", "appreciation"]
_LEVEL_DIMS = ["confidence_in_contribution", "concern_about_validity"]


def _paper_level_features(parsed: dict) -> dict:
    """Pull sentiment + reviewer_stance from parsed.core_analysis as raw strings.

    Returned values are intentionally raw (the analysis scripts apply their
    own INTENSITY_MAP / LEVEL_MAP). Missing keys land as None.
    """
    core = parsed.get("core_analysis") or {}
    sent = core.get("sentiment") or {}
    out = {
        "gemini_sentiment_score": (
            float(sent.get("score")) if isinstance(sent, dict)
            and isinstance(sent.get("score"), (int, float)) else None
        ),
        "gemini_sentiment_label": (
            sent.get("label") if isinstance(sent, dict) else None
        ),
    }
    stance = core.get("reviewer_stance") or {}
    for dim in _INTENSITY_DIMS:
        d = stance.get(dim) or {}
        intensity = d.get("intensity") if isinstance(d, dict) and d.get("present", False) else None
        out[f"gemini_stance_{dim}_intensity"] = intensity
    for dim in _LEVEL_DIMS:
        d = stance.get(dim) or {}
        level = d.get("level") if isinstance(d, dict) else None
        out[f"gemini_stance_{dim}"] = level
    return out


def _rebuttal_features(rev: dict, comments: list, include_thread: bool) -> dict:
    """Compute per-review rebuttal/thread aggregates from paper-level comments.

    Mirrors the logic in the original src/analyze/confidence_triggers.py
    and src/analyze/prediction_comparison.py:

      direct_replies   = author comments whose reply_to == review_id
      threaded         = comments whose reply_to is in {direct_replies}
      n_thread_total   = n_direct + len(threaded)

    If include_thread, also returns the JSON-encoded list of relevant
    comments (direct + threaded) so the per-row text is recoverable.
    """
    rev_id = rev.get("id")
    author_comments = [c for c in comments if c.get("participant_type") == "author"]
    direct_replies = [c for c in author_comments if c.get("reply_to") == rev_id]
    n_direct = len(direct_replies)
    total_reply_len = sum(len(c.get("text") or "") for c in direct_replies)

    direct_ids = {c.get("id") for c in direct_replies}
    threaded = [c for c in comments if c.get("reply_to") in direct_ids]
    n_thread_total = n_direct + len(threaded)

    has_rebuttal = 1 if n_direct > 0 else 0

    # First-author-reply latency in days (Jung 2025 mean response latency).
    first_reply_latency_days = float("nan")
    review_cdate = rev.get("cdate")
    if has_rebuttal and isinstance(review_cdate, (int, float)):
        reply_cdates = [c.get("cdate") for c in direct_replies
                        if isinstance(c.get("cdate"), (int, float))]
        if reply_cdates:
            first_reply_latency_days = (min(reply_cdates) - review_cdate) / 86_400_000.0

    # Max thread depth rooted at this review (Jung 2025 Depth_i).
    thread_max_depth = 1
    if has_rebuttal:
        children: dict[str, list[dict]] = {}
        for c in comments:
            parent = c.get("reply_to")
            if parent is not None:
                children.setdefault(parent, []).append(c)
        stack = [(rev_id, 1)]
        while stack:
            nid, d = stack.pop()
            if d > thread_max_depth:
                thread_max_depth = d
            for child in children.get(nid, []):
                stack.append((child.get("id"), d + 1))

    out = {
        "has_rebuttal": has_rebuttal,
        "n_author_replies": n_direct,
        "rebuttal_total_len": total_reply_len,
        "n_thread_messages": n_thread_total,
        "first_reply_latency_days": first_reply_latency_days,
        "thread_max_depth": thread_max_depth,
    }
    if include_thread:
        thread = direct_replies + threaded
        out["discussion_thread_json"] = json.dumps(thread, ensure_ascii=False)
    else:
        out["discussion_thread_json"] = None
    return out


def extract_per_review(year: int, include_thread: bool = False) -> pd.DataFrame:
    """Iterate normalized JSON + Gemini full JSON in lockstep.

    Returns one row per review, keyed by (paper_key, reviewer_suffix), with
    the text and Gemini-feature columns listed in NEW_COLS.
    """
    norm_path = NORMALIZED_DIR / f"ICLR_{year}_normalized.json"
    gem_path = BATCH_OUTPUT_DIR / GEMINI_FILES.get(year, "")

    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalized JSON: {norm_path}")
    if not gem_path.exists():
        print(f"  [warn] Gemini full file not found at {gem_path}; "
              f"Gemini-feature columns will be NaN.")

    print(f"  loading normalized JSON ({norm_path.stat().st_size / 1e6:.0f} MB)...")
    with open(norm_path) as f:
        norm = json.load(f)

    gem_by_paper: dict[str, dict] = {}
    if gem_path.exists():
        print(f"  loading Gemini full ({gem_path.stat().st_size / 1e6:.0f} MB)...")
        with open(gem_path) as f:
            preds = json.load(f)
        for p in preds:
            parsed = p.get("parsed")
            if not parsed or not isinstance(parsed, dict):
                continue
            per_rev = parsed.get("per_reviewer") or []
            gem_by_paper[str(p["custom_id"])] = {
                "per_reviewer": per_rev if isinstance(per_rev, list) else [],
                "paper_level": _paper_level_features(parsed),
            }
        del preds
        gc.collect()

    rows = []
    for paper_key, paper in norm["papers"].items():
        gem = gem_by_paper.get(str(paper_key)) or {"per_reviewer": [], "paper_level": {}}
        per_rev = gem["per_reviewer"]
        paper_level = gem["paper_level"]
        comments = paper.get("comments") or []
        for i, rev in enumerate(paper.get("reviews", [])):
            reviewer_id = rev.get("reviewer_id", "") or ""
            suffix = (
                reviewer_id.split("Reviewer_")[-1]
                if "Reviewer_" in reviewer_id
                else reviewer_id.split("/")[-1]
            )

            summary = rev.get("summary") or ""
            strengths = rev.get("strengths") or ""
            weaknesses = rev.get("weaknesses") or ""
            sw_combined = rev.get("strengths_and_weaknesses") or ""
            questions = rev.get("questions") or ""

            row = {
                "paper_key": str(paper_key),
                "reviewer_suffix": suffix,
                "review_order": i,
                "review_summary": summary,
                "review_strengths": strengths,
                "review_weaknesses": weaknesses,
                "review_strengths_and_weaknesses": sw_combined,
                "review_questions": questions,
                "summary_len": len(summary),
                "strengths_len": len(strengths) + len(sw_combined) // 2,
                "weaknesses_len": len(weaknesses) + len(sw_combined) // 2,
                "questions_len": len(questions),
            }

            if i < len(per_rev) and isinstance(per_rev[i], dict):
                gem = per_rev[i]
                tone = gem.get("tone")
                if isinstance(tone, dict):
                    row["gemini_tone_primary"] = tone.get("primary")
                    sec = tone.get("secondary")
                    row["gemini_tone_secondary"] = (
                        None if sec in (None, "null", "") else sec
                    )
                elif isinstance(tone, str):
                    row["gemini_tone_primary"] = tone
                    row["gemini_tone_secondary"] = None
                else:
                    row["gemini_tone_primary"] = None
                    row["gemini_tone_secondary"] = None

                row["gemini_engagement_quality"] = gem.get("engagement_quality")
                sc = gem.get("score_confidence")
                row["gemini_score_confidence_llm"] = (
                    float(sc) if isinstance(sc, (int, float)) else None
                )
                inds = gem.get("indicators_of_engagement") or []
                row["gemini_indicators_json"] = (
                    json.dumps(inds) if isinstance(inds, list) else "[]"
                )
            else:
                row["gemini_tone_primary"] = None
                row["gemini_tone_secondary"] = None
                row["gemini_engagement_quality"] = None
                row["gemini_score_confidence_llm"] = None
                row["gemini_indicators_json"] = None

            for k in (
                "gemini_sentiment_score",
                "gemini_sentiment_label",
                "gemini_stance_excitement_intensity",
                "gemini_stance_frustration_intensity",
                "gemini_stance_disappointment_intensity",
                "gemini_stance_appreciation_intensity",
                "gemini_stance_confidence_in_contribution",
                "gemini_stance_concern_about_validity",
            ):
                row[k] = paper_level.get(k)

            row.update(_rebuttal_features(rev, comments, include_thread))

            rows.append(row)

    del norm
    gc.collect()
    print(f"  {len(rows):,} review rows extracted")
    return pd.DataFrame(rows)


def build_year(year: int, include_thread: bool = False) -> Path:
    slim_path = PARQUET_DIR / f"iclr_{year}_reviews.parquet"
    if not slim_path.exists():
        raise FileNotFoundError(f"Slim parquet not found: {slim_path}")

    print(f"\nICLR {year}")
    slim = pd.read_parquet(slim_path)
    print(f"  slim parquet: {len(slim):,} rows x {slim.shape[1]} cols, "
          f"{slim_path.stat().st_size / 1e6:.1f} MB")

    # If a previous run wrote discussion_thread_json into the parquet but
    # this run is not asked to keep it, drop it from the input. Prevents
    # the column from silently surviving across rebuilds.
    if not include_thread and "discussion_thread_json" in slim.columns:
        slim = slim.drop(columns=["discussion_thread_json"])

    fat_only = extract_per_review(year, include_thread=include_thread)

    # Drop overlapping columns from the per-review side so the merge stays clean.
    overlap = set(fat_only.columns) & set(slim.columns) - {"paper_key", "reviewer_suffix"}
    if overlap:
        fat_only = fat_only.drop(columns=list(overlap))

    merged = slim.merge(
        fat_only, on=["paper_key", "reviewer_suffix"], how="left",
    )
    print(f"  merged: {len(merged):,} rows x {merged.shape[1]} cols")
    missing_text = merged["review_summary"].isna().sum()
    if missing_text:
        print(f"  [warn] {missing_text:,} rows missing review text "
              f"(reviewer_suffix not found in normalized JSON)")

    out_path = slim_path  # overwrite in place
    merged.to_parquet(out_path, index=False)
    new_size = out_path.stat().st_size / 1e6
    print(f"  written -> {out_path.name} ({new_size:.1f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--year", type=int, action="append",
                    help="Year(s) to build. Repeat to build several. Default: all 4.")
    ap.add_argument("--include-thread-text", action="store_true",
                    help="Also embed the full author rebuttal + threaded "
                         "follow-ups as a JSON column (discussion_thread_json). "
                         "Roughly doubles parquet size. Off by default.")
    args = ap.parse_args()

    years = args.year or sorted(GEMINI_FILES.keys())
    for y in years:
        try:
            build_year(y, include_thread=args.include_thread_text)
        except FileNotFoundError as e:
            print(f"\n[skip] ICLR {y}: {e}")


if __name__ == "__main__":
    main()
