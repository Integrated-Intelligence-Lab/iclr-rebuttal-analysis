"""
Paper: Appendix (app:ext-gt) -- ground-truth score-profile of the 1K validation sample.

Ground-truth score-profile summary for sampled vs full eligible reviews.

Changes:
- 2026-04-09: Added sample-vs-full tier/distribution report using ext ground truth.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.io import write_file, write_json
from core.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR


def classify_tier(score: int | float | None) -> str | None:
    """Map an initial review score to the low/mid/high analysis tiers.
    Returns None when the value is missing or outside the ICLR discrete bands used here.
    """
    if score is None or pd.isna(score):
        return None
    s = int(score)
    if 1 <= s <= 3:
        return "low"
    if 5 <= s <= 6:
        return "mid"
    if 8 <= s <= 10:
        return "high"
    return None


def normalize_direction(before: int | float | None, after: int | float | None, ext: str | None) -> str | None:
    """Return a normalized score-change label: increase, decrease, or same.
    Prefers `ext_direction` when present; otherwise it derives direction from before/after ratings.
    """
    if isinstance(ext, str) and ext.strip():
        e = ext.strip().lower()
        if e.startswith("inc"):
            return "increase"
        if e.startswith("dec"):
            return "decrease"
        if e.startswith("same") or e.startswith("unch"):
            return "same"
    if before is None or after is None or pd.isna(before) or pd.isna(after):
        return None
    if float(after) > float(before):
        return "increase"
    if float(after) < float(before):
        return "decrease"
    return "same"


def load_reviews_parquet(year: int) -> pd.DataFrame:
    """Load the review-level parquet for one year and attach the year column.
    Raises a clear error when the parquet file does not exist.
    """
    path = PROCESSED_DATA_DIR / "parquet" / f"iclr_{year}_reviews.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet for {year}: {path}")
    df = pd.read_parquet(path)
    df["year"] = year
    return df


def load_pairs_allowlist(path: Path) -> set[tuple[str, str]]:
    """Load sampled review pairs from a JSON list of objects.
    Each object must provide `paper_id` and `reviewer_id`, and the function returns a deduplicated set.
    """
    data = json.load(open(path))
    pairs: set[tuple[str, str]] = set()
    if not isinstance(data, list):
        raise ValueError(f"Pair allowlist must be a list: {path}")
    for item in data:
        if isinstance(item, dict):
            pid = str(item.get("forum", "") or item.get("paper_id", "")).strip()
            rid = str(item.get("reviewer_id", "")).strip()
            if pid and rid:
                pairs.add((pid, rid))
    return pairs


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean analysis frame with validated ground-truth fields and derived labels.
    Keeps only rows with valid ext before/after ratings, then adds normalized direction and tier columns.
    """
    needed = {"paper_key", "reviewer_id", "ext_rating_before", "ext_rating_after"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["paper_id"] = out["paper_key"].astype(str)
    out["forum"] = out["forum"].astype(str) if "forum" in out.columns else out["paper_id"]
    out["reviewer_id"] = out["reviewer_id"].astype(str)
    out["ext_rating_before"] = pd.to_numeric(out["ext_rating_before"], errors="coerce")
    out["ext_rating_after"] = pd.to_numeric(out["ext_rating_after"], errors="coerce")
    out = out[out["ext_rating_before"].notna() & out["ext_rating_after"].notna()].copy()

    ext_col = out["ext_direction"] if "ext_direction" in out.columns else pd.Series([None] * len(out), index=out.index)
    # Build one normalized label per row:
    # 1) use external ground-truth `ext_direction` when available,
    # 2) otherwise infer from `ext_rating_before` vs `ext_rating_after`.
    out["direction"] = [
        normalize_direction(b, a, e)
        for b, a, e in zip(out["ext_rating_before"], out["ext_rating_after"], ext_col)
    ]
    out["tier"] = out["ext_rating_before"].map(classify_tier)
    out = out[out["tier"].notna() & out["direction"].notna()].copy()
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Compute summary statistics for one dataset slice.
    Returns totals, tier distribution, and per-tier direction counts/percentages.
    """
    n_reviews = len(df)
    paper_col = "forum" if "forum" in df.columns else "paper_id"
    n_papers = int(df[paper_col].nunique()) if paper_col in df.columns else 0

    tier_counts = df["tier"].value_counts().to_dict()
    tier_pct = ((df["tier"].value_counts(normalize=True) * 100).round(2)).to_dict()

    ctab = pd.crosstab(df["tier"], df["direction"])
    for c in ["decrease", "same", "increase"]:
        if c not in ctab.columns:
            ctab[c] = 0
    ctab = ctab[["decrease", "same", "increase"]]
    ctab_pct = (ctab.div(ctab.sum(axis=1), axis=0) * 100).round(2).fillna(0.0)

    return {
        "n_reviews": int(n_reviews),
        "n_papers": n_papers,
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "tier_pct": {k: float(v) for k, v in tier_pct.items()},
        "by_tier_direction_counts": {
            tier: {k: int(v) for k, v in row.items()}
            for tier, row in ctab.to_dict(orient="index").items()
        },
        "by_tier_direction_pct": {
            tier: {k: float(v) for k, v in row.items()}
            for tier, row in ctab_pct.to_dict(orient="index").items()
        },
    }


def compare_sample_vs_full(sample_df: pd.DataFrame, full_df: pd.DataFrame) -> list[dict]:
    """Compare tier proportions between sampled and full datasets.
    Returns per-tier sample/full percentages and delta in percentage points.
    """
    tiers = ["low", "mid", "high"]
    rows: list[dict] = []
    sample_pct = sample_df["tier"].value_counts(normalize=True)
    full_pct = full_df["tier"].value_counts(normalize=True)
    for tier in tiers:
        s = float(sample_pct.get(tier, 0.0) * 100.0)
        f = float(full_pct.get(tier, 0.0) * 100.0)
        rows.append(
            {
                "tier": tier,
                "sample_pct": round(s, 2),
                "full_pct": round(f, 2),
                "delta_pct_points": round(s - f, 2),
            }
        )
    return rows


def _is_official_review(interaction: dict) -> bool:
    invitations = interaction.get("invitations", []) or []
    inv_str = " ".join(str(x) for x in invitations)
    return "Official_Review" in inv_str and "Rebuttal" not in inv_str


def _has_authors(signatures: list) -> bool:
    return any("Authors" in str(s) for s in (signatures or []))


def has_rebuttal_interaction(raw_data: dict, paper_id: str, reviewer_id: str) -> bool:
    """Check reviewer-author interaction chain exists for a sampled review.
    Returns True only when reviewer follow-up occurs after an author response under the review thread.
    """
    entry = raw_data.get(str(paper_id), {})
    interactions = sorted(entry.get("interactions", []), key=lambda x: x.get("cdate", 0))
    if not interactions:
        return False

    review = None
    for inter in interactions:
        if not _is_official_review(inter):
            continue
        if reviewer_id not in (inter.get("signatures") or []):
            continue
        review = inter
        break
    if not review:
        return False

    review_id = review.get("id")
    if not review_id:
        return False

    children_by_parent: dict[str, list[dict]] = {}
    for node in interactions:
        pid = node.get("replyto")
        if pid:
            children_by_parent.setdefault(pid, []).append(node)

    subtree = []
    queue = [review_id]
    while queue:
        pid = queue.pop(0)
        for child in children_by_parent.get(pid, []):
            subtree.append(child)
            cid = child.get("id")
            if cid:
                queue.append(cid)
    subtree.sort(key=lambda x: x.get("cdate", 0))

    first_author_cdate = None
    for node in subtree:
        if _has_authors(node.get("signatures") or []):
            first_author_cdate = node.get("cdate", 0)
            break
    if first_author_cdate is None:
        return False

    reviewer_sigs = review.get("signatures") or []
    for node in subtree:
        if (node.get("signatures") or []) == reviewer_sigs and node.get("cdate", 0) > first_author_cdate:
            return True
    return False


def build_raw_lookup(raw_data: dict) -> dict[str, str]:
    """Build a lookup from forum/raw-key identifiers to raw dataset keys.
    This lets validation work whether sampled pairs store legacy `paper_id` or OpenReview `forum`.
    """
    lookup: dict[str, str] = {}
    for key, entry in raw_data.items():
        key_s = str(key)
        lookup[key_s] = key_s
        if isinstance(entry, dict):
            forum = str(entry.get("forum") or "").strip()
            if forum:
                lookup[forum] = key_s
    return lookup


def validate_sample_interactions(sample_pairs: set[tuple[str, str]]) -> dict:
    """Validate sampled pairs have reviewer-author follow-up interactions.

    Uses the per-year parquet's `has_rebuttal` column (precomputed by
    scripts/build_fulltext_parquet.py) instead of loading the raw 3.2 GB
    OpenReview dumps.
    """
    import pandas as pd
    parquet_dir = PROCESSED_DATA_DIR / "parquet"

    rebuttal_keys: set[tuple[str, str]] = set()
    for year in (2024, 2025):
        p = parquet_dir / f"iclr_{year}_reviews.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["forum", "paper_key", "reviewer_id",
                                         "reviewer_suffix", "has_rebuttal"])
        df = df[df["has_rebuttal"] == 1]
        for _, r in df.iterrows():
            forum = str(r["forum"]) if r["forum"] else ""
            rid_full = str(r["reviewer_id"]) if r["reviewer_id"] else ""
            rid_suffix = str(r["reviewer_suffix"]) if r["reviewer_suffix"] else ""
            if forum:
                if rid_full:
                    rebuttal_keys.add((forum, rid_full))
                if rid_suffix:
                    rebuttal_keys.add((forum, rid_suffix))

    has_interactions = 0
    missing: list[dict] = []
    for paper_id, reviewer_id in sorted(sample_pairs):
        ok = (str(paper_id), str(reviewer_id)) in rebuttal_keys
        if ok:
            has_interactions += 1
        else:
            missing.append({"paper_id": paper_id, "reviewer_id": reviewer_id})

    total = len(sample_pairs)
    return {
        "total_pairs": total,
        "pairs_with_interactions": has_interactions,
        "pairs_without_interactions": total - has_interactions,
        "interaction_coverage_pct": round((has_interactions / total * 100.0), 2) if total else 0.0,
        "missing_examples": missing[:20],
    }


def render_markdown(
    sample_summary: dict,
    full_summary: dict,
    tier_comparison: list[dict],
    sample_path: Path,
    interaction_validation: dict,
) -> str:
    """Render a human-readable markdown report from computed summaries.
    Includes overall totals, sample-vs-full tier comparison, and per-tier direction tables for the sample.
    """
    lines = [
        "# Ground-Truth Score Profile: Sample vs Full Dataset",
        "",
        f"**Sample source**: `{sample_path}`",
        "**Years**: 2024 + 2025",
        "**Ground truth**: `ext_rating_before`, `ext_rating_after`, `ext_direction`",
        "",
        "## Totals",
        "",
        f"- Sample: {sample_summary['n_reviews']} reviews across {sample_summary['n_papers']} papers",
        f"- Full eligible: {full_summary['n_reviews']} reviews across {full_summary['n_papers']} papers",
        "",
        "## Tier Distribution (Sample vs Full)",
        "",
        "| Tier | Sample % | Full % | Delta (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in tier_comparison:
        lines.append(f"| {row['tier']} | {row['sample_pct']:.2f} | {row['full_pct']:.2f} | {row['delta_pct_points']:+.2f} |")

    lines.extend(
        [
            "",
            "## Interaction Validation (Sample)",
            "",
            f"- Pairs validated: {interaction_validation['total_pairs']}",
            f"- With interactions: {interaction_validation['pairs_with_interactions']}",
            f"- Without interactions: {interaction_validation['pairs_without_interactions']}",
            f"- Coverage: {interaction_validation['interaction_coverage_pct']:.2f}%",
            "",
            "## Direction Counts By Tier (Sample)",
            "",
            "| Tier | Decrease | Same | Increase |",
            "|---|---:|---:|---:|",
        ]
    )
    for tier in ["low", "mid", "high"]:
        row = sample_summary["by_tier_direction_counts"].get(tier, {})
        lines.append(
            f"| {tier} | {row.get('decrease', 0)} | {row.get('same', 0)} | {row.get('increase', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Direction Percentages By Tier (Sample)",
            "",
            "| Tier | Decrease % | Same % | Increase % |",
            "|---|---:|---:|---:|",
        ]
    )
    for tier in ["low", "mid", "high"]:
        row = sample_summary["by_tier_direction_pct"].get(tier, {})
        lines.append(
            f"| {tier} | {row.get('decrease', 0.0):.2f} | {row.get('same', 0.0):.2f} | {row.get('increase', 0.0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sample profile generation.
    Exposes input sample-pairs path and output paths for JSON and markdown artifacts.
    """
    parser = argparse.ArgumentParser(description="Ground-truth sample profile report.")
    parser.add_argument(
        "--sample-pairs-json",
        type=Path,
        default=RESULTS_DIR / "methodology" / "score_change_samples" / "iclr_2025_accepted_rejected_seed50_1k_pairs.json",
        help="JSON file with sampled pairs [{paper_id, reviewer_id}, ...].",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "methodology" / "sample_ground_truth_profile.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=RESULTS_DIR / "methodology" / "sample_ground_truth_profile.md",
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end sample profile workflow.
    Loads data, computes sample/full summaries, writes outputs, and prints a compact run summary.
    """
    args = parse_args()
    pairs = load_pairs_allowlist(args.sample_pairs_json)

    all_df = pd.concat([load_reviews_parquet(2024), load_reviews_parquet(2025)], ignore_index=True)
    all_df = prepare_dataset(all_df)

    # Prefer stable forum IDs for matching sampled pairs; fallback to legacy paper_id.
    pair_key_forum = list(zip(all_df["forum"].astype(str), all_df["reviewer_id"].astype(str)))
    pair_key_legacy = list(zip(all_df["paper_id"].astype(str), all_df["reviewer_id"].astype(str)))
    pair_series_forum = pd.Series(pair_key_forum, index=all_df.index)
    pair_series_legacy = pd.Series(pair_key_legacy, index=all_df.index)
    sample_mask = pair_series_forum.isin(pairs) | pair_series_legacy.isin(pairs)
    sample_df = all_df[sample_mask].copy()

    sample_summary = summarize(sample_df)
    full_summary = summarize(all_df)
    tier_comparison = compare_sample_vs_full(sample_df, all_df)
    interaction_validation = validate_sample_interactions(pairs)

    payload = {
        "sample_pairs_json": str(args.sample_pairs_json),
        "sample_summary": sample_summary,
        "full_summary": full_summary,
        "tier_distribution_comparison": tier_comparison,
        "interaction_validation": interaction_validation,
    }
    write_json(payload, args.output_json)
    write_file(
        render_markdown(
            sample_summary,
            full_summary,
            tier_comparison,
            args.sample_pairs_json,
            interaction_validation,
        ),
        args.output_md,
    )

    print(f"Sample reviews matched: {sample_summary['n_reviews']}")
    print(f"Full eligible reviews: {full_summary['n_reviews']}")
    print(
        "Interaction coverage in sampled pairs: "
        f"{interaction_validation['pairs_with_interactions']}/{interaction_validation['total_pairs']} "
        f"({interaction_validation['interaction_coverage_pct']:.2f}%)"
    )
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved markdown: {args.output_md}")


if __name__ == "__main__":
    main()

