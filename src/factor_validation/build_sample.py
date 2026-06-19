"""
Build a fresh held-out sample for Test 2 of the factor-classification validation.

Produces data/results/score_change/samples/classify_input_raw_threads_test2.json
with the SAME SCHEMA as classify_input_raw_threads.json, so it is a drop-in
input for classify_factors_batch.py.

Sampling strategy
-----------------
- Source: ICLR 2024 (held out from taxonomy induction, which used ICLR 2025).
- Stratification: six groups = {low, mid, high} × {increase, same_decrease}.
- Counts: by default match the per-group sizes of the existing 1000 so the
  two runs are directly comparable. Override via --counts.
- Seed: default 51 (distinct from the project-wide seed 50 already used by
  the original sample), so runs are reproducible.
- Initial-score tier is derived from (final_rating, gemini_verdict) using
  the step-up/step-down heuristic already used by cause_trend_swapped.py.

Input
-----
- data/processed/normalized/ICLR_2024_normalized.json
- data/results/score_change/gemini_verdicts_v2_full.json  (ICLR 2024 verdicts)
- data/results/score_change/samples/classify_input_raw_threads.json
    (only used to read default per-group counts)

Output
------
- data/results/score_change/samples/classify_input_raw_threads_test2.json
- data/results/score_change/samples/classify_input_raw_threads_test2_meta.json

Usage
-----
    python src/score_change/build_factor_classification_sample.py
    python src/score_change/build_factor_classification_sample.py --seed 51 --year 2024
    python src/score_change/build_factor_classification_sample.py --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.io import load_json, write_json
from core.paths import PROCESSED_DATA_DIR, RESULTS_DIR

# --- Config ---
ICLR_SCALE = [1, 3, 5, 6, 8, 10]

SCORE_TIERS = {
    "low":  (1, 3),
    "mid":  (5, 6),
    "high": (8, 10),
}

# Map gemini_verdict -> canonical verdict_group used in the existing 1000.
VERDICT_GROUP = {
    "increase": "increase",
    "same": "same_decrease",
    "decrease": "same_decrease",
}

NORMALIZED_FILES = {
    2023: PROCESSED_DATA_DIR / "normalized" / "ICLR_2023_normalized.json",
    2024: PROCESSED_DATA_DIR / "normalized" / "ICLR_2024_normalized.json",
    2025: PROCESSED_DATA_DIR / "normalized" / "ICLR_2025_normalized.json",
    2026: PROCESSED_DATA_DIR / "normalized" / "ICLR_2026_normalized.json",
}

VERDICTS_FILES = {
    2023: RESULTS_DIR / "score_change" / "gemini_verdicts_v2_iclr2023_full.json",
    2024: RESULTS_DIR / "score_change" / "gemini_verdicts_v2_full.json",
    2025: RESULTS_DIR / "score_change" / "gemini_verdicts_v2_iclr2025_full.json",
}

REFERENCE_SAMPLE = (
    RESULTS_DIR / "score_change" / "samples" / "classify_input_raw_threads.json"
)
EXPERIMENT_DIR = RESULTS_DIR / "factor_validation"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "test2_fresh1k_input.json"


# ---------------------------------------------------------------------------
# Score / tier helpers
# ---------------------------------------------------------------------------

def _step_down(score: int) -> int:
    if score in ICLR_SCALE:
        idx = ICLR_SCALE.index(score)
        if idx > 0:
            return ICLR_SCALE[idx - 1]
    return score


def _step_up(score: int) -> int:
    if score in ICLR_SCALE:
        idx = ICLR_SCALE.index(score)
        if idx < len(ICLR_SCALE) - 1:
            return ICLR_SCALE[idx + 1]
    return score


def estimate_initial_score(final_score: int, verdict: str) -> int:
    if verdict == "increase":
        return _step_down(final_score)
    if verdict == "decrease":
        return _step_up(final_score)
    return final_score


def get_tier(score: int | None) -> str | None:
    if score is None:
        return None
    for tier_name, (lo, hi) in SCORE_TIERS.items():
        if lo <= score <= hi:
            return tier_name
    return None


# ---------------------------------------------------------------------------
# Building input_text in the same format as the existing 1000
# ---------------------------------------------------------------------------

def _val(v):
    """Read a normalized content field that may be a str or {'value': str}."""
    if isinstance(v, dict):
        return (v.get("value") or "").strip()
    if v is None:
        return ""
    return str(v).strip()


def _collect_thread_comments(paper: dict, reviewer_id: str, review_id: str | None) -> list[dict]:
    """Traverse paper.comments to collect the reviewer/author thread.

    Mirrors src/analyze/sample_iclr_reviews.py::extract_reviewer_author_interactions.
    """
    comments = paper.get("comments", []) or []
    selected: set[str] = set()
    frontier: set[str] = set()

    if review_id:
        frontier.add(review_id)

    for c in comments:
        if (c.get("participant_type") == "reviewer"
                and c.get("participant_id") == reviewer_id
                and c.get("id")):
            selected.add(c["id"])
            frontier.add(c["id"])

    expanded = True
    while expanded:
        expanded = False
        for c in comments:
            cid = c.get("id")
            if not cid or cid in selected:
                continue
            if c.get("reply_to") not in frontier:
                continue
            pt = c.get("participant_type")
            if pt == "author":
                selected.add(cid); frontier.add(cid); expanded = True
            elif pt == "reviewer" and c.get("participant_id") == reviewer_id:
                selected.add(cid); frontier.add(cid); expanded = True

    return [c for c in comments if c.get("id") in selected]


def build_input_text(paper: dict, review: dict) -> str:
    """Compose the interaction text in the same layout as the existing 1000.

    [summary]
    ...
    [strengths]
    ...
    [weaknesses]
    ...
    [questions]
    ...
    [Author - comment]
    ...
    [Reviewer - comment]
    ...
    """
    parts: list[str] = []

    summary = _val(review.get("summary"))
    strengths = _val(review.get("strengths")) or _val(review.get("strengths_and_weaknesses"))
    weaknesses = _val(review.get("weaknesses"))
    questions = _val(review.get("questions"))

    if summary:
        parts.append(f"[summary]\n{summary}")
    if strengths:
        parts.append(f"[strengths]\n{strengths}")
    if weaknesses:
        parts.append(f"[weaknesses]\n{weaknesses}")
    if questions:
        parts.append(f"[questions]\n{questions}")

    thread = _collect_thread_comments(
        paper=paper,
        reviewer_id=review.get("reviewer_id"),
        review_id=review.get("id"),
    )
    for c in thread:
        pt = c.get("participant_type")
        text = _val(c.get("text"))
        if not text:
            continue
        if pt == "author":
            label = "[Author - comment]"
        elif pt == "reviewer":
            label = "[Reviewer - comment]"
        else:
            label = f"[{pt or 'comment'}]"
        parts.append(f"{label}\n{text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Enrichment: verdicts -> per-item (tier, verdict_group, final_rating)
# ---------------------------------------------------------------------------

def _build_paper_index(normalized: dict) -> dict:
    """paper_id (str) -> paper dict."""
    return {str(pid): p for pid, p in (normalized.get("papers") or {}).items()}


def _find_review(paper: dict, reviewer_id: str) -> dict | None:
    for r in paper.get("reviews", []) or []:
        if str(r.get("reviewer_id") or "") == str(reviewer_id):
            return r
    return None


def _review_rating(review: dict) -> int | None:
    rating = review.get("rating")
    if isinstance(rating, dict):
        rating = rating.get("value")
    if rating is None:
        return None
    try:
        if isinstance(rating, str) and ":" in rating:
            rating = rating.split(":")[0].strip()
        return int(rating)
    except (ValueError, TypeError):
        return None


def _has_author_reply(paper: dict, reviewer_id: str, review_id: str | None) -> bool:
    """True iff the reviewer's thread contains at least one author reply.

    We require this to avoid sampling reviewer-paper pairs that technically have
    a 'verdict' (from a before/after score delta) but no scraped rebuttal text —
    which would give the classifier an empty rebuttal to judge.
    """
    thread = _collect_thread_comments(paper=paper, reviewer_id=reviewer_id, review_id=review_id)
    for c in thread:
        if c.get("participant_type") == "author" and (c.get("text") or "").strip():
            return True
    return False


def enrich_verdicts(verdicts: list[dict], papers: dict,
                    excluded_pairs: set[tuple[str, str]]) -> tuple[list[dict], dict[str, int]]:
    """Produce candidates: (paper_id, reviewer_id, tier, verdict_group, review, paper).

    Also returns a counters dict capturing why rows were dropped, so the user
    can see how many verdicts lacked an actual rebuttal thread.
    """
    counters = {
        "total": 0, "unknown_verdict": 0, "no_paper": 0, "no_review": 0,
        "no_rating": 0, "no_tier": 0, "excluded": 0, "no_author_reply": 0,
        "kept": 0,
    }
    candidates: list[dict] = []
    for v in verdicts:
        counters["total"] += 1
        verdict = v.get("gemini_verdict")
        if verdict not in VERDICT_GROUP:
            counters["unknown_verdict"] += 1
            continue
        pid = str(v.get("paper_id") or "")
        rid = str(v.get("reviewer_id") or "")
        if not pid or not rid:
            counters["no_paper"] += 1
            continue
        paper = papers.get(pid)
        if not paper:
            counters["no_paper"] += 1
            continue
        review = _find_review(paper, rid)
        if not review:
            counters["no_review"] += 1
            continue
        final = _review_rating(review)
        if final is None:
            counters["no_rating"] += 1
            continue
        initial = estimate_initial_score(final, verdict)
        tier = get_tier(initial)
        if tier is None:
            counters["no_tier"] += 1
            continue
        forum = str(paper.get("forum") or "")
        key = (forum, rid)
        if key in excluded_pairs:
            counters["excluded"] += 1
            continue
        if not _has_author_reply(paper, rid, review.get("id")):
            counters["no_author_reply"] += 1
            continue
        counters["kept"] += 1
        candidates.append({
            "paper_id": pid,
            "forum": forum,
            "reviewer_id": rid,
            "tier": tier,
            "verdict_group": VERDICT_GROUP[verdict],
            "final_rating": final,
            "initial_rating_estimate": initial,
            "gemini_verdict": verdict,
            "_review": review,
            "_paper": paper,
        })
    return candidates, counters


# ---------------------------------------------------------------------------
# Counts + stratified sampling
# ---------------------------------------------------------------------------

def read_reference_counts(path: Path) -> dict[str, int]:
    """Per-group item counts from the existing 1000, used as the Test 2 target."""
    ref = load_json(path)
    out: dict[str, int] = {}
    for gk, g in (ref.get("groups") or {}).items():
        out[gk] = len(g.get("items", []))
    return out


def stratified_sample(candidates: list[dict], counts: dict[str, int],
                      rng: random.Random) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        gk = f"{c['tier']}_{c['verdict_group']}"
        by_group[gk].append(c)

    picked: dict[str, list[dict]] = {}
    for gk, want in counts.items():
        pool = by_group.get(gk, [])
        if not pool:
            print(f"  WARN: no candidates for {gk} (wanted {want})")
            picked[gk] = []
            continue
        if len(pool) < want:
            print(f"  WARN: only {len(pool)} candidates for {gk} (wanted {want}); taking all")
            picked[gk] = pool[:]
        else:
            picked[gk] = rng.sample(pool, want)
    return picked


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_output(picked: dict[str, list[dict]], year: int, seed: int) -> dict:
    groups = {}
    total = 0
    for gk, items in picked.items():
        parts = gk.split("_", 1)
        tier = parts[0]
        verdict_group = parts[1] if len(parts) > 1 else "?"
        built_items = []
        for it in items:
            input_text = build_input_text(it["_paper"], it["_review"])
            built_items.append({
                "paper_id": it["paper_id"],
                "forum": it["forum"],
                "reviewer_id": it["reviewer_id"],
                "input_text": input_text,
            })
        groups[gk] = {
            "tier": tier,
            "verdict_group": verdict_group,
            "n": len(built_items),
            # system_prompt intentionally omitted — factor classification uses its own prompt.
            "items": built_items,
        }
        total += len(built_items)

    return {
        "description": (
            f"Test 2 held-out sample for factor-classification validation. "
            f"ICLR {year}, stratified by (tier, verdict_group), disjoint from the "
            f"ICLR 2025 taxonomy-induction sample. seed={seed}."
        ),
        "n_total": total,
        "year": year,
        "seed": seed,
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# Pair-exclusion loader (from reference sample)
# ---------------------------------------------------------------------------

def read_excluded_pairs(path: Path) -> set[tuple[str, str]]:
    """Return (forum, reviewer_id) pairs from the reference sample.

    Useful as a guard even when sampling from a different year — forums are
    year-scoped in practice, but the intersection check costs nothing.
    """
    data = load_json(path)
    pairs: set[tuple[str, str]] = set()
    for g in (data.get("groups") or {}).values():
        for item in g.get("items") or []:
            pairs.add((str(item.get("forum") or ""), str(item.get("reviewer_id") or "")))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_counts_arg(raw: str | None, default: dict[str, int]) -> dict[str, int]:
    if not raw:
        return default
    out: dict[str, int] = {}
    for part in raw.split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out or default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024, choices=sorted(NORMALIZED_FILES.keys()))
    parser.add_argument("--seed", type=int, default=51,
                        help="RNG seed (default 51; distinct from project seed 50).")
    parser.add_argument("--counts", type=str, default=None,
                        help="Override per-group counts, e.g. "
                             "'low_increase:114,mid_increase:271,...'. Defaults to the "
                             "counts observed in the existing 1000.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE_SAMPLE,
                        help="Existing 1000-sample used to read default counts "
                             "and to exclude overlapping (forum, reviewer) pairs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show group sizes and sampling plan; do not write output.")
    args = parser.parse_args()

    normalized_path = NORMALIZED_FILES[args.year]
    verdicts_path = VERDICTS_FILES.get(args.year)
    if not normalized_path.exists():
        print(f"Missing normalized dataset: {normalized_path}")
        sys.exit(1)
    if not verdicts_path or not verdicts_path.exists():
        print(f"Missing verdicts file for ICLR {args.year}: {verdicts_path}")
        sys.exit(1)

    print(f"Loading normalized ICLR {args.year} ...")
    normalized = load_json(normalized_path)
    papers = _build_paper_index(normalized)
    print(f"  {len(papers)} papers")

    print(f"Loading verdicts ...")
    verdicts = load_json(verdicts_path)
    print(f"  {len(verdicts)} verdict rows")

    excluded = set()
    default_counts: dict[str, int] = {
        "low_increase": 114, "low_same_decrease": 107,
        "mid_increase": 271, "mid_same_decrease": 353,
        "high_increase": 58, "high_same_decrease": 97,
    }
    if args.reference.exists():
        excluded = read_excluded_pairs(args.reference)
        ref_counts = read_reference_counts(args.reference)
        if ref_counts:
            default_counts = ref_counts
        print(f"  Excluding {len(excluded)} (forum, reviewer_id) pairs from reference sample")
    else:
        print(f"  (reference sample not found at {args.reference} — using hardcoded default counts)")

    counts = parse_counts_arg(args.counts, default_counts)
    print(f"  Target counts: {counts}  (total {sum(counts.values())})")

    print("Enriching candidates ...")
    candidates, counters = enrich_verdicts(verdicts, papers, excluded)
    print(f"  drop-reason counts:")
    for k in ("total", "unknown_verdict", "no_paper", "no_review", "no_rating",
              "no_tier", "excluded", "no_author_reply", "kept"):
        print(f"    {k:<18} {counters[k]}")
    print(f"  {len(candidates)} eligible candidates (verdicts with non-empty rebuttal thread)")

    per_group_available: dict[str, int] = defaultdict(int)
    for c in candidates:
        per_group_available[f"{c['tier']}_{c['verdict_group']}"] += 1
    for gk in counts:
        avail = per_group_available.get(gk, 0)
        want = counts[gk]
        flag = "" if avail >= want else "  INSUFFICIENT"
        print(f"    {gk}: want {want}, available {avail}{flag}")

    if args.dry_run:
        print("--dry-run: stopping before sampling.")
        return

    rng = random.Random(args.seed)
    picked = stratified_sample(candidates, counts, rng)

    out = assemble_output(picked, year=args.year, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, args.output)
    print(f"Wrote {out['n_total']} items -> {args.output}")

    meta_path = args.output.with_name(args.output.stem + "_meta.json")
    meta = {
        "source_year": args.year,
        "seed": args.seed,
        "counts_requested": counts,
        "counts_actual": {gk: g["n"] for gk, g in out["groups"].items()},
        "excluded_pairs_from_reference": len(excluded),
        "reference_sample": str(args.reference),
        "normalized_path": str(normalized_path),
        "verdicts_path": str(verdicts_path),
    }
    write_json(meta, meta_path)
    print(f"Wrote meta -> {meta_path}")


if __name__ == "__main__":
    main()
