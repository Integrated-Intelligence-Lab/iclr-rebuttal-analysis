"""
Build the full-corpus partition for 44-factor classification at scale.

Companion to ``build_sample.py``, which builds a stratified 1K sample. This
script enumerates **every** reviewer-paper interaction with an author reply,
across ICLR 2024 and 2025, excludes the ~2K already classified in the
test1_same1k and test2_fresh1k batches, and emits five JSON files in the
same schema ``classify.py::flatten_items`` already expects (groups -> items).

Partition strategy
------------------
- ``--batch-id b1`` -> all ICLR 2024 (~5.9K items, year-pure validation gate).
- ``--batch-id b2..b5`` -> ICLR 2025, partitioned deterministically by
  ``hash(forum) mod 4`` so reviews of one paper never split across batches
  and dedup is trivial.
- Items above ``--token-budget`` (default 900,000) are SKIPPED, not truncated,
  and logged to ``data/results/32_factor_taxonomy/skipped_interactions.csv``.

Cost gate
---------
Each batch is ~3.6K interactions, ~22M input + 2.9M output tokens, which
maps to roughly $7-14 per batch on Gemini Flash batch pricing — small
enough to abort cheaply if anything looks wrong after batch 1.

Usage
-----
    # Build batch 1 (ICLR 2024) for submission:
    python -m src.factor_validation.build_full_corpus --batch-id b1

    # Dry-run to see counts and percentiles without writing the JSON:
    python -m src.factor_validation.build_full_corpus --batch-id b1 --dry-run

    # Build all five at once (still does not submit anything):
    python -m src.factor_validation.build_full_corpus --batch-id all

Outputs
-------
    data/processed/categorization_data/factor_full_corpus_{b1..b5}.json
    data/results/32_factor_taxonomy/skipped_interactions.csv  (appended)
    data/results/32_factor_taxonomy/build_full_corpus_summary.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.io import load_json, write_json
from core.paths import BATCH_OUTPUT_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

# Reuse the sampling pipeline's primitives so the input_text format is identical
# to test1_same1k / test2_fresh1k. That keeps factor outputs comparable across
# the existing 2K sample and the new 18K corpus.
from factor_validation.build_sample import (
    NORMALIZED_FILES,
    VERDICTS_FILES,
    _build_paper_index,
    enrich_verdicts,
    build_input_text,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CATEGORIZATION_DIR = PROCESSED_DATA_DIR / "categorization_data"
TAXONOMY_RESULTS_DIR = RESULTS_DIR / "32_factor_taxonomy"

# Existing 2K (test1 + test2) lives next to the batch outputs as *_items.json.
EXISTING_ITEMS_FILES = [
    BATCH_OUTPUT_DIR / "factor_validation_test1_same1k_items.json",
    BATCH_OUTPUT_DIR / "factor_validation_test2_fresh1k_items.json",
]

# Fixed batch IDs.
BATCH_IDS = ["b1", "b2", "b3", "b4", "b5"]

# Estimated tokens-per-character. Conservative: we measured ~4 chars/token
# in the existing batch, and want a safety margin to avoid surprise truncations.
CHARS_PER_TOKEN = 4

# Hard skip budget. Gemini 3 Flash supports 1M context; 900K leaves 100K for
# the prompt scaffolding (system instructions + 44-factor codebook + the user
# wrapper text). Items above this are dropped and recorded.
DEFAULT_TOKEN_BUDGET = 900_000


# ---------------------------------------------------------------------------
# Excluded-pairs loader
# ---------------------------------------------------------------------------

def load_excluded_pairs() -> set[tuple[str, str]]:
    """Return (forum, reviewer_id) pairs already classified in test1 + test2."""
    excluded: set[tuple[str, str]] = set()
    for path in EXISTING_ITEMS_FILES:
        if not path.exists():
            print(f"  WARN: existing-items file missing: {path.name}")
            continue
        items = load_json(path)
        for it in items:
            forum = str(it.get("forum") or "")
            rid = str(it.get("reviewer_id") or "")
            if forum and rid:
                excluded.add((forum, rid))
    return excluded


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_2025(forum: str, n_groups: int = 4) -> int:
    """Deterministic 0..n_groups-1 hash on forum so batches stay paper-stable."""
    h = hashlib.md5(forum.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h, 16) % n_groups


def assign_batch(year: int, forum: str) -> str:
    if year == 2024:
        return "b1"
    if year == 2025:
        idx = partition_2025(forum, n_groups=4)
        return BATCH_IDS[1 + idx]  # b2..b5
    raise ValueError(f"Unsupported year for full corpus: {year}")


# ---------------------------------------------------------------------------
# Token estimation + skip ledger
# ---------------------------------------------------------------------------

def estimate_tokens(input_text: str) -> int:
    """Rough chars/CHARS_PER_TOKEN estimate for the user-message thread text only.

    Underestimates total prompt tokens because it ignores the system block and
    the 44-factor list. The 900K budget already accounts for that headroom.
    """
    return len(input_text) // CHARS_PER_TOKEN


def append_skipped_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year", "paper_id", "forum", "reviewer_id",
                "n_chars", "n_tokens_est", "reason",
            ],
        )
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Per-year candidate enumeration
# ---------------------------------------------------------------------------

def enumerate_year(
    year: int,
    excluded_pairs: set[tuple[str, str]],
    token_budget: int,
) -> tuple[list[dict], list[dict], dict]:
    """Return (kept_items, skipped_rows, counters) for a single year.

    Each kept item is the same dict shape ``classify.py::flatten_items`` returns
    when reading the existing 1K sample: paper_id, forum, reviewer_id,
    input_text, tier, verdict_group, group_key.

    Reuses ``enrich_verdicts`` from build_sample.py so the candidate definition
    (verdict in {increase, same, decrease} AND author reply present) matches
    the existing 2K exactly.
    """
    counters = defaultdict(int)
    counters["year"] = year

    norm_path = NORMALIZED_FILES[year]
    verdicts_path = VERDICTS_FILES[year]
    if not norm_path.exists():
        raise SystemExit(f"Missing normalized dataset: {norm_path}")
    if not verdicts_path.exists():
        raise SystemExit(f"Missing verdicts file: {verdicts_path}")

    print(f"  [{year}] loading normalized JSON ...")
    normalized = load_json(norm_path)
    papers = _build_paper_index(normalized)
    print(f"  [{year}] {len(papers)} papers")

    print(f"  [{year}] loading verdicts ...")
    verdicts = load_json(verdicts_path)
    print(f"  [{year}] {len(verdicts)} verdict rows")

    print(f"  [{year}] enriching candidates ...")
    candidates, drop_counters = enrich_verdicts(verdicts, papers, excluded_pairs)
    counters.update({f"enrich_{k}": v for k, v in drop_counters.items()})
    print(f"  [{year}] {len(candidates)} candidates after enrichment "
          f"(excluded {drop_counters['excluded']} from existing 2K)")

    kept: list[dict] = []
    skipped_rows: list[dict] = []
    for c in candidates:
        input_text = build_input_text(c["_paper"], c["_review"])
        n_chars = len(input_text)
        n_tokens_est = estimate_tokens(input_text)

        if n_tokens_est > token_budget:
            skipped_rows.append({
                "year": year,
                "paper_id": c["paper_id"],
                "forum": c["forum"],
                "reviewer_id": c["reviewer_id"],
                "n_chars": n_chars,
                "n_tokens_est": n_tokens_est,
                "reason": f"tokens_est_over_budget_{token_budget}",
            })
            counters["skipped_token_budget"] += 1
            continue

        if n_chars == 0:
            # No content -- defensive; enrich_verdicts already filters empty rebuttals.
            skipped_rows.append({
                "year": year,
                "paper_id": c["paper_id"],
                "forum": c["forum"],
                "reviewer_id": c["reviewer_id"],
                "n_chars": 0,
                "n_tokens_est": 0,
                "reason": "empty_input_text",
            })
            counters["skipped_empty"] += 1
            continue

        kept.append({
            "year": year,
            "paper_id": c["paper_id"],
            "forum": c["forum"],
            "reviewer_id": c["reviewer_id"],
            "input_text": input_text,
            "tier": c["tier"],
            "verdict_group": c["verdict_group"],
            "group_key": f"{c['tier']}_{c['verdict_group']}",
            "_n_chars": n_chars,
            "_n_tokens_est": n_tokens_est,
        })
        counters["kept"] += 1

    return kept, skipped_rows, dict(counters)


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def assemble_batch_payload(batch_id: str, items: list[dict]) -> dict:
    """Group items by (tier, verdict_group) into the schema flatten_items expects."""
    groups: dict[str, dict] = {}
    for it in items:
        gk = it["group_key"]
        if gk not in groups:
            tier, verdict_group = gk.split("_", 1)
            groups[gk] = {"tier": tier, "verdict_group": verdict_group, "items": []}
        groups[gk]["items"].append({
            "paper_id": it["paper_id"],
            "forum": it["forum"],
            "reviewer_id": it["reviewer_id"],
            "input_text": it["input_text"],
        })
    for gk, g in groups.items():
        g["n"] = len(g["items"])
    return {
        "description": (
            f"Full-corpus factor classification, batch {batch_id}. "
            f"Schema mirrors test1_same1k / test2_fresh1k so classify.py reads it without changes. "
            f"Items pre-classified in the existing 2K are excluded; over-budget items are skipped "
            f"(see data/results/32_factor_taxonomy/skipped_interactions.csv)."
        ),
        "batch_id": batch_id,
        "n_total": sum(g["n"] for g in groups.values()),
        "groups": groups,
    }


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": s[0],
        "p50": s[n // 2],
        "p90": s[int(0.90 * n)] if n > 1 else s[0],
        "p99": s[int(0.99 * n)] if n > 1 else s[0],
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-id", type=str, required=True,
        choices=BATCH_IDS + ["all"],
        help="Which batch to write. 'all' writes b1..b5.",
    )
    parser.add_argument(
        "--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET,
        help="Maximum estimated input tokens per item. Items above are skipped, "
             "not truncated. Default: 900,000 (Gemini 1M context minus headroom).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute partitions and percentiles, but do not write the JSON.",
    )
    parser.add_argument(
        "--skipped-csv", type=Path, default=TAXONOMY_RESULTS_DIR / "skipped_interactions.csv",
    )
    parser.add_argument(
        "--summary-json", type=Path, default=TAXONOMY_RESULTS_DIR / "build_full_corpus_summary.json",
    )
    args = parser.parse_args()

    target_batches = BATCH_IDS if args.batch_id == "all" else [args.batch_id]
    print(f"Target batch(es): {target_batches}")

    # We need both years if any 2025 batch is requested OR if 'all'. Year 2024
    # is only needed for batch b1.
    needed_years: list[int] = []
    if "b1" in target_batches:
        needed_years.append(2024)
    if any(b in target_batches for b in BATCH_IDS[1:]):
        needed_years.append(2025)
    print(f"Years to load: {needed_years}")

    print("\nLoading existing-2K exclusion set ...")
    excluded = load_excluded_pairs()
    print(f"  {len(excluded)} (forum, reviewer_id) pairs to exclude")

    print("\nEnumerating candidates ...")
    all_kept: list[dict] = []
    all_skipped: list[dict] = []
    counters_per_year: dict[int, dict] = {}
    for year in needed_years:
        kept, skipped, c = enumerate_year(year, excluded, args.token_budget)
        all_kept.extend(kept)
        all_skipped.extend(skipped)
        counters_per_year[year] = c

    # Length percentiles BEFORE batch assignment.
    char_lens = [it["_n_chars"] for it in all_kept]
    tok_lens = [it["_n_tokens_est"] for it in all_kept]
    print("\nKept-item length percentiles:")
    print(f"  chars  : {percentiles(char_lens)}")
    print(f"  tokens : {percentiles(tok_lens)}")
    print(f"  total kept: {len(all_kept)}")
    print(f"  total skipped: {len(all_skipped)}")

    # Assign each kept item to a batch.
    by_batch: dict[str, list[dict]] = defaultdict(list)
    for it in all_kept:
        bid = assign_batch(it["year"], it["forum"])
        by_batch[bid].append(it)

    print("\nPer-batch counts:")
    for bid in BATCH_IDS:
        n = len(by_batch.get(bid, []))
        if n:
            est_input_tokens = sum(it["_n_tokens_est"] for it in by_batch[bid])
            print(f"  {bid}: n={n:>5}  est_input_tokens={est_input_tokens:>11,}")

    # Write skipped log (append).
    if all_skipped:
        append_skipped_csv(all_skipped, args.skipped_csv)
        print(f"\nLogged {len(all_skipped)} skipped items -> {args.skipped_csv}")

    # Write batch JSON files.
    written: list[Path] = []
    if not args.dry_run:
        CATEGORIZATION_DIR.mkdir(parents=True, exist_ok=True)
        for bid in target_batches:
            items = by_batch.get(bid, [])
            if not items:
                print(f"\n  {bid}: no items (skipping write).")
                continue
            payload = assemble_batch_payload(bid, items)
            out_path = CATEGORIZATION_DIR / f"factor_full_corpus_{bid}.json"
            write_json(payload, out_path)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"\n  Wrote {payload['n_total']} items -> {out_path.name} ({size_mb:.1f} MB)")
            written.append(out_path)

    # Summary.
    summary = {
        "target_batches": target_batches,
        "token_budget": args.token_budget,
        "n_excluded_existing_2k": len(excluded),
        "n_kept_total": len(all_kept),
        "n_skipped_total": len(all_skipped),
        "char_length_percentiles": percentiles(char_lens),
        "token_length_percentiles": percentiles(tok_lens),
        "per_batch_counts": {bid: len(by_batch.get(bid, [])) for bid in BATCH_IDS},
        "per_batch_input_tokens_est": {
            bid: sum(it["_n_tokens_est"] for it in by_batch.get(bid, []))
            for bid in BATCH_IDS
        },
        "counters_per_year": counters_per_year,
        "files_written": [str(p) for p in written],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {args.summary_json}")

    if args.dry_run:
        print("\n--dry-run: not writing batch JSON.")


if __name__ == "__main__":
    main()
