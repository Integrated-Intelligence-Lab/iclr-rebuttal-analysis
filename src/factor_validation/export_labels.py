"""
Merge all factor_validation batch outputs into per-row F01..F44 labels.

Inputs (deduped on (forum, reviewer_id)):
  - test1_same1k     (ICLR 2025 stratified 1K, validated)
  - test2_fresh1k    (ICLR 2024 stratified 1K, validated; mislabeled in earlier plan)
  - full_b1          (ICLR 2024 batch, ~5K)
  - full_b2..b5      (intended ICLR 2025; submission shipped b1's items by mistake,
                      so these contribute almost nothing new — see analysis logs)

Year is assigned by joining (forum, reviewer_id) against the year-stamped parquet
files (no overlap by construction), not from the source tag.

For each parsed result we emit:
  year, paper_key, forum, reviewer_id, reviewer_suffix,
  ext_rating_before, ext_rating_after, ext_direction,
  tier, F01..F44, n_factors_present, source_batch

Dedup rule: when (forum, reviewer_id) appears in multiple sources, keep the row
from test1/test2 (already validated), drop duplicates from full_b*.

Outputs:
  data/results/32_factor_taxonomy/interaction_factor_labels_2024.csv
  data/results/32_factor_taxonomy/interaction_factor_labels_2025.csv

Usage
-----
    python -m src.factor_validation.export_labels
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pandas as pd

from core.io import load_json
from core.paths import BATCH_OUTPUT_DIR, DATA_DIR, RESULTS_DIR
from factor_validation.prompt import VALID_FACTOR_IDS

OUT_DIR = RESULTS_DIR / "32_factor_taxonomy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_DIR = DATA_DIR / "processed" / "parquet"

# Source priority: lower index wins on duplicates.
SOURCES = [
    ("test1_same1k", 2025),
    ("test2_fresh1k", 2025),
    ("full_b1", 2024),
    ("full_b2", 2025),
    ("full_b3", 2025),
    ("full_b4", 2025),
    ("full_b5", 2025),
]

YEAR_RE = re.compile(r"ICLR\.cc/(\d{4})/Conference")
FACTOR_COLS = sorted(VALID_FACTOR_IDS)  # F01..F44


def _year_from_reviewer_id(reviewer_id: str) -> int | None:
    m = YEAR_RE.search(reviewer_id or "")
    return int(m.group(1)) if m else None


def _load_parquet_index() -> dict[tuple[str, str], dict]:
    """Build (forum, reviewer_id) -> metadata dict for 2024 and 2025."""
    idx: dict[tuple[str, str], dict] = {}
    for year in (2024, 2025):
        df = pd.read_parquet(PARQUET_DIR / f"iclr_{year}_reviews.parquet")
        cols = ["paper_key", "forum", "reviewer_id", "reviewer_suffix",
                "ext_rating_before", "ext_rating_after", "ext_direction"]
        df = df[cols].copy()
        df["year"] = year
        for row in df.itertuples(index=False):
            key = (row.forum, row.reviewer_id)
            if key in idx:
                continue
            idx[key] = {
                "year": year,
                "paper_key": row.paper_key,
                "reviewer_suffix": row.reviewer_suffix,
                "ext_rating_before": row.ext_rating_before,
                "ext_rating_after": row.ext_rating_after,
                "ext_direction": row.ext_direction,
            }
    print(f"Indexed {len(idx)} (forum, reviewer_id) pairs from parquet")
    return idx


def _load_items_index(tag: str) -> dict[str, dict]:
    """Build forum::reviewer_id -> {paper_id, tier, verdict_group} from items.json."""
    items_path = BATCH_OUTPUT_DIR / f"factor_validation_{tag}_items.json"
    if not items_path.exists():
        return {}
    items = load_json(items_path)
    idx: dict[str, dict] = {}
    for it in items:
        forum = it.get("forum")
        rid = it.get("reviewer_id")
        if not (forum and rid):
            continue
        idx[f"{forum}::{rid}"] = {
            "tier": it.get("tier"),
            "verdict_group": it.get("verdict_group"),
        }
    return idx


def _extract_factor_ids(parsed) -> tuple[set[str], int]:
    """Return (valid_ids, n_hallucinated) from a parsed.factors_present blob."""
    if isinstance(parsed, dict):
        factors = parsed.get("factors_present") or []
    elif isinstance(parsed, list):
        factors = parsed
    else:
        return set(), 0
    ids = [f.get("factor_id") for f in factors if isinstance(f, dict)]
    valid = {fid for fid in ids if fid in VALID_FACTOR_IDS}
    n_hallu = sum(1 for fid in ids if fid and fid not in VALID_FACTOR_IDS)
    return valid, n_hallu


def _process_source(tag: str, parquet_idx: dict[tuple[str, str], dict]) -> list[dict]:
    """Return list of row dicts ready to write to the year-CSVs."""
    full_path = BATCH_OUTPUT_DIR / f"factor_validation_{tag}_full.json"
    if not full_path.exists():
        print(f"  [skip] {tag}: no _full.json")
        return []
    results = load_json(full_path)
    items_idx = _load_items_index(tag)

    rows: list[dict] = []
    n_skipped_failed = 0
    n_skipped_no_meta = 0
    n_hallu_total = 0
    for r in results:
        if r.get("error") or r.get("parsed") is None:
            n_skipped_failed += 1
            continue
        cid = r.get("custom_id", "")
        if "::" not in cid:
            n_skipped_failed += 1
            continue
        forum, reviewer_id = cid.split("::", 1)
        valid_ids, n_hallu = _extract_factor_ids(r["parsed"])
        n_hallu_total += n_hallu

        meta = parquet_idx.get((forum, reviewer_id))
        if meta is None:
            n_skipped_no_meta += 1
            continue
        item_meta = items_idx.get(cid, {})

        row = {
            "year": meta["year"],
            "paper_key": meta["paper_key"],
            "forum": forum,
            "reviewer_id": reviewer_id,
            "reviewer_suffix": meta["reviewer_suffix"],
            "ext_rating_before": meta["ext_rating_before"],
            "ext_rating_after": meta["ext_rating_after"],
            "ext_direction": meta["ext_direction"],
            "tier": item_meta.get("tier"),
            "n_factors_present": len(valid_ids),
            "source_batch": tag,
        }
        for fid in FACTOR_COLS:
            row[fid] = 1 if fid in valid_ids else 0
        rows.append(row)

    print(f"  {tag}: kept {len(rows)} rows  "
          f"(failed={n_skipped_failed}, no_parquet_meta={n_skipped_no_meta}, "
          f"hallucinated_factor_ids={n_hallu_total})")
    return rows


def main() -> None:
    parquet_idx = _load_parquet_index()

    all_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    print("Processing sources in priority order:")
    for tag, _expected_year in SOURCES:
        rows = _process_source(tag, parquet_idx)
        n_kept = 0
        n_dup = 0
        for r in rows:
            key = (r["forum"], r["reviewer_id"])
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            all_rows.append(r)
            n_kept += 1
        if n_dup:
            print(f"    -> {n_kept} new, {n_dup} duplicates dropped")

    print(f"\nTotal unique rows: {len(all_rows)}")

    df = pd.DataFrame(all_rows)
    column_order = (
        ["year", "paper_key", "forum", "reviewer_id", "reviewer_suffix",
         "ext_rating_before", "ext_rating_after", "ext_direction",
         "tier", "n_factors_present", "source_batch"]
        + FACTOR_COLS
    )
    df = df[column_order]

    for year in (2024, 2025):
        sub = df[df["year"] == year].copy()
        out_path = OUT_DIR / f"interaction_factor_labels_{year}.csv"
        sub.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
        prevalence = sub[FACTOR_COLS].mean().sort_values(ascending=False)
        print(f"\n[{year}] wrote {len(sub)} rows -> {out_path.name}")
        print(f"        median n_factors_present: {sub['n_factors_present'].median():.1f}")
        print(f"        ext_direction: "
              f"{(sub['ext_direction'] == 'increase').sum()} increase / "
              f"{(sub['ext_direction'] == 'same').sum()} same / "
              f"{(sub['ext_direction'] == 'decrease').sum()} decrease")
        print(f"        top 5 factors: " + ", ".join(
            f"{fid}={p * 100:.0f}%" for fid, p in prevalence.head(5).items()
        ))
        print(f"        bottom 5 factors: " + ", ".join(
            f"{fid}={p * 100:.1f}%" for fid, p in prevalence.tail(5).items()
        ))


if __name__ == "__main__":
    main()
