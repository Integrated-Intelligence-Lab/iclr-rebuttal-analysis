"""
Repair batch_tracker.json by reconciling with Google's batch list.

When two factor_validation processes mutate the tracker concurrently
(e.g. ``submit --no-wait`` running while another terminal calls ``status``
or ``download-all``), one can clobber the other's appends. The result is
that batch jobs continue running on Google's side but the local tracker
loses the entries — so ``status``, ``download-all``, ``retry-failed-sync``
can no longer find them.

This script lists all batches via the SDK, filters to display names that
look like ``factor_validation_<tag>_<chunk>``, and inserts any missing
tracker entries. It also restores ``output_file`` for chunks where the
on-disk JSON exists.

Idempotent: rerunning is safe; existing entries are not duplicated.

Usage
-----
    python -m src.factor_validation.repair_tracker
    python -m src.factor_validation.repair_tracker --tag full_b2
    python -m src.factor_validation.repair_tracker --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv
load_dotenv()

from core.io import load_json, write_json
from core.paths import BATCH_OUTPUT_DIR, BATCHES_DIR

TRACKER_PATH = BATCHES_DIR / "batch_tracker.json"
JOB_TYPE = "factor_validation"
DISPLAY_NAME_RE = re.compile(r"^factor_validation_(?P<tag>.+?)_(?P<chunk>chunk\d+of\d+)$")


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set in .env. Exiting.")
        sys.exit(1)
    print("Importing google.genai ...")
    from google import genai
    print("Creating Gemini client ...")
    return genai.Client(api_key=api_key)


def list_factor_validation_batches(client) -> list[dict]:
    """Return parsed records for all batches whose display_name matches our scheme."""
    records: list[dict] = []
    print("Listing batches from Gemini ...")
    for batch in client.batches.list():
        display = getattr(batch, "display_name", None) or ""
        m = DISPLAY_NAME_RE.match(display)
        if not m:
            continue
        records.append({
            "tag": m.group("tag"),
            "chunk": m.group("chunk"),
            "job_name": batch.name,
            "state": batch.state.name,
            "display_name": display,
            "create_time": getattr(batch, "create_time", None),
        })
    print(f"  Found {len(records)} batches matching factor_validation_*")
    return records


def _create_time_iso(record: dict) -> str:
    ct = record.get("create_time")
    if ct is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(ct, str):
        return ct
    if hasattr(ct, "isoformat"):
        try:
            return ct.isoformat()
        except Exception:
            pass
    return str(ct)


def _output_file_if_present(tag: str, chunk: str) -> str | None:
    candidate = BATCH_OUTPUT_DIR / f"factor_validation_{tag}_{chunk}.json"
    return candidate.name if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, default=None,
                        help="Restrict repair to one tag (e.g. full_b2).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be added/updated without writing.")
    args = parser.parse_args()

    if not TRACKER_PATH.exists():
        print(f"Tracker not found at {TRACKER_PATH}. Creating empty.")
        tracker: list[dict] = []
    else:
        tracker = load_json(TRACKER_PATH)
        print(f"Loaded {len(tracker)} entries from {TRACKER_PATH.name}")

    client = _get_client()
    records = list_factor_validation_batches(client)
    if args.tag:
        records = [r for r in records if r["tag"] == args.tag]
        print(f"  Filtered to tag={args.tag}: {len(records)} record(s)")

    # Index existing tracker entries by (tag, chunk) for quick lookup.
    by_key: dict[tuple[str, str], dict] = {}
    for e in tracker:
        if e.get("job_type") != JOB_TYPE:
            continue
        key = (e.get("tag", ""), e.get("chunk", ""))
        by_key[key] = e

    n_added = 0
    n_updated_state = 0
    n_updated_output = 0
    for r in records:
        key = (r["tag"], r["chunk"])
        existing = by_key.get(key)
        out_file = _output_file_if_present(r["tag"], r["chunk"])

        if existing is None:
            # Reconstruct missing entry. We can't know n_items without re-reading
            # the JSONL; default to 0 which is fine for status/download.
            entry = {
                "job_type": JOB_TYPE,
                "tag": r["tag"],
                "job_name": r["job_name"],
                "model": os.getenv("GEMINI_MODEL", "gemini-3-flash"),
                "n_items": 0,
                "chunk": r["chunk"],
                "submitted_at": _create_time_iso(r),
                "status": r["state"],
                "input_file": f"factor_validation_{r['tag']}_{r['chunk']}.jsonl",
                "source_input": None,
                "output_file": out_file,
            }
            if not args.dry_run:
                tracker.append(entry)
                by_key[key] = entry
            n_added += 1
            print(f"  + ADD  {r['tag']:<10} {r['chunk']:<14} {r['state']:<24} {r['job_name']}")
        else:
            updated = False
            if existing.get("status") != r["state"]:
                existing["status"] = r["state"]
                n_updated_state += 1
                updated = True
            if out_file and not existing.get("output_file"):
                existing["output_file"] = out_file
                n_updated_output += 1
                updated = True
            if updated:
                print(f"  ~ UPD  {r['tag']:<10} {r['chunk']:<14} state={r['state']:<24}")

    print()
    print(f"Added entries     : {n_added}")
    print(f"Updated states    : {n_updated_state}")
    print(f"Updated outputs   : {n_updated_output}")

    if args.dry_run:
        print("\n--dry-run: tracker NOT written.")
        return

    write_json(tracker, TRACKER_PATH)
    print(f"\nWrote {len(tracker)} entries -> {TRACKER_PATH}")


if __name__ == "__main__":
    main()
