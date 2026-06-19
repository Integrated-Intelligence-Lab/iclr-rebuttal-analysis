"""
Cancel running factor_validation batch jobs on Google's side.

Walks the local tracker, picks entries whose state is non-terminal (anything
except SUCCEEDED / FAILED / CANCELLED / EXPIRED), calls
``client.batches.cancel(name=...)`` for each, and updates the tracker so
``status`` reflects the new state.

Notes
-----
- Cancellation typically still incurs cost for tokens already processed up to
  the moment of cancellation. It saves the work that hasn't started yet.
- This script is dangerous: it cancels BATCHES THAT ARE LIVE ON GOOGLE'S SIDE.
  Always pass ``--tag`` to limit scope. Use ``--dry-run`` first to confirm
  exactly which jobs would be cancelled.

Usage
-----
    # Show what WOULD be cancelled (no API calls):
    python -m src.factor_validation.cancel --tag full_b3 --dry-run

    # Cancel everything for one tag:
    python -m src.factor_validation.cancel --tag full_b3

    # Cancel a specific chunk:
    python -m src.factor_validation.cancel --tag full_b3 --chunk chunk5of11

    # Cancel ALL running factor_validation batches across all tags
    # (must pass --all to acknowledge the blast radius):
    python -m src.factor_validation.cancel --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv
load_dotenv()

from core.io import load_json
from core.paths import BATCHES_DIR

TRACKER_PATH = BATCHES_DIR / "batch_tracker.json"
JOB_TYPE = "factor_validation"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "completed",
}


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set in .env. Exiting.")
        sys.exit(1)
    print("Importing google.genai ...")
    from google import genai
    print("Creating Gemini client ...")
    return genai.Client(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, default=None,
                        help="Only cancel batches with this tag (e.g. full_b3).")
    parser.add_argument("--chunk", type=str, default=None,
                        help="Only cancel this specific chunk label "
                             "(e.g. chunk5of11). Requires --tag.")
    parser.add_argument("--all", action="store_true",
                        help="Cancel ALL non-terminal factor_validation batches "
                             "across every tag. Required if --tag is omitted.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be cancelled without calling the API.")
    args = parser.parse_args()

    if args.chunk and not args.tag:
        print("--chunk requires --tag")
        sys.exit(1)
    if not args.tag and not args.all:
        print("Pass --tag or --all to scope the cancellation.")
        sys.exit(1)

    if not TRACKER_PATH.exists():
        print(f"No tracker at {TRACKER_PATH}. Nothing to cancel.")
        sys.exit(0)
    tracker = load_json(TRACKER_PATH)

    candidates: list[dict] = []
    for e in tracker:
        if e.get("job_type") != JOB_TYPE:
            continue
        if args.tag and e.get("tag") != args.tag:
            continue
        if args.chunk and e.get("chunk") != args.chunk:
            continue
        # Skip already-terminal entries (no-op cancels just spam the API).
        if e.get("status") in TERMINAL_STATES:
            continue
        candidates.append(e)

    if not candidates:
        print("No non-terminal batches match the filter. Nothing to cancel.")
        sys.exit(0)

    print(f"\nWould cancel {len(candidates)} batch job(s):")
    print(f"{'Tag':<14} {'Chunk':<14} {'Current state':<24} Job name")
    print("-" * 90)
    for e in candidates:
        print(f"{e.get('tag','?'):<14} {e.get('chunk','?'):<14} "
              f"{e.get('status','?'):<24} {e['job_name']}")

    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        return

    print()
    confirm = input(f"Cancel {len(candidates)} batches now? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    client = _get_client()

    n_ok = n_fail = 0
    new_states: dict[str, str] = {}
    for e in candidates:
        jn = e["job_name"]
        try:
            client.batches.cancel(name=jn)
            print(f"  CANCEL  {e.get('tag','?'):<14} {e.get('chunk','?'):<14} {jn}")
            new_states[jn] = "JOB_STATE_CANCELLED"
            n_ok += 1
        except Exception as exc:
            print(f"  FAIL    {e.get('tag','?'):<14} {e.get('chunk','?'):<14} {jn}: {exc}")
            n_fail += 1

    # Update tracker through the same lock the rest of the pipeline uses.
    if new_states:
        from factor_validation.classify import _tracker_transaction
        with _tracker_transaction() as tracker_w:
            for entry in tracker_w:
                jn = entry.get("job_name")
                if jn in new_states:
                    entry["status"] = new_states[jn]

    print(f"\nDone: {n_ok} cancelled, {n_fail} failed.")


if __name__ == "__main__":
    main()
