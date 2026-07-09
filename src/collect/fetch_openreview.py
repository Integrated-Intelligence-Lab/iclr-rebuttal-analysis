"""
Unified OpenReview data tool: download submissions, fetch interactions,
print statistics, and convert to Parquet.

Merges logic from explore_openreview.py (download, stats, parquet) and
fetch_iclr2025_interactions.py (API auto-detection, batched fetch).

Usage:
  python src/collect/fetch_openreview.py download --conference ICLR.cc --year 2024
  python src/collect/fetch_openreview.py fetch    --conference ICLR.cc --year 2025 --limit 50
  python src/collect/fetch_openreview.py stats    --conference ICLR.cc --year 2024
  python src/collect/fetch_openreview.py parquet  --conference ICML.cc --year 2025
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure src/ is on sys.path so core.* imports resolve
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.io import load_json, write_json
from core.paths import RAW_DATA_DIR
from core.openreview_utils import safe_get_all_notes, get_text_fields_from_content

SAMPLE_SIZE = 4
DEFAULT_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_path(conference: str, year: int) -> Path:
    """Return the canonical raw JSON path for a conference/year."""
    return RAW_DATA_DIR / f"{conference}_{year}.json"


def _make_client(api_version: str):
    """Create an OpenReview client for the given API version ('v1' or 'v2').

    Reads OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD from environment/.env
    for authenticated access (required since the Dec 2025 security lockdown).
    Falls back to unauthenticated if no credentials are set.
    """
    from dotenv import load_dotenv
    load_dotenv()

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")

    from openreview import api as or_api
    from openreview import openreview as or_v1

    if api_version == "v1":
        return or_v1.Client(
            baseurl="https://api.openreview.net",
            username=username,
            password=password,
        )
    return or_api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )


def _forum_id(paper: dict) -> str | None:
    """Extract the forum ID from a paper dict (handles both v1 and v2 schemas)."""
    return paper.get("id") or paper.get("forum") or None


def _invitation_summary(note) -> str:
    """Short human-readable label for a note's invitation type."""
    inv = getattr(note, "invitations", None) or getattr(note, "invitation", None)
    if isinstance(inv, list):
        s = inv[0] if inv else ""
    else:
        s = str(inv) if inv else ""
    for token in [
        "Official_Review", "Decision", "Rebuttal",
        "Meta_Review", "Submission", "Camera_Ready", "Withdrawn",
    ]:
        if token in s:
            return token
    return (s.split("/")[-1] or "?")[:40]


# ---------------------------------------------------------------------------
# API version probing
# ---------------------------------------------------------------------------

def probe_api_version(conference: str, year: int, data: dict) -> tuple:
    """Test both API endpoints on a few papers and return (best_name, best_client).

    Returns (None, None) if no interactions are found.
    """
    clients = [
        ("API v1 (api.openreview.net)", _make_client("v1")),
        ("API v2 (api2.openreview.net)", _make_client("v2")),
    ]

    best_total = 0
    best_client = (None, None)

    keys = list(data.keys())[:SAMPLE_SIZE]
    for name, client in clients:
        print(f"--- Probe: {name} ---")
        counts = []
        types_seen = []
        for k in keys:
            fid = _forum_id(data[k])
            if not fid:
                continue
            try:
                notes = client.get_all_notes(forum=fid)
            except Exception as e:
                counts.append(0)
                types_seen.append(f"err:{str(e)[:30]}")
                continue
            counts.append(len(notes))
            types_seen.extend(_invitation_summary(n) for n in notes)
            time.sleep(0.3)
        total = sum(counts)
        print(f"  Sample of {SAMPLE_SIZE} papers -> note counts: {counts}, total notes: {total}")
        print(f"  Types seen: {types_seen[:12]}")
        if total > SAMPLE_SIZE and total > best_total:
            best_total = total
            best_client = (name, client)
        print()

    if best_total <= SAMPLE_SIZE:
        print("No interactions found beyond submission-level notes.")
        return None, None

    print(f"Best client: {best_client[0]} ({best_total} notes in sample)\n")
    return best_client


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_download(conference: str, year: int, client) -> None:
    """Download submissions for a conference/year.

    Tries multiple invitation patterns (newer conferences use 'Submission',
    older ones use 'Blind_Submission').
    """
    patterns = [
        f"{conference}/{year}/Conference/-/Submission",
        f"{conference}/{year}/Conference/-/Blind_Submission",
    ]
    submissions = []
    for invitation in patterns:
        print(f"Trying invitation: {invitation}")
        try:
            submissions = client.get_all_notes(invitation=invitation)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        if submissions:
            print(f"  Found {len(submissions)} submissions")
            break
        print("  No submissions returned.")

    if not submissions:
        print("No submissions found with any known invitation pattern.")
        return
    path = _resolve_data_path(conference, year)
    data = {str(i): note.to_json() for i, note in enumerate(submissions)}
    write_json(data, path)
    print(f"Saved {len(submissions)} submissions to {path}")


def _fetch_batch(paper_keys, papers, out_path: Path, client, start_index: int = 0, force: bool = False) -> None:
    """Fetch interactions for a batch of papers. Saves to disk only if something changed."""
    changed = False
    for i, key in enumerate(paper_keys, start=1):
        paper = papers[key]
        if "interactions" in paper and not force:
            continue
        print(f"  Fetching {start_index + i}/{start_index + len(paper_keys)}: {key}")
        fid = _forum_id(paper)
        if not fid:
            print(f"    Missing forum id, skip")
            continue
        notes = safe_get_all_notes(client, fid)
        papers[key]["interactions"] = [n.to_json() for n in notes]
        changed = True
        time.sleep(1)
    if changed:
        write_json(papers, out_path)


def cmd_fetch(conference: str, year: int, client, *, limit: int = 0, batch_size: int = DEFAULT_BATCH_SIZE, force: bool = False) -> None:
    """Fetch interactions for every paper. Resumable (skips papers that already have them)."""
    path = _resolve_data_path(conference, year)
    if not path.exists():
        print(f"File not found: {path}")
        print("Run the 'download' subcommand first.")
        return

    papers = load_json(path)
    keys = list(papers.keys())
    if limit and limit > 0:
        keys = keys[:limit]
        print(f"Limit: processing only first {len(keys)} papers.\n")

    num_batches = max(1, (len(keys) + batch_size - 1) // batch_size)
    print(f"Fetching interactions for {len(keys)} papers in {num_batches} batch(es) of up to {batch_size}...\n")

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(keys))
        batch_keys = keys[start:end]
        print(f"Batch {i + 1}/{num_batches}: papers {start + 1}-{end}")
        _fetch_batch(batch_keys, papers, path, client, start_index=start, force=force)
        print(f"  Saved to {path}\n")

    total = sum(len(p.get("interactions", [])) for p in papers.values())
    print(f"Done. Total interactions: {total}")


def find_statistics(papers: dict, dataset_path: str | Path = "") -> None:
    """Compute and print acceptance, area, and comment/interaction statistics."""
    acceptance_stats = defaultdict(int)
    primary_area = defaultdict(int)
    total_papers = len(papers)
    total_comment_words = 0
    total_comments = 0
    total_interaction_words_per_paper = []

    for key, paper in papers.items():
        content = paper.get("content", {})
        decision = content.get("venueid", {}).get("value", "Unknown")
        area = content.get("primary_area", {}).get("value", "Unknown")
        primary_area[area] += 1

        if "Reject" in decision or "reject" in decision:
            acceptance_stats["Rejected"] += 1
        elif "Poster" in decision:
            acceptance_stats["Accepted - Poster"] += 1
        elif "Spotlight" in decision:
            acceptance_stats["Accepted - Spotlight"] += 1
        elif "Oral" in decision:
            acceptance_stats["Accepted - Oral"] += 1
        elif "Withdraw" in decision or "withdraw" in decision:
            acceptance_stats["Withdrawn"] += 1
        else:
            decision_field = content.get("decision", {})
            decision_value = (
                decision_field.get("value", "Unknown")
                if isinstance(decision_field, dict)
                else (str(decision_field) if decision_field else "Unknown")
            )
            acceptance_stats[f"Other: {decision_value}"] += 1

        paper_word_count = 0
        for interaction in paper.get("interactions", []):
            texts = get_text_fields_from_content(interaction.get("content", {}))
            interaction_text = " ".join(texts)
            word_count = len(interaction_text.split())
            if word_count > 0:
                total_comment_words += word_count
                total_comments += 1
                paper_word_count += word_count
        if paper_word_count > 0:
            total_interaction_words_per_paper.append(paper_word_count)

    avg_comment = total_comment_words / total_comments if total_comments > 0 else 0
    avg_paper = (
        sum(total_interaction_words_per_paper) / len(total_interaction_words_per_paper)
        if total_interaction_words_per_paper
        else 0
    )
    total_words = sum(total_interaction_words_per_paper)

    print("=" * 70)
    print("OPENREVIEW DATASET ANALYSIS")
    print("=" * 70)
    print(f"\nDataset: {dataset_path or '(not specified)'}")
    print(f"Total papers: {total_papers}")

    print("\n" + "=" * 70)
    print("ACCEPTANCE STATISTICS")
    print("=" * 70)
    total_accepted = (
        acceptance_stats["Accepted - Poster"]
        + acceptance_stats["Accepted - Spotlight"]
        + acceptance_stats["Accepted - Oral"]
    )
    for category, count in sorted(acceptance_stats.items(), key=lambda x: -x[1]):
        print(f"{category:30} {count:5} ({100 * count / total_papers:5.2f}%)")
    for area, count in primary_area.items():
        print(f"{area:30} {count:5}")
    print(f"\n{'Total Accepted':30} {total_accepted:5} ({100 * total_accepted / total_papers:5.2f}%)")

    print("\n" + "=" * 70)
    print("COMMENT & INTERACTION STATISTICS")
    print("=" * 70)
    print(f"\nTotal comments/interactions: {total_comments:,}")
    print(f"Papers with interactions: {len(total_interaction_words_per_paper):,}")
    print(f"\nAverage comment length: {avg_comment:.2f} words")
    print(f"Average interaction length per paper: {avg_paper:.2f} words")
    print(f"\nTotal words in all comments: {total_words:,}")
    print(f"Total words across all interactions: {total_comment_words:,}")

    print("\n" + "=" * 70)
    print("ADDITIONAL STATISTICS")
    print("=" * 70)
    if total_interaction_words_per_paper:
        print(f"Min interactions per paper: {min(total_interaction_words_per_paper):,} words")
        print(f"Max interactions per paper: {max(total_interaction_words_per_paper):,} words")
        s = sorted(total_interaction_words_per_paper)
        mid = len(s) // 2
        median = s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
        print(f"Median interactions per paper: {median:.2f} words")
    print("=" * 70)


def cmd_stats(conference: str, year: int) -> None:
    """Load data and print statistics (no API call needed)."""
    path = _resolve_data_path(conference, year)
    if not path.exists():
        print(f"File not found: {path}")
        return
    papers = load_json(path)
    total_int = sum(len(p.get("interactions", [])) for p in papers.values())
    print(f"Total interaction records: {total_int}\n")
    find_statistics(papers, dataset_path=path)


def cmd_parquet(conference: str, year: int) -> None:
    """Convert a keyed JSON dataset to Parquet."""
    json_path = _resolve_data_path(conference, year)
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return

    try:
        import pandas as pd
        import polars as pl
    except ImportError as e:
        print(f"Missing dependency for parquet conversion: {e}")
        print("Install with: pip install pandas polars")
        return

    data = load_json(json_path)
    records = list(data.values())
    df_flat = pd.json_normalize(records, sep=".", max_level=None)
    df = pl.from_pandas(df_flat).with_row_index("json_index")
    parquet_path = json_path.with_suffix(".parquet")
    df.write_parquet(parquet_path)
    print(f"Wrote {len(df)} rows to {parquet_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified OpenReview data tool: download, fetch interactions, stats, parquet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared parent for --conference, --year, --api-version
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--conference", required=True, help="e.g. ICLR.cc or ICML.cc")
    common.add_argument("--year", required=True, type=int, help="e.g. 2024")
    common.add_argument(
        "--api-version",
        choices=["auto", "v1", "v2"],
        default="auto",
        help="API version (default: auto-detect)",
    )

    dl_p = sub.add_parser("download", parents=[common], help="Download submissions")
    dl_p.add_argument("--with-interactions", action="store_true", help="Also fetch interactions after downloading")
    dl_p.add_argument("--limit", type=int, default=0, help="With --with-interactions: process only first N papers (0 = all)")
    dl_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"With --with-interactions: batch size (default {DEFAULT_BATCH_SIZE})")

    fetch_p = sub.add_parser("fetch", parents=[common], help="Fetch interactions for all papers")
    fetch_p.add_argument("--limit", type=int, default=0, help="Process only first N papers (0 = all)")
    fetch_p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default {DEFAULT_BATCH_SIZE})")
    fetch_p.add_argument("--force", action="store_true", help="Re-fetch interactions even if already present")

    sub.add_parser("stats", parents=[common], help="Print dataset statistics (offline)")
    sub.add_parser("parquet", parents=[common], help="Convert JSON to Parquet")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Commands that don't need the API
    if args.command == "stats":
        cmd_stats(args.conference, args.year)
        return 0
    if args.command == "parquet":
        cmd_parquet(args.conference, args.year)
        return 0

    # Commands that need an API client
    if args.command == "fetch" and args.api_version == "auto":
        path = _resolve_data_path(args.conference, args.year)
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        data = load_json(path)
        name, client = probe_api_version(args.conference, args.year, data)
        if client is None:
            print("Cannot auto-detect a working API. Try --api-version v1 or v2.")
            return 1
    elif args.api_version == "auto":
        # download: try v2 first (newer conferences), cmd_download tries both patterns
        client = _make_client("v2")
    else:
        client = _make_client(args.api_version)

    if args.command == "download":
        cmd_download(args.conference, args.year, client)
        if args.with_interactions:
            path = _resolve_data_path(args.conference, args.year)
            if not path.exists():
                print("Download produced no file; skipping interactions.")
                return 1
            data = load_json(path)
            print("\nProbing for best API to fetch interactions...\n")
            name, fetch_client = probe_api_version(args.conference, args.year, data)
            if fetch_client is None:
                print("Interactions not available yet. Run 'fetch' later.")
                return 0
            cmd_fetch(args.conference, args.year, fetch_client,
                      limit=args.limit, batch_size=args.batch_size)
    elif args.command == "fetch":
        cmd_fetch(args.conference, args.year, client, limit=args.limit, batch_size=args.batch_size, force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
