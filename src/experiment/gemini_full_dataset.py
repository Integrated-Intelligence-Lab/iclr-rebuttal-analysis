"""
Full-dataset Gemini Batch API processing for ICLR & ICML conferences.

Submits ALL papers (not just a sample) to Gemini's batch endpoint and tracks
job state in a local JSON tracker since Google AI Studio has no batch UI.

Usage:
    python src/experiment/gemini_full_dataset.py submit --year 2023                        # ICLR 2023
    python src/experiment/gemini_full_dataset.py submit --year 2025 --conference icml      # ICML 2025
    python src/experiment/gemini_full_dataset.py submit --year 2024 --chunk-size 500       # custom chunk size
    python src/experiment/gemini_full_dataset.py submit --year 2023 --no-wait              # fire-and-forget (Tier 2: safe)
    python src/experiment/gemini_full_dataset.py retry --year 2023 --file-id files/abc123  # retry failed upload
    python src/experiment/gemini_full_dataset.py status                                    # check all jobs
    python src/experiment/gemini_full_dataset.py download --job JOB_NAME                   # download single job
    python src/experiment/gemini_full_dataset.py download-all --year 2023                  # download & merge all chunks
    python src/experiment/gemini_full_dataset.py download-all --year 2025 --conference icml
    python src/experiment/gemini_full_dataset.py retry-failed --year 2023                  # re-run papers that failed to parse
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from dotenv import load_dotenv
load_dotenv()

from prompts.classification_instruction import PROMPT_INSTRUCTIONS
from core.io import load_json, write_json
from core.paths import (
    BATCH_INPUT_DIR, BATCH_OUTPUT_DIR, BATCHES_DIR,
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
)
from experiment.batch_utils import prepare_all_papers

# --- Config ---
# Canonical project default gemini-3-flash; override via GEMINI_MODEL in .env or --model.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")
TRACKER_PATH = BATCHES_DIR / "batch_tracker.json"

DATA_PATHS = {
    ("iclr", 2023): PROCESSED_DATA_DIR / "ICLR.cc_2023_as_2024_format.json",
    ("iclr", 2024): RAW_DATA_DIR / "ICLR.cc_2024.json",
    ("iclr", 2025): RAW_DATA_DIR / "ICLR.cc_2025.json",
    ("iclr", 2026): RAW_DATA_DIR / "ICLR.cc_2026.json",
    ("icml", 2025): RAW_DATA_DIR / "ICML.cc_2025.json",
}

# Legacy lookup: if conference not specified, assume ICLR
def _resolve_data_path(conference: str, year: int) -> Path | None:
    return DATA_PATHS.get((conference, year))


def _file_prefix(conference: str, year: int) -> str:
    """File naming prefix: gemini_31flash_iclr2024 or gemini_31flash_icml2025."""
    return f"gemini_31flash_{conference}{year}"


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def _load_tracker() -> list[dict]:
    if TRACKER_PATH.exists():
        return load_json(TRACKER_PATH)
    return []


def _save_tracker(entries: list[dict]) -> None:
    write_json(entries, TRACKER_PATH)


# ---------------------------------------------------------------------------
# JSONL generation (Gemini batch format)
# ---------------------------------------------------------------------------

def generate_gemini_jsonl(samples: list[list[dict]], paper_keys: list[str],
                          output_path: Path) -> None:
    """Write a JSONL file in the Gemini batch format.

    Uses paper_keys as the `key` field so results can be mapped back to
    specific papers (unlike the 100-paper sample which used integer indices).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for key, sample in zip(paper_keys, samples):
            user_content = json.dumps(sample, ensure_ascii=False)
            line = {
                "key": key,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": PROMPT_INSTRUCTIONS + "\n\n" + user_content}],
                        }
                    ],
                    "generation_config": {"temperature": 0.0},
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(samples)} requests to {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

# Tier 2 batch limit: 300M enqueued tokens. At ~6K tokens/paper, 1000 papers ≈ 6M tokens.
DEFAULT_CHUNK_SIZE = 1000
POLL_INTERVAL = 60  # seconds between status checks while waiting

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


def _wait_for_job(client, job_name: str) -> str:
    """Poll until a batch job reaches a terminal state. Returns final state name."""
    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        if state in TERMINAL_STATES:
            return state
        print(f"  {state} — waiting {POLL_INTERVAL}s ...", flush=True)
        time.sleep(POLL_INTERVAL)


def _find_tracker_entry(tracker: list[dict], year: int, chunk_label: str,
                        conference: str = "iclr") -> dict | None:
    """Find an existing tracker entry for a given conference + year + chunk."""
    for entry in tracker:
        if (entry["year"] == year
                and entry.get("chunk") == chunk_label
                and entry.get("conference", "iclr") == conference):
            return entry
    return None


def cmd_submit(args) -> None:
    year = args.year
    conference = args.conference
    chunk_size = args.chunk_size
    no_wait = args.no_wait
    data_path = _resolve_data_path(conference, year)
    if data_path is None or not data_path.exists():
        print(f"Unsupported or missing: --conference {conference} --year {year}")
        print(f"Available: {list(DATA_PATHS.keys())}")
        sys.exit(1)

    prefix = _file_prefix(conference, year)

    # 1. Prepare all papers
    print(f"Loading ALL papers from {data_path.name} ...")
    samples, paper_keys = prepare_all_papers(data_path)
    n_total = len(samples)
    print(f"  {n_total} papers with reviews")

    # 2. Split into chunks
    n_chunks = math.ceil(n_total / chunk_size)
    mode = "fire-and-forget" if no_wait else "sequential (resumable)"
    print(f"  Splitting into {n_chunks} chunk(s) of up to {chunk_size} papers ({mode})")

    submitted = 0
    skipped = 0
    failed = 0
    client = _get_client()

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_total)
        chunk_samples = samples[start:end]
        chunk_keys = paper_keys[start:end]
        chunk_label = f"chunk{i+1}of{n_chunks}"

        print(f"\n{'='*60}")
        print(f"Chunk {i+1}/{n_chunks} ({len(chunk_samples)} papers)")
        print(f"{'='*60}")

        # --- Resume logic: check if this chunk was already submitted ---
        tracker = _load_tracker()
        existing = _find_tracker_entry(tracker, year, chunk_label, conference)

        if existing:
            job_name = existing["job_name"]
            # Refresh state from API
            try:
                batch_job = client.batches.get(name=job_name)
                state = batch_job.state.name
                existing["status"] = state
                _save_tracker(tracker)
            except Exception:
                state = existing.get("status", "UNKNOWN")

            if state == "JOB_STATE_SUCCEEDED":
                print(f"Already completed ({job_name}). Skipping.")
                skipped += 1
                continue

            if state not in TERMINAL_STATES:
                # Job is still running — wait for it
                print(f"Already submitted ({job_name}), state: {state}")
                if not no_wait:
                    print(f"Waiting for it to finish ...")
                    final_state = _wait_for_job(client, job_name)
                    print(f"Finished: {final_state}")
                    existing["status"] = final_state
                    _save_tracker(_load_tracker())  # re-read in case other writes happened
                    if final_state == "JOB_STATE_SUCCEEDED":
                        skipped += 1
                        continue
                    else:
                        print(f"Chunk did not succeed ({final_state}). Stopping.")
                        failed += 1
                        break
                else:
                    print("Skipping (still running, --no-wait mode).")
                    skipped += 1
                    continue

            # Terminal but not succeeded (failed/cancelled/expired) — resubmit
            print(f"Previous attempt {state}. Resubmitting ...")

        # --- Generate JSONL and submit ---
        jsonl_file = BATCH_INPUT_DIR / f"{prefix}_{chunk_label}.jsonl"
        generate_gemini_jsonl(chunk_samples, chunk_keys, jsonl_file)

        job_name = _upload_and_create_batch(
            conference, year, len(chunk_samples), jsonl_file, chunk_label=chunk_label,
        )
        if not job_name:
            failed += 1
            print(f"Chunk {i+1} failed. You can retry with:")
            print(f"  python src/experiment/gemini_full_dataset.py retry --conference {conference} --year {year} --file-id <FILE_ID>")
            if not no_wait:
                print("Stopping — fix the issue before continuing.")
                break
            continue

        submitted += 1

        # Wait for completion before submitting next chunk (unless --no-wait)
        if not no_wait and i < n_chunks - 1:
            print(f"Waiting for chunk {i+1} to complete before submitting next ...")
            final_state = _wait_for_job(client, job_name)
            print(f"Chunk {i+1} finished: {final_state}")
            if final_state != "JOB_STATE_SUCCEEDED":
                print(f"Chunk {i+1} did not succeed ({final_state}). Stopping.")
                failed += 1
                break

    print(f"\n{'='*60}")
    print(f"Done. {submitted} submitted, {skipped} skipped (already done/running), {failed} failed.")
    print(f"Check status:  python src/experiment/gemini_full_dataset.py status")
    conf_flag = f" --conference {conference}" if conference != "iclr" else ""
    print(f"Download all:  python src/experiment/gemini_full_dataset.py download-all --year {year}{conf_flag}")


# ---------------------------------------------------------------------------
# Upload + batch creation (shared by submit & retry)
# ---------------------------------------------------------------------------

def _upload_and_create_batch(conference: str, year: int, n_papers: int, jsonl_file: Path,
                             file_id: str | None = None,
                             chunk_label: str | None = None) -> str | None:
    """Upload JSONL (if needed) and create a batch job. Returns job name or None."""
    client = _get_client()
    prefix = _file_prefix(conference, year)
    suffix = f"-{chunk_label}" if chunk_label else ""

    if file_id is None:
        from google.genai import types
        display_name = f"{prefix}{suffix}"
        uploaded = client.files.upload(
            file=str(jsonl_file),
            config=types.UploadFileConfig(
                display_name=display_name,
                mime_type="jsonl",
            ),
        )
        file_id = uploaded.name
        print(f"Uploaded file: {file_id}")
    else:
        print(f"Reusing already-uploaded file: {file_id}")

    try:
        batch_job = client.batches.create(
            model=f"models/{GEMINI_MODEL}",
            src=file_id,
            config={"display_name": f"{conference}{year}-extraction{suffix}"},
        )
    except Exception as e:
        print(f"Batch creation failed: {e}")
        print(f"Uploaded file ID: {file_id}")
        return None

    print(f"Batch job created: {batch_job.name}")
    print(f"State: {batch_job.state.name}")

    tracker = _load_tracker()
    tracker.append({
        "job_name": batch_job.name,
        "conference": conference,
        "year": year,
        "model": GEMINI_MODEL,
        "n_papers": n_papers,
        "chunk": chunk_label,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": batch_job.state.name,
        "input_file": jsonl_file.name,
        "output_file": None,
    })
    _save_tracker(tracker)
    print(f"Tracker updated: {TRACKER_PATH}")
    return batch_job.name


# ---------------------------------------------------------------------------
# Retry (reuse already-uploaded file)
# ---------------------------------------------------------------------------

def cmd_retry(args) -> None:
    year = args.year
    conference = args.conference
    file_id = args.file_id
    prefix = _file_prefix(conference, year)
    jsonl_file = BATCH_INPUT_DIR / f"{prefix}_full.jsonl"

    if not jsonl_file.exists():
        print(f"JSONL file not found: {jsonl_file}")
        print("Run 'submit' first to generate it.")
        sys.exit(1)

    # Count lines to get n_papers
    with open(jsonl_file, "r") as f:
        n_papers = sum(1 for _ in f)
    print(f"Retrying batch creation for {conference.upper()} {year} ({n_papers} papers)")
    print(f"Using uploaded file: {file_id}")

    result = _upload_and_create_batch(conference, year, n_papers, jsonl_file, file_id=file_id)
    if result:
        print(f"\nTo check status:  python src/experiment/gemini_full_dataset.py status")
    else:
        print(f"\nStill failing. Check your Google AI quota at https://ai.dev/rate-limit")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def cmd_status(args) -> None:
    tracker = _load_tracker()
    if not tracker:
        print("No batch jobs tracked yet.")
        return

    client = _get_client()

    header = f"{'Job Name':<45} {'Conf':>4} {'Year':>4}  {'Chunk':<16} {'Papers':>6}  {'Submitted':<20}  {'Status'}"
    print(header)
    print("-" * len(header))

    for entry in tracker:
        # Refresh status from API
        try:
            batch_job = client.batches.get(name=entry["job_name"])
            current_status = batch_job.state.name
            entry["status"] = current_status
        except Exception as e:
            current_status = f"ERROR: {e}"

        submitted = entry.get("submitted_at", "?")[:19]
        chunk = entry.get("chunk") or "full"
        conf = entry.get("conference", "iclr")
        print(f"{entry['job_name']:<45} {conf:>4} {entry['year']:>4}  {chunk:<16} {entry['n_papers']:>6}  {submitted:<20}  {current_status}")

    _save_tracker(tracker)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def cmd_download(args) -> None:
    job_name = args.job
    tracker = _load_tracker()

    # Find tracker entry
    entry = next((e for e in tracker if e["job_name"] == job_name), None)
    if not entry:
        print(f"Job '{job_name}' not found in tracker. Known jobs:")
        for e in tracker:
            print(f"  {e['job_name']}")
        sys.exit(1)

    client = _get_client()
    batch_job = client.batches.get(name=job_name)
    print(f"Job state: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print("Job has not succeeded yet. Current state:", batch_job.state.name)
        entry["status"] = batch_job.state.name
        _save_tracker(tracker)
        return

    if not (batch_job.dest and batch_job.dest.file_name):
        print("No output file found in batch job dest.")
        return

    # Download raw results
    print("Downloading results ...")
    raw_bytes = client.files.download(file=batch_job.dest.file_name)
    raw_text = raw_bytes.decode("utf-8")

    results = []
    for line in raw_text.strip().split("\n"):
        result_entry = json.loads(line)
        response_text = None
        parsed = None
        error = None
        try:
            response_text = (
                result_entry.get("response", {})
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text")
            )
            if response_text:
                parsed = _parse_json_response(response_text)
                if not parsed:
                    error = "json_parse_failed"
        except (IndexError, KeyError) as e:
            error = str(e)

        results.append({
            "custom_id": result_entry.get("key"),
            "raw_response": response_text,
            "parsed": parsed,
            "error": error,
        })

    year = entry["year"]
    conference = entry.get("conference", "iclr")
    prefix = _file_prefix(conference, year)
    chunk = entry.get("chunk")
    suffix = f"_{chunk}" if chunk else "_full"
    output_file = BATCH_OUTPUT_DIR / f"{prefix}{suffix}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(results, output_file)

    ok = sum(1 for r in results if r["parsed"] is not None)
    print(f"Saved {ok}/{len(results)} parsed results to {output_file}")

    # Update tracker
    entry["status"] = "completed"
    entry["output_file"] = output_file.name
    _save_tracker(tracker)
    print(f"Tracker updated.")


# ---------------------------------------------------------------------------
# Download all chunks for a year
# ---------------------------------------------------------------------------

def cmd_download_all(args) -> None:
    year = args.year
    conference = args.conference
    prefix = _file_prefix(conference, year)
    tracker = _load_tracker()
    entries = [e for e in tracker
               if e["year"] == year
               and e.get("conference", "iclr") == conference
               and e.get("output_file") is None]

    if not entries:
        print(f"No pending downloads for {conference.upper()} {year}.")
        return

    print(f"Found {len(entries)} chunk(s) to download for {conference.upper()} {year}")
    client = _get_client()

    all_results = []
    for entry in entries:
        job_name = entry["job_name"]
        chunk = entry.get("chunk") or "full"
        print(f"\n--- {chunk}: {job_name} ---")

        batch_job = client.batches.get(name=job_name)
        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"  Skipping — state: {batch_job.state.name}")
            entry["status"] = batch_job.state.name
            continue

        if not (batch_job.dest and batch_job.dest.file_name):
            print("  No output file in batch job.")
            continue

        raw_bytes = client.files.download(file=batch_job.dest.file_name)
        raw_text = raw_bytes.decode("utf-8")

        chunk_results = []
        for line in raw_text.strip().split("\n"):
            result_entry = json.loads(line)
            response_text = None
            parsed = None
            error = None
            try:
                response_text = (
                    result_entry.get("response", {})
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text")
                )
                if response_text:
                    parsed = _parse_json_response(response_text)
                    if not parsed:
                        error = "json_parse_failed"
            except (IndexError, KeyError) as e:
                error = str(e)

            chunk_results.append({
                "custom_id": result_entry.get("key"),
                "raw_response": response_text,
                "parsed": parsed,
                "error": error,
            })

        # Save individual chunk file
        suffix = f"_{chunk}" if entry.get("chunk") else "_full"
        chunk_file = BATCH_OUTPUT_DIR / f"{prefix}{suffix}.json"
        write_json(chunk_results, chunk_file)
        ok = sum(1 for r in chunk_results if r["parsed"] is not None)
        print(f"  Saved {ok}/{len(chunk_results)} parsed → {chunk_file.name}")

        entry["status"] = "completed"
        entry["output_file"] = chunk_file.name
        all_results.extend(chunk_results)

    _save_tracker(tracker)

    if all_results:
        # Save merged file
        merged_file = BATCH_OUTPUT_DIR / f"{prefix}_full.json"
        write_json(all_results, merged_file)
        ok = sum(1 for r in all_results if r["parsed"] is not None)
        print(f"\nMerged: {ok}/{len(all_results)} parsed → {merged_file.name}")


# ---------------------------------------------------------------------------
# Retry failed papers
# ---------------------------------------------------------------------------

def cmd_retry_failed(args) -> None:
    year = args.year
    conference = args.conference
    prefix = _file_prefix(conference, year)
    merged_file = BATCH_OUTPUT_DIR / f"{prefix}_full.json"

    if not merged_file.exists():
        print(f"No merged output file found: {merged_file}")
        print("Run 'download-all' first.")
        sys.exit(1)

    # Find failed paper IDs
    results = load_json(merged_file)
    failed_ids = set()
    for r in results:
        if r.get("error") or r.get("parsed") is None:
            failed_ids.add(str(r["custom_id"]))

    if not failed_ids:
        print(f"No failed papers in {merged_file.name}. All {len(results)} parsed successfully.")
        return

    print(f"Found {len(failed_ids)} failed papers out of {len(results)} total")

    # Load original data and extract reviews for failed papers only
    data_path = _resolve_data_path(conference, year)
    print(f"Loading reviews for failed papers from {data_path.name} ...")
    all_samples, all_keys = prepare_all_papers(data_path)

    retry_samples = []
    retry_keys = []
    for sample, key in zip(all_samples, all_keys):
        if str(key) in failed_ids:
            retry_samples.append(sample)
            retry_keys.append(key)

    print(f"  Matched {len(retry_samples)} papers to retry")

    if not retry_samples:
        print("Could not match any failed IDs to papers. Check custom_id format.")
        return

    # Generate JSONL and submit
    chunk_label = "retry_failed"
    jsonl_file = BATCH_INPUT_DIR / f"{prefix}_{chunk_label}.jsonl"
    generate_gemini_jsonl(retry_samples, retry_keys, jsonl_file)

    job_name = _upload_and_create_batch(
        conference, year, len(retry_samples), jsonl_file, chunk_label=chunk_label,
    )
    if job_name:
        print(f"\nRetry batch submitted. Check status:")
        print(f"  python src/experiment/gemini_full_dataset.py status")
    else:
        print(f"\nSubmission failed. Check quota at https://ai.dev/rate-limit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set in .env. Exiting.")
        sys.exit(1)
    from google import genai
    return genai.Client(api_key=api_key)


def _parse_json_response(text: str) -> dict | None:
    """Try to parse JSON from a Gemini response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full-dataset Gemini batch processing for ICLR"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    VALID_YEARS = [2023, 2024, 2025, 2026]

    def _add_conference_year(sub):
        sub.add_argument("--conference", type=str, default="iclr",
                         choices=["iclr", "icml"], help="Conference (default: iclr)")
        sub.add_argument("--year", type=int, required=True, choices=VALID_YEARS,
                         help="Conference year to process")

    # submit
    sub = subparsers.add_parser("submit", help="Prepare JSONL and submit batch job(s)")
    _add_conference_year(sub)
    sub.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                     help=f"Max papers per batch chunk (default: {DEFAULT_CHUNK_SIZE})")
    sub.add_argument("--no-wait", action="store_true",
                     help="Submit all chunks without waiting (Tier 2: safe to use)")
    sub.set_defaults(func=cmd_submit)

    # retry
    sub = subparsers.add_parser("retry", help="Retry batch creation from an already-uploaded file")
    _add_conference_year(sub)
    sub.add_argument("--file-id", type=str, required=True,
                     help="Google file ID from a previous upload (e.g. files/pxnpuyikuxdb)")
    sub.set_defaults(func=cmd_retry)

    # status
    sub = subparsers.add_parser("status", help="Check status of all tracked batch jobs")
    sub.set_defaults(func=cmd_status)

    # download
    sub = subparsers.add_parser("download", help="Download results for a single job")
    sub.add_argument("--job", type=str, required=True,
                     help="Batch job name (from tracker or submit output)")
    sub.set_defaults(func=cmd_download)

    # download-all
    sub = subparsers.add_parser("download-all", help="Download & merge all chunks for a year")
    _add_conference_year(sub)
    sub.set_defaults(func=cmd_download_all)

    # retry-failed
    sub = subparsers.add_parser("retry-failed", help="Re-submit only papers that failed to parse")
    _add_conference_year(sub)
    sub.set_defaults(func=cmd_retry_failed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
