"""
Gemini batch pipeline: classify rebuttal interactions against a 44-factor codebook.

Purpose
-------
Validates the LLM-induced factor list (src/factor_validation/codebook.json) by
asking a second Gemini model which factors are clearly present in each interaction.
Two runs are supported:

  Test 1 (within-sample, operationalisability):
    --input data/results/score_change/samples/classify_input_raw_threads.json
    --tag test1_same1k
    Measures whether LLM-B agrees with the factor taxonomy on the same 1000
    interactions the taxonomy was mined from. High hit rates here are expected
    by construction; the value is Cohen's-kappa-style reliability, not discovery.

  Test 2 (held-out, discriminative validity):
    --input data/results/score_change/samples/classify_input_raw_threads_test2.json
    --tag test2_fresh1k
    Measures whether the factor-category associations replicate on a fresh
    random sample disjoint from the taxonomy-induction corpus. This is the
    scientifically load-bearing test.

The prompt is outcome-blinded (tier and verdict_group are stripped before send),
factor IDs are shuffled per request, and Gemini's response_schema enforces that
factor_id takes one of F01..F44 so the model cannot hallucinate new factors.

Usage
-----
    python src/factor_validation/classify.py submit \\
        --input data/results/score_change/samples/classify_input_raw_threads.json \\
        --tag test1_same1k

    python src/factor_validation/classify.py status
    python src/factor_validation/classify.py download-all --tag test1_same1k
    python src/factor_validation/classify.py retry-failed --tag test1_same1k

Output
------
    data/batches/input/factor_validation_{tag}_chunk{i}of{n}.jsonl
    data/batches/output/factor_validation_{tag}_chunk{i}of{n}.json
    data/batches/output/factor_validation_{tag}_full.json  (merged)
    data/batches/batch_tracker.json  (extended with job_type='factor_validation')
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv
load_dotenv()

from core.io import load_json, write_json
from core.paths import BATCH_INPUT_DIR, BATCH_OUTPUT_DIR, BATCHES_DIR, RESULTS_DIR
from factor_validation.prompt import build_request_body


# --- Config ---
# Reads GEMINI_MODEL from .env (same variable the existing scripts use) so the
# model stays consistent across the project. Override per-run with --model.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")
GEMINI_MODEL = DEFAULT_MODEL  # runtime value, possibly overridden by --model
TRACKER_PATH = BATCHES_DIR / "batch_tracker.json"
CODEBOOK_PATH = Path(__file__).parent / "codebook.json"
EXPERIMENT_DIR = RESULTS_DIR / "factor_validation"
FILE_PREFIX = "factor_validation"
JOB_TYPE = "factor_validation"

DEFAULT_CHUNK_SIZE = 500
POLL_INTERVAL = 60
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


# ---------------------------------------------------------------------------
# Codebook + input loading
# ---------------------------------------------------------------------------

def load_codebook() -> list[dict]:
    """Load and validate the 44-factor codebook."""
    data = load_json(CODEBOOK_PATH)
    factors = data.get("factors", [])
    if len(factors) != 44:
        raise ValueError(f"Expected 44 factors, got {len(factors)}")
    for f in factors:
        for key in ("id", "label", "polarity"):
            if key not in f:
                raise ValueError(f"Factor missing key '{key}': {f}")
    return factors


def flatten_items(classify_input: dict) -> list[dict]:
    """Flatten the 6-group structure of classify_input_raw_threads.json.

    Attaches tier and verdict_group to each item as metadata so results can be
    aggregated by category AFTER classification. This metadata is NOT sent to
    the LLM; it is stripped by build_request_body.
    """
    flat: list[dict] = []
    groups = classify_input.get("groups", {})
    for group_key, group in groups.items():
        tier = group.get("tier")
        verdict_group = group.get("verdict_group")
        for item in group.get("items", []):
            flat.append({
                "paper_id": item["paper_id"],
                "forum": item["forum"],
                "reviewer_id": item["reviewer_id"],
                "input_text": item["input_text"],
                "tier": tier,
                "verdict_group": verdict_group,
                "group_key": group_key,
            })
    return flat


def make_batch_key(item: dict) -> str:
    """Stable key that round-trips through Gemini's `key` field.

    Forum IDs are short and URL-safe, reviewer IDs contain slashes but survive
    OK as the key field. We join on '::' which does not appear in either.
    """
    return f"{item['forum']}::{item['reviewer_id']}"


# ---------------------------------------------------------------------------
# JSONL generation
# ---------------------------------------------------------------------------

def generate_jsonl(items: list[dict], factors: list[dict],
                   output_path: Path, shuffle_seed_base: int) -> None:
    """Write a JSONL file in the Gemini batch format.

    shuffle_seed_base + item index is used as the per-request RNG seed so the
    factor order is reproducible but still varies across items.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Generating JSONL for {len(items)} items -> {output_path.name} ...", flush=True)
    progress_every = max(1, len(items) // 10)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            rng = random.Random(shuffle_seed_base + idx)
            request_body = build_request_body(
                interaction_text=item["input_text"],
                factors=factors,
                rng=rng,
                temperature=0.0,
            )
            line = {
                "key": make_batch_key(item),
                "request": request_body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if (idx + 1) % progress_every == 0:
                print(f"    ... {idx + 1}/{len(items)}", flush=True)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {len(items)} requests ({size_mb:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def _load_tracker() -> list[dict]:
    if TRACKER_PATH.exists():
        return load_json(TRACKER_PATH)
    return []


def _save_tracker(entries: list[dict]) -> None:
    write_json(entries, TRACKER_PATH)


@contextmanager
def _tracker_transaction():
    """File-locked atomic read-modify-write of the tracker.

    Acquires an exclusive flock on a sentinel file before loading the tracker,
    yields the loaded list for in-place mutation, then writes it back and
    releases the lock. This prevents the race where one process reads a stale
    snapshot while another is appending entries (typical when `submit` runs
    concurrently with `status` or `download-all`).

    Usage:
        with _tracker_transaction() as tracker:
            tracker.append(...)  # or mutate existing entries

    Backwards-compatible callers can still use `_load_tracker()` /
    `_save_tracker()` directly for read-only inspection, but any mutation that
    must be persisted MUST go through this context manager.
    """
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = TRACKER_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            tracker = _load_tracker()
            yield tracker
            _save_tracker(tracker)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _find_tracker_entry(tracker: list[dict], tag: str, chunk_label: str) -> dict | None:
    for entry in tracker:
        if (entry.get("job_type") == JOB_TYPE
                and entry.get("tag") == tag
                and entry.get("chunk") == chunk_label):
            return entry
    return None


# ---------------------------------------------------------------------------
# Client + parsing
# ---------------------------------------------------------------------------

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set in .env. Exiting.", flush=True)
        sys.exit(1)
    print("  Importing google.genai ...", flush=True)
    from google import genai
    print("  Creating Gemini client ...", flush=True)
    return genai.Client(api_key=api_key)


def _parse_json_response(text: str) -> dict | None:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _wait_for_job(client, job_name: str) -> str:
    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        if state in TERMINAL_STATES:
            return state
        print(f"  {state} — waiting {POLL_INTERVAL}s ...", flush=True)
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Upload + batch creation
# ---------------------------------------------------------------------------

def _upload_and_create_batch(tag: str, chunk_label: str, n_items: int,
                             jsonl_file: Path, input_file_label: str,
                             file_id: str | None = None) -> str | None:
    client = _get_client()

    if file_id is None:
        from google.genai import types
        display_name = f"factor_validation_{tag}_{chunk_label}"
        size_mb = jsonl_file.stat().st_size / (1024 * 1024)
        print(f"  Uploading {jsonl_file.name} ({size_mb:.1f} MB) to Google ...", flush=True)
        uploaded = client.files.upload(
            file=str(jsonl_file),
            config=types.UploadFileConfig(
                display_name=display_name,
                mime_type="jsonl",
            ),
        )
        file_id = uploaded.name
        print(f"  Uploaded file: {file_id}", flush=True)
    else:
        print(f"  Reusing already-uploaded file: {file_id}", flush=True)

    print(f"  Creating batch job ...", flush=True)

    try:
        batch_job = client.batches.create(
            model=f"models/{GEMINI_MODEL}",
            src=file_id,
            config={"display_name": f"factor_validation_{tag}_{chunk_label}"},
        )
    except Exception as e:
        print(f"Batch creation failed: {e}")
        print(f"Uploaded file ID: {file_id}")
        return None

    print(f"Batch job created: {batch_job.name}")
    print(f"State: {batch_job.state.name}")

    with _tracker_transaction() as tracker:
        tracker.append({
            "job_type": JOB_TYPE,
            "tag": tag,
            "job_name": batch_job.name,
            "model": GEMINI_MODEL,
            "n_items": n_items,
            "chunk": chunk_label,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "status": batch_job.state.name,
            "input_file": jsonl_file.name,
            "source_input": input_file_label,
            "output_file": None,
        })
    print(f"Tracker updated: {TRACKER_PATH}")
    return batch_job.name


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_submit(args) -> None:
    global GEMINI_MODEL
    if getattr(args, "model", None):
        GEMINI_MODEL = args.model
    print(f"Model: {GEMINI_MODEL}", flush=True)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    print(f"Loading input from {input_path.name} ...")
    classify_input = load_json(input_path)
    items = flatten_items(classify_input)
    n_total = len(items)
    print(f"  {n_total} interactions")

    factors = load_codebook()
    print(f"  {len(factors)} factors in codebook")

    # Save the flattened metadata for later joins (tier, verdict_group).
    meta_path = BATCH_OUTPUT_DIR / f"factor_validation_{args.tag}_items.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        [{k: v for k, v in item.items() if k != "input_text"} for item in items],
        meta_path,
    )
    print(f"  Saved item metadata to {meta_path.name}")

    n_chunks = math.ceil(n_total / args.chunk_size)
    mode = "fire-and-forget" if args.no_wait else "sequential (resumable)"
    print(f"  Splitting into {n_chunks} chunk(s) of up to {args.chunk_size} ({mode})")

    client = _get_client()
    submitted = skipped = failed = 0

    for i in range(n_chunks):
        start = i * args.chunk_size
        end = min(start + args.chunk_size, n_total)
        chunk_items = items[start:end]
        chunk_label = f"chunk{i+1}of{n_chunks}"

        print(f"\n{'='*60}")
        print(f"{args.tag} — {chunk_label} ({len(chunk_items)} items)")
        print(f"{'='*60}")

        # Read-only check; mutation goes through _tracker_transaction below.
        existing = _find_tracker_entry(_load_tracker(), args.tag, chunk_label)

        if existing:
            job_name = existing["job_name"]
            try:
                batch_job = client.batches.get(name=job_name)
                state = batch_job.state.name
                with _tracker_transaction() as tracker_w:
                    fresh_existing = _find_tracker_entry(tracker_w, args.tag, chunk_label)
                    if fresh_existing is not None:
                        fresh_existing["status"] = state
            except Exception:
                state = existing.get("status", "UNKNOWN")

            if state == "JOB_STATE_SUCCEEDED":
                print(f"Already completed ({job_name}). Skipping.")
                skipped += 1
                continue

            if state not in TERMINAL_STATES:
                print(f"Already submitted ({job_name}), state: {state}")
                if not args.no_wait:
                    print("Waiting for it to finish ...")
                    final_state = _wait_for_job(client, job_name)
                    print(f"Finished: {final_state}")
                    with _tracker_transaction() as tracker_w:
                        fresh_existing = _find_tracker_entry(tracker_w, args.tag, chunk_label)
                        if fresh_existing is not None:
                            fresh_existing["status"] = final_state
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

            print(f"Previous attempt {state}. Resubmitting ...")

        # Generate JSONL
        jsonl_file = BATCH_INPUT_DIR / f"factor_validation_{args.tag}_{chunk_label}.jsonl"
        generate_jsonl(
            chunk_items, factors, jsonl_file,
            shuffle_seed_base=args.shuffle_seed + start,
        )

        job_name = _upload_and_create_batch(
            tag=args.tag,
            chunk_label=chunk_label,
            n_items=len(chunk_items),
            jsonl_file=jsonl_file,
            input_file_label=input_path.name,
        )
        if not job_name:
            failed += 1
            print(f"Chunk {i+1} failed. Retry later.")
            if not args.no_wait:
                print("Stopping — fix the issue before continuing.")
                break
            continue

        submitted += 1

        if not args.no_wait and i < n_chunks - 1:
            print(f"Waiting for chunk {i+1} to complete before submitting next ...")
            final_state = _wait_for_job(client, job_name)
            print(f"Chunk {i+1} finished: {final_state}")
            if final_state != "JOB_STATE_SUCCEEDED":
                print(f"Chunk {i+1} did not succeed ({final_state}). Stopping.")
                failed += 1
                break

    print(f"\n{'='*60}")
    print(f"Done. {submitted} submitted, {skipped} skipped, {failed} failed.")
    print(f"Check status:  python src/factor_validation/classify.py status")
    print(f"Download:      python src/factor_validation/classify.py download-all --tag {args.tag}")


def cmd_status(args) -> None:
    tracker = _load_tracker()
    entries = [e for e in tracker if e.get("job_type") == JOB_TYPE]
    if args.tag:
        entries = [e for e in entries if e.get("tag") == args.tag]
    if not entries:
        print("No factor_validation jobs tracked.")
        return

    client = _get_client()
    header = f"{'Tag':<18} {'Chunk':<16} {'Items':>6}  {'Submitted':<20}  {'Status':<28} {'Job Name'}"
    print(header)
    print("-" * len(header))

    # Collect state updates without writing yet, so we can do a single atomic
    # merge at the end (avoids clobbering entries appended by a concurrent
    # `submit` while this loop runs).
    state_by_job_name: dict[str, str] = {}
    for e in entries:
        try:
            batch_job = client.batches.get(name=e["job_name"])
            state = batch_job.state.name
            state_by_job_name[e["job_name"]] = state
        except Exception as exc:
            state = f"ERROR: {exc}"
        submitted = e.get("submitted_at", "?")[:19]
        print(f"{e.get('tag','?'):<18} {e.get('chunk','?'):<16} {e.get('n_items','?'):>6}  "
              f"{submitted:<20}  {state:<28} {e['job_name']}")

    if state_by_job_name:
        with _tracker_transaction() as tracker_w:
            for entry in tracker_w:
                jn = entry.get("job_name")
                if jn in state_by_job_name:
                    entry["status"] = state_by_job_name[jn]


def _download_one_chunk(client, entry: dict) -> list[dict]:
    job_name = entry["job_name"]
    batch_job = client.batches.get(name=job_name)
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"  Skipping {entry.get('chunk')} — state: {batch_job.state.name}")
        entry["status"] = batch_job.state.name
        return []
    if not (batch_job.dest and batch_job.dest.file_name):
        print(f"  No output file for {entry.get('chunk')}.")
        return []

    raw_bytes = client.files.download(file=batch_job.dest.file_name)
    raw_text = raw_bytes.decode("utf-8")

    results = []
    for line in raw_text.strip().split("\n"):
        if not line.strip():
            continue
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
            else:
                error = "no_response_text"
        except (IndexError, KeyError) as e:
            error = f"extraction_error: {e}"

        results.append({
            "custom_id": result_entry.get("key"),
            "raw_response": response_text,
            "parsed": parsed,
            "error": error,
        })

    return results


def cmd_download_all(args) -> None:
    tag = args.tag
    tracker = _load_tracker()
    entries = [e for e in tracker
               if e.get("job_type") == JOB_TYPE
               and e.get("tag") == tag
               and e.get("output_file") is None]

    if not entries:
        print(f"No pending downloads for tag '{tag}'.")
        return

    print(f"Found {len(entries)} chunk(s) to download for tag '{tag}'")
    client = _get_client()

    # Collect updates here, then merge once at the end so concurrent submits
    # don't clobber tracker entries appended during this loop.
    updates_by_job_name: dict[str, dict] = {}
    all_results: list[dict] = []
    for entry in entries:
        chunk = entry.get("chunk") or "full"
        print(f"\n--- {chunk}: {entry['job_name']} ---")
        chunk_results = _download_one_chunk(client, entry)
        if not chunk_results:
            continue

        chunk_file = BATCH_OUTPUT_DIR / f"factor_validation_{tag}_{chunk}.json"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(chunk_results, chunk_file)
        ok = sum(1 for r in chunk_results if r["parsed"] is not None)
        print(f"  Saved {ok}/{len(chunk_results)} parsed -> {chunk_file.name}")

        updates_by_job_name[entry["job_name"]] = {
            "status": "completed",
            "output_file": chunk_file.name,
        }
        all_results.extend(chunk_results)

    if updates_by_job_name:
        with _tracker_transaction() as tracker_w:
            for entry in tracker_w:
                jn = entry.get("job_name")
                if jn in updates_by_job_name:
                    entry.update(updates_by_job_name[jn])

    if all_results:
        merged_file = BATCH_OUTPUT_DIR / f"factor_validation_{tag}_full.json"
        write_json(all_results, merged_file)
        ok = sum(1 for r in all_results if r["parsed"] is not None)
        print(f"\nMerged: {ok}/{len(all_results)} parsed -> {merged_file.name}")


def cmd_list_models(args) -> None:
    """List models visible to the current API key (filtered to Flash / batch-capable names)."""
    client = _get_client()
    print("Models visible to this API key (best-effort filter for batch-capable):", flush=True)
    seen = set()
    for m in client.models.list():
        name = getattr(m, "name", "") or ""
        if not name or name in seen:
            continue
        seen.add(name)
        methods = getattr(m, "supported_generation_methods", None) or \
                  getattr(m, "supported_actions", None) or []
        print(f"  {name}   methods={list(methods) if methods else '?'}", flush=True)


def cmd_retry_failed(args) -> None:
    global GEMINI_MODEL
    if getattr(args, "model", None):
        GEMINI_MODEL = args.model
    print(f"Model: {GEMINI_MODEL}", flush=True)

    tag = args.tag
    merged_file = BATCH_OUTPUT_DIR / f"factor_validation_{tag}_full.json"
    if not merged_file.exists():
        print(f"No merged output at {merged_file}. Run download-all first.")
        sys.exit(1)

    results = load_json(merged_file)
    failed_keys = {str(r["custom_id"]) for r in results
                   if r.get("error") or r.get("parsed") is None}

    if not failed_keys:
        print(f"No failed items in {merged_file.name}. All {len(results)} parsed.")
        return

    print(f"Found {len(failed_keys)} failed items out of {len(results)}")

    input_hint = _infer_source_input(tag)
    if not input_hint:
        print("Could not infer source input path. Re-run with --input.")
        sys.exit(1)
    input_path = input_hint
    print(f"Re-loading source input: {input_path}")

    classify_input = load_json(input_path)
    items = flatten_items(classify_input)
    retry_items = [it for it in items if make_batch_key(it) in failed_keys]
    print(f"  Matched {len(retry_items)} failed items to source.")

    if not retry_items:
        print("No matches — check that source input matches the original submission.")
        return

    factors = load_codebook()
    chunk_label = "retry_failed"
    jsonl_file = BATCH_INPUT_DIR / f"factor_validation_{tag}_{chunk_label}.jsonl"
    generate_jsonl(
        retry_items, factors, jsonl_file,
        shuffle_seed_base=args.shuffle_seed + 10_000_000,
    )

    _upload_and_create_batch(
        tag=tag,
        chunk_label=chunk_label,
        n_items=len(retry_items),
        jsonl_file=jsonl_file,
        input_file_label=input_path.name,
    )


def _infer_source_input(tag: str) -> Path | None:
    """Look at tracker for the most recent source_input for this tag.

    Checks the experiment dir first, then falls back to legacy locations where
    the Test 1 file lives.
    """
    tracker = _load_tracker()
    candidates_roots = [
        EXPERIMENT_DIR,
        RESULTS_DIR / "score_change" / "samples",
    ]
    for e in reversed(tracker):
        if e.get("job_type") == JOB_TYPE and e.get("tag") == tag:
            src_name = e.get("source_input")
            if not src_name:
                continue
            for root in candidates_roots:
                candidate = root / src_name
                if candidate.exists():
                    return candidate
    return None


# ---------------------------------------------------------------------------
# Sync retry for a small number of failed items
# ---------------------------------------------------------------------------

def _try_repair_json(raw: str) -> dict | None:
    """Attempt to salvage a truncated JSON response by closing brackets.

    Strategy:
      1. Strip trailing whitespace / incomplete tokens.
      2. Drop an incomplete trailing object (text after the last complete '}').
      3. Close any still-open brackets in the correct nesting order.
    Returns the parsed dict if a repair produced valid JSON, else None.
    """
    if not raw:
        return None
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
    s = s.strip()
    # Drop any trailing text past the last '}' so we can recover partial arrays.
    last_obj_close = s.rfind("}")
    if last_obj_close != -1:
        candidate = s[: last_obj_close + 1]
    else:
        candidate = s
    # Count unclosed brackets and append closers.
    depth_obj = depth_arr = 0
    in_str = False
    esc = False
    for ch in candidate:
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1
    # Trim trailing comma if present (after dropping trailing junk above).
    candidate = candidate.rstrip(",")
    closers = ("]" * max(0, depth_arr)) + ("}" * max(0, depth_obj))
    repaired = candidate + closers
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def cmd_retry_failed_sync(args) -> None:
    """Fix failed items by direct (non-batch) API calls. Patches the merged file in place."""
    global GEMINI_MODEL
    if getattr(args, "model", None):
        GEMINI_MODEL = args.model
    print(f"Model: {GEMINI_MODEL}", flush=True)

    tag = args.tag
    merged_file = BATCH_OUTPUT_DIR / f"{FILE_PREFIX}_{tag}_full.json"
    if not merged_file.exists():
        print(f"No merged output at {merged_file}. Run download-all first.")
        sys.exit(1)

    results = load_json(merged_file)
    failed_idx = [i for i, r in enumerate(results)
                  if r.get("error") or r.get("parsed") is None]
    if not failed_idx:
        print(f"No failed items in {merged_file.name}.")
        return
    print(f"Found {len(failed_idx)} failed items out of {len(results)}")

    # --- Stage 1: free repair pass on existing raw_response ---
    repaired = 0
    still_failed: list[int] = []
    for i in failed_idx:
        raw = results[i].get("raw_response") or ""
        parsed = _try_repair_json(raw)
        if parsed and isinstance(parsed.get("factors_present"), list):
            results[i]["parsed"] = parsed
            results[i]["error"] = None
            results[i]["recovered_by"] = "repair_parse"
            repaired += 1
        else:
            still_failed.append(i)
    print(f"  Stage 1 (local repair): salvaged {repaired}, still failed {len(still_failed)}")

    if not still_failed:
        write_json(results, merged_file)
        print(f"Wrote {merged_file}")
        return

    # --- Stage 2: sync API calls for the rest ---
    input_hint = Path(args.input).expanduser().resolve() if args.input else _infer_source_input(tag)
    if not input_hint or not input_hint.exists():
        print("Need --input <path-to-classify-input.json> (source with input_text).")
        sys.exit(1)
    print(f"  Stage 2 source: {input_hint}")
    source = load_json(input_hint)
    items = flatten_items(source)
    by_key = {make_batch_key(it): it for it in items}
    factors = load_codebook()

    client = _get_client()
    for n, i in enumerate(still_failed, 1):
        key = str(results[i].get("custom_id") or "")
        item = by_key.get(key)
        if not item:
            print(f"  [{n}/{len(still_failed)}] {key} NOT FOUND in source — skipping")
            continue
        # Reproduce the same shuffled factor order the batch used.
        # Item index in the flat list matches the original submit order.
        try:
            item_idx = next(idx for idx, it in enumerate(items)
                            if make_batch_key(it) == key)
        except StopIteration:
            item_idx = 0
        rng = random.Random(args.shuffle_seed + item_idx)
        body = build_request_body(
            interaction_text=item["input_text"],
            factors=factors,
            rng=rng,
            temperature=0.0,
        )
        print(f"  [{n}/{len(still_failed)}] calling API for {key} ...", flush=True)
        try:
            from google.genai import types
            cfg = types.GenerateContentConfig(temperature=0.0)
            resp = client.models.generate_content(
                model=f"models/{GEMINI_MODEL}",
                contents=body["contents"],
                config=cfg,
            )
            raw_text = getattr(resp, "text", None) or ""
            parsed = _parse_json_response(raw_text) or _try_repair_json(raw_text)
            if parsed and isinstance(parsed.get("factors_present"), list):
                results[i]["raw_response"] = raw_text
                results[i]["parsed"] = parsed
                results[i]["error"] = None
                results[i]["recovered_by"] = "sync_api"
                print(f"    ok: {len(parsed['factors_present'])} factors")
            else:
                results[i]["raw_response"] = raw_text
                results[i]["error"] = "json_parse_failed_after_sync"
                print(f"    still unparseable, raw len={len(raw_text)}")
        except Exception as exc:
            results[i]["error"] = f"sync_exception: {exc}"
            print(f"    EXC: {exc}")

    # Final write + summary
    write_json(results, merged_file)
    still = sum(1 for r in results if r.get("error") or r.get("parsed") is None)
    print(f"Wrote {merged_file.name}. Remaining failures: {still}/{len(results)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini batch classification of rebuttal interactions against a 44-factor codebook."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # submit
    sub = subparsers.add_parser("submit", help="Generate JSONL and submit batch job(s).")
    sub.add_argument("--input", type=str, required=True,
                     help="Path to classify_input_raw_threads*.json (Test 1 or Test 2).")
    sub.add_argument("--tag", type=str, required=True,
                     help="Short tag used to name outputs (e.g. test1_same1k, test2_fresh1k).")
    sub.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                     help=f"Items per batch chunk (default {DEFAULT_CHUNK_SIZE}).")
    sub.add_argument("--shuffle-seed", type=int, default=50,
                     help="Base seed for per-item factor shuffle (default 50).")
    sub.add_argument("--no-wait", action="store_true",
                     help="Submit all chunks without waiting between them.")
    sub.add_argument("--model", type=str, default=DEFAULT_MODEL,
                     help=f"Gemini model name for batch creation (default: {DEFAULT_MODEL}).")
    sub.set_defaults(func=cmd_submit)

    # status
    sub = subparsers.add_parser("status", help="Check status of factor_validation jobs.")
    sub.add_argument("--tag", type=str, default=None, help="Filter to one tag.")
    sub.set_defaults(func=cmd_status)

    # download-all
    sub = subparsers.add_parser("download-all", help="Download and merge all chunks for a tag.")
    sub.add_argument("--tag", type=str, required=True)
    sub.set_defaults(func=cmd_download_all)

    # retry-failed (batch, for many failures)
    sub = subparsers.add_parser("retry-failed", help="Resubmit items that failed to parse (via batch API).")
    sub.add_argument("--tag", type=str, required=True)
    sub.add_argument("--shuffle-seed", type=int, default=50)
    sub.add_argument("--model", type=str, default=DEFAULT_MODEL,
                     help=f"Gemini model name (default: {DEFAULT_MODEL}).")
    sub.set_defaults(func=cmd_retry_failed)

    # retry-failed-sync (direct sync calls, ideal for <~50 failures)
    sub = subparsers.add_parser("retry-failed-sync",
        help="Fix failed items via direct (non-batch) API calls and patch the merged file in place. "
             "First tries a free local JSON-repair pass, then calls the API for anything still broken.")
    sub.add_argument("--tag", type=str, required=True)
    sub.add_argument("--input", type=str, default=None,
                     help="Source classify_input_*.json (defaults to the path stored in tracker).")
    sub.add_argument("--shuffle-seed", type=int, default=50,
                     help="Must match the seed used at submit time (default 50).")
    sub.add_argument("--model", type=str, default=DEFAULT_MODEL,
                     help=f"Gemini model name (default: {DEFAULT_MODEL}).")
    sub.set_defaults(func=cmd_retry_failed_sync)

    # list-models: quick helper to enumerate batch-capable models your key can see
    sub = subparsers.add_parser("list-models",
        help="List Gemini models available via the batch API for the current key.")
    sub.set_defaults(func=cmd_list_models)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
