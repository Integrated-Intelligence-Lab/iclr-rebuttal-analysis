"""
Step 1: Extract regex-based score-change verdicts (increase / decrease / same).

For (paper, reviewer) with chain [Official_Review → Author Response → Reviewer
follow-up(s)], runs analyze_sentiment on the chosen follow-up text. "Later" =
**after** the author's first rebuttal in time (newer, more recent), not older.
Outputs one row per (paper_id, reviewer_id) only when the regex matches
increase/decrease/same. No row when: the reviewer left no follow-up after the
rebuttal, or their follow-up(s) do not contain our score-change phrases.

Usage:
    python src/score_change/extract_regex_verdicts.py             # 100-paper sample ICLR 2024 (legacy)
    python src/score_change/extract_regex_verdicts.py --full      # full dataset ICLR 2024
    python src/score_change/extract_regex_verdicts.py --full --year 2023
    python src/score_change/extract_regex_verdicts.py --full --year 2025

Changes:
- 2026-04-07: Added --tag outputs and automatic *_new_additions.json generation.
- 2026-04-07: Added optional diagnostics fields with regex matched pattern IDs.
- 2026-04-09: Added exclusion and allowlist controls with metadata sidecar outputs.
"""

import argparse
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from core.io import load_json, write_json
from core.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from core.sampling import get_sampled_indices
from core.paper_filters import is_withdrawn_or_desk_rejected
from score_change.sentiment import analyze_sentiment, analyze_sentiment_with_reason, get_text_content

# --- Config ---
SAMPLE_SIZE = 100
SEED = 50

DATA_PATHS = {
    2023: PROCESSED_DATA_DIR / "ICLR.cc_2023_as_2024_format.json",
    2024: RAW_DATA_DIR / "ICLR.cc_2024.json",
    2025: RAW_DATA_DIR / "ICLR.cc_2025.json",
}

def _normalize_tag(tag: str) -> str:
    if not tag:
        return ""
    return tag if tag.startswith("_") else f"_{tag}"


def _output_paths(year: int, tag: str = ""):
    tag = _normalize_tag(tag)
    suffix = f"_iclr{year}" if year != 2024 else ""
    sample = RESULTS_DIR / "score_change" / f"regex_verdicts{suffix}{tag}.json"
    full = RESULTS_DIR / "score_change" / f"regex_verdicts{suffix}{tag}_full.json"
    return sample, full


def _additions_path(output_file: Path) -> Path:
    stem = output_file.stem
    return output_file.with_name(f"{stem}_new_additions.json")


def _metadata_path(output_file: Path) -> Path:
    stem = output_file.stem
    return output_file.with_name(f"{stem}_meta.json")


def _parse_allowlist(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Allowlist file not found: {p}")

    if p.suffix.lower() == ".json":
        data = load_json(p)
        if isinstance(data, list):
            return {str(x).strip() for x in data if str(x).strip()}
        if isinstance(data, dict):
            out = set()
            for key in ("paper_ids", "allowlist", "ids"):
                vals = data.get(key)
                if isinstance(vals, list):
                    out.update(str(x).strip() for x in vals if str(x).strip())
            if out:
                return out
        raise ValueError(f"Unsupported JSON allowlist format: {p}")

    lines = p.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def _parse_pair_allowlist(path: str | None) -> set[tuple[str, str]] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pair allowlist file not found: {p}")

    pairs: set[tuple[str, str]] = set()
    if p.suffix.lower() == ".json":
        data = load_json(p)
        if not isinstance(data, list):
            raise ValueError(f"Pair allowlist JSON must be a list: {p}")
        for item in data:
            if isinstance(item, dict):
                pid = str(item.get("paper_id", "")).strip()
                rid = str(item.get("reviewer_id", "")).strip()
                if pid and rid:
                    pairs.add((pid, rid))
            elif isinstance(item, str) and "||" in item:
                pid, rid = item.split("||", 1)
                pid = pid.strip()
                rid = rid.strip()
                if pid and rid:
                    pairs.add((pid, rid))
    else:
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "||" in s:
                pid, rid = s.split("||", 1)
            elif "," in s:
                pid, rid = s.split(",", 1)
            else:
                raise ValueError(f"Invalid pair line (expected 'paper_id||reviewer_id' or CSV): {s}")
            pid = pid.strip()
            rid = rid.strip()
            if pid and rid:
                pairs.add((pid, rid))
    return pairs


def _is_official_review(invitations: list) -> bool:
    inv_str = " ".join(str(x) for x in (invitations or []))
    return "Official_Review" in inv_str and "Rebuttal" not in inv_str


def _has_authors(signatures: list) -> bool:
    return any("Authors" in str(s) for s in (signatures or []))


def _same_reviewer(sigs1: list, sigs2: list) -> bool:
    return (sigs1 or []) == (sigs2 or [])


def _collect_subtree(root_id: str, interactions: list) -> list[dict]:
    """Collect all notes in the reply subtree rooted at root_id (BFS), sorted by cdate."""
    children_by_parent: dict[str, list[dict]] = {}
    for n in interactions:
        pid = n.get("replyto")
        if pid:
            children_by_parent.setdefault(pid, []).append(n)

    result = []
    queue = [root_id]
    while queue:
        pid = queue.pop(0)
        for child in children_by_parent.get(pid, []):
            result.append(child)
            queue.append(child["id"])

    result.sort(key=lambda x: x.get("cdate", 0))
    return result


def _get_all_later_notes(data: dict, paper_id: str, reviewer_id: str) -> list[dict]:
    """
    Return all follow-up notes by this reviewer in the reply subtree under
    their Official_Review, posted **after** the first author response.
    Sorted by cdate ascending. Returns [] if the chain is missing.
    """
    entry = data.get(paper_id)
    if not entry or "interactions" not in entry:
        return []
    interactions = sorted(entry["interactions"], key=lambda x: x.get("cdate", 0))
    review = None
    for inv in interactions:
        if not _is_official_review(inv.get("invitations") or []):
            continue
        if reviewer_id not in (inv.get("signatures") or []):
            continue
        review = inv
        break
    if not review:
        return []
    reviewer_sigs = review.get("signatures") or []

    # Walk the full reply subtree under this review
    subtree = _collect_subtree(review["id"], interactions)

    # Find the first author response in the subtree
    first_author_cdate = None
    for n in subtree:
        if _has_authors(n.get("signatures") or []):
            first_author_cdate = n.get("cdate", 0)
            break
    if first_author_cdate is None:
        return []

    # Return reviewer notes in the subtree posted after the first author response
    candidates = [
        n for n in subtree
        if _same_reviewer(n.get("signatures"), reviewer_sigs)
        and n.get("cdate", 0) > first_author_cdate
    ]
    return candidates


def get_later_note_text(data: dict, paper_id: str, reviewer_id: str) -> str | None:
    """
    Return the text of the chosen reviewer follow-up **after** the author's first
    rebuttal (newer in time). Rules:
    1) Prefer increase/decrease over "same": if any follow-up has POSITIVE or
       NEGATIVE, use the most recent such note. This avoids a final "keeping my
       score" (same) overriding an earlier "I decreased/increased" when the
       newer note refers back to that change.
    2) If none has POSITIVE or NEGATIVE: use the most recent note with NEUTRAL,
       or the last note's text if none. Returns None if the reviewer posted no
       follow-up after the rebuttal.
    Used by gemini_verdict_check and extract_regex_verdicts.
    """
    candidates = _get_all_later_notes(data, paper_id, reviewer_id)
    if not candidates:
        return None
    # 1) Newest to oldest: prefer POSITIVE or NEGATIVE (explicit direction)
    for i in range(len(candidates) - 1, -1, -1):
        text = get_text_content(candidates[i])
        if analyze_sentiment(text) in ("POSITIVE", "NEGATIVE"):
            return text
    # 2) Else: most recent note with NEUTRAL, or last note
    for i in range(len(candidates) - 1, -1, -1):
        text = get_text_content(candidates[i])
        if analyze_sentiment(text) == "NEUTRAL":
            return text
    return get_text_content(candidates[-1])


def extract_regex_verdicts(
    data: dict,
    paper_keys: list,
    include_diagnostics: bool = False,
    pair_allowlist: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    For each (paper, reviewer) with [Official_Review → Author Response] and at
    least one reviewer follow-up after the rebuttal, set regex_verdict from
    analyze_sentiment on the note chosen by get_later_note_text. Prefer
    explicit increase/decrease over "same" when any follow-up has it. Verdict:
    increase, decrease, same. No row when: the reviewer left no follow-up after
    the rebuttal, or none of their follow-ups match our score-change regex.
    At most one row per (paper_id, reviewer_id).
    """
    verdict_map = {"POSITIVE": "increase", "NEGATIVE": "decrease", "NEUTRAL": "same"}
    rows = []
    seen = set()

    for paper_id in paper_keys:
        entry = data.get(paper_id)
        if not entry or "interactions" not in entry:
            continue
        interactions = sorted(entry["interactions"], key=lambda x: x.get("cdate", 0))

        for inv in interactions:
            inv_list = inv.get("invitations") or []
            if not _is_official_review(inv_list):
                continue
            review_id = inv.get("id")
            reviewer_sigs = inv.get("signatures") or []
            reviewer_sig = reviewer_sigs[0] if reviewer_sigs else None
            if not reviewer_sig:
                continue

            author_resps = [
                n for n in interactions
                if n.get("replyto") == review_id and _has_authors(n.get("signatures") or [])
            ]
            if not author_resps:
                continue

            text = get_later_note_text(data, paper_id, reviewer_sig)
            if not text:
                continue
            sent, pattern_id = analyze_sentiment_with_reason(text)
            verdict = verdict_map.get(sent) if sent in verdict_map else None

            if verdict is not None:
                key = (paper_id, reviewer_sig)
                if pair_allowlist is not None and key not in pair_allowlist:
                    continue
                if key not in seen:
                    seen.add(key)
                    row = {
                        "paper_id": paper_id,
                        "forum": str((entry or {}).get("forum") or ""),
                        "reviewer_id": reviewer_sig,
                        "regex_verdict": verdict,
                    }
                    if include_diagnostics:
                        row["regex_pattern_id"] = pattern_id
                        row["followup_text_len"] = len(text)
                    rows.append(row)

    return rows


def _print_summary(rows, output_file):
    print(f"Extracted {len(rows)} (paper, reviewer) with regex verdict (increase/decrease/same).")
    by_verdict = {}
    for r in rows:
        v = r["regex_verdict"]
        by_verdict[v] = by_verdict.get(v, 0) + 1
    for v, c in sorted(by_verdict.items()):
        print(f"  {v}: {c}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(rows, output_file)
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract regex-based score-change verdicts")
    parser.add_argument("--full", action="store_true",
                        help="Run on ALL papers (not just the 100-paper sample)")
    parser.add_argument("--year", type=int, default=2024, choices=[2023, 2024, 2025],
                        help="ICLR year to process (default: 2024)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag for output filenames (e.g., v2)")
    parser.add_argument("--include-diagnostics", action="store_true",
                        help="Include regex_pattern_id and text length in output rows.")
    parser.add_argument("--exclude-withdrawn-desk", action="store_true",
                        help="Exclude withdrawn and desk-rejected papers.")
    parser.add_argument("--paper-id-allowlist", type=str, default=None,
                        help="Path to paper-id allowlist (.txt or .json).")
    parser.add_argument("--pair-allowlist", type=str, default=None,
                        help="Path to (paper_id, reviewer_id) allowlist (.txt or .json).")
    args = parser.parse_args()

    dataset_path = DATA_PATHS[args.year]
    output_sample, output_full = _output_paths(args.year, tag=args.tag)
    baseline_sample, baseline_full = _output_paths(args.year, tag="")

    print(f"Loading {dataset_path}...")
    data = load_json(dataset_path)
    keys = list(data.keys())

    paper_allowlist = _parse_allowlist(args.paper_id_allowlist)
    pair_allowlist = _parse_pair_allowlist(args.pair_allowlist)
    if args.exclude_withdrawn_desk:
        before = len(keys)
        keys = [k for k in keys if not is_withdrawn_or_desk_rejected(data.get(k, {}))]
        print(f"Excluded withdrawn/desk-reject papers: {before - len(keys)} removed, {len(keys)} remain.")
    if paper_allowlist is not None:
        before = len(keys)
        keys = [k for k in keys if str(k) in paper_allowlist]
        print(f"Applied paper allowlist: {before} -> {len(keys)} papers.")

    if args.full:
        paper_keys = keys
        output_file = output_full
        print(f"Running on ALL {len(paper_keys)} papers (ICLR {args.year}).")
    else:
        indices = get_sampled_indices(len(keys), sample_size=SAMPLE_SIZE, seed=SEED)
        paper_keys = [keys[i] for i in indices]
        output_file = output_sample
        print(f"Restricting to {len(paper_keys)} papers (ICLR {args.year}, sample_size={SAMPLE_SIZE}, seed={SEED}).")

    rows = extract_regex_verdicts(
        data,
        paper_keys,
        include_diagnostics=args.include_diagnostics,
        pair_allowlist=pair_allowlist,
    )
    _print_summary(rows, output_file)

    metadata = {
        "year": args.year,
        "full": args.full,
        "tag": args.tag,
        "seed": SEED if not args.full else None,
        "sample_size": SAMPLE_SIZE if not args.full else None,
        "exclude_withdrawn_desk": args.exclude_withdrawn_desk,
        "paper_allowlist_path": args.paper_id_allowlist,
        "pair_allowlist_path": args.pair_allowlist,
        "input_dataset": str(dataset_path),
        "n_papers_after_filters": len(paper_keys),
        "n_rows": len(rows),
    }
    meta_path = _metadata_path(output_file)
    write_json(metadata, meta_path)
    print(f"Saved metadata to {meta_path}")

    if _normalize_tag(args.tag):
        baseline_file = baseline_full if args.full else baseline_sample
        if baseline_file.exists():
            baseline_rows = load_json(baseline_file)
            baseline_keys = {
                (r.get("paper_id"), r.get("reviewer_id"))
                for r in baseline_rows
            }
            additions = [
                r for r in rows
                if (r.get("paper_id"), r.get("reviewer_id")) not in baseline_keys
            ]
            additions_path = _additions_path(output_file)
            write_json(additions, additions_path)
            print(f"New additions vs baseline: {len(additions)}")
            print(f"Saved additions to {additions_path}")
        else:
            print(f"Baseline file not found for additions diff: {baseline_file}")


if __name__ == "__main__":
    main()
