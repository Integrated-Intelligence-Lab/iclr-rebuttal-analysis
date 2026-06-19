"""
Double-check regex verdicts with Gemini: did the reviewer increase, decrease, or
keep the same their score vs their first review?

Loads regex_verdicts.json and the reviewer's later-note text, calls Gemini
(gemini-2.0-flash or GEMINI_MODEL), extracts verdict via pattern matching.
Stores raw responses and parsed verdict in gemini_verdicts.json. Uses
GEMINI_API_KEY from .env.

Changes:
- Added: gemini_verdict_check; uses score_change.extract_regex_verdicts
  get_later_note_text, core.paths, core.io.
"""

import os
import re
import sys
import time
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from dotenv import load_dotenv
load_dotenv()

from core.io import load_json, write_json
from core.paths import RAW_DATA_DIR, RESULTS_DIR
from score_change.extract_regex_verdicts import get_later_note_text

# --- Config ---
REGEX_VERDICTS_FILE = RESULTS_DIR / "score_change" / "regex_verdicts.json"
ORIGINAL_DATASET = RAW_DATA_DIR / "ICLR.cc_2024.json"
OUTPUT_FILE = RESULTS_DIR / "score_change" / "gemini_verdicts.json"
# Model: canonical project default gemini-3-flash; override via GEMINI_MODEL in .env.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")
MAX_TEXT_LEN = 60000
REQUEST_DELAY_S = 0.5
# Run on first N items when set; None = all
LIMIT = None

INSTRUCTION = """You are analyzing a reviewer's follow-up comment after the authors' rebuttal.

TASK: Did the reviewer indicate they INCREASED, DECREASED, or kept the SAME their score compared to their initial review?

REPLY WITH EXACTLY ONE WORD: increased, decreased, or same.

REVIEWER'S FOLLOW-UP TEXT:
"""
# Pattern matching (fallback if the model does not follow instructions)
_PATTERN = re.compile(
    r"\b(increased?|decreased?|same|unchanged)\b",
    re.IGNORECASE,
)
_MAP = {
    "increase": "increase",
    "increased": "increase",
    "decrease": "decrease",
    "decreased": "decrease",
    "same": "same",
    "unchanged": "same",
}


def _parse_verdict(raw: str) -> tuple[str | None, bool]:
    """Extract increase/decrease/same from raw text. Returns (verdict, pattern_matched)."""
    if not raw or not isinstance(raw, str):
        return None, False
    m = _PATTERN.search(raw)
    if not m:
        return None, False
    word = m.group(1).lower()
    if word in _MAP:
        return _MAP[word], True
    return None, False


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set in .env. Exiting.")
        return

    try:
        import google.generativeai as genai
    except ImportError:
        print("Install: pip install google-generativeai")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    if not REGEX_VERDICTS_FILE.exists():
        print(f"Run extract_regex_verdicts first. Missing: {REGEX_VERDICTS_FILE}")
        return

    verdicts = load_json(REGEX_VERDICTS_FILE)
    if LIMIT is not None:
        verdicts = verdicts[:LIMIT]
        print(f"Limited to first {LIMIT} items.")
    print(f"Loading {ORIGINAL_DATASET}...")
    data = load_json(ORIGINAL_DATASET)

    results = []
    for i, v in enumerate(verdicts):
        paper_id = v["paper_id"]
        reviewer_id = v["reviewer_id"]
        regex = v["regex_verdict"]
        text = get_later_note_text(data, paper_id, reviewer_id)
        if not text or not text.strip():
            results.append({
                "paper_id": paper_id,
                "reviewer_id": reviewer_id,
                "regex_verdict": regex,
                "gemini_raw": None,
                "gemini_verdict": None,
                "pattern_matched": False,
                "verdict": None,
                "text_raw": text,
            })
            continue
        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN] + "\n\n[truncated]"
        prompt = INSTRUCTION + text

        raw = None
        try:
            resp = model.generate_content(prompt)
            raw = getattr(resp, "text", None) or ""
        except Exception as e:
            raw = f"[Error: {e!r}]"

        parsed, matched = _parse_verdict(raw or "")
        results.append({
            "paper_id": paper_id,
            "reviewer_id": reviewer_id,
            "regex_verdict": regex,
            "gemini_raw": raw,
            "gemini_verdict": parsed,
            "pattern_matched": matched,
            "verdict": (parsed == regex) if (parsed is not None) else None,
            "text_raw": text,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(verdicts)}")
        time.sleep(REQUEST_DELAY_S)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json(results, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}.")

    # Summary (regex_verdict vs gemini_verdict)
    n = len(results)
    matched = sum(1 for r in results if r["pattern_matched"])
    with_parsed = [r for r in results if r["gemini_verdict"] is not None]
    with_parsed_n = len(with_parsed)
    agree = [r for r in with_parsed if r["regex_verdict"] == r["gemini_verdict"]]
    disagree = [r for r in with_parsed if r["regex_verdict"] != r["gemini_verdict"]]

    print("\n--- Summary ---")
    print(f"  Total: {n}")
    print(f"  pattern_matched: {matched}/{n}")
    print(f"  gemini_verdict parsed: {with_parsed_n}/{n}")
    if with_parsed_n:
        pct = 100 * len(agree) / with_parsed_n
        print(f"  regex vs gemini agreement: {len(agree)}/{with_parsed_n} ({pct:.1f}%)  [compares regex_verdict to gemini_verdict]")
        if disagree:
            print(f"  Disagreements: {len(disagree)}")
            print("  Disagreeing rows (paper_id, reviewer_id, regex_verdict -> gemini_verdict):")
            for r in disagree:
                rid = (r.get("reviewer_id") or "").split("/")[-1] or r.get("reviewer_id", "")
                print(f"    {r['paper_id']}  {rid}: {r['regex_verdict']} -> {r['gemini_verdict']}")


if __name__ == "__main__":
    main()
