"""
Offline salvage pass for Gemini responses that failed JSON parsing.

Three strategies, applied in order per response:

  1. Fence-strip + brace-extract + bad-backslash escape + json.loads.
     Handles the common ```json ... ``` wrapper, missing outer `{`, and
     unescaped LaTeX backslashes (\\log, \\hat, \\epsilon, \\u2265, \\').
  2. Best-effort truncation repair via closing unclosed brackets.
  3. Regex fallback: extract FNN mentions from prose (for responses where
     the model abandoned the JSON protocol and wrote explanatory text).

Each recovered record is tagged with `recovered_by` so the audit trail
is preserved: "salvage_json" (valid JSON after escape fix), "salvage_repair"
(required bracket closing), or "salvage_regex" (prose fallback, no evidence
quotes available).

Usage:
    python src/factor_validation/salvage.py --tag test1_same1k
    python src/factor_validation/salvage.py --tag test1_same1k --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.io import load_json, write_json
from core.paths import BATCH_OUTPUT_DIR
from factor_validation.prompt import VALID_FACTOR_IDS


VALID_JSON_ESCAPE_CHARS = set('"\\/bfnrtu')

# Matches any factor id F01..F44 as a whole token (not a substring of a bigger id).
FACTOR_ID_RE = re.compile(r"\b(F(?:0[1-9]|[1-3][0-9]|4[0-4]))\b")


# ---------------------------------------------------------------------------
# Strategy 1: lenient JSON parse
# ---------------------------------------------------------------------------

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _extract_brace_body(s: str) -> str:
    """Return content from the first `{` (or prepend `{` if absent) to the matching `}`.

    If the string begins with `"factors_present":` we wrap it in braces. If it
    contains an outer `{...}` we trim leading/trailing garbage. If neither,
    we return the input unchanged for the caller to fall through.
    """
    s = s.strip()
    if s.startswith('"factors_present"'):
        return "{" + s + "}"
    first = s.find("{")
    if first == -1:
        return s
    last = s.rfind("}")
    if last <= first:
        return s[first:]
    return s[first:last + 1]


def _escape_bad_backslashes(s: str) -> str:
    """Replace `\\X` with `\\\\X` when X is not a valid JSON escape character.

    JSON allows: \\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX. Anything else
    (\\log, \\hat, \\epsilon, \\', \\ ) is invalid and we pre-escape it.
    For \\u we additionally require 4 hex digits to follow, else we escape.
    """
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue
        nxt = s[i + 1]
        if nxt == "u":
            # Valid only if followed by exactly 4 hex digits.
            hex4 = s[i + 2:i + 6]
            if len(hex4) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex4):
                out.append(s[i:i + 6])
                i += 6
            else:
                out.append("\\\\")
                i += 1
            continue
        if nxt in VALID_JSON_ESCAPE_CHARS:
            out.append(s[i:i + 2])
            i += 2
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


def _close_unclosed_brackets(s: str) -> str:
    """Append closing brackets if truncation left them open. String-aware."""
    depth_obj = depth_arr = 0
    in_str = False
    esc = False
    for ch in s:
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
    s = s.rstrip().rstrip(",")
    return s + ("]" * max(0, depth_arr)) + ("}" * max(0, depth_obj))


def salvage_json(raw: str) -> tuple[dict | None, str | None]:
    """Return (parsed, recovered_by_label) or (None, None) if all strategies fail."""
    if not raw or not raw.strip():
        return None, None
    s = _strip_fences(raw)
    s = _extract_brace_body(s)

    # First attempt: as-is after fence/brace cleanup.
    try:
        parsed = json.loads(s)
        if _valid_shape(parsed):
            return parsed, "salvage_trim"
    except json.JSONDecodeError:
        pass

    # Second: escape bad backslashes.
    s2 = _escape_bad_backslashes(s)
    try:
        parsed = json.loads(s2)
        if _valid_shape(parsed):
            return parsed, "salvage_json"
    except json.JSONDecodeError:
        pass

    # Third: close unclosed brackets after escaping.
    s3 = _close_unclosed_brackets(s2)
    try:
        parsed = json.loads(s3)
        if _valid_shape(parsed):
            return parsed, "salvage_repair"
    except json.JSONDecodeError:
        pass

    return None, None


def _valid_shape(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    fp = obj.get("factors_present")
    return isinstance(fp, list)


# ---------------------------------------------------------------------------
# Strategy 2: regex extraction from prose
# ---------------------------------------------------------------------------

def salvage_regex(raw: str) -> dict | None:
    """Extract F01..F44 mentions from prose as a last resort.

    We return {factors_present: [{factor_id, evidence_quote: ""}, ...]} with
    empty evidence quotes because we can't reliably associate one.
    """
    if not raw:
        return None
    ids = sorted(set(FACTOR_ID_RE.findall(raw)))
    ids = [fid for fid in ids if fid in VALID_FACTOR_IDS]
    if not ids:
        return None
    return {
        "factors_present": [{"factor_id": fid, "evidence_quote": ""} for fid in ids]
    }


# ---------------------------------------------------------------------------
# Apply to merged results
# ---------------------------------------------------------------------------

def salvage_file(merged_file: Path, dry_run: bool = False) -> dict:
    results = load_json(merged_file)
    failed_idx = [i for i, r in enumerate(results)
                  if r.get("error") or r.get("parsed") is None]
    stats = {"total_failed": len(failed_idx),
             "salvage_trim": 0, "salvage_json": 0, "salvage_repair": 0,
             "salvage_regex": 0, "unrecoverable": 0}
    for i in failed_idx:
        raw = results[i].get("raw_response") or ""
        parsed, label = salvage_json(raw)
        if parsed is not None:
            stats[label] += 1
            if not dry_run:
                results[i]["parsed"] = parsed
                results[i]["error"] = None
                results[i]["recovered_by"] = label
            continue
        regex_parsed = salvage_regex(raw)
        if regex_parsed is not None:
            stats["salvage_regex"] += 1
            if not dry_run:
                results[i]["parsed"] = regex_parsed
                results[i]["error"] = None
                results[i]["recovered_by"] = "salvage_regex"
            continue
        stats["unrecoverable"] += 1

    if not dry_run:
        write_json(results, merged_file)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be recovered without writing.")
    args = parser.parse_args()

    merged_file = BATCH_OUTPUT_DIR / f"factor_validation_{args.tag}_full.json"
    if not merged_file.exists():
        print(f"No merged output at {merged_file}. Run download-all first.")
        sys.exit(1)

    print(f"Salvaging {merged_file.name} (dry-run={args.dry_run}) ...")
    stats = salvage_file(merged_file, dry_run=args.dry_run)
    print("Results:")
    for k, v in stats.items():
        print(f"  {k:<18}  {v}")
    recovered = sum(v for k, v in stats.items()
                    if k.startswith("salvage_"))
    print(f"  ----")
    print(f"  recovered         {recovered}/{stats['total_failed']}")


if __name__ == "__main__":
    main()
