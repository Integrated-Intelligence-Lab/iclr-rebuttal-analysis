"""
Per-batch validation gate — go/no-go check after each full-corpus batch.

Run after ``classify download-all --tag full_b{i}`` finishes. Reports:
  - Parse-rate (must be >= 99%).
  - n_factors_present distribution (median should be 5-20, max <= 44).
  - Truncation heuristic: % of raw_response strings ending in non-'}'.
  - Items with hallucinated factor IDs (IDs outside F01..F44).
  - Top-3 errors by category.

Exit code 0 if all gates pass, 1 otherwise. Use it in a shell loop:

    if python -m src.factor_validation.validate_batch --tag full_b1; then
        echo "OK to submit next batch"
    else
        echo "GATE FAILED — investigate before submitting next"
    fi

Usage
-----
    python -m src.factor_validation.validate_batch --tag full_b1
    python -m src.factor_validation.validate_batch --tag full_b1 --strict
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.io import load_json
from core.paths import BATCH_OUTPUT_DIR
from factor_validation.prompt import VALID_FACTOR_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, required=True,
                        help="Batch tag, e.g. full_b1.")
    parser.add_argument("--parse-rate-threshold", type=float, default=0.99,
                        help="Minimum acceptable parse-rate (default 0.99).")
    parser.add_argument("--truncation-threshold", type=float, default=0.50,
                        help="Among items that FAILED to parse, max fraction that look "
                             "truncated (raw_response not ending in '}}' after fence-strip). "
                             "Default 50%%. Above this we assume max_output_tokens too low.")
    parser.add_argument("--strict", action="store_true",
                        help="Treat any warning as failure.")
    args = parser.parse_args()

    merged = BATCH_OUTPUT_DIR / f"factor_validation_{args.tag}_full.json"
    if not merged.exists():
        print(f"FAIL: merged output not found: {merged}")
        print(f"      run: python -m src.factor_validation.classify download-all --tag {args.tag}")
        sys.exit(2)

    results = load_json(merged)
    n = len(results)
    if n == 0:
        print(f"FAIL: {merged.name} is empty")
        sys.exit(2)

    # 1. Parse-rate.
    n_parsed = sum(1 for r in results if r.get("parsed") is not None and not r.get("error"))
    parse_rate = n_parsed / n

    # 2. n_factors_present distribution + factor-ID validation.
    n_factors_list: list[int] = []
    n_with_hallucination = 0
    factor_id_counts: Counter[str] = Counter()
    for r in results:
        parsed = r.get("parsed")
        if parsed is None:
            continue
        if isinstance(parsed, dict):
            factors = parsed.get("factors_present") or []
        elif isinstance(parsed, list):
            factors = parsed
        else:
            factors = []
        ids = [f.get("factor_id") for f in factors if isinstance(f, dict)]
        valid_ids = {fid for fid in ids if fid in VALID_FACTOR_IDS}
        hallucinated = [fid for fid in ids if fid and fid not in VALID_FACTOR_IDS]
        n_factors_list.append(len(valid_ids))
        for fid in valid_ids:
            factor_id_counts[fid] += 1
        if hallucinated:
            n_with_hallucination += 1

    if n_factors_list:
        median_factors = statistics.median(n_factors_list)
        max_factors = max(n_factors_list)
        min_factors = min(n_factors_list)
    else:
        median_factors = max_factors = min_factors = 0

    # 3. Truncation heuristic. Only meaningful for items that FAILED to parse:
    # successful ones are valid JSON by definition. We also strip markdown code
    # fences (```json ... ```) the way classify._parse_json_response does, so a
    # well-formed fenced response is not flagged as truncated.
    def _strip_for_truncation_check(raw: str) -> str:
        s = (raw or "").strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else s
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3]
        return s.strip()

    n_failed = n - n_parsed
    n_truncated_among_failed = sum(
        1 for r in results
        if (r.get("parsed") is None or r.get("error"))
        and r.get("raw_response")
        and not _strip_for_truncation_check(r.get("raw_response") or "").endswith("}")
    )
    trunc_rate = (n_truncated_among_failed / n_failed) if n_failed else 0.0

    # 4. Error breakdown.
    err_counts = Counter()
    for r in results:
        e = r.get("error")
        if e:
            err_counts[str(e)[:60]] += 1

    print(f"Validation report for tag '{args.tag}'")
    print("=" * 60)
    print(f"  total items                : {n}")
    print(f"  parsed (no error)          : {n_parsed} ({parse_rate * 100:.2f}%)")
    print(f"  n_factors_present  min/med/max : {min_factors} / {median_factors:.1f} / {max_factors}")
    print(f"  hallucinated factor IDs    : {n_with_hallucination}")
    print(f"  failed items               : {n_failed}")
    if n_failed:
        print(f"  failed AND look truncated  : {n_truncated_among_failed} ({trunc_rate * 100:.1f}% of failures)")
    print(f"  unique factor IDs observed : {len(factor_id_counts)} of 44")
    if err_counts:
        print(f"  top errors:")
        for err, c in err_counts.most_common(5):
            print(f"    {c:>5}  {err}")

    # Gate evaluation.
    failures: list[str] = []
    warnings: list[str] = []

    if parse_rate < args.parse_rate_threshold:
        failures.append(
            f"parse-rate {parse_rate * 100:.2f}% < threshold "
            f"{args.parse_rate_threshold * 100:.2f}%"
        )
    # Truncation gate: only fires if the truncation rate AMONG FAILURES exceeds
    # the threshold AND there were enough failures to be meaningful.
    if n_failed >= 5 and trunc_rate > args.truncation_threshold:
        failures.append(
            f"of {n_failed} parse failures, {trunc_rate * 100:.1f}% look truncated "
            f"(> threshold {args.truncation_threshold * 100:.1f}%) — consider raising max_output_tokens"
        )
    if max_factors > 44:
        failures.append(f"max n_factors_present={max_factors} (> 44 — schema violation)")
    if median_factors < 1:
        failures.append(f"median n_factors_present={median_factors} (suspiciously low)")
    if median_factors > 30:
        warnings.append(f"median n_factors_present={median_factors} (suspiciously high)")
    if n_with_hallucination > 0:
        warnings.append(f"{n_with_hallucination} item(s) with hallucinated factor IDs (will be filtered downstream)")
    if len(factor_id_counts) < 30:
        warnings.append(f"only {len(factor_id_counts)}/44 factor IDs observed (some never selected)")

    print("=" * 60)
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
    if warnings:
        print("WARN:")
        for w in warnings:
            print(f"  - {w}")
    if not failures and not warnings:
        print("PASS: all checks green.")

    if failures or (args.strict and warnings):
        print("\nGate failed. Suggested next steps:")
        print(f"  python -m src.factor_validation.classify retry-failed-sync --tag {args.tag}")
        print(f"  python -m src.factor_validation.validate_batch --tag {args.tag}")
        sys.exit(1)
    print(f"\nGate PASSED — safe to submit next batch.")
    sys.exit(0)


if __name__ == "__main__":
    main()
