"""
Analyse Gemini factor-classification results.

Joins the batch output (per-item list of factor_ids) with the per-item metadata
(tier, verdict_group) saved by classify_factors_batch.py::cmd_submit and
produces:

  1. Per-group factor prevalence  (count and rate).
  2. Per-factor discriminative test:
       verdict_group = increase vs. same_decrease
       chi-squared + Fisher's exact; odds ratio with Wilson 95% CI.
  3. Permutation null: shuffle verdict_group labels K times, recompute the
     mean OR per factor, report two-sided empirical p-value.
  4. Correspondence check: for each factor, is its observed direction of
     association (higher in `increase` vs `same_decrease`) consistent with
     the polarity declared in factor_codebook.json?

Outputs:
  - {out_dir}/{tag}_factor_prevalence.csv
  - {out_dir}/{tag}_factor_tests.csv
  - {out_dir}/{tag}_report.md

Usage
-----
    python src/factor_validation/analyze.py --tag test1_same1k
    python src/factor_validation/analyze.py --tag test2_fresh1k --n-permutations 2000
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.io import load_json, write_file
from core.paths import BATCH_OUTPUT_DIR, RESULTS_DIR
from factor_validation.prompt import VALID_FACTOR_IDS


CODEBOOK_PATH = Path(__file__).resolve().parent / "codebook.json"
DEFAULT_OUT_DIR = RESULTS_DIR / "factor_validation"
FILE_PREFIX = "factor_validation"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_inputs(tag: str) -> tuple[list[dict], list[dict], dict]:
    results_path = BATCH_OUTPUT_DIR / f"{FILE_PREFIX}_{tag}_full.json"
    items_path = BATCH_OUTPUT_DIR / f"{FILE_PREFIX}_{tag}_items.json"
    codebook = load_json(CODEBOOK_PATH)
    if not results_path.exists():
        raise SystemExit(f"Missing results: {results_path}. Run download-all first.")
    if not items_path.exists():
        raise SystemExit(f"Missing item metadata: {items_path}. Re-run submit to regenerate.")
    return load_json(results_path), load_json(items_path), codebook


_VERDICT_NORM = {
    "increase": "increase",
    "same+decrease": "same_decrease",
    "same_decrease": "same_decrease",
    "same/decrease": "same_decrease",
}


def _normalise_verdict(vg: str | None) -> str | None:
    """Canonicalise verdict_group strings. The Test 1 source file mixes
    `same+decrease` and `same_decrease` across fields; we collapse to `same_decrease`.
    """
    if not vg:
        return None
    return _VERDICT_NORM.get(vg, vg)


def join_results_with_meta(results: list[dict], items: list[dict]) -> list[dict]:
    """Attach (tier, verdict_group) metadata to each parsed result.

    Since we removed the response_schema enum guard on the batch side, we
    validate factor_id values here and drop any that aren't in F01..F44.
    Counts are tracked via the returned `_hallucinated` counter on each row.
    """
    by_key: dict[str, dict] = {}
    for it in items:
        key = f"{it['forum']}::{it['reviewer_id']}"
        by_key[key] = it
    joined = []
    for r in results:
        meta = by_key.get(str(r.get("custom_id") or ""))
        if not meta:
            continue
        parsed = r.get("parsed")
        # Accept both shapes the model produces: {"factors_present": [...]}
        # or a bare list [...]
        if isinstance(parsed, dict):
            factors = parsed.get("factors_present") or []
        elif isinstance(parsed, list):
            factors = parsed
        else:
            factors = []
        raw_ids = [f.get("factor_id") for f in factors if isinstance(f, dict)]
        valid_ids = sorted({fid for fid in raw_ids if fid in VALID_FACTOR_IDS})
        hallucinated = sorted({fid for fid in raw_ids
                               if fid and fid not in VALID_FACTOR_IDS})
        joined.append({
            "paper_id": meta.get("paper_id"),
            "forum": meta.get("forum"),
            "reviewer_id": meta.get("reviewer_id"),
            "tier": meta.get("tier"),
            "verdict_group": _normalise_verdict(meta.get("verdict_group")),
            "group_key": meta.get("group_key"),
            "factor_ids": valid_ids,
            "n_factors": len(valid_ids),
            "hallucinated_ids": hallucinated,
            "parse_error": r.get("error"),
        })
    return joined


# ---------------------------------------------------------------------------
# Statistics (pure-python, no SciPy required)
# ---------------------------------------------------------------------------

def _chi_sq_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Pearson chi-squared with continuity correction on a 2x2. Returns (chi2, p)."""
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if 0 in (row1, row2, col1, col2):
        return 0.0, 1.0
    expected = [(row1 * col1) / n, (row1 * col2) / n,
                (row2 * col1) / n, (row2 * col2) / n]
    observed = [a, b, c, d]
    chi2 = 0.0
    for o, e in zip(observed, expected):
        diff = abs(o - e) - 0.5
        if diff < 0:
            diff = 0
        chi2 += (diff ** 2) / e
    # survival of chi2 with df=1 -> erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
    return chi2, p


def _odds_ratio(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Odds ratio with Wolf log-CI. Haldane-Anscombe correction if any zero."""
    a_, b_, c_, d_ = a, b, c, d
    if 0 in (a_, b_, c_, d_):
        a_ += 0.5; b_ += 0.5; c_ += 0.5; d_ += 0.5
    or_ = (a_ * d_) / (b_ * c_)
    se_log = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    lo = math.exp(math.log(or_) - 1.96 * se_log)
    hi = math.exp(math.log(or_) + 1.96 * se_log)
    return or_, lo, hi


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

GROUPS = ["low_increase", "low_same_decrease",
          "mid_increase", "mid_same_decrease",
          "high_increase", "high_same_decrease"]


def compute_prevalence(joined: list[dict]) -> dict:
    """Count per-factor presence broken down by group_key and by verdict_group."""
    n_by_group = defaultdict(int)
    n_by_verdict = defaultdict(int)
    for j in joined:
        if j["group_key"]:
            n_by_group[j["group_key"]] += 1
        if j["verdict_group"]:
            n_by_verdict[j["verdict_group"]] += 1

    cnt_by_group: dict[tuple[str, str], int] = defaultdict(int)
    cnt_by_verdict: dict[tuple[str, str], int] = defaultdict(int)
    for j in joined:
        for fid in j["factor_ids"]:
            cnt_by_group[(fid, j["group_key"])] += 1
            cnt_by_verdict[(fid, j["verdict_group"])] += 1

    return {
        "n_by_group": dict(n_by_group),
        "n_by_verdict": dict(n_by_verdict),
        "cnt_by_group": cnt_by_group,
        "cnt_by_verdict": cnt_by_verdict,
    }


def permutation_null_or(joined: list[dict], factor_ids: list[str],
                        n_permutations: int, seed: int) -> dict[str, float]:
    """Empirical two-sided p-value for each factor's OR (increase vs same_decrease).

    Shuffles the verdict_group label across items (breaking any true association)
    and recomputes the OR. Reports fraction of permuted ORs at least as extreme
    (in log-OR magnitude) as observed.
    """
    rng = random.Random(seed)
    labels = [j["verdict_group"] for j in joined]
    presence: dict[str, list[int]] = {
        fid: [1 if fid in set(j["factor_ids"]) else 0 for j in joined]
        for fid in factor_ids
    }

    def _contingency(pres: list[int], labs: list[str]) -> tuple[int, int, int, int]:
        a = b = c = d = 0
        for p, l in zip(pres, labs):
            if l == "increase":
                if p: a += 1
                else: b += 1
            else:
                if p: c += 1
                else: d += 1
        return a, b, c, d

    observed_log_or = {}
    for fid in factor_ids:
        a, b, c, d = _contingency(presence[fid], labels)
        or_, _, _ = _odds_ratio(a, b, c, d)
        observed_log_or[fid] = math.log(or_) if or_ > 0 else 0.0

    extreme_counts = {fid: 0 for fid in factor_ids}
    shuffled = labels[:]
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        for fid in factor_ids:
            a, b, c, d = _contingency(presence[fid], shuffled)
            or_, _, _ = _odds_ratio(a, b, c, d)
            log_or = math.log(or_) if or_ > 0 else 0.0
            if abs(log_or) >= abs(observed_log_or[fid]):
                extreme_counts[fid] += 1

    return {
        fid: (extreme_counts[fid] + 1) / (n_permutations + 1)
        for fid in factor_ids
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_prevalence_csv(path: Path, prev: dict, factor_meta: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = (["factor_id", "polarity", "label"]
            + [f"n_{g}" for g in GROUPS]
            + [f"rate_{g}" for g in GROUPS]
            + ["n_increase", "n_same_decrease",
               "rate_increase", "rate_same_decrease"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for fid in sorted(factor_meta):
            meta = factor_meta[fid]
            row = [fid, meta.get("polarity"), meta.get("label", "")]
            for g in GROUPS:
                k = prev["cnt_by_group"].get((fid, g), 0)
                row.append(k)
            for g in GROUPS:
                k = prev["cnt_by_group"].get((fid, g), 0)
                n = prev["n_by_group"].get(g, 0)
                row.append(f"{(k / n) if n else 0.0:.4f}")
            for v in ("increase", "same_decrease"):
                k = prev["cnt_by_verdict"].get((fid, v), 0)
                row.append(k)
            for v in ("increase", "same_decrease"):
                k = prev["cnt_by_verdict"].get((fid, v), 0)
                n = prev["n_by_verdict"].get(v, 0)
                row.append(f"{(k / n) if n else 0.0:.4f}")
            writer.writerow(row)


def write_tests_csv(path: Path, tests: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["factor_id", "polarity", "label",
            "a_inc_pres", "b_inc_abs", "c_same_pres", "d_same_abs",
            "odds_ratio", "or_ci_lo", "or_ci_hi",
            "chi2", "chi2_p", "perm_p",
            "observed_direction", "polarity_match"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for t in tests:
            writer.writerow([t[k] for k in cols])


def render_report(tag: str, tests: list[dict], n_total: int,
                  n_permutations: int) -> str:
    total_pos = sum(1 for t in tests if t["polarity"] == "positive")
    total_neg = sum(1 for t in tests if t["polarity"] == "negative")

    match_pos = sum(1 for t in tests
                    if t["polarity"] == "positive" and t["polarity_match"] == "yes")
    match_neg = sum(1 for t in tests
                    if t["polarity"] == "negative" and t["polarity_match"] == "yes")

    sig_perm = sum(1 for t in tests if t["perm_p"] < 0.05)
    bonf = 0.05 / max(1, len(tests))
    sig_bonf = sum(1 for t in tests if t["perm_p"] < bonf)

    lines = []
    lines.append(f"# Factor-classification analysis: {tag}\n")
    lines.append(f"- Items analysed: {n_total}")
    lines.append(f"- Factors: {len(tests)} ({total_pos} positive, {total_neg} negative)")
    lines.append(f"- Permutations: {n_permutations}")
    lines.append(f"- Bonferroni threshold (alpha=0.05): {bonf:.4f}")
    lines.append("")

    lines.append("## Polarity correspondence")
    lines.append("")
    lines.append("For each factor, we check whether its observed direction "
                 "(more prevalent in `increase` vs `same_decrease`) matches the "
                 "polarity declared during taxonomy induction.")
    lines.append("")
    lines.append(f"- Positive factors with expected direction: **{match_pos}/{total_pos}**")
    lines.append(f"- Negative factors with expected direction: **{match_neg}/{total_neg}**")
    lines.append(f"- Overall match rate: **{(match_pos + match_neg)}/{len(tests)}** "
                 f"({100 * (match_pos + match_neg) / max(1, len(tests)):.1f}%)")
    lines.append("")

    lines.append("## Statistical significance")
    lines.append(f"- Factors with permutation p < 0.05 (uncorrected): **{sig_perm}/{len(tests)}**")
    lines.append(f"- Factors surviving Bonferroni: **{sig_bonf}/{len(tests)}**")
    lines.append("")

    lines.append("## Per-factor test table")
    lines.append("")
    lines.append("| ID | Pol | OR | 95% CI | chi2 p | perm p | Match? | Label |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in sorted(tests, key=lambda x: (x["polarity"], x["factor_id"])):
        ci = f"[{t['or_ci_lo']:.2f}, {t['or_ci_hi']:.2f}]"
        lines.append(f"| {t['factor_id']} | {t['polarity'][:3]} | "
                     f"{t['odds_ratio']:.2f} | {ci} | "
                     f"{t['chi2_p']:.3g} | {t['perm_p']:.3g} | "
                     f"{t['polarity_match']} | {t['label'][:60]} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", type=str, required=True,
                        help="Tag used when calling classify_factors_batch.py (e.g. test1_same1k).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()

    results, items, codebook = load_inputs(args.tag)
    factor_meta = {f["id"]: f for f in codebook["factors"]}
    factor_ids = [f["id"] for f in codebook["factors"]]

    joined = join_results_with_meta(results, items)
    print(f"Loaded {len(results)} results, joined with {len(joined)} items.")

    joined = [j for j in joined if j["verdict_group"] in ("increase", "same_decrease")]
    print(f"After filtering to known verdict_groups: {len(joined)} items.")

    prev = compute_prevalence(joined)
    print(f"Prevalence computed. n_by_verdict: {prev['n_by_verdict']}")

    tests = []
    for fid in factor_ids:
        meta = factor_meta[fid]
        a = prev["cnt_by_verdict"].get((fid, "increase"), 0)
        b = prev["n_by_verdict"].get("increase", 0) - a
        c = prev["cnt_by_verdict"].get((fid, "same_decrease"), 0)
        d = prev["n_by_verdict"].get("same_decrease", 0) - c
        or_, lo, hi = _odds_ratio(a, b, c, d)
        chi2, p = _chi_sq_2x2(a, b, c, d)
        obs_dir = "increase" if or_ > 1 else ("same_decrease" if or_ < 1 else "flat")
        expected = "increase" if meta.get("polarity") == "positive" else "same_decrease"
        match = "yes" if obs_dir == expected else "no"
        tests.append({
            "factor_id": fid,
            "polarity": meta.get("polarity"),
            "label": meta.get("label", ""),
            "a_inc_pres": a, "b_inc_abs": b,
            "c_same_pres": c, "d_same_abs": d,
            "odds_ratio": or_, "or_ci_lo": lo, "or_ci_hi": hi,
            "chi2": chi2, "chi2_p": p,
            "perm_p": None,
            "observed_direction": obs_dir,
            "polarity_match": match,
        })

    print(f"Running {args.n_permutations} permutations ...")
    perm_p = permutation_null_or(joined, factor_ids, args.n_permutations, args.seed)
    for t in tests:
        t["perm_p"] = perm_p[t["factor_id"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prev_csv = args.out_dir / f"{args.tag}_factor_prevalence.csv"
    tests_csv = args.out_dir / f"{args.tag}_factor_tests.csv"
    report_md = args.out_dir / f"{args.tag}_report.md"

    write_prevalence_csv(prev_csv, prev, factor_meta)
    write_tests_csv(tests_csv, tests)
    report = render_report(args.tag, tests,
                           n_total=len(joined),
                           n_permutations=args.n_permutations)
    write_file(report, report_md)

    print(f"\nWrote:")
    print(f"  {prev_csv}")
    print(f"  {tests_csv}")
    print(f"  {report_md}")


if __name__ == "__main__":
    main()
