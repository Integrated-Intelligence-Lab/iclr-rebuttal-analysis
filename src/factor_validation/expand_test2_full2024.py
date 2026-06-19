"""Expand Test 2 to the full classified ICLR 2024 cohort.

Recomputes per-factor odds ratios, Wolf log-CIs, chi-squared p-values, and
two-sided permutation p-values (Bonferroni-significant at alpha = 0.05/44)
on the full set of ICLR 2024 rebuttals classified under the 44-factor
codebook (n ~ 5,710), mirroring the methodology of
src/factor_validation/analyze.py used for the original n=952 Test 2.

This script is read-only with respect to the paper text. It writes:
  data/results/factor_validation/test2_full2024_factor_tests.csv
  data/results/factor_validation/test2_full2024_comparison.csv
  data/results/factor_validation/test2_full2024_report.md

The comparison CSV places original n=952 ORs alongside the new n~5,710 ORs.
The report.md summarises differences and counts of doubly-Bonferroni-
significant factors under the new test.

Run:
    python src/factor_validation/expand_test2_full2024.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_2024 = REPO_ROOT / "data" / "results" / "32_factor_taxonomy" / "interaction_factor_labels_2024.csv"
ORIG_TEST1 = REPO_ROOT / "data" / "results" / "factor_validation" / "test1_same1k_factor_tests.csv"
ORIG_TEST2 = REPO_ROOT / "data" / "results" / "factor_validation" / "test2_fresh1k_factor_tests.csv"
OUT_DIR = REPO_ROOT / "data" / "results" / "factor_validation"

ALPHA_BONF = 0.05 / 44
N_PERMUTATIONS = 2000
SEED = 50


def chi_sq_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
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
    p = math.erfc(math.sqrt(chi2 / 2.0)) if chi2 > 0 else 1.0
    return chi2, p


def odds_ratio(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    a_, b_, c_, d_ = a, b, c, d
    if 0 in (a_, b_, c_, d_):
        a_ += 0.5; b_ += 0.5; c_ += 0.5; d_ += 0.5
    or_ = (a_ * d_) / (b_ * c_)
    se_log = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    lo = math.exp(math.log(or_) - 1.96 * se_log)
    hi = math.exp(math.log(or_) + 1.96 * se_log)
    return or_, lo, hi


def _contingency_log_or(pres: list[int], labs: list[str]) -> float:
    a = b = c = d = 0
    for p, l in zip(pres, labs):
        if l == "increase":
            if p: a += 1
            else: b += 1
        else:
            if p: c += 1
            else: d += 1
    or_, _, _ = odds_ratio(a, b, c, d)
    return math.log(or_) if or_ > 0 else 0.0


def permutation_null_or(presence: dict[str, list[int]], labels: list[str],
                        factor_ids: list[str], n_perm: int, seed: int) -> dict[str, float]:
    """Naive permutation: shuffle row-level labels freely. Treats each row as
    independent. Underestimates p-values when rows are clustered within paper.
    """
    rng = random.Random(seed)
    obs = {fid: _contingency_log_or(presence[fid], labels) for fid in factor_ids}
    extreme = {fid: 0 for fid in factor_ids}
    shuffled = labels[:]
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        for fid in factor_ids:
            log_or = _contingency_log_or(presence[fid], shuffled)
            if abs(log_or) >= abs(obs[fid]):
                extreme[fid] += 1
    return {fid: (extreme[fid] + 1) / (n_perm + 1) for fid in factor_ids}


def cluster_permutation_null_or(presence: dict[str, list[int]], labels: list[str],
                                paper_keys: list[str], factor_ids: list[str],
                                n_perm: int, seed: int) -> dict[str, float]:
    """Paper-clustered (block) permutation: shuffle outcome BLOCKS by paper.

    For each paper, the block of per-reviewer outcomes is swapped with the block
    of another paper. When block sizes differ, mod indexing reuses the swap
    paper's outcomes. This preserves within-paper outcome correlation while
    breaking the link between a paper's factor profile and its outcome,
    which is the right exchangeable null when reviews within a paper are not
    independent. Effective sample size is the number of unique papers, not
    the number of rows.
    """
    rng = random.Random(seed)
    paper_idx: dict[str, list[int]] = {}
    for i, p in enumerate(paper_keys):
        paper_idx.setdefault(p, []).append(i)
    papers = list(paper_idx.keys())
    paper_blocks = {p: [labels[i] for i in paper_idx[p]] for p in papers}

    obs = {fid: _contingency_log_or(presence[fid], labels) for fid in factor_ids}
    extreme = {fid: 0 for fid in factor_ids}
    shuffled = labels[:]
    for _ in range(n_perm):
        perm = papers[:]
        rng.shuffle(perm)
        for orig, swap in zip(papers, perm):
            orig_indices = paper_idx[orig]
            swap_block = paper_blocks[swap]
            ks = len(swap_block)
            for j, row_i in enumerate(orig_indices):
                shuffled[row_i] = swap_block[j % ks]
        for fid in factor_ids:
            log_or = _contingency_log_or(presence[fid], shuffled)
            if abs(log_or) >= abs(obs[fid]):
                extreme[fid] += 1
    return {fid: (extreme[fid] + 1) / (n_perm + 1) for fid in factor_ids}


def main() -> None:
    df = pd.read_csv(LABELS_2024)
    print(f"Loaded {len(df):,} rows from interaction_factor_labels_2024.csv")

    df = df[df["ext_direction"].isin(["increase", "same", "decrease"])].copy()
    df["verdict_group"] = df["ext_direction"].map(
        lambda d: "increase" if d == "increase" else "same_decrease"
    )
    print(f"After filtering to valid ext_direction: {len(df):,} rows")
    print(f"Increase: {(df['verdict_group'] == 'increase').sum():,} | "
          f"Same/decrease: {(df['verdict_group'] == 'same_decrease').sum():,}")

    factor_ids = [f"F{i:02d}" for i in range(1, 45)]
    polarity_map = pd.read_csv(ORIG_TEST2)[["factor_id", "polarity", "label"]].set_index("factor_id").to_dict(orient="index")

    presence = {fid: df[fid].astype(int).tolist() for fid in factor_ids}
    labels = df["verdict_group"].tolist()
    paper_keys = df["paper_key"].tolist()
    n_papers = df["paper_key"].nunique()
    print(f"Unique papers: {n_papers:,} (rows-per-paper avg = {len(df)/n_papers:.2f})")

    rows = []
    for fid in factor_ids:
        a = sum(1 for p, l in zip(presence[fid], labels) if l == "increase" and p == 1)
        b = sum(1 for p, l in zip(presence[fid], labels) if l == "increase" and p == 0)
        c = sum(1 for p, l in zip(presence[fid], labels) if l == "same_decrease" and p == 1)
        d = sum(1 for p, l in zip(presence[fid], labels) if l == "same_decrease" and p == 0)
        or_, lo, hi = odds_ratio(a, b, c, d)
        chi2, chi2_p = chi_sq_2x2(a, b, c, d)
        obs_dir = "increase" if or_ > 1 else ("same_decrease" if or_ < 1 else "flat")
        meta = polarity_map.get(fid, {})
        expected_dir = "increase" if meta.get("polarity") == "positive" else "same_decrease"
        rows.append({
            "factor_id": fid,
            "polarity": meta.get("polarity"),
            "label": meta.get("label", ""),
            "a_inc_pres": a, "b_inc_abs": b, "c_same_pres": c, "d_same_abs": d,
            "odds_ratio": or_, "or_ci_lo": lo, "or_ci_hi": hi,
            "chi2": chi2, "chi2_p": chi2_p,
            "observed_direction": obs_dir,
            "polarity_match": "yes" if obs_dir == expected_dir else "no",
        })

    print(f"Running {N_PERMUTATIONS} naive (row-level) permutations ...")
    perm = permutation_null_or(presence, labels, factor_ids, N_PERMUTATIONS, SEED)
    for r in rows:
        r["perm_p"] = perm[r["factor_id"]]

    print(f"Running {N_PERMUTATIONS} paper-clustered (block) permutations ...")
    cluster_perm = cluster_permutation_null_or(
        presence, labels, paper_keys, factor_ids, N_PERMUTATIONS, SEED
    )
    for r in rows:
        r["perm_p_clustered"] = cluster_perm[r["factor_id"]]

    new = pd.DataFrame(rows)
    out_csv = OUT_DIR / "test2_full2024_factor_tests.csv"
    new.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    orig_t1 = pd.read_csv(ORIG_TEST1)
    orig_t2 = pd.read_csv(ORIG_TEST2)
    cols = ["factor_id", "polarity", "odds_ratio", "or_ci_lo", "or_ci_hi", "perm_p", "polarity_match"]
    cmp = orig_t1[cols].rename(columns={c: f"{c}_t1" for c in cols if c != "factor_id"})
    cmp = cmp.merge(orig_t2[cols].rename(columns={c: f"{c}_t2_orig952" for c in cols if c != "factor_id"}),
                    on="factor_id")
    cmp = cmp.merge(new[cols].rename(columns={c: f"{c}_t2_full2024" for c in cols if c != "factor_id"}),
                    on="factor_id")
    cmp = cmp.merge(
        new[["factor_id", "perm_p_clustered"]].rename(
            columns={"perm_p_clustered": "perm_p_t2_full2024_clustered"}),
        on="factor_id",
    )
    cmp["bonf_t1"] = cmp["perm_p_t1"] < ALPHA_BONF
    cmp["bonf_t2_orig952"] = cmp["perm_p_t2_orig952"] < ALPHA_BONF
    cmp["bonf_t2_full2024"] = cmp["perm_p_t2_full2024"] < ALPHA_BONF
    cmp["bonf_t2_full2024_clustered"] = cmp["perm_p_t2_full2024_clustered"] < ALPHA_BONF
    cmp["doubly_orig"] = cmp["bonf_t1"] & cmp["bonf_t2_orig952"]
    cmp["doubly_new"] = cmp["bonf_t1"] & cmp["bonf_t2_full2024"]
    cmp["doubly_new_clustered"] = cmp["bonf_t1"] & cmp["bonf_t2_full2024_clustered"]
    cmp["delta_log_or"] = cmp.apply(
        lambda r: (math.log(r["odds_ratio_t2_full2024"]) - math.log(r["odds_ratio_t2_orig952"]))
        if r["odds_ratio_t2_full2024"] > 0 and r["odds_ratio_t2_orig952"] > 0 else float("nan"),
        axis=1,
    )

    cmp_csv = OUT_DIR / "test2_full2024_comparison.csv"
    cmp.to_csv(cmp_csv, index=False)
    print(f"Saved: {cmp_csv}")

    n_inc = int((df["verdict_group"] == "increase").sum())
    n_sd = int((df["verdict_group"] == "same_decrease").sum())
    base_rate_full = n_inc / (n_inc + n_sd)
    base_rate_orig = orig_t2["a_inc_pres"].iloc[0] + orig_t2["b_inc_abs"].iloc[0]
    n_orig = orig_t2["a_inc_pres"].iloc[0] + orig_t2["b_inc_abs"].iloc[0] + orig_t2["c_same_pres"].iloc[0] + orig_t2["d_same_abs"].iloc[0]
    base_rate_orig_pct = (orig_t2["a_inc_pres"].iloc[0] + orig_t2["b_inc_abs"].iloc[0]) / n_orig

    polarity_preserved_orig = int((orig_t2["polarity_match"] == "yes").sum())
    polarity_preserved_new = int((new["polarity_match"] == "yes").sum())
    n_doubly_orig = int(cmp["doubly_orig"].sum())
    n_doubly_new = int(cmp["doubly_new"].sum())
    n_doubly_new_clust = int(cmp["doubly_new_clustered"].sum())
    n_doubly_orig_pos = int(cmp[cmp["doubly_orig"] & (cmp["polarity_t1"] == "positive")].shape[0])
    n_doubly_orig_neg = int(cmp[cmp["doubly_orig"] & (cmp["polarity_t1"] == "negative")].shape[0])
    n_doubly_new_pos = int(cmp[cmp["doubly_new"] & (cmp["polarity_t1"] == "positive")].shape[0])
    n_doubly_new_neg = int(cmp[cmp["doubly_new"] & (cmp["polarity_t1"] == "negative")].shape[0])
    n_doubly_new_clust_pos = int(cmp[cmp["doubly_new_clustered"] & (cmp["polarity_t1"] == "positive")].shape[0])
    n_doubly_new_clust_neg = int(cmp[cmp["doubly_new_clustered"] & (cmp["polarity_t1"] == "negative")].shape[0])

    flipped_polarity_orig = orig_t2[orig_t2["polarity_match"] == "no"]["factor_id"].tolist()
    flipped_polarity_new = new[new["polarity_match"] == "no"]["factor_id"].tolist()

    gained = cmp[cmp["doubly_new"] & ~cmp["doubly_orig"]]["factor_id"].tolist()
    lost = cmp[~cmp["doubly_new"] & cmp["doubly_orig"]]["factor_id"].tolist()
    gained_clust = cmp[cmp["doubly_new_clustered"] & ~cmp["doubly_orig"]]["factor_id"].tolist()
    lost_clust = cmp[~cmp["doubly_new_clustered"] & cmp["doubly_orig"]]["factor_id"].tolist()

    lines = [
        "# Test 2 expansion: n=952 -> full classified ICLR 2024 cohort\n",
        "## Cohort sizes\n",
        f"- Original Test 2 (stratified): n = {n_orig:,} (base rate {base_rate_orig_pct*100:.1f}% increase)\n",
        f"- Expanded Test 2 (full 2024 classified): n = {n_inc + n_sd:,} (base rate {base_rate_full*100:.1f}% increase)\n",
        f"- Net additional rows: {n_inc + n_sd - n_orig:,}\n",
        "\n## Polarity preservation (vs codebook-declared polarity)\n",
        f"- Original Test 2: {polarity_preserved_orig} of 44 factors match codebook polarity\n",
        f"- Expanded Test 2: {polarity_preserved_new} of 44\n",
        f"- Original mismatched factor IDs: {flipped_polarity_orig}\n",
        f"- Expanded mismatched factor IDs: {flipped_polarity_new}\n",
        f"\n## Doubly-Bonferroni-significant (Test 1 perm_p < {ALPHA_BONF:.4f} AND Test 2 perm_p < {ALPHA_BONF:.4f})\n",
        f"- With original n=952 Test 2 (naive perm): {n_doubly_orig} ({n_doubly_orig_pos} positive, {n_doubly_orig_neg} negative)\n",
        f"- Expanded full-2024, naive (row-level) perm: {n_doubly_new} ({n_doubly_new_pos} positive, {n_doubly_new_neg} negative)\n",
        f"- Expanded full-2024, paper-clustered (block) perm: {n_doubly_new_clust} ({n_doubly_new_clust_pos} positive, {n_doubly_new_clust_neg} negative)\n",
        f"- Newly doubly-Bonf under expanded NAIVE: {gained}\n",
        f"- Lost from doubly-Bonf under expanded NAIVE: {lost}\n",
        f"- Newly doubly-Bonf under expanded CLUSTERED: {gained_clust}\n",
        f"- Lost from doubly-Bonf under expanded CLUSTERED: {lost_clust}\n",
        "\n## Per-factor changes (sorted by absolute change in log-OR)\n",
        "| factor | pol | OR (952) | OR (full) | dlogOR | perm_p (952) | perm_p naive (full) | perm_p clust (full) | doubly orig | doubly new naive | doubly new clust |\n",
        "|--------|-----|----------|-----------|--------|--------------|---------------------|---------------------|-------------|------------------|------------------|\n",
    ]
    cmp_sorted = cmp.reindex(cmp["delta_log_or"].abs().sort_values(ascending=False).index)
    for _, r in cmp_sorted.iterrows():
        lines.append(
            f"| {r['factor_id']} | {r['polarity_t1']} | "
            f"{r['odds_ratio_t2_orig952']:.2f} | {r['odds_ratio_t2_full2024']:.2f} | "
            f"{r['delta_log_or']:+.2f} | "
            f"{r['perm_p_t2_orig952']:.3g} | "
            f"{r['perm_p_t2_full2024']:.3g} | "
            f"{r['perm_p_t2_full2024_clustered']:.3g} | "
            f"{'yes' if r['doubly_orig'] else 'no'} | "
            f"{'yes' if r['doubly_new'] else 'no'} | "
            f"{'yes' if r['doubly_new_clustered'] else 'no'} |\n"
        )

    report = OUT_DIR / "test2_full2024_report.md"
    report.write_text("".join(lines))
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
