# Data layout

This folder holds **everything the analysis scripts read or write**, except for
the bulky raw OpenReview JSON dumps and the intermediate normalized JSON, which
are not redistributed here. Download instructions for those are below.

## Hugging Face dataset

The full dataset (raw JSON, parquets, Gemini batch outputs, results) is mirrored at:

**https://huggingface.co/datasets/MlouisBE/iclr-rebuttal-analysis**

To pull just the parquets:

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/MlouisBE/iclr-rebuttal-analysis/data/processed/parquet/iclr_2024_reviews.parquet")
```

To pull the raw OpenReview JSON dumps (3.0 GB total):

```bash
hf download MlouisBE/iclr-rebuttal-analysis --repo-type dataset \
    --include "data/raw/*.json" --local-dir .
```

## What is included

```
data/
├── batches/
│   └── output/                 # Gemini/factor batch outputs -- in the Zenodo/HF record, NOT the
│                               # git repo; needed only for `make sec32`. Download before running it.
├── processed/
│   └── parquet/                # Analysis-ready, per-year tables (~155 MB total)
│       ├── iclr_2023_reviews.parquet   (19 MB, 14k rows x 48 cols)
│       ├── iclr_2024_reviews.parquet   (47 MB, 28k rows x 59 cols)
│       ├── iclr_2025_reviews.parquet   (85 MB, 47k rows x 61 cols)
│       └── iclr_2026_reviews.parquet   (4 MB,  not used in paper)
└── results/
    ├── 31_score_prediction/    # Sec 3.1 -- Gemini score-decoding accuracy
    ├── 31_text_score_gap/      # Sec 3.1 -- text-vs-score gap and offset funnel
    ├── 32_factor_taxonomy/     # Sec 3.2 -- 44-factor Claude + Gemini validation
    │   ├── interaction_factor_labels_2024.csv  # row-level F01..F44 labels (input to 3.3 + 3.4)
    │   ├── interaction_factor_labels_2025.csv
    │   └── factor_vs_8cat_consistency.csv      # sanity cross-tab vs 8-category coding
    ├── horizon_inference/      # Sec 3.3 -- paper-clustered logistic + recommendation gate
    │   ├── figures/forest_recommendation.{pdf,png}    # Figure 4 in the paper
    │   ├── figures/calibration_temporal.{pdf,png}     # Figure 8 in the paper
    │   ├── tables/coefficients_headline.csv           # joint rebuttal-thread-content fit
    │   ├── tables/coefficients_marginal_per_factor.csv
    │   ├── tables/coefficients_negatives_only.csv     # polarity-isolated refit
    │   ├── tables/coefficients_ablation_no_structural.csv
    │   ├── tables/coefficients_ablation_strata_fe.csv
    │   ├── tables/factor_recommendation_table.csv     # the recommendation list
    │   ├── tables/calibration_temporal.csv
    │   └── tables/scenario_predictions.csv            # Table 1 (scenarios) in the paper
    ├── rebuttal_rf_prediction/ # Sec 3.4 -- cascade pre/draft/post + RF ceiling
    │   ├── figures/figure_horizons.{pdf,png}           # Figure 6 in the paper
    │   ├── tables/temporal_split_metrics_{pre,draft,post}.csv
    │   ├── tables/grouped_cv_metrics_{pre,draft,post}.csv
    │   ├── tables/incremental_value_summary.csv
    │   └── tables/permutation_importance_{pre,draft,post}.csv
    └── methodology/            # Appendix supporting analyses
        ├── reviewer_agreement/
        ├── external_gt_reconciliation/
        ├── inclusion_funnel/
        ├── score_change_validation/
        └── score_change_samples/  # 1K-pair seed=50 sample for app:ext-gt
```

Each `results/` subfolder contains the final figures (PDF + PNG) and tables
(CSV) that the paper points at. Intermediate caches and exploratory artefacts
have been stripped.

Packaging note: in the development checkout, some bulky release inputs may be
symlinked from the working repository. Before making a zip or archival upload,
resolve those symlinks or upload the corresponding files through the Hugging
Face dataset so `data/batches/output/` and `data/processed/parquet/` are real
readable files for external users.

### Re-running Section 3.3 / 3.4 from this snapshot

The Sec 3.3 inferential model (`src/analyze/exp33_inference_horizons.py`) and
the Sec 3.4 cascade-prediction model (`src/analyze/exp34_prediction_horizons.py`)
assemble their review-level feature matrices from the bundled parquet and the
bundled factor-label tables. They no longer require
`data/processed/normalized/ICLR_*_normalized.json` for the default public-release
path. The normalized JSON is still needed only if you want to rebuild the
parquets themselves or rerun text/thread extraction from raw OpenReview dumps.

The leakage tests (`tests/test_horizon_leakage.py`) run against the same public
feature builders and are included in `make appendix`.

### What is in the fat parquet

The per-year parquet is a strict superset of the original slim version. Each
row is one review with these columns (varies slightly by year because ICLR
2023 has slightly different review fields):

- **Paper metadata**: paper_key, forum, title, primary_area, decision
- **Review scores**: rating, confidence, soundness, presentation, contribution
- **Review text** (full, not truncated): review_summary, review_strengths,
  review_weaknesses, review_strengths_and_weaknesses, review_questions
- **Text-length features**: summary_len, strengths_len, weaknesses_len, questions_len
- **External ground truth (Kargaran et al.)**: ext_rating_before, ext_rating_after,
  ext_direction, ext_confidence_before, ext_confidence_after
- **Slim Gemini outputs**: gemini_pred_score, gemini_pred_confidence, gemini_pred_sentiment
- **Full Gemini per-reviewer features**: gemini_tone_primary, gemini_tone_secondary,
  gemini_engagement_quality, gemini_score_confidence_llm, gemini_indicators_json
- **Paper-level Gemini stance**: gemini_sentiment_score/label, gemini_stance_*
  (excitement, frustration, disappointment, appreciation, confidence_in_contribution,
  concern_about_validity)
- **Score-change verdicts**: gemini_verdict, regex_verdict, gemini_verdict_confidence,
  gemini_verdict_causes
- **Rebuttal aggregates**: has_rebuttal, n_author_replies, rebuttal_total_len,
  n_thread_messages
- **Bookkeeping**: review_order, reviewer_id, reviewer_suffix, review_cdate

The lossless raw rebuttal text per review (every author reply and threaded
follow-up) is *not* bundled in the shipped parquet because the resulting
files exceed GitHub's 100 MB per-file limit. The build script can recreate
them locally with `--include-thread-text` (see "Reproducing the parquet
tables" below); no analysis script in this release reads that column.

## What is NOT included

| Folder | Size | Why excluded | How to recreate |
|--------|------|--------------|-----------------|
| `data/raw/` | ~3.0 GB | Bulk OpenReview JSON dumps, redistributable but large | **Download from Hugging Face** (`hf download MlouisBE/iclr-rebuttal-analysis --include "data/raw/*.json" --repo-type dataset --local-dir .`) or run `python src/collect/fetch_openreview.py` |
| `data/processed/normalized/` | ~1.9 GB | Per-conference normalized JSON, deterministically derived from `raw/` | Run `python -m src.normalize.cli normalize-all` after `data/raw/` is populated |
| `data/processed/categorization_data/` | ~330 KB | Intermediate batch inputs for the factor taxonomy | Run `python src/factor_validation/build_sample.py` |
| `data/results/score_change/` | ~273 MB | Intermediate per-batch JSONL caches from the score-change pipeline | Run `python src/score_change/extract_regex_verdicts.py` then `python src/score_change/gemini_verdict_check.py` |

## External ground truth

The paper uses pre- and post-rebuttal review scores from the
[papercopilot/iclr-insights](https://github.com/papercopilot/iclr-insights)
project (Kargaran et al.). The matched reviewer-level before/after scores are
already included in the shipped parquet columns (`ext_rating_before`,
`ext_rating_after`, `ext_direction`). The raw external paper-level metadata is
not redistributed here. This matters only for the empty-entry accounting in the
appendix: empty external entries contain paper metadata but no reviewer rows, so
they cannot be reconstructed from a review-row parquet. The public script
`src/analyze/methodology_external_gt_reconciliation.py` reproduces all row-level
checks from parquet and records the archived external snapshot counts used for
the paper-level empty-entry split.

## Reproducing the parquet tables

The shipped parquet tables in `processed/parquet/` were built by joining
three sources:

1. The slim per-year parquet (paper metadata + scores + GT + verdicts).
2. The normalized JSON at `data/processed/normalized/ICLR_{year}_normalized.json`
   (full review text + author rebuttals + reviewer follow-ups).
3. The Gemini Flash 3 batch output JSON in `data/batches/output/`
   (per-reviewer tone, engagement, indicators, score_confidence; plus
   paper-level sentiment and reviewer stance).

To regenerate from scratch:

```bash
# After data/raw/ is populated and Gemini batch outputs are downloaded:
python -m src.normalize.cli normalize-all              # raw -> normalized JSON
python scripts/build_fulltext_parquet.py               # default: ~155 MB total
python scripts/build_fulltext_parquet.py --include-thread-text  # ~360 MB total, with raw thread text
```

The default build (no flag) computes all the structured columns the analysis
scripts need, including the rebuttal aggregates (has_rebuttal,
n_author_replies, rebuttal_total_len, n_thread_messages). It is what is
shipped in this repository.

Pass `--include-thread-text` to additionally embed every author reply +
threaded follow-up as a JSON column (`discussion_thread_json`) for
downstream readers who want to analyze the raw rebuttal text. The
resulting files are too large for direct GitHub hosting (the 2025 parquet
balloons to ~208 MB, above the 100 MB per-file limit), so this output is
power-user-only and not part of the public bundle.

## Want CSV instead of parquet?

The full review tables are shipped only in parquet (about 4x smaller than CSV
on this dataset). If you want CSV for inspection in Excel or another tool:

```bash
python scripts/parquet_to_csv.py            # convert all four years
python scripts/parquet_to_csv.py --year 2024
```

Output lands next to the parquet files by default. Pass `--out-dir` to
redirect.
