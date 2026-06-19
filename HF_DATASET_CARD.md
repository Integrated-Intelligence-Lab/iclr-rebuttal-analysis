---
license: cc-by-4.0
task_categories:
  - text-classification
  - tabular-classification
language:
  - en
tags:
  - peer-review
  - rebuttal
  - ICLR
  - academic-publishing
  - LLM-annotation
  - scientific-review
size_categories:
  - 10K<n<100K
pretty_name: ICLR Rebuttal Dynamics Dataset
---

# ICLR Rebuttal Dynamics Dataset

Companion dataset for the paper *"Rebuttal dynamics in machine-learning peer review: an LLM-instrumented analysis of ICLR 2024-2025"*.

## Overview

This dataset contains 164,974 peer reviews from ICLR 2023-2026, enriched with LLM-extracted features for studying how author rebuttals influence reviewer score changes. Each review includes the full review text, pre- and post-rebuttal scores, Gemini-predicted scores, score-change verdicts, and 44-factor rebuttal content classifications.

## Dataset structure

### Analysis-ready tables (parquet)

The primary data lives in `data/processed/parquet/`. Each row is one review.

| File | Reviews | Papers | Year |
|------|---------|--------|------|
| `iclr_2024_reviews.parquet` | 28,028 | 7,262 | 2024 |
| `iclr_2025_reviews.parquet` | 46,748 | 11,520 | 2025 |
| `iclr_2023_reviews.parquet` | 14,339 | 3,793 | 2023 |
| `iclr_2026_reviews.parquet` | 75,859 | 19,474 | 2026 |

**Key columns:**

- **Paper metadata**: `paper_key`, `forum`, `title`, `primary_area`, `decision`
- **Review text** (full, not truncated): `review_summary`, `review_strengths`, `review_weaknesses`, `review_questions`
- **Review scores**: `rating`, `confidence`, `soundness`, `presentation`, `contribution`
- **External ground truth** (Kargaran et al.): `ext_rating_before`, `ext_rating_after`, `ext_direction`
- **Gemini score predictions**: `gemini_pred_score`, `gemini_pred_confidence`, `gemini_pred_sentiment`
- **Gemini per-reviewer features**: `gemini_tone_primary`, `gemini_engagement_quality`, `gemini_indicators_json`
- **Score-change verdicts**: `gemini_verdict`, `regex_verdict`, `gemini_verdict_causes`
- **Rebuttal aggregates**: `has_rebuttal`, `n_author_replies`, `rebuttal_total_len`, `n_thread_messages`

### Gemini batch outputs

Raw LLM outputs in `data/batches/`, consolidated into clean per-year files:

- **Score predictions** (`score_predictions_iclr{2023,2024,2025}.json`): per-paper Gemini predictions with full raw responses
- **Verdicts** (`verdicts_iclr{2023,2024,2025}.json`): per-review score-change classifications
- **Factor validation** (`factor_validation_*.json`): 44-factor rebuttal content classifications

### Results

Pre-computed figures (PDF + PNG) and tables (CSV) in `data/results/`, organized by paper section.

## Quick start

```python
import pandas as pd

# Load directly from Hugging Face
df_2024 = pd.read_parquet("hf://datasets/MlouisBE/iclr-rebuttal-analysis/data/processed/parquet/iclr_2024_reviews.parquet")
df_2025 = pd.read_parquet("hf://datasets/MlouisBE/iclr-rebuttal-analysis/data/processed/parquet/iclr_2025_reviews.parquet")

# Combine
df = pd.concat([df_2024, df_2025], ignore_index=True)
print(f"{len(df)} reviews across {df['paper_key'].nunique()} papers")

# Score change distribution
print(df['ext_direction'].value_counts(normalize=True))
```

## Data provenance

- **Review text and metadata**: collected from the OpenReview API using `openreview-py`
- **Pre/post-rebuttal scores**: from [papercopilot/iclr-insights](https://github.com/papercopilot/iclr-insights) (Kargaran et al.)
- **Gemini predictions**: Gemini Flash 3.0 (ICLR 2024) and Gemini Flash 3.1 (ICLR 2023, 2025) at temperature 0
- **44-factor taxonomy**: induced by Claude (5 runs, manually aggregated), validated by Gemini classification

## Code

The full analysis pipeline is available in the companion code repository. Each analysis script is named after the paper subsection it produces.

## Citation

```bibtex
@article{louis2026rebuttal,
  title  = {Rebuttal dynamics in machine-learning peer review:
            an LLM-instrumented analysis of ICLR 2024-2025},
  author = {Louis, Mathieu and others},
  year   = {2026}
}
```

## License

License is set per artefact tier, in line with [OpenReview's terms of use](https://openreview.net/legal/terms):

- **Review comments, discussion messages, and configuration records** (including our LLM-derived per-review labels): redistributed under **CC-BY-4.0**, matching OpenReview's comment licence.
- **Submission and venue metadata** (paper_key, forum, decision tier, timestamps): redistributed under **CC0** where OpenReview classifies it as such.
- **Article / submission content** (PDFs, abstracts, primary submission body): **NOT redistributed** here. Readers who need the underlying paper content must retrieve it from OpenReview directly under that paper's own licence field.
- **Code** (analysis scripts, prompts, LaTeX source, Makefile) in the companion `public_release/` bundle: **MIT**.

The HF dataset top-level `license: cc-by-4.0` field reflects the dominant tier (review/comment text and derived labels). Tier-specific licences above govern the actual artefacts.
