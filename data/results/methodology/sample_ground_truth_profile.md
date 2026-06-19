# Ground-Truth Score Profile: Sample vs Full Dataset

**Sample source**: `/Users/mathieulouis/Documents/0. Mathieu/1. Travail/0. VUB/DATALAB/II/Research Exp./openreview_dataset_analysis/public_release/data/results/methodology/score_change_samples/iclr_2025_accepted_rejected_seed50_1k_pairs.json`
**Years**: 2024 + 2025
**Ground truth**: `ext_rating_before`, `ext_rating_after`, `ext_direction`

## Totals

- Sample: 1000 reviews across 936 papers
- Full eligible: 73231 reviews across 18403 papers

## Tier Distribution (Sample vs Full)

| Tier | Sample % | Full % | Delta (pp) |
|---|---:|---:|---:|
| low | 22.10 | 32.31 | -10.21 |
| mid | 62.40 | 58.87 | +3.53 |
| high | 15.50 | 8.81 | +6.69 |

## Interaction Validation (Sample)

- Pairs validated: 1000
- With interactions: 999
- Without interactions: 1
- Coverage: 99.90%

## Direction Counts By Tier (Sample)

| Tier | Decrease | Same | Increase |
|---|---:|---:|---:|
| low | 0 | 107 | 114 |
| mid | 16 | 337 | 271 |
| high | 5 | 92 | 58 |

## Direction Percentages By Tier (Sample)

| Tier | Decrease % | Same % | Increase % |
|---|---:|---:|---:|
| low | 0.00 | 48.42 | 51.58 |
| mid | 2.56 | 54.01 | 43.43 |
| high | 3.23 | 59.35 | 37.42 |
