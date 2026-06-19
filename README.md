# Rebuttals Move Peer-Review Scores, but Initial-Review Structure Bounds the Movement

Companion code and data for the paper:
*"Rebuttals Move Peer-Review Scores, but Initial-Review Structure Bounds the Movement"*.

This is the public code release for the paper. It contains the scripts, prompts, bundled
analysis-ready data (the per-year parquet tables and released figures/tables), and
reproducibility checks needed to rebuild the released tables and figures without rerunning
the original LLM batch jobs. The bulky data (raw OpenReview dumps, normalized JSON, LLM
batch outputs) is archived separately. See **Data availability** below.

---

## Repository layout

```
public_release/
├── docs/paper/                # title-named paper PDF, TeX source, references, style files
├── src/
│   ├── core/                  # Shared utilities: paths, IO, sampling, plotting
│   ├── prompts/               # Verbatim Gemini score-decoder prompts
│   ├── collect/               # OpenReview crawl helpers
│   ├── normalize/             # Per-conference raw JSON -> uniform schema
│   ├── experiment/            # Gemini batch submit/download helpers
│   ├── score_change/          # Regex + Gemini score-change verdict checks
│   ├── factor_validation/     # Claude induction + Gemini 44-factor validation
│   └── analyze/               # Paper analysis scripts
├── data/
│   ├── batches/output/        # Bundled LLM batch outputs needed for offline stats
│   ├── processed/parquet/     # Full-text derived tables
│   ├── results/               # Released figures, tables, model outputs
│   └── README.md              # Data layout and regeneration notes
├── scripts/
│   ├── build_fulltext_parquet.py
│   └── parquet_to_csv.py
├── tests/                     # Leakage and reproducibility tests
├── Makefile
└── requirements.txt
```

## Paper -> script map

| Paper component | Script | Outputs |
|-----------------|--------|---------|
| Score decoder from score-stripped reviews | `src/analyze/exp31_score_prediction.py` | `data/results/31_score_prediction/` |
| Text-score offset and score movement | `src/analyze/exp31_text_score_gap.py` | `data/results/31_text_score_gap/` |
| 44-factor exchange taxonomy | `src/factor_validation/{build_sample,classify,analyze,expand_test2_full2024}.py` | `data/results/32_factor_taxonomy/`, `data/results/factor_validation/` |
| Exchange-factor conditional inference | `src/analyze/exp33_inference_horizons.py` | `data/results/horizon_inference/` |
| Three-horizon prediction benchmark | `src/analyze/exp34_prediction_horizons.py` | `data/results/rebuttal_rf_prediction/` |
| Appendix: dataset funnel | `src/analyze/methodology_inclusion_funnel.py` | `data/results/methodology/inclusion_funnel/` |
| Appendix: external GT reconciliation | `src/analyze/methodology_external_gt_reconciliation.py` | `data/results/methodology/external_gt_reconciliation/` |
| Appendix: external ground-truth profile | `src/analyze/methodology_sample_gt_profile.py` | `data/results/methodology/` |
| Appendix: reviewer agreement | `src/analyze/methodology_reviewer_agreement.py` | `data/results/methodology/reviewer_agreement/` |
| Appendix: regex vs Gemini score-change | `src/analyze/methodology_regex_vs_gemini.py` | `data/results/methodology/score_change_validation/` |

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run selected analyses
make sec31
make sec33
make sec34
make appendix

# Run the public leakage/reproducibility guard
python -m pytest tests/test_horizon_leakage.py -q
```

Each `exp*.py` script also accepts `python <script>.py --help` for direct invocation.

## Status of each section

- **Score decoder and text-score offset analyses** rebuild from the bundled parquet (the
  Gemini score predictions are already merged in as columns).
- **44-factor taxonomy statistics** are recomputed with `make sec32` from the consolidated
  factor-classification outputs. Those batch outputs are not bundled in this repo; fetch them
  from the Zenodo record or the Hugging Face mirror into `data/batches/output/` first.
  Re-running Gemini classification is deliberately separated into `make sec32-batch` because
  it launches a paid API batch job; Claude induction remains a provenance step documented by
  the shipped prompts and serialized induction sample.
- **Exchange-factor inference and three-horizon prediction** are implemented in the public
  release and use the same feature provenance boundaries described in the paper. The
  temporal prediction test uses the stratified, rare-cell-enriched ICLR 2025 slice; the
  accompanying paper-clustered GroupKFold results are the population-mixed robustness check.
- **Appendix methodology scripts** reproduce the released data-funnel, external-score,
  reviewer-agreement, and score-change-validation tables. The external-GT reconciliation
  script reproduces the review-level match from parquet and records the paper-level empty
  external entries from the archived external snapshot, because empty entries have no
  review rows to recover from parquet.

## Reproducibility notes

- Random seed: a fixed seed is used for every stochastic step (`50` for the prediction-horizon
  models and sampling, `42` for the legacy text-gap and prediction-comparison models).
- Cross-validation: prediction models use `GroupKFold` keyed on `paper_key`.
- Leakage guard: `tests/test_horizon_leakage.py` verifies that H0, H1, and H2 feature
  families stay inside their declared observation horizons.
- Score-field stripping removes numeric review scores and confidence before Gemini score
  decoding.
- Taxonomy validation uses Bonferroni correction at `alpha = 0.05 / 44 = 0.0011`.
- LLM provenance:
  - Score decoder: Gemini Flash 3 batch outputs, temperature 0.
  - Factor induction: Claude Opus 4.6 with extended thinking enabled.
  - Factor classification: Gemini Flash 3, outcome-blinded with shuffled factor IDs.
- Prompts are in `src/prompts/classification_instruction.py`,
  `src/factor_validation/prompt.py`, and `src/factor_validation/induction_prompt.md`.

## Data availability

This repository bundles only the small, analysis-ready data under `data/`:

- `data/processed/parquet/` -- per-year review tables (full review text, reviewer scores,
  external before/after ground truth, Gemini per-review features, rebuttal aggregates).
- `data/results/` -- the released figures (PDF/PNG) and tables (CSV).

The bulky inputs are archived on Zenodo and mirrored on Hugging Face:

- **Zenodo (archival, canonical):** DOI assigned on publication -- raw OpenReview records
  (ICLR 2023-2026), normalized JSON, consolidated LLM batch outputs, plus the same parquet
  and results, with a Croissant metadata file, datasheet, and responsible-use note.
- **Hugging Face (convenience mirror):**
  https://huggingface.co/datasets/MlouisBE/iclr-rebuttal-analysis

A plain clone is enough to rebuild **sec31, sec33, and sec34** (score decoder, exchange-feature
inference, and the three-horizon prediction benchmark): those `Makefile` targets read only the
bundled parquet. **sec32** (the 44-factor taxonomy statistics) additionally needs the
consolidated factor-classification batch outputs, which are not in this repo -- download them
from the Zenodo record or the Hugging Face mirror into `data/batches/output/` first. The
external before/after rebuttal scores come from
[papercopilot/iclr-insights](https://github.com/papercopilot/iclr-insights) (Kargaran et al.)
and are already merged into the parquet columns.

## Citing

```bibtex
@article{louis2026rebuttals,
  title  = {Rebuttals Move Peer-Review Scores, but Initial-Review Structure Bounds the Movement},
  author = {Louis, Mathieu and Vanleke, Tibo and Ginis, Vincent and Algaba, Andres},
  year   = {2026}
}
```

See `docs/paper/references.bib` for the bibliography, and `CITATION.cff` for machine-readable
citation metadata.

## License

- **Code:** MIT (see `LICENSE`).
- **Data** (bundled `data/` and the Zenodo record): review comments, discussion messages,
  and LLM-derived per-review labels under CC-BY-4.0; submission and venue metadata under CC0
  where OpenReview classifies them as such. The underlying OpenReview records remain subject
  to the [OpenReview Terms of Use](https://openreview.net/legal/terms).
- **Paper text and figures:** per the publishing venue.
