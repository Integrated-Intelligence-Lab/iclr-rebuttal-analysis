# ===========================================================================
# Makefile -- Public release of the OpenReview rebuttal-dynamics analysis
# ===========================================================================
#
# Each target corresponds to one released paper component.
#
#   make help        Show all targets
#   make sec31       Section 3.1: LLM as instrument (score decoding + text-vs-score gap)
#   make sec32       Section 3.2: offline 44-factor validation statistics
#   make sec32-batch Section 3.2: rebuild held-out sample and submit Gemini batch
#   make sec33       Section 3.3: M1-M8 logistic ladder + three triggers
#   make sec34       Section 3.4: random-forest prediction ceiling
#   make methodology Appendix: reviewer agreement, regex-vs-Gemini, funnel, GT profile
#   make appendix    Appendix: methodology, external-GT reconciliation, MDE, leakage tests
#   make all         Run everything in order
#
# Prerequisites:
#   - Bundled parquets in data/processed/parquet/
#   - Bundled Gemini/factor outputs under data/batches/output/ and data/results/
#   - Raw + normalized OpenReview data are only needed to rebuild parquets
# ===========================================================================

PYTHON ?= python

.PHONY: help all collect normalize sec31 sec32 sec32-batch sec33 sec34 methodology appendix leakage-tests

help:
	@echo ""
	@echo "  Data preparation"
	@echo "    collect          Fetch OpenReview raw data (requires API access)"
	@echo "    normalize        Normalize raw -> uniform schema"
	@echo ""
	@echo "  Paper experiments (each maps to a Results subsection)"
	@echo "    sec31            Section 3.1: score decoding + text-vs-score gap"
	@echo "    sec32            Section 3.2: offline 44-factor validation statistics"
	@echo "    sec32-batch      Section 3.2: rebuild held-out sample and submit Gemini batch"
	@echo "    sec33            Section 3.3: M1-M8 ladder + three triggers"
	@echo "    sec34            Section 3.4: random-forest prediction ceiling"
	@echo ""
	@echo "  Appendix"
	@echo "    methodology      Reviewer agreement, regex-vs-Gemini, funnel, GT profile"
	@echo "    appendix         Methodology + external-GT reconciliation + MDE + leakage tests"
	@echo "    leakage-tests    Run horizon leakage guard"
	@echo ""
	@echo "  Aggregate"
	@echo "    all              Everything in order"
	@echo ""

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
collect:
	$(PYTHON) src/collect/fetch_openreview.py

normalize:
	$(PYTHON) -m src.normalize.cli normalize-all

# ---------------------------------------------------------------------------
# Section 3.1: LLM as instrument
# ---------------------------------------------------------------------------
sec31:
	$(PYTHON) src/analyze/exp31_score_prediction.py
	$(PYTHON) src/analyze/exp31_text_score_gap.py

# ---------------------------------------------------------------------------
# Section 3.2: rebuttal taxonomy
# ---------------------------------------------------------------------------
sec32:
	$(PYTHON) src/factor_validation/analyze.py --tag test1_same1k
	$(PYTHON) src/factor_validation/analyze.py --tag test2_fresh1k
	$(PYTHON) src/factor_validation/expand_test2_full2024.py

sec32-batch:
	$(PYTHON) src/factor_validation/build_sample.py
	$(PYTHON) src/factor_validation/classify.py submit --input data/results/factor_validation/test2_fresh1k_input.json --tag test2_fresh1k

# ---------------------------------------------------------------------------
# Section 3.3: inferential analysis -- which behaviors are independently
#               associated with score increase (paper-clustered logistic
#               with cluster-robust SEs, recommendation gate, scenarios).
# ---------------------------------------------------------------------------
sec33:
	$(PYTHON) -m src.analyze.exp33_inference_horizons

# ---------------------------------------------------------------------------
# Section 3.4: cascade prediction across pre / draft / post horizons
#               (logistic + RF ceiling check; same engine as sec33).
# ---------------------------------------------------------------------------
sec34:
	$(PYTHON) -m src.analyze.exp34_prediction_horizons

# ---------------------------------------------------------------------------
# Appendix / methodology
# ---------------------------------------------------------------------------
methodology:
	$(PYTHON) src/analyze/methodology_inclusion_funnel.py
	$(PYTHON) src/analyze/methodology_external_gt_reconciliation.py
	$(PYTHON) src/analyze/methodology_sample_gt_profile.py
	$(PYTHON) src/analyze/methodology_reviewer_agreement.py
	$(PYTHON) src/analyze/methodology_regex_vs_gemini.py

appendix: methodology
	$(PYTHON) src/analyze/null_triplet_mde.py
	$(PYTHON) -m pytest tests/test_horizon_leakage.py -q

leakage-tests:
	$(PYTHON) -m pytest tests/test_horizon_leakage.py -q

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
all: sec31 sec32 sec33 sec34 appendix
