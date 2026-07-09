"""Factor-taxonomy validation experiment.

Tests whether the 44-factor rebuttal taxonomy induced from 1000 ICLR 2025
reviewer-author interactions (a) is operationalisable by an independent LLM
and (b) discriminates outcome groups on held-out data (ICLR 2024).

Scripts:
    prompt.py         — outcome-blinded classification prompt and schema.
    codebook.json     — 44 factors with polarity metadata.
    build_sample.py   — Test 2 held-out sample builder (ICLR 2024).
    classify.py       — Gemini batch pipeline: submit / status / download-all / retry-failed.
    analyze.py        — Prevalence, odds ratios, permutation null, polarity correspondence.
"""
