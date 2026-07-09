# Factor-classification analysis: test2_fresh1k

- Items analysed: 952
- Factors: 44 (23 positive, 21 negative)
- Permutations: 5
- Bonferroni threshold (alpha=0.05): 0.0011

## Polarity correspondence

For each factor, we check whether its observed direction (more prevalent in `increase` vs `same_decrease`) matches the polarity declared during taxonomy induction.

- Positive factors with expected direction: **23/23**
- Negative factors with expected direction: **20/21**
- Overall match rate: **43/44** (97.7%)

## Statistical significance
- Factors with permutation p < 0.05 (uncorrected): **0/44**
- Factors surviving Bonferroni: **0/44**

## Per-factor test table

| ID | Pol | OR | 95% CI | chi2 p | perm p | Match? | Label |
|---|---|---|---|---|---|---|---|
| F24 | neg | 0.14 | [0.06, 0.36] | 4.76e-06 | 0.167 | yes | Defensive or dismissive tone. |
| F25 | neg | 0.23 | [0.13, 0.42] | 5.13e-07 | 0.167 | yes | Superficial or partial addressal of reviewer concerns. |
| F26 | neg | 0.28 | [0.06, 1.28] | 0.144 | 0.167 | yes | Missing reproducibility details. |
| F27 | neg | 0.49 | [0.31, 0.77] | 0.00268 | 0.167 | yes | Missing or insufficient baseline comparisons. |
| F28 | neg | 0.46 | [0.32, 0.66] | 3.24e-05 | 0.167 | yes | Lack of quantitative/empirical evidence in rebuttal. |
| F29 | neg | 0.48 | [0.34, 0.69] | 5.14e-05 | 0.167 | yes | No new requested experiments / analyses / ablations / review |
| F30 | neg | 0.86 | [0.62, 1.20] | 0.42 | 0.833 | yes | Vague future-oriented promises instead of concrete commitmen |
| F31 | neg | 0.61 | [0.44, 0.86] | 0.00527 | 0.167 | yes | Narrow or insufficient evaluation scope. |
| F32 | neg | 0.85 | [0.61, 1.19] | 0.404 | 0.833 | yes | Incomplete empirical coverage even after rebuttal additions. |
| F33 | neg | 0.46 | [0.31, 0.68] | 0.000125 | 0.167 | yes | Unresolved quality-of-execution concerns, methodological or  |
| F34 | neg | 0.96 | [0.51, 1.80] | 1 | 1 | yes | Silent acceptance of limitations with no mitigation. |
| F35 | neg | 0.08 | [0.00, 1.42] | 0.0422 | 0.167 | yes | Lack of structured rebuttal format, no explicit rebuttal sec |
| F36 | neg | 0.36 | [0.21, 0.61] | 0.000149 | 0.167 | yes | Weak novelty justification. |
| F37 | neg | 0.25 | [0.09, 0.73] | 0.0112 | 0.167 | yes | Brevity or low-effort rebuttal. |
| F38 | neg | 0.26 | [0.14, 0.47] | 3.33e-06 | 0.167 | yes | Section-referencing or paper-rephrasing instead of real clar |
| F39 | neg | 0.37 | [0.26, 0.53] | 3.93e-08 | 0.167 | yes | Inadequate explanation or persistent clarity issues. |
| F40 | neg | 1.43 | [0.72, 2.83] | 0.396 | 0.333 | no | Marginal quantitative improvements. |
| F41 | neg | 0.35 | [0.04, 3.15] | 0.601 | 0.333 | yes | Presentation or notation overload that worsens clarity. |
| F42 | neg | 0.27 | [0.15, 0.48] | 3.63e-06 | 0.167 | yes | Vague or insufficient technical explanation. |
| F43 | neg | 0.58 | [0.37, 0.89] | 0.0164 | 0.167 | yes | No explicit commitment to revise or improve the paper. |
| F44 | neg | 0.78 | [0.36, 1.70] | 0.663 | 0.667 | yes | Surfacing a previously unnoticed problem without proactive f |
| F01 | pos | 3.08 | [1.03, 9.22] | 0.0592 | 0.167 | yes | Structured, point-by-point response format. |
| F02 | pos | 3.57 | [1.89, 6.75] | 5.98e-05 | 0.167 | yes | Non-defensive, respectful, collaborative tone. |
| F03 | pos | 3.38 | [1.48, 7.73] | 0.00396 | 0.167 | yes | Respectful opening / appreciation toward the reviewer. |
| F04 | pos | 5.19 | [2.44, 11.04] | 4.39e-06 | 0.167 | yes | The rebuttal signals sincere engagement through detailed res |
| F05 | pos | 2.34 | [1.46, 3.75] | 0.000458 | 0.167 | yes | Methodological detail and reproducibility clarification. |
| F06 | pos | 1.87 | [1.40, 2.48] | 2.12e-05 | 0.167 | yes | Concrete empirical evidence tied to reviewer concerns. |
| F07 | pos | 2.12 | [1.63, 2.76] | 2.49e-08 | 0.167 | yes | Baseline completion / comparator strengthening. |
| F08 | pos | 2.56 | [1.64, 3.99] | 3.3e-05 | 0.167 | yes | Demonstrating understanding and acknowledging valid points b |
| F09 | pos | 3.36 | [1.77, 6.37] | 0.000161 | 0.167 | yes | Comprehensive concern coverage. |
| F10 | pos | 1.78 | [1.20, 2.64] | 0.00533 | 0.167 | yes | Explicit cross-references to figures, tables, and sections b |
| F11 | pos | 1.40 | [1.05, 1.85] | 0.0242 | 0.167 | yes | Ablation studies showing component-level impact. |
| F12 | pos | 1.11 | [0.84, 1.48] | 0.509 | 0.5 | yes | (Proactive and) honest limitation acknowledgment. |
| F13 | pos | 1.53 | [1.10, 2.13] | 0.0133 | 0.167 | yes | Explicit manuscript revision signaling. |
| F14 | pos | 1.86 | [1.41, 2.45] | 1.18e-05 | 0.167 | yes | Supplementary material and new visual evidence. |
| F15 | pos | 1.48 | [1.07, 2.07] | 0.0239 | 0.333 | yes | Explaining novelty and broader impact. |
| F16 | pos | 2.06 | [1.41, 3.01] | 0.000209 | 0.167 | yes | Conceptual explanations alongside empirical results. |
| F17 | pos | 4.32 | [2.10, 8.89] | 3.08e-05 | 0.167 | yes | Clarification-oriented response strategy. |
| F18 | pos | 1.67 | [0.89, 3.13] | 0.146 | 0.167 | yes | Concerns-addressed checklist or summary. |
| F19 | pos | 1.81 | [1.39, 2.37] | 1.55e-05 | 0.167 | yes | Rewriting and simplifying for clarity. |
| F20 | pos | 1.62 | [1.25, 2.10] | 0.000377 | 0.167 | yes | Theoretical or mathematical grounding. |
| F21 | pos | 2.61 | [1.99, 3.44] | 5.72e-12 | 0.167 | yes | Active dialogue and continued engagement. |
| F22 | pos | 1.55 | [1.11, 2.18] | 0.0134 | 0.167 | yes | Concrete revision commitments. |
| F23 | pos | 2.11 | [1.62, 2.74] | 3.85e-08 | 0.167 | yes | Going beyond the explicit ask and anticipating follow-ups. |
