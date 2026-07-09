# Factor-classification analysis: test1_same1k

- Items analysed: 1000
- Factors: 44 (23 positive, 21 negative)
- Permutations: 2000
- Bonferroni threshold (alpha=0.05): 0.0011

## Polarity correspondence

For each factor, we check whether its observed direction (more prevalent in `increase` vs `same_decrease`) matches the polarity declared during taxonomy induction.

- Positive factors with expected direction: **22/23**
- Negative factors with expected direction: **21/21**
- Overall match rate: **43/44** (97.7%)

## Statistical significance
- Factors with permutation p < 0.05 (uncorrected): **32/44**
- Factors surviving Bonferroni: **23/44**

## Per-factor test table

| ID | Pol | OR | 95% CI | chi2 p | perm p | Match? | Label |
|---|---|---|---|---|---|---|---|
| F24 | neg | 0.25 | [0.12, 0.54] | 0.000307 | 0.0005 | yes | Defensive or dismissive tone. |
| F25 | neg | 0.12 | [0.05, 0.27] | 8.55e-09 | 0.0005 | yes | Superficial or partial addressal of reviewer concerns. |
| F26 | neg | 0.07 | [0.00, 1.27] | 0.0296 | 0.009 | yes | Missing reproducibility details. |
| F27 | neg | 0.29 | [0.16, 0.51] | 8.81e-06 | 0.0005 | yes | Missing or insufficient baseline comparisons. |
| F28 | neg | 0.26 | [0.17, 0.40] | 3.99e-11 | 0.0005 | yes | Lack of quantitative/empirical evidence in rebuttal. |
| F29 | neg | 0.34 | [0.23, 0.50] | 3.87e-08 | 0.0005 | yes | No new requested experiments / analyses / ablations / review |
| F30 | neg | 0.36 | [0.24, 0.54] | 3.77e-07 | 0.0005 | yes | Vague future-oriented promises instead of concrete commitmen |
| F31 | neg | 0.20 | [0.13, 0.32] | 4.28e-13 | 0.0005 | yes | Narrow or insufficient evaluation scope. |
| F32 | neg | 0.35 | [0.22, 0.55] | 4.33e-06 | 0.0005 | yes | Incomplete empirical coverage even after rebuttal additions. |
| F33 | neg | 0.20 | [0.12, 0.33] | 1.37e-10 | 0.0005 | yes | Unresolved quality-of-execution concerns, methodological or  |
| F34 | neg | 0.29 | [0.13, 0.67] | 0.00389 | 0.0045 | yes | Silent acceptance of limitations with no mitigation. |
| F35 | neg | 0.11 | [0.01, 2.05] | 0.122 | 0.0875 | yes | Lack of structured rebuttal format, no explicit rebuttal sec |
| F36 | neg | 0.20 | [0.11, 0.35] | 4.85e-09 | 0.0005 | yes | Weak novelty justification. |
| F37 | neg | 0.15 | [0.04, 0.67] | 0.00876 | 0.0055 | yes | Brevity or low-effort rebuttal. |
| F38 | neg | 0.18 | [0.08, 0.41] | 8.41e-06 | 0.0005 | yes | Section-referencing or paper-rephrasing instead of real clar |
| F39 | neg | 0.21 | [0.14, 0.33] | 4.46e-13 | 0.0005 | yes | Inadequate explanation or persistent clarity issues. |
| F40 | neg | 0.41 | [0.13, 1.29] | 0.189 | 0.126 | yes | Marginal quantitative improvements. |
| F41 | neg | 0.42 | [0.02, 10.29] | 1 | 1 | yes | Presentation or notation overload that worsens clarity. |
| F42 | neg | 0.11 | [0.04, 0.30] | 5.96e-07 | 0.0005 | yes | Vague or insufficient technical explanation. |
| F43 | neg | 0.34 | [0.20, 0.58] | 7.04e-05 | 0.0005 | yes | No explicit commitment to revise or improve the paper. |
| F44 | neg | 0.32 | [0.12, 0.87] | 0.0328 | 0.0205 | yes | Surfacing a previously unnoticed problem without proactive f |
| F01 | pos | 1.33 | [0.32, 5.59] | 0.975 | 0.728 | yes | Structured, point-by-point response format. |
| F02 | pos | 6.05 | [2.35, 15.56] | 4.79e-05 | 0.0005 | yes | Non-defensive, respectful, collaborative tone. |
| F03 | pos | 1.24 | [0.57, 2.67] | 0.727 | 0.696 | yes | Respectful opening / appreciation toward the reviewer. |
| F04 | pos | 6.03 | [2.10, 17.28] | 0.000311 | 0.0005 | yes | The rebuttal signals sincere engagement through detailed res |
| F05 | pos | 2.87 | [1.62, 5.08] | 0.000294 | 0.0005 | yes | Methodological detail and reproducibility clarification. |
| F06 | pos | 1.99 | [1.49, 2.67] | 4.71e-06 | 0.0005 | yes | Concrete empirical evidence tied to reviewer concerns. |
| F07 | pos | 2.13 | [1.65, 2.74] | 6.65e-09 | 0.0005 | yes | Baseline completion / comparator strengthening. |
| F08 | pos | 2.00 | [1.25, 3.19] | 0.00457 | 0.0035 | yes | Demonstrating understanding and acknowledging valid points b |
| F09 | pos | 3.08 | [1.51, 6.27] | 0.00191 | 0.0025 | yes | Comprehensive concern coverage. |
| F10 | pos | 1.35 | [0.88, 2.07] | 0.205 | 0.171 | yes | Explicit cross-references to figures, tables, and sections b |
| F11 | pos | 1.15 | [0.89, 1.49] | 0.322 | 0.321 | yes | Ablation studies showing component-level impact. |
| F12 | pos | 0.90 | [0.68, 1.19] | 0.483 | 0.486 | no | (Proactive and) honest limitation acknowledgment. |
| F13 | pos | 2.19 | [1.52, 3.14] | 2.46e-05 | 0.0005 | yes | Explicit manuscript revision signaling. |
| F14 | pos | 2.05 | [1.55, 2.72] | 7.44e-07 | 0.0005 | yes | Supplementary material and new visual evidence. |
| F15 | pos | 1.22 | [0.90, 1.67] | 0.231 | 0.219 | yes | Explaining novelty and broader impact. |
| F16 | pos | 1.89 | [1.27, 2.81] | 0.00202 | 0.0015 | yes | Conceptual explanations alongside empirical results. |
| F17 | pos | 1.49 | [0.75, 2.96] | 0.33 | 0.312 | yes | Clarification-oriented response strategy. |
| F18 | pos | 1.38 | [0.80, 2.36] | 0.307 | 0.269 | yes | Concerns-addressed checklist or summary. |
| F19 | pos | 1.42 | [1.10, 1.83] | 0.00907 | 0.0065 | yes | Rewriting and simplifying for clarity. |
| F20 | pos | 1.02 | [0.80, 1.31] | 0.923 | 0.888 | yes | Theoretical or mathematical grounding. |
| F21 | pos | 1.81 | [1.37, 2.39] | 3.64e-05 | 0.0005 | yes | Active dialogue and continued engagement. |
| F22 | pos | 2.12 | [1.44, 3.12] | 0.000167 | 0.0005 | yes | Concrete revision commitments. |
| F23 | pos | 1.30 | [1.01, 1.67] | 0.0479 | 0.0475 | yes | Going beyond the explicit ask and anticipating follow-ups. |
