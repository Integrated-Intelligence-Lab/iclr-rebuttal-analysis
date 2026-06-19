## Validation of the factor taxonomy

The central question of this paper is which rebuttal moves associate with an upward revision of the reviewer score. We induce a taxonomy of such moves from ICLR reviewer-author interactions using a large language model (Claude, hereafter "LLM-A"). Two conditions must hold before the taxonomy can support any downstream claim: (a) the factors reflect structure in the interaction text rather than patterns internal to LLM-A, and (b) their associations with score changes replicate beyond the induction sample. We check both in three stages: (i) an induction procedure that includes a memory-control check, (ii) a within-sample reliability test (Test 1) that asks whether a second, independent model can recover the induced factors from the same interactions, and (iii) a held-out discriminative validity test (Test 2) that asks whether factor-outcome associations replicate on data disjoint from the induction corpus.

**Sample for taxonomy induction.** From ICLR 2025 we draw 1000 reviewer-paper pairs stratified by reviewer initial rating (low, medium, high) crossed with rebuttal outcome (`increase` versus `same_decrease`), restricted to pairs with a non-empty author rebuttal thread. The `high_increase` cell is rare in ICLR 2025 and we therefore take all 58 available pairs rather than subsample, which matches the strategy later applied to the ICLR 2024 hold-out. The remaining five cells are filled by random sampling.

**Factor induction.** In a single prompt we present LLM-A with all six cells of the sample (low, medium, and high initial rating, each split into `increase` and `same_decrease` interactions) and instruct the model to perform the contrast within each tier (low-increase versus low-same_decrease, and analogously for medium and high). This within-tier framing preserves tier-specific patterns and prevents the model from collapsing the task into a single cross-tier contrast that could be dominated by score-level differences rather than by rebuttal content. We repeat the procedure five times to attenuate run-to-run variability, then manually aggregate the extracted categories across runs and across tiers, merging entries that are pure rephrasings or semantically close. The aggregation yields 44 factors, each labelled with a declared polarity (positive or negative) from the pooled induction output. To rule out the hypothesis that LLM-A is recalling patterns already internal to the model rather than reading the interactions, we re-run the induction under two memory-control manipulations. The first swaps the low-tier and high-tier interactions within the same prompt structure. The second silently swaps the `increase` and `same_decrease` outcome labels on the same 1000 interactions, holding tier assignment fixed. In both cases the resulting factor content shifts with the presented inputs rather than tracking the label, consistent with extraction from the interaction text. The second swap is reported in Appendix C because it operates on the axis where the original induction did its discriminative work and therefore provides the sharper stress test of the two.

**Classification protocol.** A second, independent Gemini call ("LLM-B") classifies each interaction against the fixed 44-factor list. Classification is outcome-blinded: reviewer tier and rebuttal outcome are stripped from the prompt, factor identifiers are shuffled per request so that ordering cannot cue the model, and the Gemini response schema constrains each returned factor_id to the set F01...F44, which blocks hallucinated categories. For every factor we compute an odds ratio between the `increase` and `same_decrease` strata with a 95 percent confidence interval, and a two-sided permutation test with 2000 permutations. Significance is reported at the Bonferroni-corrected threshold alpha = 0.05 / 44 = 0.0011.

**Tests.** Test 1 (operationalisability) applies LLM-B to the same 1000 ICLR 2025 interactions used for induction, and probes whether a second model can recover the induced factors from the same text. Test 2 (discriminative validity) applies the same protocol to 952 held-out ICLR 2024 interactions disjoint from the induction corpus, and probes whether the factor-outcome associations replicate on fresh data. Test 2 was targeted at 1000 with the same six-cell stratification, but the `high_increase` cell contains only 10 eligible pairs in ICLR 2024 because high-tier reviewers rarely raise their score further. The remaining five cells are filled as specified.

**Results.** The taxonomy shows the declared directional structure in both tests. In Test 1, 43 of 44 factors are more prevalent on the side of the `increase` versus `same_decrease` contrast declared at induction (22 of 23 positive factors, 21 of 21 negative factors), and 23 of 44 survive the Bonferroni-corrected threshold. In Test 2, 43 of 44 factors again match their declared direction (23 of 23 positive, 20 of 21 negative), and 23 of 44 again survive Bonferroni. The single disagreement falls on different factors in the two tests: F12 (proactive limitation acknowledgement) inverts polarity only in Test 1 with OR = 0.90 (95 percent CI 0.68 to 1.19), and F40 (marginal quantitative improvements) inverts polarity only in Test 2 with OR = 1.52 (95 percent CI 0.76 to 3.05), so no factor in the taxonomy inverts its direction on both the within-sample and held-out samples. These results indicate that the 44 factors are operationalisable by a model independent of the one that induced them, and that their associations with reviewer score changes hold on data disjoint from the induction corpus.

**Conclusion.** Both conditions declared at the start of this section are satisfied. Condition (a), that the factors reflect structure in the interaction text rather than patterns internal to LLM-A, is supported by Test 1 together with the memory-control swap in the induction stage. Condition (b), that the factor-outcome associations replicate beyond the induction sample, is supported by Test 2 on held-out ICLR 2024 interactions. Of the 44 factors, 16 cross the Bonferroni-corrected significance threshold in both tests (7 positive, 9 negative), a further 7 cross only in Test 1, and a further 7 cross only in Test 2. The 16 doubly-significant factors form the most conservative core of the taxonomy and anchor the paper's subsequent claims about which rebuttal moves associate with an upward revision of the reviewer score. Factors that reach Bonferroni in only one test are retained but reported as weaker evidence, and the two factors with a polarity inversion in a single test (F12 in Test 1, F40 in Test 2) are flagged accordingly.

## Appendix A: Factor groupings from the two validation tests

**Group A — Bonferroni-significant in both Test 1 and Test 2 (n = 16).** The conservative core of the taxonomy.

| ID  | Polarity | Label |
|-----|----------|-------|
| F02 | positive | Non-defensive, respectful, collaborative tone. |
| F04 | positive | The rebuttal signals sincere engagement through detailed responses, length, added appendix content, or extensive explanation. |
| F05 | positive | Methodological detail and reproducibility clarification. |
| F06 | positive | Concrete empirical evidence tied to reviewer concerns. |
| F07 | positive | Baseline completion / comparator strengthening. |
| F14 | positive | Supplementary material and new visual evidence. |
| F21 | positive | Active dialogue and continued engagement. |
| F24 | negative | Defensive or dismissive tone. |
| F25 | negative | Superficial or partial addressal of reviewer concerns. |
| F28 | negative | Lack of quantitative/empirical evidence in rebuttal. |
| F29 | negative | No new requested experiments / analyses / ablations / reviewer 'would-have-liked' checks. |
| F33 | negative | Unresolved quality-of-execution concerns, methodological or experimental design issues that additional experiments do not fix. |
| F36 | negative | Weak novelty justification. |
| F38 | negative | Section-referencing or paper-rephrasing instead of real clarification. |
| F39 | negative | Inadequate explanation or persistent clarity issues. |
| F42 | negative | Vague or insufficient technical explanation. |

**Group B — Bonferroni-significant in Test 1 only (n = 7).** Replicates on the induction sample, but does not cross Bonferroni on the held-out ICLR 2024 sample.

| ID  | Polarity | Label |
|-----|----------|-------|
| F13 | positive | Explicit manuscript revision signaling. |
| F22 | positive | Concrete revision commitments. |
| F27 | negative | Missing or insufficient baseline comparisons. |
| F30 | negative | Vague future-oriented promises instead of concrete commitments. |
| F31 | negative | Narrow or insufficient evaluation scope. |
| F32 | negative | Incomplete empirical coverage even after rebuttal additions. |
| F43 | negative | No explicit commitment to revise or improve the paper. |

**Group C — Bonferroni-significant in Test 2 only (n = 7).** Crosses Bonferroni on the held-out sample, but not on the induction sample.

| ID  | Polarity | Label |
|-----|----------|-------|
| F08 | positive | Demonstrating understanding and acknowledging valid points before responding. |
| F09 | positive | Comprehensive concern coverage. |
| F16 | positive | Conceptual explanations alongside empirical results. |
| F17 | positive | Clarification-oriented response strategy. |
| F19 | positive | Rewriting and simplifying for clarity. |
| F20 | positive | Theoretical or mathematical grounding. |
| F23 | positive | Going beyond the explicit ask and anticipating follow-ups. |

**Group D — Polarity inversion in a single test (n = 2).** Observed direction disagrees with the induction-declared polarity in exactly one of the two tests.

| ID  | Declared polarity | Inverts in | Label |
|-----|-------------------|------------|-------|
| F12 | positive | Test 1 | (Proactive and) honest limitation acknowledgment. |
| F40 | negative | Test 2 | Marginal quantitative improvements. |

## Appendix B: Tier-specific attributions from LLM-A and tier-localisation test

Alongside the binary polarity label, LLM-A produced at induction a sub-claim about which initial-score tiers each factor is most prevalent in. We record these sub-claims in Group E (all 44 factors) and check them with an auxiliary test (Test 3) on both the `increase` and `same_decrease` strata.

**Group E — LLM-A tier attributions for all 44 factors.** "All" means LLM-A reports the factor at low, medium, and high initial-score tiers; other entries list the tiers in which LLM-A reports the factor as more prevalent.

Positive factors (n = 23):

| ID  | Attributed tier(s) | Label |
|-----|--------------------|-------|
| F01 | low, medium, high | Structured, point-by-point response format. |
| F02 | low, medium, high | Non-defensive, respectful, collaborative tone. |
| F03 | low, medium, high | Respectful opening / appreciation toward the reviewer. |
| F04 | low, medium, high | The rebuttal signals sincere engagement through detailed responses, length, added appendix content, or extensive explanation. |
| F05 | low, medium, high | Methodological detail and reproducibility clarification. |
| F06 | low, medium, high | Concrete empirical evidence tied to reviewer concerns. |
| F07 | low, medium, high | Baseline completion / comparator strengthening. |
| F08 | low, medium, high | Demonstrating understanding and acknowledging valid points before responding. |
| F09 | medium, high | Comprehensive concern coverage. |
| F10 | medium, high | Explicit cross-references to figures, tables, and sections by number. |
| F11 | medium, high | Ablation studies showing component-level impact. |
| F12 | medium, high | (Proactive and) honest limitation acknowledgment. |
| F13 | low, medium | Explicit manuscript revision signaling. |
| F14 | low, medium | Supplementary material and new visual evidence. |
| F15 | low, high | Explaining novelty and broader impact. |
| F16 | medium | Conceptual explanations alongside empirical results. |
| F17 | medium | Clarification-oriented response strategy. |
| F18 | medium | Concerns-addressed checklist or summary. |
| F19 | low | Rewriting and simplifying for clarity. |
| F20 | low | Theoretical or mathematical grounding. |
| F21 | low | Active dialogue and continued engagement. |
| F22 | high | Concrete revision commitments. |
| F23 | high | Going beyond the explicit ask and anticipating follow-ups. |

Negative factors (n = 21):

| ID  | Attributed tier(s) | Label |
|-----|--------------------|-------|
| F24 | low, medium, high | Defensive or dismissive tone. |
| F25 | low, medium, high | Superficial or partial addressal of reviewer concerns. |
| F26 | low, medium, high | Missing reproducibility details. |
| F27 | low, medium, high | Missing or insufficient baseline comparisons. |
| F28 | low, medium, high | Lack of quantitative/empirical evidence in rebuttal. |
| F29 | low, medium, high | No new requested experiments / analyses / ablations / reviewer 'would-have-liked' checks. |
| F30 | low, medium, high | Vague future-oriented promises instead of concrete commitments. |
| F31 | low, medium, high | Narrow or insufficient evaluation scope. |
| F32 | low, medium, high | Incomplete empirical coverage even after rebuttal additions. |
| F33 | low, medium, high | Unresolved quality-of-execution concerns, methodological or experimental design issues that additional experiments do not fix. |
| F34 | low, medium, high | Silent acceptance of limitations with no mitigation. |
| F35 | medium, high | Lack of structured rebuttal format. |
| F36 | low, medium | Weak novelty justification. |
| F37 | low, high | Brevity or low-effort rebuttal. |
| F38 | low, high | Section-referencing or paper-rephrasing instead of real clarification. |
| F39 | low | Inadequate explanation or persistent clarity issues. |
| F40 | low | Marginal quantitative improvements. |
| F41 | low | Presentation or notation overload that worsens clarity. |
| F42 | low | Vague or insufficient technical explanation. |
| F43 | high | No explicit commitment to revise or improve the paper. |
| F44 | high | Surfacing a previously unnoticed problem without proactive framing. |

**Test 3 (aggregate tier localisation).** For each initial-score tier t and each polarity (positive or negative) we define a bucket B_t containing all same-polarity factors whose LLM-A attribution includes t. For every interaction in the relevant outcome stratum we count how many factors from each bucket LLM-B flagged as present, take the mean across interactions grouped by the interaction's own initial-score tier, and divide by the bucket size to obtain a detection rate (the fraction of attributed factors that LLM-B detected on average). Under LLM-A's attribution, the detection rate for bucket B_t should peak in tier-t interactions (diagonal dominance of the bucket-by-tier matrix). Because the three buckets overlap heavily (the eleven "all-tier" positive factors F01-F08 and the eleven "all-tier" negative factors F24-F34 belong to all three buckets respectively), we report two views per polarity. The full bucket uses all attributed factors. The tier-specific bucket excludes the all-tier factors and isolates the signal from factors LLM-A did not attribute to every tier.

**Results on the `increase` stratum (positive-factor buckets).** The matrix rows are buckets, the columns are the initial-score tier of the interactions, and the bold entry in each row marks the tier with the highest detection rate.

Test 1 (n_low = 114, n_mid = 271, n_high = 58, ICLR 2025).

| Bucket | Bucket size | Low-tier | Medium-tier | High-tier | Peaks where attributed? |
|--------|-------------|----------|-------------|-----------|-------------------------|
| Full low bucket           | 14 | **83.6%** | 81.6%     | 76.7%     | yes |
| Full medium bucket        | 17 | 81.7%     | **81.9%** | 78.8%     | yes |
| Full high bucket          | 15 | **84.7%** | 83.3%     | 79.1%     | no (peak low) |
| Tier-specific low bucket  | 6  | **74.5%** | 69.2%     | 60.0%     | yes |
| Tier-specific mid bucket  | 9  | **74.1%** | 73.9%     | 69.6%     | no (peak low, mid within 0.2 pp) |
| Tier-specific high bucket | 7  | **78.1%** | 74.4%     | 67.4%     | no (peak low) |

Test 2 (n_low = 113, n_mid = 270, n_high = 10, ICLR 2024).

| Bucket | Bucket size | Low-tier | Medium-tier | High-tier | Peaks where attributed? |
|--------|-------------|----------|-------------|-----------|-------------------------|
| Full low bucket           | 14 | 80.5%     | 80.4%     | **82.1%** | no (peak high, n = 10) |
| Full medium bucket        | 17 | 79.1%     | 79.8%     | **81.8%** | no (peak high, n = 10) |
| Full high bucket          | 15 | 82.2%     | 81.6%     | **82.7%** | yes |
| Tier-specific low bucket  | 6  | **71.2%** | 68.5%     | 68.3%     | yes |
| Tier-specific mid bucket  | 9  | 71.6%     | 71.4%     | **72.2%** | no (peak high, n = 10) |
| Tier-specific high bucket | 7  | **76.1%** | 72.7%     | 71.4%     | no (peak low) |

**Results on the `same_decrease` stratum (negative-factor buckets).**

Test 1 (n_low = 107, n_mid = 353, n_high = 97, ICLR 2025).

| Bucket | Bucket size | Low-tier | Medium-tier | High-tier | Peaks where attributed? |
|--------|-------------|----------|-------------|-----------|-------------------------|
| Full low bucket           | 18 | **26.7%** | 9.2%      | 4.5%      | yes |
| Full medium bucket        | 13 | **27.7%** | 10.5%     | 5.5%      | no (peak low) |
| Full high bucket          | 16 | **22.9%** | 9.3%      | 5.5%      | no (peak low) |
| Tier-specific low bucket  | 7  | **23.9%** | 5.4%      | 1.6%      | yes |
| Tier-specific mid bucket  | 2  | **23.4%** | 4.7%      | 0.5%      | no (peak low) |
| Tier-specific high bucket | 5  | **10.5%** | 4.1%      | 3.7%      | no (peak low) |

Test 2 (n_low = 105, n_mid = 350, n_high = 96, ICLR 2024).

| Bucket | Bucket size | Low-tier | Medium-tier | High-tier | Peaks where attributed? |
|--------|-------------|----------|-------------|-----------|-------------------------|
| Full low bucket           | 18 | **26.7%** | 11.7%     | 5.7%      | yes |
| Full medium bucket        | 13 | **27.3%** | 12.5%     | 6.2%      | no (peak low) |
| Full high bucket          | 16 | **23.4%** | 11.3%     | 7.0%      | no (peak low) |
| Tier-specific low bucket  | 7  | **23.0%** | 8.3%      | 3.6%      | yes |
| Tier-specific mid bucket  | 2  | **17.6%** | 4.7%      | 2.1%      | no (peak low) |
| Tier-specific high bucket | 5  | **10.9%** | 5.7%      | 7.1%      | no (peak low) |

**Interpretation.** The aggregate matrices behave asymmetrically across polarity, and the pattern is best read in the context of a strong baseline effect that pulls detection rates toward the low column in both strata.

On the `increase` side the low-attribution bucket peaks in low-tier interactions in both Test 1 and Test 2, with a monotonic decline across tiers in Test 1 (74.5, 69.2, 60.0 percent). The medium and high buckets do not show diagonal dominance, and the magnitudes of the remaining tier effects are modest (the largest gap is about 14.5 percentage points for the low bucket in Test 1).

On the `same_decrease` side every bucket peaks in the low column in both tests, including the high-attribution bucket. The gap between low and high-tier interactions is an order of magnitude larger than on the `increase` side. Mean negative factors per `same_decrease` interaction are 5.07 at low versus 0.96 at high in Test 1, a five-to-one ratio.

Three candidate explanations account for most of what we observe.

First, rebuttal content is highly asymmetric by initial score. Low-tier reviewers raise more and sharper concerns, and both positive and negative factors are detected at substantially higher rates in low-tier interactions regardless of outcome. This baseline effect is modest on the `increase` side (about 6 percentage points between low and high in Test 1) and dominant on the `same_decrease` side (17 to 25 percentage-point gaps between low and high). Under this baseline any bucket drawn from the same-polarity factors is pulled toward the low column before any tier-specific signal can be added.

Second, LLM-A's attributions were produced from a contrastive induction prompt that compared `increase` with `same_decrease` within each tier. An attribution that reads "this factor is most prevalent at high" is therefore more naturally a within-tier discrimination claim (the factor distinguishes successful from unsuccessful rebuttals when the initial score is high) than a cross-tier prevalence claim (the factor is more common in high-tier interactions than in low-tier interactions). Our matrix measures the second quantity. A within-tier analysis of the negative-factor side shows sharper tier signals in several cells (for example F43 "no explicit commitment to revise" reaches 11, 12, 21 percent at low, medium, and high `same_decrease` interactions in Test 2, with peak at high as attributed), but these do not translate into diagonal dominance at the bucket level once the baseline effect is applied.

Third, many factors are near their detection-rate ceiling across all tiers (notably F01-F08 on the positive side), and the remaining tier-exclusive factors (F18, F41 among others) are rare in the data, so the signal carried by tier-specific factors is limited by both ceilings and base rates.

Taken together, the aggregate matrix supports the polarity layer of the taxonomy without supporting a clean three-way cross-tier localisation for either polarity. The most defensible reading of LLM-A's tier sub-claims is as a within-tier emphasis on which factors help distinguish successful from unsuccessful rebuttals at a given initial score, not as a claim that the factor is more prevalent in that tier in absolute terms. This reading is consistent with both the polarity-replication results (Tests 1 and 2) and with the observed shape of the same-polarity bucket matrices (Test 3).

## Appendix C: Memory-control test by outcome-label swap

The tier swap referenced in the main text operates on a weak axis of the induction, because the `increase` stratum behaves similarly across tiers (Test 3 shows modest cross-tier gaps of about six percentage points in the positive-factor buckets) and therefore leaves little polarity for the swap to disturb. A sharper stress test swaps the outcome labels that the induction actually uses to contrast the two sides of each tier. We report that test here.

**Protocol.** On the same 1000 ICLR 2025 reviewer-author interactions used for the original induction, we silently swap the `increase` and `same_decrease` labels within each of the three tier blocks, holding the tier assignment of each interaction fixed. Under this manipulation, interactions truly belonging to `same_decrease` are presented to LLM-A as "successful", and interactions truly belonging to `increase` are presented as "failed/unchanged". The prompt structure, the per-tier contrastive framing, and the model are identical to the original induction. The raw output of this run is archived in `swapped_induction_outcome_labels.md` alongside this appendix.

**Expected behaviour under the two hypotheses.** If LLM-A extracts factor structure from the interaction text, the features it attributes to "successful" on the swapped input should match the features the original induction attributed to `same_decrease`, and the features it attributes to "failed" should match the original `increase` side. If instead LLM-A retrieves canonical "good rebuttal" patterns from its prior, the same positive-factor set (for example new experiments, active dialogue, baseline completion, structured point-by-point responses) should be attributed to "successful" regardless of the label swap.

**Result.** The empirical content LLM-A cites in the swapped run is unchanged from the original induction; only the label attached to each side flips. Features LLM-A attributes to "failed" under the swap (true outcome = `increase`) include more new experiments at the low tier (+15 percentage points), more inline tables (+19 pp) and more new figures (+16 pp), more author replies (4.3 versus 3.4 at low), more thorough point-by-point structure at the mid tier (61 versus 57 percent), and more code-release mentions at the high tier (35 versus 22 percent). These features map directly onto the original positive factors F07 (baseline completion and comparator strengthening), F06 (concrete empirical evidence), F14 (supplementary material and new visual evidence), F21 (active dialogue and continued engagement), F01 (structured, point-by-point response format), and F09 (comprehensive concern coverage). The original induction had assigned the same features to the `increase` side of the same interactions. Under the label swap, LLM-A continues to locate these features on the same actual data, which now bears the opposite label.

A retrieval-driven run would have attributed these features to "successful" regardless of the swap, because they constitute the canonical good-rebuttal profile in LLM-A's prior. That is not what we observe.

**Narrative rationalisation is label-driven.** The swapped run does produce a narrative layer that differs from the original, organised around three post-hoc rationalisations ("clarification premium", "urgency tax", "verbosity penalty") that construct a plausible story for why the data features now associated with "successful" should be regarded as effective. This narrative-level re-framing is expected when an LLM is handed real data under false labels, and it does not change the factor-level statistics the model reports. The reading we take from the swap is therefore two-layered: the factor-level content is extracted from the interaction text and is stable under the swap, and the narrative framing smooths to the presented label without disturbing the underlying feature attributions.

**Scope and limitations.** The swap is run once on the same 1000 ICLR 2025 interactions used for the original induction; a more exhaustive version would repeat the run across seeds and would feed the swapped-induction factor list into the Gemini classifier (LLM-B) to check that the odds ratios invert on the true verdict. Within the scope reported here, the swap supports the claim made in the main text that the 44-factor taxonomy reflects structure in the reviewer-author interactions rather than retrieval of prior patterns about peer review.
