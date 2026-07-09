# Memory-control test: outcome-label swap induction

This file records the raw qualitative output of a second memory-control run of the factor induction, in which the `increase` and `same_decrease` outcome labels were silently swapped on the same 1000 ICLR 2025 reviewer-author interactions used for the original induction. Tier assignment (low, medium, high) was left untouched. The prompt structure, per-tier contrastive framing, and model (Claude, "LLM-A") were identical to the original induction. Under this manipulation, interactions truly belonging to `same_decrease` were presented to LLM-A as "successful", and interactions truly belonging to `increase` were presented as "failed/unchanged".

The purpose of this run is to test whether LLM-A extracts factor structure from the interaction text or retrieves canonical patterns of "what makes a rebuttal work" from its prior. If extraction, the features LLM-A attributes to "successful" here should match the features the original induction attributed to `same_decrease`, and vice versa. If retrieval, LLM-A should attribute the same canonical features (new experiments, dialogue, structured responses, etc.) to "successful" regardless of the label swap.

Below is the narrative output returned by LLM-A on the swapped input. The numbering follows LLM-A's own sectioning. Note that "successful" in this output refers to the presented label, which is the true `same_decrease` outcome, and "failed" refers to the true `increase` outcome.

---

## 3.1 Low Tier (Initial Scores 1--3): 107 Increased, 114 Unchanged/Decreased

The challenge at this tier: Reviewers assigned scores of 1--3 typically identified fundamental issues with novelty, significance, or methodological soundness. These are not concerns that additional experiments can easily resolve.

What distinguished successful rebuttals:

Successful low-tier rebuttals focused on clarification over volume. The strongest differentiator was the use of explicit clarification language ("to clarify," "let us clarify") at 52% vs. 48% in the failed group. Meanwhile, failed rebuttals provided significantly more new experiments (+15 pp), inline tables (+19 pp), and new figures (+16 pp), all without moving the score.

The most dramatic gap was in urgency language: 68% of failed rebuttals used deadline-pressure language vs. only 47% of successful ones, a 22 percentage-point difference. This was the largest single pattern gap in the entire dataset.

Successful authors appeared to identify the reviewer's core conceptual objection, often a misunderstanding of the paper's scope, contribution, or positioning, and addressed it directly. Failed authors tried to overwhelm the objection with evidence, which did not resolve the underlying concern.

Key statistics:

- Successful threads averaged 3.4 author replies; failed threads averaged 4.3
- 53% of successful threads ended with the reviewer expressing appreciation; 42% of failed threads ended with the reviewer explicitly discussing their score decision
- Defensiveness was similar between groups (~31%), suggesting tone alone was not the differentiator

## 3.2 Mid Tier (Initial Scores 5--6): 353 Increased, 271 Unchanged/Decreased

The challenge at this tier: Mid-tier papers are borderline. Reviewers see genuine merit but have specific concerns about execution, evaluation, or completeness. This is the tier where rebuttals have the most leverage, and the largest absolute number of interactions (624 total).

What distinguished successful rebuttals:

The mid tier showed evidence that strategic targeting outperforms comprehensive coverage. Failed rebuttals were more likely to address all listed weaknesses (40% vs. 34%) using a thorough point-by-point structure (61% vs. 57%), yet this completeness did not translate into score increases. Successful rebuttals appeared to identify the 1--2 pivotal concerns and address those with precision.

The reviewer final comment pattern was the starkest differentiator at this tier: in failed threads, 68% of final reviewer comments explicitly discussed their score decision, the highest rate in the entire dataset. In successful threads, only 10% did. This suggests that in successful interactions, the quality of the response spoke for itself; the reviewer simply raised the score without extensive deliberation. In failed interactions, reviewers felt compelled to explain why they were not changing their score, often acknowledging the effort while stating the core concern remained.

Key statistics:

- Clarification language: 39% (successful) vs. 34% (failed)
- Revision references: 43% (successful) vs. 51% (failed)
- Defensive tone: 15% vs. 14% (nearly identical, mid tier was the least defensive overall)
- Promises of future work: 32% vs. 35%

## 3.3 High Tier (Initial Scores 8--10): 97 Increased, 58 Unchanged/Decreased

The challenge at this tier: High-tier papers are already strong. Reviewer concerns are typically about presentation, scope, specific technical assumptions, or positioning. The bar for further improvement is high because the reviewer is already positively disposed.

What distinguished successful rebuttals:

High-tier successful rebuttals exhibited the most restrained, confident communication style of any group. Only 5% used defensive language (lowest of all groups), they averaged only 2.2 author replies per thread (lowest), and 21% kept total author content under 3,000 characters (highest concise rate). These authors trusted the strength of their work and addressed concerns with precision.

The most counterintuitive pattern at this tier was the code data release gap: 35% of failed high-tier rebuttals mentioned code release vs. only 22% of successful ones. Similarly, 43% of failed rebuttals promised future work (vs. 29% successful), the highest future-promise rate of any group. This suggests that over-investment in peripheral signals (reproducibility, future extensions) distracted from the targeted response the reviewer needed.

Key statistics:

- 62% of successful threads ended with reviewer appreciation (highest of any group)
- New experiments: 36% (successful) vs. 41% (failed)
- Avg reply length: 2,608 chars (successful) vs. 2,529 chars (failed), similar
- Avg replies: 2.2 (successful) vs. 2.7 (failed)

## 4. Cross-Tier Patterns

### 4.1 The "Clarification Premium"

Across all tiers, successful rebuttals showed a slight but consistent advantage in clarification-focused language. At the low tier, the gap was 52% vs. 48%; at mid, 39% vs. 34%; at high, the rates were equal (15% each, because high-tier papers had fewer misunderstandings to clarify). This suggests the most effective rebuttal action is not adding new evidence but ensuring the reviewer correctly understands the existing contribution.

### 4.2 The "Urgency Tax"

Deadline-pressure language and score-nudging correlated negatively with success at every tier. The pattern was most dramatic at the low tier (47% vs. 68%) but present throughout. Reviewers appear to penalize, or at least not reward, authors who signal desperation about the outcome.

### 4.3 The "Verbosity Penalty"

Across all tiers, failed rebuttals used more words, more replies, more structural formatting, and more inline evidence. The relationship between response volume and success was consistently negative. This is perhaps the most actionable finding: shorter, more targeted rebuttals outperform longer, more comprehensive ones.

### 4.4 Defensive Tone: Not the Problem It's Assumed to Be

Contrary to common advice, defensiveness rates were remarkably similar between successful and failed groups at every tier (low: 31% vs. 32%; mid: 15% vs. 14%; high: 5% vs. 7%). The issue is not whether authors push back on criticism, but whether they push back with new substance or merely restate their original claims.

### 4.5 The Reviewer's Mind Is Often Already Made Up

A striking pattern across the same/decrease groups: reviewers' final comments frequently expressed appreciation for the effort while maintaining their score. At the low tier, 42% of failed threads ended with explicit score decisions; at the mid tier, 68%. This suggests that for a significant fraction of interactions, the reviewer's assessment was driven by the paper itself, and no rebuttal, however thorough, could shift a concern that was fundamentally about the paper's contribution or design.

---

## Mapping to the original 44-factor taxonomy

The empirical patterns LLM-A cites under the swapped labels are unchanged from the original induction; only the label attached to each side flipped. Below we map the features LLM-A attributes to each side of the swapped run onto the original 44-factor taxonomy (in which F01--F23 are positive and F24--F44 are negative, as declared by LLM-A on the unswapped run).

Features LLM-A attributed to "failed" rebuttals in the swapped run (true outcome = `increase`):

| Swapped-run feature                        | Original-taxonomy factor (declared polarity) |
|--------------------------------------------|----------------------------------------------|
| More new experiments (+15 pp low)          | F07 baseline completion (positive), F06 concrete empirical evidence (positive) |
| More inline tables (+19 pp low)            | F14 supplementary material and new visual evidence (positive) |
| More new figures (+16 pp low)              | F14 supplementary material and new visual evidence (positive) |
| More author replies (4.3 vs 3.4 at low)    | F21 active dialogue and continued engagement (positive) |
| More point-by-point structure (61% vs 57% mid) | F01 structured, point-by-point response format (positive) |
| More code release mentions (35% vs 22% high)   | F14 supplementary material and new visual evidence (positive) |
| Addresses all listed weaknesses (40% vs 34% mid) | F09 comprehensive concern coverage (positive) |

Features LLM-A attributed to "successful" rebuttals in the swapped run (true outcome = `same_decrease`):

| Swapped-run feature                        | Original-taxonomy factor (declared polarity) |
|--------------------------------------------|----------------------------------------------|
| Fewer replies, shorter content             | Low values of F21 active dialogue (positive), i.e., factors on the `same_decrease` side |
| Less point-by-point, less comprehensive    | Low values of F01, F09 (positive) |
| Less new experiments, less code release    | Low values of F07, F14 (positive) |
| Restrained, confident style at high tier   | Absence of the effortful positive factors; consistent with high-tier `same_decrease` where the paper already stands on its own |

The empirical content of the swapped run therefore matches the original induction: the features that mark the `increase` side are the same set of positive factors under both runs. What the swap does is pin the label "successful" onto the other side, and LLM-A then generates a narrative (clarification premium, urgency tax, verbosity penalty) that rationalises why these features constitute success. The factor-level statistics do not change with the label; the narrative framing does.

This is consistent with extraction from the interaction text rather than retrieval of prior patterns. A retrieval-driven run would have attributed new experiments, dialogue, baseline completion, and point-by-point structure to "successful" regardless of which side the label was pinned to, because those are the canonical features of a good rebuttal in LLM-A's prior. That is not what we observe.

The two narrative artefacts of the swap ("urgency tax" and "verbosity penalty") are expected post-hoc rationalisations that an LLM will generate when handed real data under false labels, and they do not undermine the extraction claim at the factor level.
