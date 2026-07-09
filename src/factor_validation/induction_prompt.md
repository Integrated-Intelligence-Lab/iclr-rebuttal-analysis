# Claude factor-induction prompt (LLM-A, Section 3.2)

The 44-factor codebook in `codebook.json` was induced by Claude from a
contrastive sample of 1,000 ICLR 2025 reviewer-author interactions, organised
into six tier x outcome cells. The induction was driven by **two layers** of
prompts: a brief outer prompt that the user typed into the chat, and six
group-specific system prompts that lived inside the input JSON file.

## Outer prompt (verbatim, sent in the chat)

```
Hello Claude,
You are a meta-reviewer analyzing patterns in academic peer review at ICLR 2025. In the file attached, you will find a .json formatting with 6 groups. Each group has an own prompt. Please read each prompt and execute the task for each group then outpot a structured, complete, report in .md format with the 6 .json output files for
each category.
Thank you
```

## Attached file

`classify_input_raw_threads.json` (1,000 interactions, 6 groups). This file
is shipped at:

```
data/results/32_factor_taxonomy/induction_input_1k_threads_iclr2025.json
```

Top-level structure:

```json
{
  "description": "Contrastive cause analysis: one entry per interaction. Raw review threads.",
  "n_total": 1000,
  "groups": {
    "low_increase":       {"tier": "low",  "verdict_group": "increase",       "n": 114, "system_prompt": "...", "items": [...]},
    "low_same_decrease":  {"tier": "low",  "verdict_group": "same+decrease",  "n": 107, "system_prompt": "...", "items": [...]},
    "mid_increase":       {"tier": "mid",  "verdict_group": "increase",       "n": 271, "system_prompt": "...", "items": [...]},
    "mid_same_decrease":  {"tier": "mid",  "verdict_group": "same+decrease",  "n": 353, "system_prompt": "...", "items": [...]},
    "high_increase":      {"tier": "high", "verdict_group": "increase",       "n":  58, "system_prompt": "...", "items": [...]},
    "high_same_decrease": {"tier": "high", "verdict_group": "same+decrease",  "n":  97, "system_prompt": "...", "items": [...]}
  }
}
```

Each `items` entry is one full reviewer-author interaction (review +
rebuttal + any follow-up).

## Per-group system prompts

The six `system_prompt` fields share a common template, varying only in
tier label, outcome direction, and the n's. For tier T (low / mid / high),
outcome group G in {increase, same+decrease}, and contrastive count n_G
(plus the contrast group's count n_other), the prompt is:

```
You are a meta-reviewer analyzing patterns in academic peer review at ICLR 2025.

CONTEXT:
- You are provided with ALL <n_G> reviewer-author interactions from the
  <T (range): description> initial score range where the reviewer
  <INCREASED their score after the rebuttal | did NOT increase their
  score (kept it the same or decreased it)>.
- For comparison, a parallel set of <n_other> interactions from the same
  <T (range): description> tier where the reviewer
  <did NOT raise the score | DID raise the score> has been analyzed separately.
- ICLR uses a discrete rating scale: 1, 3, 5, 6, 8, 10.

YOUR TASK:
Read and analyze EVERY interaction thread provided below, not just a sample.
Your goal is to identify recurring <patterns, strategies, and factors |
patterns of failure, missed opportunities, or ineffective strategies>
across ALL <n_G> threads that distinguish these
<successful rebuttals from unsuccessful ones | unsuccessful rebuttals from
successful ones> in the same initial score tier.

Specifically:
- What <rebuttal strategies, evidence types, or communication approaches |
  patterns of failure, missed opportunities, or ineffective strategies>
  appear consistently across these <n_G> interactions where the score
  <increased | did not increase>?
- What do these authors <do differently from | fail to do, or do poorly,
  compared to> peers in the same <T (range): description> range whose
  rebuttals <failed to move the score | succeeded in raising the score>?
- Ground every factor in concrete patterns you observe across multiple
  threads. Do not speculate or rely on general knowledge about peer review.

Structure your response as a JSON object:
{
  "differentiating_factors": [
    {
      "factor": "<short name for what distinguished this outcome>",
      "description": "<1-2 sentences: what specifically happened and why it mattered>"
    }
  ],
  "what_worked": ["<rebuttal action that helped or would have helped>", ...],
  "what_failed_or_was_missing": ["<action that hurt, was ineffective, or was absent>", ...]
}

Return ONLY the JSON object.
```

Verbatim per-group prompts are recoverable by reading the `system_prompt`
field of any group in the bundled JSON, e.g.:

```python
import json
data = json.load(open("data/results/32_factor_taxonomy/induction_input_1k_threads_iclr2025.json"))
print(data["groups"]["low_increase"]["system_prompt"])
```

## Tier and outcome cell sizes

| Cell | n | Description |
|------|---|-------------|
| low_increase       | 114 | weak papers (1-3) where reviewer raised score |
| low_same_decrease  | 107 | weak papers where reviewer did not raise |
| mid_increase       | 271 | borderline (5-6) where reviewer raised |
| mid_same_decrease  | 353 | borderline where reviewer did not raise |
| high_increase      |  58 | strong (8-10) where reviewer raised |
| high_same_decrease |  97 | strong where reviewer did not raise |
| **Total**          | **1000** | |

## Run protocol

- **Model:** Claude Opus 4.6 with extended thinking enabled (LLM-A).
- **Channel:** Claude.ai (web), with the JSON above attached as a file
  upload, and the brief outer prompt typed in the chat.
- **Repeats:** 5 independent runs to assess factor-list stability.
- **Memory-control swaps:** two additional runs (tier-swap and
  outcome-label-swap) to confirm extraction tracks the data not the
  labels. See `data/results/32_factor_taxonomy/swapped_induction_outcome_labels.md`.
- **Output reconciliation:** the 6 per-group JSON outputs from each of the
  5 runs were manually reconciled by the researcher into the final
  44-factor codebook (`codebook.json`), with polarity labels assigned
  during post-induction reconciliation. 23 factors cross the Bonferroni
  threshold in both Test 1 (in-sample, n=1000 ICLR 2025) and the expanded
  Test 2 (held-out, n=5730 ICLR 2024) using the Gemini Flash 3
  classification prompt in `prompt.py`.

## Remaining provenance to record

- Run date (YYYY-MM-DD):
- Whether each of the 5 repeats was a fresh chat or the same conversation:
- Any deviations from the outer prompt above between runs:
