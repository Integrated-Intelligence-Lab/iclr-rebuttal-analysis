"""
Factor-presence classification prompt for rebuttal interactions.

Used by src/score_change/classify_factors_batch.py to validate a 44-factor
codebook induced from 1000 ICLR 2025 rebuttal interactions. The classifier
receives one interaction plus a shuffled list of 44 factors and returns
which factors are present with a short verbatim evidence quote each.

Design choices (rationale in conversation notes 2026-04-18):
- Outcome-blinded: the prompt never reveals tier, verdict_group, or polarity.
- Closed-set: response_schema restricts factor_id to the F01..F44 enum so
  the model cannot invent factors.
- Evidence-grounded: every selected factor must carry a short verbatim
  excerpt from the interaction so results are auditable.
- All-that-apply: explicit instruction that every clearly-present factor
  must be returned, empty list is valid.
"""

from __future__ import annotations

import random
from typing import Any


VALID_FACTOR_IDS = frozenset(f"F{i:02d}" for i in range(1, 45))


SYSTEM_INSTRUCTIONS = """You are an expert annotator coding peer-review rebuttal interactions at ICLR.

You will see ONE reviewer-author interaction (the reviewer's initial review, the authors' rebuttal, any follow-up messages) and a FIXED LIST of candidate factors. Each factor has a stable ID (F01..F44), a short label, and an operational definition.

YOUR TASK
Return every factor from the list that is CLEARLY PRESENT in the interaction, based only on its operational definition.

RULES
1. Use ONLY the provided factor IDs. Do not invent, rename, merge, or split factors. The factor_id field is restricted to F01..F44.
2. "Clearly present" means the operational definition is satisfied by specific text in the interaction. If it is merely plausible, implied, or weakly suggested, do NOT select it.
3. Return ALL applicable factors, not just the most prominent ones. An empty list is a valid answer if none clearly apply.
4. Do NOT consider the outcome of the interaction (whether the score went up, stayed, or decreased). Judge only what is written.
5. For each selected factor, include a SHORT verbatim or near-verbatim quote (<= 25 words) from the interaction that grounds the selection. The quote must come from the interaction text, not paraphrased.
6. Do not select a factor on the basis of the reviewer's wording alone. Factors describe properties of the AUTHORS' rebuttal / response behavior, unless the label explicitly concerns dialogue or reviewer engagement (e.g., F21 "Active dialogue and continued engagement").

OUTPUT FORMAT (strict)
Return a single JSON object and nothing else. No prose, no markdown fences, no preamble, no trailing text.
The object MUST have exactly one key: `factors_present`, whose value is a JSON array.
Each array element MUST be an object with exactly two keys:
  - `factor_id`: a string, one of F01, F02, F03, F04, F05, F06, F07, F08, F09, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F31, F32, F33, F34, F35, F36, F37, F38, F39, F40, F41, F42, F43, F44. Any other value is invalid.
  - `evidence_quote`: a string, a short (<= 25 words) verbatim or near-verbatim excerpt from the interaction.
An empty array is valid. Do not include any factor more than once."""


def render_factor_list(factors: list[dict], rng: random.Random) -> str:
    """Render the 44 factors as a shuffled bullet list for the user message.

    Shuffling per-request neutralises primacy/recency bias. IDs remain stable
    so downstream aggregation maps cleanly back.
    """
    shuffled = factors[:]
    rng.shuffle(shuffled)
    lines = []
    for f in shuffled:
        desc = f.get("description") or ""
        if desc:
            lines.append(f"- {f['id']}: {f['label']} — {desc}")
        else:
            lines.append(f"- {f['id']}: {f['label']}")
    return "\n".join(lines)


def build_user_content(interaction_text: str, factors: list[dict],
                       rng: random.Random) -> str:
    """Compose the user message containing the factor list and the interaction."""
    factor_block = render_factor_list(factors, rng)
    return (
        "FACTOR LIST (shuffled order; IDs are stable):\n"
        f"{factor_block}\n\n"
        "=== BEGIN INTERACTION ===\n"
        f"{interaction_text}\n"
        "=== END INTERACTION ===\n\n"
        "Return the JSON object now."
    )


def build_request_body(
    interaction_text: str,
    factors: list[dict],
    rng: random.Random,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Build the `request` body for a Gemini batch line.

    Uses the same minimal shape as the existing gemini_full_dataset.py script
    (only `generation_config.temperature`). Structured-output fields
    (`response_mime_type`, `response_schema`) are NOT set here because the
    batch JSONL endpoint rejects them. We rely on strict prompt instructions
    plus downstream JSON parsing + factor-id validation.
    """
    user_content = build_user_content(interaction_text, factors, rng)
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": SYSTEM_INSTRUCTIONS + "\n\n" + user_content}],
            }
        ],
        "generation_config": {
            "temperature": temperature,
            "max_output_tokens": 65536,
        },
    }
