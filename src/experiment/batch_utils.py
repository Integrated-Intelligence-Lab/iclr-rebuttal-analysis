"""
Shared utilities for LLM batches: review cleaning, sampling, JSONL generation, upload.
"""

import json
import random
from pathlib import Path
from typing import Any, List

# Ensure Scripts is on path for Prompts and core
import sys
_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from core.io import load_json
from core.sampling import get_sampled_indices
from core.openreview_utils import get_official_reviews


def remove_score_fields(review_content: dict) -> dict:
    """Remove scoring fields so they are not seen by the LLM. Modifies a copy.

    Strips all known numeric/scoring fields across ICLR years:
    - 2024+: soundness, presentation, contribution, rating, recommendation
    - 2023:  correctness, technical_novelty_and_significance,
             empirical_novelty_and_significance,
             clarity,_quality,_novelty_and_reproducibility,
             confidence, recommendation
    """
    if not isinstance(review_content, dict):
        return review_content
    cleaned = review_content.copy()
    score_fields = {
        # ICLR 2024+
        "soundness", "presentation", "contribution", "rating", "recommendation",
        # ICLR 2023 (numeric sub-scores on 1-4 or 1-5 scales)
        "confidence", "correctness",
        "technical_novelty_and_significance",
        "empirical_novelty_and_significance",
        "clarity,_quality,_novelty_and_reproducibility",
    }
    for k in list(cleaned.keys()):
        if k.lower().replace(" ", "_") in score_fields or k in score_fields:
            del cleaned[k]
    return cleaned


def select_random_samples(amount: int, filename: Path | str, seed: int = 50) -> List[List[dict]]:
    """
    Load dataset, sample `amount` papers with random.seed(seed), and return a list of
    official review contents (score fields removed) per paper.
    """
    data = load_json(filename)
    keys = list(data.keys())
    indices = get_sampled_indices(len(keys), sample_size=amount, seed=seed)
    out = []
    for i in range(min(amount, len(indices))):
        key = keys[indices[i]]
        paper = data[key]
        official = get_official_reviews(paper.get("interactions", []))
        out.append([remove_score_fields(r.get("content", {})) for r in official])
    return out


def prepare_all_papers(data_path: Path | str) -> tuple[list[list[dict]], list[str]]:
    """Load all papers and return (review_lists, paper_keys).

    Like select_random_samples() but processes ALL papers (no sampling).
    Skips papers with no official reviews.
    """
    data = load_json(data_path)
    keys = list(data.keys())
    reviews = []
    valid_keys = []
    for key in keys:
        paper = data[key]
        official = get_official_reviews(paper.get("interactions", []))
        if official:
            reviews.append([remove_score_fields(r.get("content", {})) for r in official])
            valid_keys.append(key)
    return reviews, valid_keys


def generate_jsonl_line(sample: List[dict], custom_id: int, prompt_instructions: str, model: str = "gpt-5") -> dict:
    """Build one JSONL request for the chat completions batch API."""
    return {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt_instructions},
                {"role": "user", "content": json.dumps(sample, ensure_ascii=False)},
            ],
        },
    }


def generate_upload_file(samples: List[List[dict]], output_path: Path | str, prompt_instructions: str, model: str = "gpt-5") -> None:
    """Write JSONL file for batch upload. One line per sample."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            line = generate_jsonl_line(sample, i, prompt_instructions, model)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def upload_file(filepath: Path | str, client, expires_seconds: int = 2592000):
    """Upload a file for batch processing. Returns the file object."""
    return client.files.create(
        file=open(filepath, "rb"),
        purpose="batch",
        expires_after={"anchor": "created_at", "seconds": expires_seconds},
    )


def process_batch(file_id: str, client, endpoint: str = "/v1/chat/completions", completion_window: str = "24h"):
    """Create a batch job from an uploaded file."""
    return client.batches.create(
        input_file_id=file_id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
