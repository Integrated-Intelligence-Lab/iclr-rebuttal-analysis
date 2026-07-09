"""
Extract reviewer-author interactions from OpenReview data for the categorization task.
Excludes meta-reviews and decisions. Builds batches of papers for LLM categorization.

Uses core.paths, core.openreview_utils.extract_reviews_and_responses.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from core.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from core.openreview_utils import extract_reviews_and_responses

DATA_FILE = RAW_DATA_DIR / "ICML.cc_2025.parquet"
OUTPUT_DIR = PROCESSED_DATA_DIR / "categorization_data"
BATCH_SIZE = 15
MAX_PAPERS = 15


def format_paper_for_llm(paper_id: str, title: str, interactions: Dict[str, Any]) -> Dict[str, Any]:
    """Format one paper's reviews and author responses as text for the LLM."""
    parts = [f"=== PAPER {paper_id}: {title} ===\n"]
    for i, r in enumerate(interactions["reviews"], 1):
        parts.append(f"\n--- Reviewer {i} ---")
        for k, v in r.items():
            parts.append(f"{k.upper()}: {v}\n")
    if interactions["author_responses"]:
        parts.append("\n--- Author Responses ---")
        for i, resp in enumerate(interactions["author_responses"], 1):
            parts.append(f"\nResponse {i}: {resp['text']}\n")
    return {
        "paper_id": paper_id,
        "title": title,
        "formatted_text": "\n".join(parts),
        "num_reviews": len(interactions["reviews"]),
        "num_responses": len(interactions["author_responses"]),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    df = pd.read_parquet(DATA_FILE)
    print(f"Total papers: {len(df)}")

    df_with = df[df["interactions"].apply(len) > 0]
    print(f"Papers with interactions: {len(df_with)}")
    df_sample = df_with.head(MAX_PAPERS)
    print(f"Processing {len(df_sample)} papers\n")

    formatted = []
    for idx, row in df_sample.iterrows():
        paper_id = f"P{len(formatted):04d}"
        title = row.get("content.title.value", "Untitled") or "Untitled"
        inter = extract_reviews_and_responses(row["interactions"])
        if not inter["reviews"]:
            continue
        rec = format_paper_for_llm(paper_id, title, inter)
        formatted.append(rec)
        print(f"{paper_id}: {title[:60]}...")
        print(f"  Reviews: {rec['num_reviews']}, Responses: {rec['num_responses']}")

    print(f"\n=== EXTRACTED {len(formatted)} PAPERS ===")
    batches = [formatted[i : i + BATCH_SIZE] for i in range(0, len(formatted), BATCH_SIZE)]
    print(f"Created {len(batches)} batch(es) of up to {BATCH_SIZE} papers each\n")

    for bi, batch in enumerate(batches):
        p = OUTPUT_DIR / f"batch_{bi}_formatted.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)
        print(f"Saved batch {bi} to {p}")
        print(f"  Papers: {[x['paper_id'] for x in batch]}")
        total_chars = sum(len(x["formatted_text"]) for x in batch)
        print(f"  Approximate tokens: {total_chars // 4:,}")

    print("\n=== EXTRACTION COMPLETE ===")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
