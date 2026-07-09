"""
Prepare categorization batch for OpenAI Batch API: build JSONL from formatted batches.

Uses core.paths, prompts.categorization_prompt.
"""

import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from prompts.categorization_prompt import CATEGORIZATION_PROMPT
from core.paths import PROCESSED_DATA_DIR

DATA_DIR_CAT = PROCESSED_DATA_DIR / "categorization_data"
OUTPUT_DIR = PROCESSED_DATA_DIR / "categorization_batches"
MODEL = "gpt-4o"


def create_batch_input(batch_idx: int):
    batch_file = DATA_DIR_CAT / f"batch_{batch_idx}_formatted.json"
    if not batch_file.exists():
        print(f"Batch file not found: {batch_file}")
        return None

    with open(batch_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"\n=== PREPARING BATCH {batch_idx} ===")
    print(f"Papers: {len(papers)}")
    combined = "\n\n".join(p["formatted_text"] for p in papers)
    print(f"Approximate input tokens: {len(combined) // 4:,}")

    req = {
        "custom_id": f"categorization_batch_{batch_idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": CATEGORIZATION_PROMPT},
                {"role": "user", "content": combined},
            ],
            "temperature": 0.3,
            "max_tokens": 8000,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUTPUT_DIR / f"categorization_batch_{batch_idx}_input.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"Created: {out_jsonl}")

    readable = OUTPUT_DIR / f"categorization_batch_{batch_idx}_readable.json"
    with open(readable, "w", encoding="utf-8") as f:
        json.dump(req, f, indent=2, ensure_ascii=False)
    print(f"Created: {readable}")
    return out_jsonl


def main():
    print("=== CATEGORIZATION BATCH PREPARATION ===\n")
    batch_files = sorted(DATA_DIR_CAT.glob("batch_*_formatted.json"))
    print(f"Found {len(batch_files)} batch(es)\n")

    outputs = []
    for bf in batch_files:
        bi = int(bf.stem.split("_")[1])
        p = create_batch_input(bi)
        if p:
            outputs.append(p)

    print(f"\n=== PREPARATION COMPLETE ===")
    print(f"Created {len(outputs)} batch input file(s)")
    print(f"Output: {OUTPUT_DIR}")
    print("\n=== NEXT STEPS ===")
    print("1. Upload: https://platform.openai.com/batches")
    print("2. openai api batches.create -f <input_file.jsonl>")
    print("3. openai api batches.retrieve -i <batch_id>")
    print("4. openai api batches.retrieve-output -i <batch_id> > output.jsonl")


if __name__ == "__main__":
    main()
