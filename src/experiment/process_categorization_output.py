"""
Process categorization batch output: extract JSON, save results and summaries.

Uses core.paths, core.json_utils.parse_json_content.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from core.paths import RESULTS_DIR, PROCESSED_DATA_DIR
from core.json_utils import parse_json_content

OUTPUT_DIR = RESULTS_DIR / "categorization"
BATCH_DIR = PROCESSED_DATA_DIR / "categorization_batches"


def process_batch_output(output_file: Path) -> list:
    results = []
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"\n=== PROCESSING {output_file.name} ===")
    print(f"Found {len(lines)} result(s)")

    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        cid = item["custom_id"]
        try:
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            cat = parse_json_content(content)
            if cat is None:
                raise ValueError("parse_json_content returned None")
            results.append({"custom_id": cid, "categorization": cat, "timestamp": datetime.now().isoformat()})
            ss = cat.get("summary_statistics", {})
            print(f"\n✓ {cid}")
            print(f"  Category groups: {ss.get('num_category_groups')}")
            print(f"  Total papers: {ss.get('total_papers')}")
        except Exception as e:
            print(f"\n✗ Error {cid}: {e}")
            err_path = OUTPUT_DIR / f"{cid}_error.txt"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            raw = (item.get("response") or {}).get("body", {}).get("choices", [{}])
            raw = (raw[0] if raw else {}).get("message", {}).get("content", str(item))
            with open(err_path, "w", encoding="utf-8") as ef:
                ef.write(raw if isinstance(raw, str) else str(item))
            print(f"  Saved: {err_path}")

    return results


def save_results(results: list, base_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # JSON
    jp = OUTPUT_DIR / f"{base_name}_results.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {jp}")

    # Summary text
    sp = OUTPUT_DIR / f"{base_name}_summary.txt"
    with open(sp, "w", encoding="utf-8") as f:
        f.write("=== CATEGORIZATION SUMMARY ===\n\n")
        for r in results:
            c = r["categorization"]
            f.write(f"Batch: {r['custom_id']}\n")
            f.write(f"Timestamp: {r['timestamp']}\n")
            f.write(f"Papers: {c['summary_statistics']['total_papers']}\n")
            f.write(f"Category groups: {c['summary_statistics']['num_category_groups']}\n")
            f.write(f"Key insights: {c['summary_statistics']['key_insights']}\n\n")
            for g in c.get("category_groups", []):
                f.write(f"\n## {g['group_name']} ({g['dimension']})\n")
                f.write(f"{g['description']}\n")
                for cat in g.get("categories", []):
                    f.write(f"\n  - {cat['category_name']}: {len(cat['paper_ids'])} papers\n")
                    f.write(f"    {cat['definition']}\n")
                    f.write(f"    Papers: {', '.join(cat['paper_ids'])}\n")
            f.write("\n" + "=" * 60 + "\n\n")
    print(f"Saved: {sp}")

    # CSV mappings
    cp = OUTPUT_DIR / f"{base_name}_mappings.csv"
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["batch_id", "paper_id", "group_name", "category_name"])
        for r in results:
            bid = r["custom_id"]
            for g in r["categorization"].get("category_groups", []):
                for cat in g.get("categories", []):
                    for pid in cat["paper_ids"]:
                        w.writerow([bid, pid, g["group_name"], cat["category_name"]])
    print(f"Saved: {cp}")


def main():
    print("=== CATEGORIZATION OUTPUT PROCESSOR ===")
    outputs = list(BATCH_DIR.glob("*_output.jsonl")) if BATCH_DIR.exists() else []
    if not outputs:
        print(f"\nNo *_output.jsonl in {BATCH_DIR}")
        return

    print(f"\nFound {len(outputs)} output file(s)\n")
    all_results = []
    for p in outputs:
        all_results.extend(process_batch_output(p))

    if all_results:
        save_results(all_results, f"categorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print("\n=== PROCESSING COMPLETE ===")
        print(f"Processed {len(all_results)} batch(es) -> {OUTPUT_DIR}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
