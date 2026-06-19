"""
CLI for OpenReview dataset normalization.

Usage:
    python -m src.normalize.cli normalize --input data/raw/ICLR.cc_2024.json
    python -m src.normalize.cli normalize-all --input-dir data/raw
    python -m src.normalize.cli info --input data/raw/ICLR.cc_2024.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.io import load_json, write_json
from src.core.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.normalize.converters import (
    ICLR2024Converter,
    ICLR2025Converter,
    ICLR2026Converter,
    ICLR2023Converter,
    ICML2025Converter,
)


# Registry mapping filename patterns to converters
CONVERTER_REGISTRY = {
    "ICLR.cc_2023": ICLR2023Converter,
    "ICLR.cc_2024": ICLR2024Converter,
    "ICLR.cc_2025": ICLR2025Converter,
    "ICLR.cc_2026": ICLR2026Converter,
    "ICML.cc_2025": ICML2025Converter,
}

# Output directory
NORMALIZED_DIR = PROCESSED_DATA_DIR / "normalized"


def detect_converter(filepath: Path):
    """Detect appropriate converter based on filename."""
    stem = filepath.stem
    for pattern, converter_cls in CONVERTER_REGISTRY.items():
        if pattern in stem:
            return converter_cls
    return None


def get_output_path(input_path: Path) -> Path:
    """Generate output path for normalized file."""
    # ICLR.cc_2024.json -> ICLR_2024_normalized.json
    stem = input_path.stem
    parts = stem.replace(".cc_", "_").replace(".", "_")
    return NORMALIZED_DIR / f"{parts}_normalized.json"


def normalize_dataset(input_path: Path, output_path: Path | None = None) -> dict:
    """Normalize a single dataset file.

    Returns statistics dict with paper_count, review_count, etc.
    """
    converter_cls = detect_converter(input_path)
    if converter_cls is None:
        raise ValueError(f"No converter found for {input_path.name}")

    print(f"Loading {input_path.name}...")
    data = load_json(input_path)
    print(f"  Found {len(data)} papers")

    print(f"Converting with {converter_cls.__name__}...")
    converter = converter_cls(input_path)
    normalized = converter.convert(data)

    if output_path is None:
        output_path = get_output_path(input_path)

    print(f"Writing to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(normalized.to_dict(), output_path)

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "conference": normalized._meta.conference,
        "year": normalized._meta.year,
        "paper_count": normalized._meta.paper_count,
        "review_count": normalized._meta.review_count,
        "comment_count": normalized._meta.comment_count,
    }

    print(
        f"  Normalized: {stats['paper_count']} papers, "
        f"{stats['review_count']} reviews, {stats['comment_count']} comments"
    )
    return stats


def normalize_all(input_dir: Path) -> list[dict]:
    """Normalize all supported datasets in a directory."""
    results = []

    for pattern in CONVERTER_REGISTRY.keys():
        # Find matching files
        for ext in [".json"]:
            files = list(input_dir.glob(f"*{pattern}*{ext}"))
            for filepath in files:
                try:
                    stats = normalize_dataset(filepath)
                    results.append(stats)
                except Exception as e:
                    print(f"Error normalizing {filepath}: {e}")
                    results.append({"input": str(filepath), "error": str(e)})

    # Write manifest
    manifest_path = NORMALIZED_DIR / "manifest.json"
    manifest = {
        "normalized_files": [
            {
                "filename": Path(r["output"]).name if "output" in r else None,
                "conference": r.get("conference"),
                "year": r.get("year"),
                "paper_count": r.get("paper_count"),
                "review_count": r.get("review_count"),
                "comment_count": r.get("comment_count"),
            }
            for r in results
            if "output" in r
        ]
    }
    write_json(manifest, manifest_path)
    print(f"\nManifest written to {manifest_path}")

    return results


def show_info(input_path: Path) -> None:
    """Show information about a dataset without converting."""
    print(f"Analyzing {input_path.name}...")
    data = load_json(input_path)

    print(f"  Papers: {len(data)}")

    # Count reviews
    review_count = 0
    papers_with_reviews = 0

    for paper_id, paper in data.items():
        interactions = paper.get("interactions", [])
        paper_reviews = 0
        for i in interactions:
            # Check for official review
            inv_str = " ".join(str(x) for x in i.get("invitations", []))
            if not inv_str:
                inv_str = str(i.get("invitation", ""))
            if "Official_Review" in inv_str and "Rebuttal" not in inv_str:
                paper_reviews += 1

        if paper_reviews > 0:
            papers_with_reviews += 1
        review_count += paper_reviews

    print(f"  Papers with reviews: {papers_with_reviews}")
    print(f"  Total reviews: {review_count}")
    print(f"  Avg reviews/paper: {review_count/len(data) if data else 0:.1f}")

    # Detect converter
    converter_cls = detect_converter(input_path)
    if converter_cls:
        print(f"  Suggested converter: {converter_cls.__name__}")
    else:
        print("  No converter found for this file")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize OpenReview datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Normalize a single file
    python -m src.normalize.cli normalize --input data/raw/ICLR.cc_2024.json

    # Normalize all supported files in data/raw
    python -m src.normalize.cli normalize-all

    # Show info about a dataset
    python -m src.normalize.cli info --input data/raw/ICLR.cc_2024.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # normalize command
    norm_parser = subparsers.add_parser("normalize", help="Normalize a single dataset")
    norm_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input JSON file"
    )
    norm_parser.add_argument("--output", "-o", type=Path, help="Output JSON file")

    # normalize-all command
    all_parser = subparsers.add_parser(
        "normalize-all", help="Normalize all supported datasets"
    )
    all_parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw data files",
    )

    # info command
    info_parser = subparsers.add_parser("info", help="Show dataset info")
    info_parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input JSON file"
    )

    args = parser.parse_args()

    if args.command == "normalize":
        normalize_dataset(args.input, args.output)
    elif args.command == "normalize-all":
        results = normalize_all(args.input_dir)
        print(f"\nNormalized {len([r for r in results if 'output' in r])} datasets")
    elif args.command == "info":
        show_info(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
