"""
I/O helpers: JSON, JSONL, and generic read/write.
"""

import json
from pathlib import Path
from typing import Any, List


def load_json(filepath: Path | str) -> Any:
    """Load a JSON file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(filepath: Path | str) -> List[dict]:
    """Load a JSONL file as a list of dicts. Skips empty lines."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(content: Any, filepath: Path | str, indent: int = 4) -> None:
    """Write content to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=indent, ensure_ascii=False)


def write_file(content: str, filepath: Path | str) -> None:
    """Write raw string to a file. For non-JSON use."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_normalized_dataset(name: str) -> dict:
    """Load a normalized dataset by name.

    Args:
        name: Dataset name like "ICLR_2024" or "ICML_2025"

    Returns:
        Normalized dataset dict with "_meta" and "papers" keys

    Raises:
        FileNotFoundError: If normalized file doesn't exist
    """
    from .paths import NORMALIZED_DATA_DIR

    # Handle various naming conventions
    # "ICLR_2024" -> "ICLR_2024_normalized.json"
    # "ICLR.cc_2024" -> "ICLR_cc_2024_normalized.json"
    normalized_name = name.replace(".", "_").replace(".cc", "_cc")
    if not normalized_name.endswith("_normalized"):
        normalized_name = f"{normalized_name}_normalized"

    filepath = NORMALIZED_DATA_DIR / f"{normalized_name}.json"

    if not filepath.exists():
        # Try alternative naming
        alt_path = NORMALIZED_DATA_DIR / f"{name}_normalized.json"
        if alt_path.exists():
            filepath = alt_path
        else:
            available = list(NORMALIZED_DATA_DIR.glob("*_normalized.json"))
            available_names = [f.stem.replace("_normalized", "") for f in available]
            raise FileNotFoundError(
                f"Normalized dataset '{name}' not found at {filepath}. "
                f"Available: {available_names}"
            )

    return load_json(filepath)
