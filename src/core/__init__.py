"""
Core utilities for the OpenReview dataset analysis project.
Shared components: paths, I/O, JSON parsing, OpenReview data extraction.
"""

from .paths import BASE_DIR, DATA_DIR, SCRIPTS_DIR, NORMALIZED_DATA_DIR
from .io import load_json, load_jsonl, write_json
from .json_utils import clean_json_from_markdown, parse_json_content

__all__ = [
    "BASE_DIR",
    "DATA_DIR", 
    "SCRIPTS_DIR",
    "NORMALIZED_DATA_DIR",
    "load_json",
    "load_jsonl",
    "write_json",
    "clean_json_from_markdown",
    "parse_json_content",
]
