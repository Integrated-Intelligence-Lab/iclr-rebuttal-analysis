"""
Paths and directory constants.
Centralises project roots so scripts work from src/ or project root.
"""

from pathlib import Path

# Project root (parent of src)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories (new structure)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NORMALIZED_DATA_DIR = PROCESSED_DATA_DIR / "normalized"
RESULTS_DIR = DATA_DIR / "results"
SCORE_CHANGE_DIR = RESULTS_DIR / "score_change"
CACHE_DIR = DATA_DIR / "cache"

# Batch JSONL: inputs (to upload) and outputs (API results)
BATCHES_DIR = DATA_DIR / "batches"
BATCH_INPUT_DIR = BATCHES_DIR / "input"
BATCH_OUTPUT_DIR = BATCHES_DIR / "output"

# Source directories
SRC_DIR = BASE_DIR / "src"
TEMPLATES_DIR = SRC_DIR / "templates"
PROMPTS_DIR = SRC_DIR / "prompts"

# Apps directory
APPS_DIR = BASE_DIR / "apps"

# Documentation
DOCS_DIR = BASE_DIR / "docs"

# Legacy compatibility (for transition period)
SCRIPTS_DIR = SRC_DIR  # Maps old SCRIPTS_DIR to new SRC_DIR
