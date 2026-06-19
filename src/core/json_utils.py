"""
JSON parsing helpers for LLM outputs (markdown-wrapped, raw, etc.).
"""

import json
from typing import Any, Optional


def clean_json_from_markdown(content: str) -> str:
    """
    Strip ```json and ``` wrappers from LLM output.
    Handles: ```json\n{...}\n``` and ```\n{...}\n```
    """
    if not content or not isinstance(content, str):
        return content
    cleaned = content.strip()
    if "```json" in cleaned:
        start = cleaned.index("```json") + 7
        cleaned = cleaned[start:]
        if "```" in cleaned:
            cleaned = cleaned[: cleaned.rindex("```")]
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3]
    return cleaned.strip()


def parse_json_content(content: str) -> Optional[dict]:
    """
    Clean markdown-wrapped JSON and parse. Returns None on failure.
    """
    cleaned = clean_json_from_markdown(content)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
