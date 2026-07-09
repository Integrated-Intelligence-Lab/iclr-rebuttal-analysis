"""
Shared helpers for paper-level inclusion/exclusion policies.

Changes:
- 2026-04-09: Added withdrawn/desk-reject exclusion and accepted/rejected helpers.
"""

from typing import Any

try:
    # Test/module mode (project root on sys.path)
    from src.core.openreview_utils import get_final_decision
except ImportError:  # pragma: no cover
    # Script mode (src/ injected on sys.path)
    from core.openreview_utils import get_final_decision


def _extract_decision_text(paper: Any) -> str:
    """Collect decision-related text from raw and normalized paper shapes."""
    if not isinstance(paper, dict):
        return str(paper or "")

    chunks: list[str] = []

    for key in ("decision", "decision_raw", "venue", "venueid"):
        value = paper.get(key)
        if isinstance(value, dict):
            chunks.append(str(value.get("value", "") or ""))
        else:
            chunks.append(str(value or ""))

    content = paper.get("content")
    if isinstance(content, dict):
        for key in ("decision", "decision_raw", "venue", "venueid"):
            value = content.get(key)
            if isinstance(value, dict):
                chunks.append(str(value.get("value", "") or ""))
            else:
                chunks.append(str(value or ""))

    return " ".join(chunks).strip()


def is_withdrawn_or_desk_rejected(paper: Any) -> bool:
    """Return True when paper should be excluded by policy."""
    text = _extract_decision_text(paper).lower()
    if "withdraw" in text:
        return True

    desk_patterns = (
        "desk reject",
        "desk-reject",
        "desk rejected",
        "desk_reject",
        "desk_rejected",
    )
    return any(pattern in text for pattern in desk_patterns)


def is_accepted_or_rejected_non_desk(paper: Any) -> bool:
    """
    Return True for accepted or standard rejected papers.

    Excludes withdrawn and desk-rejected papers.
    """
    if is_withdrawn_or_desk_rejected(paper):
        return False

    decision = get_final_decision(paper)
    accepted = {"Poster", "Spotlight", "Oral", "Accepted"}
    return decision in accepted or decision == "Rejected"

