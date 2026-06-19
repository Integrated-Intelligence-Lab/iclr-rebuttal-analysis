"""
OpenReview data helpers: ratings, decisions, interactions, and content extraction.

Changes:
- 2026-02-04: Hardened `get_final_decision` to accept full paper dicts, content dicts,
  and raw venue strings, and to classify into standard decision buckets.
"""

import re
import time
from typing import Any, Dict, List

# Optional: openreview client used only in safe_get_all_notes
# Caller must pass client; we avoid importing at module level.


def safe_get_all_notes(client, forum_id: str, max_retries: int = 5) -> List[Any]:
    """Fetch all notes for a forum with exponential backoff. Returns [] on failure."""
    for attempt in range(max_retries):
        try:
            return client.get_all_notes(forum=forum_id)
        except Exception as e:
            wait = 2 ** attempt
            print(f"⚠️ Error fetching notes for {forum_id}: {e} (retrying in {wait}s)")
            time.sleep(wait)
    print(f"❌ Failed to fetch notes for {forum_id} after {max_retries} attempts.")
    return []


def extract_rating_number(rating_string: Any) -> int | None:
    """Extract numeric rating from various formats.

    Supports:
    - ICLR format: '6: marginally above the acceptance threshold'
    - ICML format: plain integer (2) or string ('2')
    - Any string starting with a number
    """
    if rating_string is None:
        return None

    s = str(rating_string).strip()

    # Try ICLR format first: "6: description..."
    match = re.match(r"^(\d+):", s)
    if match:
        return int(match.group(1))

    # Try plain number (ICML 2025 style)
    match = re.match(r"^(\d+)$", s)
    if match:
        return int(match.group(1))

    # Last resort: extract first number from string
    match = re.search(r"(\d+)", s)
    if match:
        return int(match.group(1))

    return None


def get_final_decision(paper: Any) -> str:
    """
    Derive a normalized final decision label from OpenReview data.

    Accepts:
    - Full paper dicts with a nested ``content.venueid.value`` field (ICLR 2024 style)
    - Content dicts that contain ``venueid`` directly
    - Raw venue/decision strings

    Returns one of: "Rejected", "Poster", "Spotlight", "Oral", "Accepted",
    "Withdrawn", or "Pending".
    """

    venueid: str = ""
    venue: str = ""

    if isinstance(paper, dict):
        # Either a full paper entry (with "content") or a bare content dict.
        if "content" in paper:
            content = paper.get("content") or {}
        else:
            content = paper

        # Check venueid field (contains rejection/withdrawal status)
        venue_field = content.get("venueid") or content.get("decision") or ""
        if isinstance(venue_field, dict):
            venueid = str(venue_field.get("value", "") or "")
        else:
            venueid = str(venue_field or "")

        # Check venue field (contains acceptance category: poster/spotlight/oral)
        venue_val = content.get("venue") or ""
        if isinstance(venue_val, dict):
            venue = str(venue_val.get("value", "") or "")
        else:
            venue = str(venue_val or "")
    elif isinstance(paper, str):
        venueid = paper
    else:
        venueid = ""

    # Combine venueid and venue for matching
    text = f"{venueid} {venue}".lower()

    if "withdraw" in text:
        return "Withdrawn"
    if "reject" in text:
        return "Rejected"
    if "oral" in text:
        return "Oral"
    if "spotlight" in text:
        return "Spotlight"
    if "poster" in text:
        return "Poster"
    if "accept" in text:
        return "Accepted"
    return "Pending"


def _invitations_contain(interaction: dict, *substrings: str) -> bool:
    inv = interaction.get("invitations", [])
    s = " ".join(str(x) for x in inv)
    return any(part in s for part in substrings)


def get_official_reviews(interactions: List[dict]) -> List[dict]:
    """Return interactions that are Official_Review (excludes Rebuttal)."""
    return [
        i
        for i in interactions
        if _invitations_contain(i, "Official_Review") and not _invitations_contain(i, "Rebuttal")
    ]


def get_text_fields_from_content(content: dict) -> List[str]:
    """
    Collect all textual values from common OpenReview content fields.
    Used for word counting and similar. Returns list of non-empty strings.
    """
    text_fields = []
    field_names = [
        "comment", "review", "summary", "strengths", "weaknesses",
        "questions", "soundness", "presentation", "contribution",
        "title", "metareview", "official_comment",
    ]
    for name in field_names:
        val = content.get(name)
        if isinstance(val, dict) and "value" in val:
            txt = val.get("value", "")
        else:
            txt = str(val) if val is not None else ""
        if isinstance(txt, str) and txt.strip():
            text_fields.append(txt)
    return text_fields


def extract_reviews_and_responses(paper_interactions: List[dict]) -> Dict[str, Any]:
    """
    Extract structured reviews and author responses from paper interactions.
    Excludes Decision and Meta_Review.
    Returns: { "reviews": [...], "author_responses": [...] }
    """
    reviews = []
    author_responses = []

    for interaction in paper_interactions:
        inv = interaction.get("invitations", [])
        inv_str = " ".join(str(x) for x in inv)
        if "Decision" in inv_str or "Meta_Review" in inv_str:
            continue

        content = interaction.get("content", {})

        if "Official_Review" in inv_str and "Rebuttal" not in inv_str:
            rec = {}
            for field in [
                "summary", "strengths", "weaknesses", "questions",
                "overall_recommendation", "confidence", "soundness",
                "presentation", "contribution",
            ]:
                v = content.get(field)
                if v is None:
                    continue
                if isinstance(v, dict) and "value" in v:
                    rec[field] = v["value"]
                elif not isinstance(v, dict):
                    rec[field] = v
            if rec:
                reviews.append(rec)

        elif "Rebuttal" in inv_str or "Official_Comment" in inv_str:
            text = None
            for key in ("comment", "rebuttal"):
                v = content.get(key)
                if v is None:
                    continue
                if isinstance(v, dict) and "value" in v:
                    text = v["value"]
                elif not isinstance(v, dict):
                    text = v
                if text:
                    break
            if text:
                author_responses.append({
                    "text": text,
                    "type": "rebuttal" if "Rebuttal" in inv_str else "comment",
                })

    return {"reviews": reviews, "author_responses": author_responses}
