"""
Base converter class for OpenReview dataset normalization.

Provides common utilities for value unwrapping, rating extraction,
and decision normalization. Subclasses implement dataset-specific logic.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.core.openreview_utils import (
    extract_rating_number as _extract_rating_number,
    get_final_decision as _get_final_decision,
)

from .schema import (
    DatasetMeta,
    RatingValue,
    ConfidenceValue,
    Review,
    AuthorResponse,
    Comment,
    Paper,
    NormalizedDataset,
)


class BaseConverter(ABC):
    """Abstract base class for dataset converters."""

    # Subclasses must define these
    CONFERENCE: str = ""
    YEAR: int = 0
    RATING_SCALE_MIN: int = 1
    RATING_SCALE_MAX: int = 10
    CONFIDENCE_SCALE_MIN: int = 1
    CONFIDENCE_SCALE_MAX: int = 5

    # Rating field name in raw data
    RATING_FIELD: str = "rating"

    def __init__(self, source_path: Path):
        self.source_path = source_path

    # -------------------------------------------------------------------------
    # Value extraction utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def unwrap_value(field: Any) -> Any:
        """Extract value from {"value": x} wrapper, or return as-is."""
        if isinstance(field, dict) and "value" in field:
            return field["value"]
        return field

    @staticmethod
    def extract_rating_number(rating_string: Any) -> int | None:
        """Extract numeric rating from various formats.

        Supports:
        - ICLR format: '6: marginally above the acceptance threshold'
        - ICML format: plain integer (2) or string ('2')
        """
        if isinstance(rating_string, int):
            return rating_string
        return _extract_rating_number(rating_string)

    @staticmethod
    def normalize_decision(paper: Any) -> tuple[str, str | None]:
        """Normalize decision to standard categories.

        Delegates to core.openreview_utils.get_final_decision() for the
        canonical keyword-matching logic, and extracts the raw string for
        provenance tracking.

        Returns: (normalized_decision, raw_decision_string)
        """
        # Extract raw string for provenance
        raw: str | None = None
        if isinstance(paper, dict):
            content = paper.get("content") or paper
            if "content" in paper:
                content = paper.get("content") or {}
            venueid_field = content.get("venueid") or content.get("decision") or ""
            if isinstance(venueid_field, dict):
                venueid = str(venueid_field.get("value", "") or "")
            else:
                venueid = str(venueid_field or "")
            venue_val = content.get("venue") or ""
            if isinstance(venue_val, dict):
                venue = str(venue_val.get("value", "") or "")
            else:
                venue = str(venue_val or "")
            raw = venueid if venueid else venue if venue else None
        elif isinstance(paper, str):
            raw = paper

        decision = _get_final_decision(paper)
        return decision, raw

    def get_string_field(self, content: dict, field_name: str) -> str | None:
        """Extract a string field, handling {"value": x} wrapper."""
        val = content.get(field_name)
        if val is None:
            return None
        unwrapped = self.unwrap_value(val)
        if unwrapped is None:
            return None
        return str(unwrapped) if unwrapped else None

    def get_list_field(self, content: dict, field_name: str) -> list:
        """Extract a list field, handling {"value": x} wrapper."""
        val = content.get(field_name)
        if val is None:
            return []
        unwrapped = self.unwrap_value(val)
        if isinstance(unwrapped, list):
            return unwrapped
        return []

    # -------------------------------------------------------------------------
    # Rating and confidence extraction
    # -------------------------------------------------------------------------

    def extract_rating(self, content: dict) -> RatingValue:
        """Extract rating from review content."""
        raw_field = content.get(self.RATING_FIELD)
        raw_val = self.unwrap_value(raw_field)
        numeric = self.extract_rating_number(raw_val)
        raw_str = str(raw_val) if raw_val is not None else None

        return RatingValue(
            value=numeric,
            raw=raw_str,
            scale_min=self.RATING_SCALE_MIN,
            scale_max=self.RATING_SCALE_MAX,
        )

    def extract_confidence(self, content: dict) -> ConfidenceValue | None:
        """Extract confidence from review content."""
        raw_field = content.get("confidence")
        if raw_field is None:
            return None
        raw_val = self.unwrap_value(raw_field)
        numeric = self.extract_rating_number(raw_val)
        raw_str = str(raw_val) if raw_val is not None else None

        return ConfidenceValue(
            value=numeric,
            raw=raw_str,
            scale_min=self.CONFIDENCE_SCALE_MIN,
            scale_max=self.CONFIDENCE_SCALE_MAX,
        )

    # -------------------------------------------------------------------------
    # Interaction classification
    # -------------------------------------------------------------------------

    def is_official_review(self, interaction: dict) -> bool:
        """Check if interaction is an official review."""
        inv_str = self._get_invitation_string(interaction)
        return "Official_Review" in inv_str and "Rebuttal" not in inv_str

    def is_comment_interaction(self, interaction: dict) -> bool:
        """Check if interaction is a comment (rebuttal or official comment)."""
        inv_str = self._get_invitation_string(interaction)
        return "Rebuttal" in inv_str or "Official_Comment" in inv_str

    # Backward-compatible alias
    def is_author_response(self, interaction: dict) -> bool:
        """Deprecated: use is_comment_interaction() instead."""
        return self.is_comment_interaction(interaction)

    @staticmethod
    def classify_participant(interaction: dict) -> tuple[str, str | None]:
        """Classify who wrote a comment based on its signatures.

        Returns: (participant_type, participant_id)
        """
        signatures = interaction.get("signatures", [])
        if not signatures:
            return ("unknown", None)
        sig = str(signatures[0])
        if "Authors" in sig:
            return ("author", sig)
        if "Reviewer" in sig:
            return ("reviewer", sig)
        if "Area_Chair" in sig:
            return ("area_chair", sig)
        if "Program_Chair" in sig:
            return ("program_chair", sig)
        return ("unknown", sig or None)

    def _get_invitation_string(self, interaction: dict) -> str:
        """Get invitations as a single string for matching."""
        # Handle both 2023 (invitation) and 2024+ (invitations) formats
        if "invitations" in interaction:
            return " ".join(str(x) for x in interaction.get("invitations", []))
        elif "invitation" in interaction:
            return str(interaction.get("invitation", ""))
        return ""

    def get_reviewer_id(self, interaction: dict) -> str | None:
        """Extract reviewer ID from interaction."""
        signatures = interaction.get("signatures", [])
        if signatures:
            return str(signatures[0])
        return None

    # -------------------------------------------------------------------------
    # Abstract methods for subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def convert_review(self, interaction: dict) -> Review | None:
        """Convert a raw interaction to a normalized Review."""
        pass

    @abstractmethod
    def convert_comment(self, interaction: dict) -> Comment | None:
        """Convert a raw interaction to a normalized Comment."""
        pass

    def convert_author_response(self, interaction: dict) -> Comment | None:
        """Deprecated: use convert_comment() instead."""
        return self.convert_comment(interaction)

    @abstractmethod
    def convert_paper(self, paper_id: str, paper_data: dict) -> Paper:
        """Convert a raw paper entry to a normalized Paper."""
        pass

    # -------------------------------------------------------------------------
    # Main conversion method
    # -------------------------------------------------------------------------

    def convert(self, data: dict) -> NormalizedDataset:
        """Convert entire dataset to normalized format."""
        papers = {}
        total_reviews = 0
        total_comments = 0

        for paper_id, paper_data in data.items():
            paper = self.convert_paper(paper_id, paper_data)
            papers[paper_id] = paper
            total_reviews += len(paper.reviews)
            total_comments += len(paper.comments)

        meta = DatasetMeta.create(
            source_file=self.source_path.name,
            conference=self.CONFERENCE,
            year=self.YEAR,
            paper_count=len(papers),
            review_count=total_reviews,
            comment_count=total_comments,
        )

        return NormalizedDataset(_meta=meta, papers=papers)
