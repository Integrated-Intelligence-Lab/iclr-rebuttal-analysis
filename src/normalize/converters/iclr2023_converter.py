"""
Converter for ICLR 2023 dataset format.

This is the "legacy" OpenReview format with:
- invitation (string) instead of invitations (array)
- Content fields NOT wrapped (direct values)
- Combined strength_and_weaknesses field
- recommendation field instead of rating
- summary_of_the_paper instead of summary
"""

from pathlib import Path
from typing import Any

from ..base_converter import BaseConverter
from ..schema import Review, Comment, Paper


class ICLR2023Converter(BaseConverter):
    """Converter for ICLR 2023 format."""

    CONFERENCE = "ICLR"
    YEAR = 2023
    RATING_SCALE_MIN = 1
    RATING_SCALE_MAX = 10
    CONFIDENCE_SCALE_MIN = 1
    CONFIDENCE_SCALE_MAX = 5
    RATING_FIELD = "recommendation"  # ICLR 2023 uses "recommendation"

    @staticmethod
    def normalize_decision(paper: Any) -> tuple[str, str | None]:
        """Override to handle ICLR 2023's non-standard venue strings.

        ICLR 2023 uses:
        - "ICLR 2023 notable top 5%"  -> Oral
        - "ICLR 2023 notable top 25%" -> Spotlight
        - "ICLR 2023 poster"          -> Poster (handled by base)
        - "Submitted to ICLR 2023"    -> Rejected
        """
        # Try base logic first (handles poster, withdrawn, etc.)
        decision, raw = BaseConverter.normalize_decision(paper)
        if decision != "Pending":
            return decision, raw

        # Extract venue text for 2023-specific patterns
        venue = ""
        if isinstance(paper, dict):
            content = paper.get("content") or {}
            if "content" in paper:
                content = paper["content"]
            venue_val = content.get("venue") or ""
            if isinstance(venue_val, dict):
                venue = str(venue_val.get("value", "") or "")
            else:
                venue = str(venue_val or "")

        if "notable top 5%" in venue:
            return "Oral", venue
        if "notable top 25%" in venue:
            return "Spotlight", venue
        if "Submitted to" in venue:
            return "Rejected", venue

        return "Pending", raw

    def _get_invitation_string(self, interaction: dict) -> str:
        """Override to handle 2023's single invitation field."""
        if "invitation" in interaction:
            return str(interaction.get("invitation", ""))
        return super()._get_invitation_string(interaction)

    def is_official_review(self, interaction: dict) -> bool:
        """Check if interaction is an official review (2023 format)."""
        inv_str = self._get_invitation_string(interaction)
        return "Official_Review" in inv_str

    def convert_review(self, interaction: dict) -> Review | None:
        """Convert ICLR 2023 review interaction."""
        if not self.is_official_review(interaction):
            return None

        content = interaction.get("content", {})

        # ICLR 2023 has direct values (not wrapped in {"value": x})
        # Use get_string_field which handles both cases

        # Map 2023 field names to normalized names
        # summary_of_the_paper -> summary
        # strength_and_weaknesses -> strengths_and_weaknesses
        # recommendation -> rating (handled by RATING_FIELD)

        return Review(
            reviewer_id=self.get_reviewer_id(interaction),
            rating=self.extract_rating(content),
            confidence=self.extract_confidence(content),
            summary=self.get_string_field(content, "summary_of_the_paper"),
            strengths=None,  # ICLR 2023 has combined field
            weaknesses=None,
            strengths_and_weaknesses=self.get_string_field(content, "strength_and_weaknesses"),
            questions=self.get_string_field(content, "questions"),
            cdate=interaction.get("cdate"),
            id=interaction.get("id"),
            soundness=self.get_string_field(content, "soundness"),
            presentation=self.get_string_field(content, "presentation"),
            contribution=self.get_string_field(content, "contribution"),
        )

    def convert_comment(self, interaction: dict) -> Comment | None:
        """Convert ICLR 2023 comment interaction (any participant)."""
        if not self.is_comment_interaction(interaction):
            return None

        content = interaction.get("content", {})
        inv_str = self._get_invitation_string(interaction)

        # Extract response text
        text = None
        for field in ("comment", "rebuttal"):
            text = self.get_string_field(content, field)
            if text:
                break

        if not text:
            return None

        comment_type = "rebuttal" if "Rebuttal" in inv_str else "comment"
        participant_type, participant_id = self.classify_participant(interaction)

        return Comment(
            id=interaction.get("id"),
            participant_type=participant_type,
            participant_id=participant_id,
            reply_to=interaction.get("replyto"),
            text=text,
            cdate=interaction.get("cdate"),
            comment_type=comment_type,
        )

    def convert_paper(self, paper_id: str, paper_data: dict) -> Paper:
        """Convert ICLR 2023 paper entry."""
        content = paper_data.get("content", {})
        interactions = paper_data.get("interactions", [])

        # Extract reviews and comments
        reviews = []
        comments = []

        for interaction in interactions:
            review = self.convert_review(interaction)
            if review:
                reviews.append(review)
                continue

            comment = self.convert_comment(interaction)
            if comment:
                comments.append(comment)

        # Get decision
        decision, decision_raw = self.normalize_decision(paper_data)

        # ICLR 2023 uses a different field name for primary_area
        primary_area = (
            self.get_string_field(content, "primary_area")
            or self.get_string_field(
                content, "Please_choose_the_closest_area_that_your_submission_falls_into"
            )
        )

        return Paper(
            id=paper_data.get("id", paper_id),
            forum=paper_data.get("forum", paper_id),
            title=self.get_string_field(content, "title"),
            abstract=self.get_string_field(content, "abstract"),
            authors=self.get_list_field(content, "authors"),
            keywords=self.get_list_field(content, "keywords"),
            primary_area=primary_area,
            decision=decision,
            decision_raw=decision_raw,
            reviews=reviews,
            comments=comments,
        )
