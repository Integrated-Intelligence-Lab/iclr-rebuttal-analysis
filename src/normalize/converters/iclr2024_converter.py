"""
Converter for ICLR 2024 and 2025 dataset format.

This is the "modern" OpenReview format with:
- invitations (array) instead of invitation (string)
- Content fields wrapped as {"value": x}
- Separate strengths/weaknesses fields
- rating field for reviewer scores
"""

from pathlib import Path

from ..base_converter import BaseConverter
from ..schema import Review, Comment, Paper, RatingValue, ConfidenceValue


class ICLR2024Converter(BaseConverter):
    """Converter for ICLR 2024/2025 format."""

    CONFERENCE = "ICLR"
    YEAR = 2024
    RATING_SCALE_MIN = 1
    RATING_SCALE_MAX = 10
    CONFIDENCE_SCALE_MIN = 1
    CONFIDENCE_SCALE_MAX = 5
    RATING_FIELD = "rating"

    def convert_review(self, interaction: dict) -> Review | None:
        """Convert ICLR 2024 review interaction."""
        if not self.is_official_review(interaction):
            return None

        content = interaction.get("content", {})

        return Review(
            reviewer_id=self.get_reviewer_id(interaction),
            rating=self.extract_rating(content),
            confidence=self.extract_confidence(content),
            summary=self.get_string_field(content, "summary"),
            strengths=self.get_string_field(content, "strengths"),
            weaknesses=self.get_string_field(content, "weaknesses"),
            strengths_and_weaknesses=None,  # ICLR 2024 has separate fields
            questions=self.get_string_field(content, "questions"),
            cdate=interaction.get("cdate"),
            id=interaction.get("id"),
            soundness=self.get_string_field(content, "soundness"),
            presentation=self.get_string_field(content, "presentation"),
            contribution=self.get_string_field(content, "contribution"),
        )

    def convert_comment(self, interaction: dict) -> Comment | None:
        """Convert ICLR 2024 comment interaction (any participant)."""
        if not self.is_comment_interaction(interaction):
            return None

        content = interaction.get("content", {})
        inv_str = self._get_invitation_string(interaction)

        # Extract response text from comment or rebuttal field
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
        """Convert ICLR 2024 paper entry."""
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

        return Paper(
            id=paper_data.get("id", paper_id),
            forum=paper_data.get("forum", paper_id),
            title=self.get_string_field(content, "title"),
            abstract=self.get_string_field(content, "abstract"),
            authors=self.get_list_field(content, "authors"),
            keywords=self.get_list_field(content, "keywords"),
            primary_area=self.get_string_field(content, "primary_area"),
            decision=decision,
            decision_raw=decision_raw,
            reviews=reviews,
            comments=comments,
        )


class ICLR2025Converter(ICLR2024Converter):
    """Converter for ICLR 2025 - same format as 2024."""

    YEAR = 2025
