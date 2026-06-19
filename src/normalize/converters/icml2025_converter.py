"""
Converter for ICML 2025 dataset format.

ICML 2025 uses the modern OpenReview format but with different field names:
- overall_recommendation instead of rating (1-6 scale, not 1-10)
- other_strengths_and_weaknesses (combined field)
- Different review structure fields
"""

from pathlib import Path

from ..base_converter import BaseConverter
from ..schema import Review, Comment, Paper


class ICML2025Converter(BaseConverter):
    """Converter for ICML 2025 format."""

    CONFERENCE = "ICML"
    YEAR = 2025
    RATING_SCALE_MIN = 1
    RATING_SCALE_MAX = 6  # ICML uses 1-6 scale
    CONFIDENCE_SCALE_MIN = 1
    CONFIDENCE_SCALE_MAX = 5
    RATING_FIELD = "overall_recommendation"  # ICML 2025 uses this

    def convert_review(self, interaction: dict) -> Review | None:
        """Convert ICML 2025 review interaction."""
        if not self.is_official_review(interaction):
            return None

        content = interaction.get("content", {})

        # Check if this review has actual content (not just a comment)
        if not content.get(self.RATING_FIELD):
            return None

        return Review(
            reviewer_id=self.get_reviewer_id(interaction),
            rating=self.extract_rating(content),
            confidence=self.extract_confidence(content),
            summary=self.get_string_field(content, "summary"),
            strengths=None,  # ICML 2025 uses combined field
            weaknesses=None,
            strengths_and_weaknesses=self.get_string_field(
                content, "other_strengths_and_weaknesses"
            ),
            questions=self.get_string_field(content, "questions_for_authors"),
            cdate=interaction.get("cdate"),
            id=interaction.get("id"),
            # ICML has different additional fields
            soundness=None,
            presentation=None,
            contribution=None,
        )

    def convert_comment(self, interaction: dict) -> Comment | None:
        """Convert ICML 2025 comment interaction (any participant)."""
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
        """Convert ICML 2025 paper entry."""
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
