"""
Normalized schema definitions for OpenReview datasets.

Uses dataclasses for clean serialization and validation.
All fields follow the unified schema documented in the normalization plan.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class DatasetMeta:
    """Metadata about the normalized dataset."""
    schema_version: str
    source_file: str
    conference: str
    year: int
    normalized_date: str
    paper_count: int
    review_count: int
    comment_count: int = 0

    @classmethod
    def create(
        cls,
        source_file: str,
        conference: str,
        year: int,
        paper_count: int,
        review_count: int,
        comment_count: int = 0,
    ) -> "DatasetMeta":
        """Factory method with auto-populated fields."""
        return cls(
            schema_version="1.1.0",
            source_file=source_file,
            conference=conference,
            year=year,
            normalized_date=datetime.now().strftime("%Y-%m-%d"),
            paper_count=paper_count,
            review_count=review_count,
            comment_count=comment_count,
        )


@dataclass
class RatingValue:
    """Normalized rating with scale metadata."""
    value: int | None
    raw: str | None
    scale_min: int
    scale_max: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfidenceValue:
    """Normalized confidence with scale metadata."""
    value: int | None
    raw: str | None
    scale_min: int
    scale_max: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Review:
    """Normalized review data."""
    reviewer_id: str | None
    rating: RatingValue
    confidence: ConfidenceValue | None
    summary: str | None
    strengths: str | None
    weaknesses: str | None
    strengths_and_weaknesses: str | None
    questions: str | None
    cdate: int | None
    # Additional fields that may be present
    id: str | None = None
    soundness: str | None = None
    presentation: str | None = None
    contribution: str | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "reviewer_id": self.reviewer_id,
            "rating": self.rating.to_dict(),
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "strengths_and_weaknesses": self.strengths_and_weaknesses,
            "questions": self.questions,
            "cdate": self.cdate,
        }
        # Only include optional fields if they have values
        if self.soundness:
            d["soundness"] = self.soundness
        if self.presentation:
            d["presentation"] = self.presentation
        if self.contribution:
            d["contribution"] = self.contribution
        return d


@dataclass
class AuthorResponse:
    """Deprecated: use Comment instead. Kept for backward compatibility."""
    reply_to: str | None
    text: str | None
    cdate: int | None
    response_type: str = "comment"  # "rebuttal" or "comment"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Comment:
    """Normalized comment with participant tracking.

    Replaces AuthorResponse to capture ALL Official_Comment interactions
    (author, reviewer, area_chair, program_chair) with their identity.
    """
    id: str | None
    participant_type: str  # "author" | "reviewer" | "area_chair" | "program_chair" | "unknown"
    participant_id: str | None  # Full signature string (links reviewer comments to their Review)
    reply_to: str | None  # Parent interaction ID (review or comment)
    text: str | None
    cdate: int | None
    comment_type: str = "comment"  # "rebuttal" or "comment"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Paper:
    """Normalized paper data with reviews and comments."""
    id: str
    forum: str
    title: str | None
    abstract: str | None
    authors: list[str]
    keywords: list[str]
    primary_area: str | None
    decision: str
    decision_raw: str | None
    reviews: list[Review]
    comments: list[Comment]

    @property
    def author_responses(self) -> list[Comment]:
        """Backward-compatible accessor: returns only author comments."""
        return [c for c in self.comments if c.participant_type == "author"]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "forum": self.forum,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "keywords": self.keywords,
            "primary_area": self.primary_area,
            "decision": self.decision,
            "decision_raw": self.decision_raw,
            "reviews": [r.to_dict() for r in self.reviews],
            "comments": [c.to_dict() for c in self.comments],
        }


@dataclass
class NormalizedDataset:
    """Complete normalized dataset with metadata and papers."""
    _meta: DatasetMeta
    papers: dict[str, Paper]

    def to_dict(self) -> dict:
        return {
            "_meta": asdict(self._meta),
            "papers": {k: v.to_dict() for k, v in self.papers.items()},
        }

    def review_count(self) -> int:
        """Count total reviews across all papers."""
        return sum(len(p.reviews) for p in self.papers.values())

    def comment_count(self) -> int:
        """Count total comments across all papers."""
        return sum(len(p.comments) for p in self.papers.values())


def validate_rating(rating: RatingValue) -> list[str]:
    """Validate a rating value. Returns list of error messages."""
    errors = []
    if rating.value is not None:
        if rating.value < rating.scale_min or rating.value > rating.scale_max:
            errors.append(
                f"Rating {rating.value} outside scale [{rating.scale_min}, {rating.scale_max}]"
            )
    return errors


def validate_paper(paper: Paper) -> list[str]:
    """Validate a paper. Returns list of error messages."""
    errors = []
    if not paper.id:
        errors.append("Paper missing id")
    if not paper.forum:
        errors.append("Paper missing forum")
    for i, review in enumerate(paper.reviews):
        rating_errors = validate_rating(review.rating)
        for err in rating_errors:
            errors.append(f"Review {i}: {err}")
    return errors
