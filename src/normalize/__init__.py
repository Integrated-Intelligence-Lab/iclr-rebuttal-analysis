"""
Data normalization module for OpenReview datasets.

Provides a unified normalized format for ICLR and ICML conference data,
enabling cross-dataset comparisons and simplified analysis code.
"""

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
from .base_converter import BaseConverter

__all__ = [
    "DatasetMeta",
    "RatingValue",
    "ConfidenceValue",
    "Review",
    "AuthorResponse",
    "Comment",
    "Paper",
    "NormalizedDataset",
    "BaseConverter",
]
