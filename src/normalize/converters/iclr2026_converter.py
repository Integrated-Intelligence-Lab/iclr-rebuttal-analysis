"""
Converter for ICLR 2026 dataset format.

Same format as ICLR 2024/2025 (modern OpenReview format).
"""

from .iclr2024_converter import ICLR2024Converter


class ICLR2026Converter(ICLR2024Converter):
    """Converter for ICLR 2026 - same format as 2024/2025."""

    YEAR = 2026
