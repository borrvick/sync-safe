"""
services/validation/_pure.py
Pure validation rule functions — no class dependencies, no I/O.
"""
from __future__ import annotations

import re

from core.config import CONSTANTS

_ISRC_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}\d{7}$")


def validate_isrc(isrc: str) -> bool:
    """
    Return True when *isrc* conforms to ISO 3901 (ignoring embedded dashes).

    Pure function — no I/O.

    Args:
        isrc: Raw ISRC string, e.g. "US-ABC-23-12345" or "USABC2312345".

    Returns:
        True if the ISRC matches the ISO 3901 pattern, False otherwise.
    """
    cleaned = isrc.strip().replace("-", "").upper()
    return bool(_ISRC_RE.match(cleaned))


def validate_splits(splits: list[float]) -> bool:
    """
    Return True when the writer split percentages sum to 100 % within tolerance.

    Pure function — no I/O.

    Args:
        splits: List of percentage values (e.g. [50.0, 25.0, 25.0]).

    Returns:
        True if abs(sum(splits) - 100.0) <= CONSTANTS.SPLIT_TOLERANCE.
    """
    if not splits:
        return False
    return abs(sum(splits) - 100.0) <= CONSTANTS.SPLIT_TOLERANCE
