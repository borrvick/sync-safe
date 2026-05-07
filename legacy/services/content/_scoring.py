"""
services/content/_scoring.py
Pure verdict mapping for AI lyric authorship detection — no I/O, no GPU.
"""
from __future__ import annotations

from core.config import CONSTANTS


def _compute_verdict(ai_signals: int) -> str:
    """
    Map a signal count to a verdict string.

    Thresholds from CONSTANTS:
        AI_SIGNAL_COUNT_CERTAIN   → "Likely AI"
        AI_SIGNAL_COUNT_UNCERTAIN → "Uncertain"
        else                      → "Likely Human"
    """
    if ai_signals >= CONSTANTS.AI_SIGNAL_COUNT_CERTAIN:
        return "Likely AI"
    if ai_signals >= CONSTANTS.AI_SIGNAL_COUNT_UNCERTAIN:
        return "Uncertain"
    return "Likely Human"
