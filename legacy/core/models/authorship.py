"""
core/models/authorship.py
AI lyric authorship detection output models.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._types import AIVerdict


class SectionAuthorshipResult(BaseModel):
    """
    Lightweight per-section authorship result (#156).

    No RoBERTa score — running the GPU classifier per section is prohibitively
    expensive. Sections with < 4 lines or < 80 chars are omitted from the dict
    entirely; callers use .get() and treat missing keys as 'Insufficient data'.
    """

    model_config = ConfigDict(frozen=True)

    verdict: AIVerdict                  = "Likely Human"
    signal_count: float                 = 0.0  # float to support 0.5-weight phrase signal (#160)
    feature_notes: list[str]            = Field(default_factory=list)
    scores: dict[str, Optional[float]]  = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AuthorshipResult(BaseModel):
    """Output of the AI lyric authorship detection stage."""

    model_config = ConfigDict(frozen=True)

    verdict: AIVerdict                  = "Likely Human"
    signal_count: float                 = 0.0  # float to support 0.5-weight phrase signal (#160)
    roberta_score: Optional[float]      = None  # None when model not run or insufficient data
    feature_notes: list[str]            = Field(default_factory=list)
    scores: dict[str, Optional[float]]  = Field(default_factory=dict)
    skip_reason: Optional[str]          = None  # "instrumental" | "too_short" | "short" | None
    # Per-section breakdown keyed by normalised label e.g. "CHORUS" (#156).
    # Sections below the minimum data threshold are omitted; treat missing = 'Insufficient data'.
    per_section: dict[str, SectionAuthorshipResult] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
