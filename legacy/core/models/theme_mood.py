"""
core/models/theme_mood.py
Theme and mood analysis output model.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ThemeMoodResult(BaseModel):
    """Output of the theme and mood analysis stage."""

    model_config = ConfigDict(frozen=True)

    themes: list[str]               = Field(default_factory=list)
    mood: str                       = ""
    confidence: float               = 0.0
    groq_enriched: bool             = False
    raw_keywords: list[str]         = Field(default_factory=list)
    # Per-theme confidence scores for UI bars (#167)
    theme_scores: dict[str, float]  = Field(default_factory=dict)
    # Category of the top-ranked theme — "energy" | "emotional" | "seasonal" | "" (#167)
    top_category: str               = ""
    # Optional one-to-two sentence mood summary from Groq enrichment (#169)
    mood_summary: Optional[str]     = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
