"""
core/models/transcription.py
Whisper transcript segment model.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """A single time-stamped segment from Whisper."""

    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0.0)
    end: float   = Field(ge=0.0)
    text: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
