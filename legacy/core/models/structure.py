"""
core/models/structure.py
allin1 structural analysis output models.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    """An allin1 structural segment (intro, verse, chorus, etc.)."""

    model_config = ConfigDict(frozen=True)

    label: str
    start: float = Field(ge=0.0)
    end: float   = Field(ge=0.0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class StructureResult(BaseModel):
    """Full output of the structure analysis stage."""

    model_config = ConfigDict(frozen=True)

    bpm: float | str            # float when detected; str error message when not
    key: str                    # e.g. "C# Major"
    sections: list[Section]     = Field(default_factory=list)
    beats: list[float]          = Field(default_factory=list)
    metadata: dict[str, str]    = Field(default_factory=dict)  # title, artist

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
