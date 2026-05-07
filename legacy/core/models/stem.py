"""
core/models/stem.py
Stem and alternate-mix validation output model.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StemValidationResult(BaseModel):
    """Output of stereo/mono compatibility and phase alignment analysis."""

    model_config = ConfigDict(frozen=True)

    mono_compatible: bool        # True if cancellation < MONO_CANCELLATION_DB_WARN
    phase_correlation: float     # Pearson L/R correlation [-1, 1]
    cancellation_db: float       # dB loss in mono sum vs stereo RMS; negative = cancellation
    mid_side_ratio: float        # Side/Mid energy ratio; -1.0 if mono or undefined
    flags: list[str]             = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
