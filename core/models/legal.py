"""
core/models/legal.py
Rights metadata validation and PRO legal links output models.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class MetadataValidationResult(BaseModel):
    """
    Result of pre-flight track rights metadata validation.

    Produced by services/metadata_validator.py before the scan pipeline runs.
    Stored in AnalysisResult so the compliance report can surface intake issues.
    """

    model_config = ConfigDict(frozen=True)

    valid: bool                         # True when all checks pass
    missing_fields: list[str]           = Field(default_factory=list)
    split_total: float                  = 0.0   # sum of supplied writer splits
    split_error: Optional[float]        = None  # abs deviation from 100.0, or None if not supplied
    isrc_valid: bool                    = True  # True when ISRC matches ISO 3901 or was not provided
    rejection_reason: Optional[str]     = None  # human-readable summary; None when valid

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class LegalLinks(BaseModel):
    """PRO repertory search URLs and inferred PRO match for a given track."""

    model_config = ConfigDict(frozen=True)

    ascap: str  = ""
    bmi: str    = ""
    sesac: str  = ""

    # Populated by services/pro_lookup.py — None when MusicBrainz returns no hit
    isrc: Optional[str]           = None   # e.g. "US-ABC-23-12345"
    pro_match: Optional[str]      = None   # e.g. "ASCAP/BMI (US)", "PRS (UK)"
    pro_confidence: Optional[str] = None   # "High"|"Medium"|"Low"; None when pro_match is None (#118)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
