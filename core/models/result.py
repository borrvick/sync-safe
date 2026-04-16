"""
core/models/result.py
Top-level pipeline result model — the single object app.py needs to render the full UI.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .audio import AudioBuffer
from .authorship import AuthorshipResult
from .compliance import ComplianceReport
from .discovery import AudioQualityResult, PlacementProfile, PopularityResult, SyncCut, TrackCandidate
from .forensics import ForensicsResult
from .legal import LegalLinks, MetadataValidationResult
from .stem import StemValidationResult
from .structure import StructureResult
from .theme_mood import ThemeMoodResult
from .transcription import TranscriptSegment


class AnalysisResult(BaseModel):
    """
    The complete output of a single pipeline run.

    This is the only object app.py needs to render the full UI.
    Serialise with .to_dict() (excludes audio.raw) for JSON storage or
    a future REST API response.
    """

    audio: AudioBuffer
    structure: Optional[StructureResult]                    = None
    forensics: Optional[ForensicsResult]                    = None
    transcript: list[TranscriptSegment]                     = Field(default_factory=list)
    compliance: Optional[ComplianceReport]                  = None
    authorship: Optional[AuthorshipResult]                  = None
    similar_tracks: list[TrackCandidate]                    = Field(default_factory=list)
    legal: Optional[LegalLinks]                             = None
    popularity: Optional[PopularityResult]                  = None
    audio_quality: Optional[AudioQualityResult]             = None
    metadata_validation: Optional[MetadataValidationResult] = None
    sync_cuts: list[SyncCut]                                = Field(default_factory=list)
    stem_validation: Optional[StemValidationResult]         = None
    theme_mood: Optional[ThemeMoodResult]                   = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to a plain dict suitable for JSON storage or an API response.
        Raw audio bytes are excluded — they are not a domain value.
        """
        data = self.model_dump(exclude={"audio": {"raw"}})
        return data

    def to_json(self) -> str:
        """Serialise to a JSON string (audio bytes excluded)."""
        return self.model_dump_json(exclude={"audio": {"raw"}})
