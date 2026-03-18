"""
core/models.py
Typed domain models for the Sync-Safe pipeline.

Design rules:
- Every model uses Pydantic BaseModel for validation + serialisation.
- Categorical fields use Literal types — a typo becomes a ValidationError,
  not a silent bug that reaches the UI.
- frozen=True on immutable results: a StructureResult produced by allin1
  should never be mutated by a downstream service.
- AudioBuffer holds raw bytes for in-process use; call .to_bytesio() when
  a service needs an io.BytesIO. Exclude `raw` when serialising to JSON
  (it's binary data, not a domain value): model.model_dump(exclude={'raw'}).
- All models expose .to_dict() as a convenience alias over model_dump() so
  future API layers don't need to know the Pydantic internals.
"""
from __future__ import annotations

import io
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Type aliases — used as Literal constraints throughout
# ---------------------------------------------------------------------------

IssueType    = Literal["EXPLICIT", "BRAND", "LOCATION", "VIOLENCE", "DRUGS"]
Confidence   = Literal["confirmed", "potential"]
EndingType   = Literal["sting", "fade", "cut"]
AIVerdict    = Literal["Likely Human", "Uncertain", "Likely AI", "Insufficient data"]
ForensicVerdict = Literal["Human", "Uncertain", "AI"]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class AudioBuffer(BaseModel):
    """
    In-memory audio representation passed between pipeline stages.

    `raw` holds the WAV/MP3 bytes as ingested — no resampling is done here.
    Each service is responsible for resampling to its required rate via
    librosa.load(buffer.to_bytesio(), sr=CONSTANTS.SAMPLE_RATE).
    """

    model_config = ConfigDict(frozen=True)

    raw: bytes = Field(repr=False)          # excluded from repr; can be 50 MB
    sample_rate: int = Field(default=22_050)
    label: str = Field(default="")          # display name shown in the UI

    def to_bytesio(self) -> io.BytesIO:
        """Return a fresh BytesIO cursor at position 0."""
        return io.BytesIO(self.raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialise without the raw bytes (not meaningful in JSON)."""
        return self.model_dump(exclude={"raw"})


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class TranscriptSegment(BaseModel):
    """A single time-stamped segment from Whisper."""

    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0.0)
    end: float   = Field(ge=0.0)
    text: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Structure Analysis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Forensics
# ---------------------------------------------------------------------------

class ForensicsResult(BaseModel):
    """Output of the AI-humanity forensics stage."""

    model_config = ConfigDict(frozen=True)

    # Individual signal scores (0.0–1.0 where 1.0 = most AI-like)
    c2pa_flag: bool         = False     # True → born-AI assertion found in manifest
    ibi_variance: float     = 1.0       # inter-beat interval variance
    loop_score: float       = 0.0       # highest cross-correlation across 4-bar windows
    spectral_slop: float    = 0.0       # anomalous energy above SPECTRAL_SLOP_HZ
    synthid_score: float    = 0.0       # phase coherence in 18–22 kHz band

    flags: list[str]        = Field(default_factory=list)  # human-readable flag labels
    verdict: ForensicVerdict = "Human"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

class ComplianceFlag(BaseModel):
    """A single issue surfaced during lyric/structural compliance checking."""

    model_config = ConfigDict(frozen=True)

    timestamp_s: int                        # seconds from track start
    issue_type: IssueType
    text: str                               # flagged excerpt or brand name
    recommendation: str                     # supervisor action guidance
    confidence: Confidence = "confirmed"    # confirmed = NER hit; potential = keyword

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class StingResult(BaseModel):
    """Result of the sting/ending-type check."""

    model_config = ConfigDict(frozen=True)

    ending_type: EndingType
    sync_ready: bool
    final_energy_ratio: float   = Field(ge=0.0)
    flag: bool                  = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EnergyEvolutionResult(BaseModel):
    """Result of the 4-8 bar energy evolution check."""

    model_config = ConfigDict(frozen=True)

    stagnant_windows: int   = 0     # count of windows below ENERGY_DELTA_MIN
    total_windows: int      = 0
    flag: bool              = False
    detail: str             = ""    # e.g. "3 of 12 windows below 10% delta"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class IntroResult(BaseModel):
    """Result of the intro-length check."""

    model_config = ConfigDict(frozen=True)

    intro_seconds: float    = 0.0
    flag: bool              = False
    source: str             = ""    # "allin1" | "whisper_fallback" | "none"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ComplianceReport(BaseModel):
    """Aggregated output of all Gallo-Method compliance checks."""

    model_config = ConfigDict(frozen=True)

    flags: list[ComplianceFlag]         = Field(default_factory=list)
    sting: StingResult                  = Field(default_factory=StingResult.model_construct)
    evolution: EnergyEvolutionResult    = Field(default_factory=EnergyEvolutionResult.model_construct)
    intro: IntroResult                  = Field(default_factory=IntroResult.model_construct)
    grade: str                          = "N/A"     # A–F or "N/A"

    @property
    def confirmed_flags(self) -> list[ComplianceFlag]:
        return [f for f in self.flags if f.confidence == "confirmed"]

    @property
    def potential_flags(self) -> list[ComplianceFlag]:
        return [f for f in self.flags if f.confidence == "potential"]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Authorship
# ---------------------------------------------------------------------------

class AuthorshipResult(BaseModel):
    """Output of the AI lyric authorship detection stage."""

    model_config = ConfigDict(frozen=True)

    verdict: AIVerdict                  = "Likely Human"
    signal_count: int                   = 0
    roberta_score: Optional[float]      = None  # None when model not run or insufficient data
    feature_notes: list[str]            = Field(default_factory=list)
    scores: dict[str, Optional[float]]  = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Discovery & Legal
# ---------------------------------------------------------------------------

class TrackCandidate(BaseModel):
    """A similar track returned by the discovery service."""

    model_config = ConfigDict(frozen=True)

    title: str
    artist: str
    youtube_url: Optional[str]  = None  # None when yt-dlp URL lookup fails
    similarity: float           = 0.0   # 0.0–1.0; rank-derived from Last.fm order

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class LegalLinks(BaseModel):
    """PRO repertory search URLs for a given track."""

    model_config = ConfigDict(frozen=True)

    ascap: str  = ""
    bmi: str    = ""
    sesac: str  = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Top-level pipeline result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """
    The complete output of a single pipeline run.

    This is the only object app.py needs to render the full UI.
    Serialise with .to_dict() (excludes audio.raw) for JSON storage or
    a future REST API response.
    """

    audio: AudioBuffer
    structure: Optional[StructureResult]        = None
    forensics: Optional[ForensicsResult]        = None
    transcript: list[TranscriptSegment]         = Field(default_factory=list)
    compliance: Optional[ComplianceReport]      = None
    authorship: Optional[AuthorshipResult]      = None
    similar_tracks: list[TrackCandidate]        = Field(default_factory=list)
    legal: Optional[LegalLinks]                 = None

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
