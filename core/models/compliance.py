"""
core/models/compliance.py
Sync-readiness compliance check output models.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._types import Confidence, EndingType, IssueType, Severity


class ComplianceFlag(BaseModel):
    """A single issue surfaced during lyric/structural compliance checking."""

    model_config = ConfigDict(frozen=True)

    timestamp_s: int                        # seconds from track start
    issue_type: IssueType
    text: str                               # flagged excerpt or brand name
    recommendation: str                     # supervisor action guidance
    confidence: Confidence = "confirmed"    # confirmed = NER hit; potential = keyword
    severity: Severity     = "soft"         # hard = deal-breaker in any context;
                                            # soft = placement-dependent, director's call

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class StingResult(BaseModel):
    """Result of the sting/ending-type check."""

    model_config = ConfigDict(frozen=True)

    ending_type: EndingType
    sync_ready: bool
    final_energy_ratio: float       = Field(ge=0.0)
    flag: bool                      = False
    # Fade-specific enrichment (#103)
    fade_severity: float            = 0.0   # 0.0 (instant cut) → 1.0 (long gentle fade)
    fade_tail_seconds: float        = 0.0   # duration (s) where RMS > tail threshold
    # Cut-specific enrichment (#104)
    cut_type: Optional[Literal["clean_cut", "mid_phrase_cut"]] = None
    # Raw diagnostic fields for JSON export (#108)
    norm_slope: float               = 0.0   # normalised RMS slope at track end
    onset_spike_factor: float       = 0.0   # final-onset energy vs track mean

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EnergyEvolutionResult(BaseModel):
    """Result of the 4-8 bar energy evolution check."""

    model_config = ConfigDict(frozen=True)

    stagnant_windows: int   = 0     # count of windows below ENERGY_DELTA_MIN
    total_windows: int      = 0
    flag: bool              = False
    detail: str             = ""    # e.g. "3 of 12 windows below 10% delta"
    # Per-section breakdown (#106)
    section_details: list[dict[str, str | int | bool]] = Field(default_factory=list)
    ending_section: Optional[str]                      = None  # section label containing track end
    # Raw diagnostic fields for JSON export (#108)
    stagnant_timestamps: list[float]    = Field(default_factory=list)  # start_s of each stagnant window
    per_window_contrasts: list[float]   = Field(default_factory=list)  # energy delta per window

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class IntroResult(BaseModel):
    """Result of the intro-length check."""

    model_config = ConfigDict(frozen=True)

    intro_seconds: float    = 0.0
    flag: bool              = False
    source: str             = ""    # "allin1" | "whisper" | "onset" | "whisper_fallback" | "none"
    confidence: str         = ""    # "High" | "Medium" | "Low" | "" (pre-#105 results)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ComplianceReport(BaseModel):
    """Aggregated output of all sync readiness compliance checks."""

    model_config = ConfigDict(frozen=True)

    flags: list[ComplianceFlag]         = Field(default_factory=list)
    sting: StingResult                  = Field(default_factory=lambda: StingResult(ending_type="cut", sync_ready=False, final_energy_ratio=0.0))
    evolution: EnergyEvolutionResult    = Field(default_factory=EnergyEvolutionResult)
    intro: IntroResult                  = Field(default_factory=IntroResult)
    grade: str                          = "N/A"     # A–F or "N/A"

    @property
    def confirmed_flags(self) -> list[ComplianceFlag]:
        return [f for f in self.flags if f.confidence == "confirmed"]

    @property
    def hard_flags(self) -> list[ComplianceFlag]:
        """Confirmed flags that are absolute deal-breakers in any placement context."""
        return [f for f in self.flags if f.confidence == "confirmed" and f.severity == "hard"]

    @property
    def soft_flags(self) -> list[ComplianceFlag]:
        """Confirmed flags that are placement-dependent — sync director's call."""
        return [f for f in self.flags if f.confidence == "confirmed" and f.severity == "soft"]

    @property
    def potential_flags(self) -> list[ComplianceFlag]:
        return [f for f in self.flags if f.confidence == "potential"]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
