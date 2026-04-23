"""
backend/core/models.py
Pydantic models for the ML pipeline payload.

These types flow between Django, Modal, and services/:
  AudioBuffer        — raw WAV bytes + sample rate (ingestion output)
  TranscriptSegment  — one Whisper segment with timestamps
  Section            — one allin1 structural section
  ForensicsResult    — AI-origin detection signals
  StructureResult    — full allin1 structure analysis
  ComplianceReport   — Gallo-Method sync-readiness flags
  AnalysisResult     — top-level container stored in Analysis.result_json

All models serialise to plain dicts via .to_dict() so they can be stored
in Analysis.result_json and passed through the Modal webhook payload.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AudioBuffer(BaseModel):
    """Raw WAV bytes + native sample rate produced by ingestion."""

    raw: bytes
    sample_rate: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable dict. raw bytes are intentionally excluded —
        callers must pass audio data out-of-band (e.g. as a separate function arg)."""
        return {"sample_rate": self.sample_rate}


class TranscriptSegment(BaseModel):
    """One Whisper transcript segment with start/end timestamps."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


class Section(BaseModel):
    """One allin1 structural section (intro, verse, chorus, outro…)."""

    label: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "start": self.start, "end": self.end}


class ForensicsResult(BaseModel):
    """AI-origin detection output from services/forensics.py."""

    c2pa_verdict: str = "UNKNOWN"           # CLEAN | AI_GENERATED | UNKNOWN
    ibi_variance: float = 0.0               # inter-beat interval variance
    perfect_quantization: bool = False      # zero IBI variance → likely machine-quantized
    loop_detected: bool = False
    loop_score: float = 0.0                 # spectral cross-correlation 0–1
    spectral_anomaly: bool = False
    flags: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class StructureResult(BaseModel):
    """allin1 structure analysis: BPM, key, beat grid, sections."""

    bpm: float = 0.0
    key: str = ""
    duration_s: float = 0.0
    beats: list[float] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": self.bpm,
            "key": self.key,
            "duration_s": self.duration_s,
            "beats": self.beats,
            "sections": [s.to_dict() for s in self.sections],
        }


class ComplianceReport(BaseModel):
    """Gallo-Method sync-readiness check results from services/compliance.py."""

    sting_pass: bool = True
    bar_rule_pass: bool = True
    intro_pass: bool = True
    overall_pass: bool = True
    lyric_flags: list[dict[str, Any]] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AnalysisResult(BaseModel):
    """Top-level result container — stored verbatim in Analysis.result_json."""

    job_id: str
    source_url: str = ""
    title: str = ""
    artist: str = ""
    forensics: ForensicsResult = Field(default_factory=ForensicsResult)
    structure: StructureResult = Field(default_factory=StructureResult)
    transcription: list[TranscriptSegment] = Field(default_factory=list)
    compliance: ComplianceReport = Field(default_factory=ComplianceReport)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_url": self.source_url,
            "title": self.title,
            "artist": self.artist,
            "forensics": self.forensics.to_dict(),
            "structure": self.structure.to_dict(),
            "transcription": [t.to_dict() for t in self.transcription],
            "compliance": self.compliance.to_dict(),
        }
