"""
core/models/__init__.py
Public API for the core.models package.

Re-exports every public name so all existing ``from core.models import X``
call-sites continue to work without modification after the package split.
"""
from __future__ import annotations

# Shared type aliases
from ._types import (
    AIVerdict,
    AudioSource,
    Confidence,
    EndingType,
    ForensicVerdict,
    IssueType,
    Severity,
)

# Audio ingestion
from .audio import AudioBuffer

# Transcription
from .transcription import TranscriptSegment

# Structure analysis
from .structure import Section, StructureResult

# Forensics
from .forensics import AiSegment, ForensicsResult, SectionRepetition

# Compliance
from .compliance import (
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    IntroResult,
    StingResult,
)

# Authorship
from .authorship import AuthorshipResult, SectionAuthorshipResult

# Theme & Mood
from .theme_mood import ThemeMoodResult

# Discovery, popularity, placement, sync cuts
from .discovery import (
    AudioQualityResult,
    PlacementProfile,
    PopularityResult,
    SyncCut,
    TrackCandidate,
)

# Legal / rights
from .legal import LegalLinks, MetadataValidationResult

# Stem validation
from .stem import StemValidationResult

# Top-level pipeline result
from .result import AnalysisResult

__all__ = [
    # _types
    "AIVerdict",
    "AudioSource",
    "Confidence",
    "EndingType",
    "ForensicVerdict",
    "IssueType",
    "Severity",
    # audio
    "AudioBuffer",
    # transcription
    "TranscriptSegment",
    # structure
    "Section",
    "StructureResult",
    # forensics
    "AiSegment",
    "ForensicsResult",
    "SectionRepetition",
    # compliance
    "ComplianceFlag",
    "ComplianceReport",
    "EnergyEvolutionResult",
    "IntroResult",
    "StingResult",
    # authorship
    "AuthorshipResult",
    "SectionAuthorshipResult",
    # theme & mood
    "ThemeMoodResult",
    # discovery
    "AudioQualityResult",
    "PlacementProfile",
    "PopularityResult",
    "SyncCut",
    "TrackCandidate",
    # legal
    "LegalLinks",
    "MetadataValidationResult",
    # stem
    "StemValidationResult",
    # result
    "AnalysisResult",
]
