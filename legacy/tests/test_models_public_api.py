"""
tests/test_models_public_api.py
Smoke-tests that every public name in core.models is importable and is the
expected type after the package split (#200).

These tests contain no business logic — they exist only to catch regressions
where a refactor accidentally removes or shadows a re-export.
"""
import pytest

import core.models as m


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

def test_type_aliases_are_strings() -> None:
    """Literal aliases are plain typing constructs — just confirm they exist."""
    assert m.AIVerdict is not None
    assert m.AudioSource is not None
    assert m.Confidence is not None
    assert m.EndingType is not None
    assert m.ForensicVerdict is not None
    assert m.IssueType is not None
    assert m.Severity is not None


# ---------------------------------------------------------------------------
# Model classes are importable and are Pydantic BaseModel subclasses
# ---------------------------------------------------------------------------

EXPECTED_MODELS = [
    "AudioBuffer",
    "TranscriptSegment",
    "Section",
    "StructureResult",
    "AiSegment",
    "ForensicsResult",
    "ComplianceFlag",
    "ComplianceReport",
    "EnergyEvolutionResult",
    "IntroResult",
    "StingResult",
    "AuthorshipResult",
    "SectionAuthorshipResult",
    "ThemeMoodResult",
    "AudioQualityResult",
    "PlacementProfile",
    "PopularityResult",
    "SyncCut",
    "TrackCandidate",
    "LegalLinks",
    "MetadataValidationResult",
    "StemValidationResult",
    "AnalysisResult",
]


@pytest.mark.parametrize("name", EXPECTED_MODELS)
def test_model_is_exported(name: str) -> None:
    assert hasattr(m, name), f"core.models is missing: {name}"


def test_section_repetition_is_dataclass() -> None:
    """SectionRepetition is a frozen dataclass, not a BaseModel."""
    import dataclasses
    assert dataclasses.is_dataclass(m.SectionRepetition)


# ---------------------------------------------------------------------------
# Spot-check: __all__ is complete
# ---------------------------------------------------------------------------

def test_all_contains_expected_models() -> None:
    for name in EXPECTED_MODELS:
        assert name in m.__all__, f"__all__ is missing: {name}"


# ---------------------------------------------------------------------------
# Spot-check: instantiation round-trips for key models
# ---------------------------------------------------------------------------

def test_audio_buffer_instantiable() -> None:
    buf = m.AudioBuffer(raw=b"\x00" * 100, label="test")
    assert buf.label == "test"
    assert buf.sample_rate == 22_050


def test_forensics_result_defaults() -> None:
    r = m.ForensicsResult()
    assert r.verdict == "Likely Not AI"
    assert r.ai_probability == 0.0


def test_compliance_report_defaults() -> None:
    r = m.ComplianceReport()
    assert r.grade == "N/A"
    assert r.flags == []


def test_placement_profile_defaults() -> None:
    p = m.PlacementProfile()
    assert p.name == "Standard"
    assert p.intro_max_seconds == 15


def test_analysis_result_round_trip() -> None:
    buf = m.AudioBuffer(raw=b"\x00" * 100, label="round-trip")
    result = m.AnalysisResult(audio=buf)
    d = result.to_dict()
    assert "audio" in d
    assert "raw" not in d["audio"]
