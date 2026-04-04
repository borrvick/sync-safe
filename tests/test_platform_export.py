"""
tests/test_platform_export.py
Unit tests for services/platform_export.py pure functions.
"""
from __future__ import annotations

import csv
import io

import pytest

from core.models import (
    AnalysisResult,
    AudioBuffer,
    AudioQualityResult,
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    ForensicsResult,
    IntroResult,
    LegalLinks,
    StingResult,
    StructureResult,
)
from services.export import (
    PLATFORM_SCHEMAS,
    _extract_track_data,
    _to_platform_row,
    to_platform_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    title: str = "Test Track",
    artist: str = "Test Artist",
    bpm: float | str = 120.0,
    key: str = "C Major",
    isrc: str = "US-ABC-23-12345",
    grade: str = "A",
    ai_probability: float = 0.1,
    verdict: str = "Likely Not AI",
    flags: list[ComplianceFlag] | None = None,
    lufs: float = -14.0,
) -> AnalysisResult:
    compliance = ComplianceReport(
        flags=flags or [],
        sting=StingResult(ending_type="fade", sync_ready=True, final_energy_ratio=0.1),
        evolution=EnergyEvolutionResult(stagnant_windows=0, total_windows=8, detail=""),
        intro=IntroResult(intro_seconds=4.0),
        grade=grade,
    )
    forensics = ForensicsResult(
        ai_probability=ai_probability,
        verdict=verdict,  # type: ignore[arg-type]
    )
    structure = StructureResult(bpm=bpm, key=key, sections=[], beats=[])
    legal     = LegalLinks(isrc=isrc, pro_match="ASCAP (US)")
    audio_quality = AudioQualityResult(
        integrated_lufs=lufs,
        true_peak_dbfs=-1.5,
        loudness_range_lu=8.0,
        delta_spotify=0.0,
        delta_apple_music=2.0,
        delta_youtube=0.0,
        delta_broadcast=9.0,
        true_peak_warning=False,
        gain_spotify_db=0.0,
        gain_apple_music_db=-2.0,
        gain_youtube_db=0.0,
        gain_broadcast_db=-9.0,
        loudness_verdict="Streaming-ready",
        dialogue_score=0.75,
        dialogue_label="Dialogue-Ready",
    )
    return AnalysisResult(
        audio=AudioBuffer(
            raw=b"\x00" * 44,
            label="Test Track",
            metadata={"title": title, "artist": artist},
        ),
        compliance=compliance,
        forensics=forensics,
        structure=structure,
        legal=legal,
        audio_quality=audio_quality,
    )


# ---------------------------------------------------------------------------
# _extract_track_data
# ---------------------------------------------------------------------------

class TestExtractTrackData:
    def test_basic_fields(self) -> None:
        result = _make_result()
        data   = _extract_track_data(result)
        assert data["title"]  == "Test Track"
        assert data["artist"] == "Test Artist"
        assert data["bpm"]    == "120.0"
        assert data["key"]    == "C Major"
        assert data["isrc"]   == "US-ABC-23-12345"
        assert data["grade"]  == "A"

    def test_verdict_and_ai_probability(self) -> None:
        result = _make_result(ai_probability=0.456)
        data   = _extract_track_data(result)
        assert data["ai_probability_pct"] == "45.6"
        assert data["verdict"]            == "Likely Not AI"

    def test_cleared_true_when_passing(self) -> None:
        result = _make_result(grade="A", verdict="Likely Not AI")
        assert _extract_track_data(result)["sync_safe_cleared"] == "true"

    def test_cleared_false_when_ai_verdict(self) -> None:
        result = _make_result(grade="A", verdict="Likely AI")
        assert _extract_track_data(result)["sync_safe_cleared"] == "false"

    def test_cleared_false_when_bad_grade(self) -> None:
        result = _make_result(grade="D", verdict="Likely Not AI")
        assert _extract_track_data(result)["sync_safe_cleared"] == "false"

    def test_flags_pipe_separated(self) -> None:
        flag = ComplianceFlag(
            timestamp_s=10, issue_type="EXPLICIT",
            text="bad", recommendation="remove",
        )
        result = _make_result(flags=[flag])
        data   = _extract_track_data(result)
        assert "EXPLICIT" in data["flags"]

    def test_no_flags_shows_none(self) -> None:
        result = _make_result(flags=[])
        assert _extract_track_data(result)["flags"] == "none"

    def test_bpm_string_falls_back_to_empty(self) -> None:
        result = _make_result(bpm="Detection failed")
        data   = _extract_track_data(result)
        assert data["bpm"] == ""

    def test_lufs_extracted(self) -> None:
        result = _make_result(lufs=-16.0)
        assert _extract_track_data(result)["lufs_integrated"] == "-16.0"


# ---------------------------------------------------------------------------
# to_platform_csv — generic
# ---------------------------------------------------------------------------

class TestToPlatformCsvGeneric:
    def test_returns_bytes(self) -> None:
        out = to_platform_csv(_make_result(), "generic")
        assert isinstance(out, bytes)

    def test_bom_present(self) -> None:
        out = to_platform_csv(_make_result(), "generic")
        assert out[:3] == b"\xef\xbb\xbf"

    def test_header_matches_schema(self) -> None:
        out    = to_platform_csv(_make_result(), "generic")
        reader = csv.DictReader(io.StringIO(out.decode("utf-8-sig")))
        assert set(reader.fieldnames or []) == set(PLATFORM_SCHEMAS["generic"])

    def test_title_in_output(self) -> None:
        out  = to_platform_csv(_make_result(title="MyTrack"), "generic")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert rows[0]["title"] == "MyTrack"

    def test_sync_safe_cleared_field_present(self) -> None:
        out  = to_platform_csv(_make_result(), "generic")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert rows[0]["sync_safe_cleared"] in ("true", "false")


# ---------------------------------------------------------------------------
# to_platform_csv — disco
# ---------------------------------------------------------------------------

class TestToPlatformCsvDisco:
    def test_header_matches_schema(self) -> None:
        out    = to_platform_csv(_make_result(), "disco")
        reader = csv.DictReader(io.StringIO(out.decode("utf-8-sig")))
        assert set(reader.fieldnames or []) == set(PLATFORM_SCHEMAS["disco"])

    def test_tags_contains_grade(self) -> None:
        out  = to_platform_csv(_make_result(grade="B"), "disco")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert "GRADE:B" in rows[0]["tags"]

    def test_cleared_tag_present_when_passing(self) -> None:
        out  = to_platform_csv(_make_result(grade="A", verdict="Likely Not AI"), "disco")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert "SYNC_SAFE_CLEARED" in rows[0]["tags"]

    def test_notes_contains_ai_probability(self) -> None:
        out  = to_platform_csv(_make_result(ai_probability=0.25), "disco")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert "25.0%" in rows[0]["notes"]


# ---------------------------------------------------------------------------
# to_platform_csv — synchtank
# ---------------------------------------------------------------------------

class TestToPlatformCsvSynchtank:
    def test_header_matches_schema(self) -> None:
        out    = to_platform_csv(_make_result(), "synchtank")
        reader = csv.DictReader(io.StringIO(out.decode("utf-8-sig")))
        assert set(reader.fieldnames or []) == set(PLATFORM_SCHEMAS["synchtank"])

    def test_track_title_mapped(self) -> None:
        out  = to_platform_csv(_make_result(title="MyTrack"), "synchtank")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert rows[0]["track_title"] == "MyTrack"

    def test_tempo_mapped_from_bpm(self) -> None:
        out  = to_platform_csv(_make_result(bpm=140.0), "synchtank")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert rows[0]["tempo"] == "140.0"

    def test_description_contains_grade(self) -> None:
        out  = to_platform_csv(_make_result(grade="C"), "synchtank")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert "Grade: C" in rows[0]["description"]

    def test_pre_cleared_in_description(self) -> None:
        out  = to_platform_csv(_make_result(grade="A", verdict="Not AI"), "synchtank")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert "Pre-cleared: yes" in rows[0]["description"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestPlatformExportErrors:
    def test_unknown_platform_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown platform"):
            to_platform_csv(_make_result(), "unknown_platform")

    def test_missing_structure_does_not_crash(self) -> None:
        result = AnalysisResult(
            audio=AudioBuffer(raw=b"\x00" * 44, label="x"),
        )
        out = to_platform_csv(result, "generic")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert rows[0]["bpm"] == ""

    def test_missing_forensics_does_not_crash(self) -> None:
        result = AnalysisResult(
            audio=AudioBuffer(raw=b"\x00" * 44, label="x"),
        )
        out = to_platform_csv(result, "generic")
        rows = list(csv.DictReader(io.StringIO(out.decode("utf-8-sig"))))
        assert rows[0]["verdict"] == ""
