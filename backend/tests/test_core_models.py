"""
tests/test_core_models.py
Unit tests for backend/core/models.py Pydantic ML payload types.

All tests are pure (no I/O, no DB, no GPU) and complete in < 1 second.
Covers: AudioBuffer, TranscriptSegment, Section, ForensicsResult,
        StructureResult, ComplianceReport, AnalysisResult.
"""
from __future__ import annotations

import pytest

from core.models import (
    AnalysisResult,
    AudioBuffer,
    ComplianceReport,
    ForensicsResult,
    Section,
    StructureResult,
    TranscriptSegment,
)


# ---------------------------------------------------------------------------
# AudioBuffer
# ---------------------------------------------------------------------------

class TestAudioBuffer:
    def test_stores_raw_and_sample_rate(self) -> None:
        buf = AudioBuffer(raw=b"\x00\x01", sample_rate=22050)
        assert buf.raw == b"\x00\x01"
        assert buf.sample_rate == 22050

    def test_to_dict_excludes_raw_bytes(self) -> None:
        buf = AudioBuffer(raw=b"\xff" * 100, sample_rate=44100)
        d = buf.to_dict()
        assert "raw" not in d

    def test_to_dict_preserves_sample_rate(self) -> None:
        buf = AudioBuffer(raw=b"", sample_rate=16000)
        assert buf.to_dict()["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------

class TestTranscriptSegment:
    def test_construction(self) -> None:
        seg = TranscriptSegment(start=0.5, end=3.2, text="hello world")
        assert seg.start == 0.5
        assert seg.end == 3.2
        assert seg.text == "hello world"

    def test_to_dict_roundtrip(self) -> None:
        seg = TranscriptSegment(start=1.0, end=4.0, text="sync safe")
        d = seg.to_dict()
        assert d == {"start": 1.0, "end": 4.0, "text": "sync safe"}

    def test_to_dict_keys(self) -> None:
        d = TranscriptSegment(start=0.0, end=1.0, text="").to_dict()
        assert set(d.keys()) == {"start", "end", "text"}


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------

class TestSection:
    def test_construction(self) -> None:
        s = Section(label="intro", start=0.0, end=15.0)
        assert s.label == "intro"
        assert s.start == 0.0
        assert s.end == 15.0

    def test_to_dict_roundtrip(self) -> None:
        s = Section(label="chorus", start=30.0, end=60.0)
        assert s.to_dict() == {"label": "chorus", "start": 30.0, "end": 60.0}


# ---------------------------------------------------------------------------
# ForensicsResult
# ---------------------------------------------------------------------------

class TestForensicsResult:
    def test_defaults(self) -> None:
        r = ForensicsResult()
        assert r.c2pa_verdict == "UNKNOWN"
        assert r.ibi_variance == 0.0
        assert r.perfect_quantization is False
        assert r.loop_detected is False
        assert r.loop_score == 0.0
        assert r.spectral_anomaly is False
        assert r.flags == []

    def test_flags_list_is_independent_per_instance(self) -> None:
        a = ForensicsResult()
        b = ForensicsResult()
        a.flags.append("AI_ORIGIN")
        assert b.flags == []

    def test_to_dict_is_dict(self) -> None:
        r = ForensicsResult(c2pa_verdict="AI_GENERATED", loop_detected=True)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["c2pa_verdict"] == "AI_GENERATED"
        assert d["loop_detected"] is True

    def test_to_dict_contains_all_fields(self) -> None:
        d = ForensicsResult().to_dict()
        expected = {"c2pa_verdict", "ibi_variance", "perfect_quantization",
                    "loop_detected", "loop_score", "spectral_anomaly", "flags"}
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# StructureResult
# ---------------------------------------------------------------------------

class TestStructureResult:
    def test_defaults(self) -> None:
        r = StructureResult()
        assert r.bpm == 0.0
        assert r.key == ""
        assert r.duration_s == 0.0
        assert r.beats == []
        assert r.sections == []

    def test_to_dict_serialises_sections(self) -> None:
        r = StructureResult(
            bpm=120.0,
            key="C major",
            duration_s=180.0,
            beats=[0.5, 1.0, 1.5],
            sections=[Section(label="intro", start=0.0, end=15.0)],
        )
        d = r.to_dict()
        assert d["bpm"] == 120.0
        assert d["key"] == "C major"
        assert d["beats"] == [0.5, 1.0, 1.5]
        assert d["sections"] == [{"label": "intro", "start": 0.0, "end": 15.0}]

    def test_to_dict_keys(self) -> None:
        d = StructureResult().to_dict()
        assert set(d.keys()) == {"bpm", "key", "duration_s", "beats", "sections"}

    def test_sections_list_is_independent_per_instance(self) -> None:
        a = StructureResult()
        b = StructureResult()
        a.sections.append(Section(label="verse", start=0.0, end=30.0))
        assert b.sections == []


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------

class TestComplianceReport:
    def test_defaults_all_pass(self) -> None:
        r = ComplianceReport()
        assert r.sting_pass is True
        assert r.bar_rule_pass is True
        assert r.intro_pass is True
        assert r.overall_pass is True
        assert r.lyric_flags == []
        assert r.flags == []

    def test_to_dict_contains_all_fields(self) -> None:
        d = ComplianceReport().to_dict()
        expected = {"sting_pass", "bar_rule_pass", "intro_pass",
                    "overall_pass", "lyric_flags", "flags"}
        assert set(d.keys()) == expected

    def test_failure_propagates(self) -> None:
        r = ComplianceReport(overall_pass=False, flags=["INTRO_TOO_LONG"])
        d = r.to_dict()
        assert d["overall_pass"] is False
        assert "INTRO_TOO_LONG" in d["flags"]


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

class TestAnalysisResult:
    def test_construction_with_defaults(self) -> None:
        r = AnalysisResult(job_id="abc-123")
        assert r.job_id == "abc-123"
        assert r.source_url == ""
        assert r.title == ""
        assert r.artist == ""
        assert isinstance(r.forensics, ForensicsResult)
        assert isinstance(r.structure, StructureResult)
        assert r.transcription == []
        assert isinstance(r.compliance, ComplianceReport)

    def test_to_dict_top_level_keys(self) -> None:
        r = AnalysisResult(job_id="x")
        d = r.to_dict()
        assert set(d.keys()) == {
            "job_id", "source_url", "title", "artist",
            "forensics", "structure", "transcription", "compliance",
        }

    def test_to_dict_nested_serialisation(self) -> None:
        r = AnalysisResult(
            job_id="test-job",
            title="Test Track",
            transcription=[TranscriptSegment(start=0.0, end=2.0, text="hello")],
        )
        d = r.to_dict()
        assert d["title"] == "Test Track"
        assert len(d["transcription"]) == 1
        assert d["transcription"][0]["text"] == "hello"

    def test_to_dict_is_json_serialisable(self) -> None:
        import json
        r = AnalysisResult(job_id="json-test", title="Song", source_url="https://example.com")
        # Must not raise — all values must be JSON-compatible primitives
        json.dumps(r.to_dict())

    def test_forensics_instances_are_independent(self) -> None:
        a = AnalysisResult(job_id="a")
        b = AnalysisResult(job_id="b")
        a.forensics.flags.append("AI_ORIGIN")
        assert b.forensics.flags == []
