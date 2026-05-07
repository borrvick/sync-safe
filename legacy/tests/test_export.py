"""
tests/test_export.py
Unit tests for export pure functions in ui/pages/report.py.
Streamlit render functions (_render_export_buttons) require a live Streamlit
context and are tested manually.
"""
from __future__ import annotations

import csv
import io

import pytest

from core.models import (
    AnalysisResult,
    AudioBuffer,
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    IntroResult,
    LegalLinks,
    StingResult,
    SyncCut,
)
from services.export import _build_davinci_drt, _build_premiere_xml, _seconds_to_timecode
from ui.pages.report import _analysis_to_pdf, _compliance_flags_to_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    flags: list[ComplianceFlag] | None = None,
    grade: str = "B",
    forensics: None = None,
    legal: LegalLinks | None = None,
) -> AnalysisResult:
    compliance = ComplianceReport(
        flags=flags or [],
        sting=StingResult(ending_type="fade", sync_ready=True, final_energy_ratio=0.1),
        evolution=EnergyEvolutionResult(stagnant_windows=0, total_windows=8, detail=""),
        intro=IntroResult(intro_seconds=4.0),
        grade=grade,
    )
    return AnalysisResult(
        audio=AudioBuffer(
            raw=b"\x00" * 44,
            label="Test Track",
            metadata={"title": "Test Track", "artist": "Test Artist"},
        ),
        compliance=compliance,
        forensics=forensics,
        legal=legal,
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestComplianceFlagsToCsv:
    def test_returns_bytes(self) -> None:
        result = _make_result()
        out = _compliance_flags_to_csv(result)
        assert isinstance(out, bytes)

    def test_header_row_present(self) -> None:
        result = _make_result()
        reader = csv.DictReader(io.StringIO(_compliance_flags_to_csv(result).decode("utf-8-sig")))
        assert "check" in (reader.fieldnames or [])
        assert "status" in (reader.fieldnames or [])

    def test_flag_serialised_correctly(self) -> None:
        flag = ComplianceFlag(
            timestamp_s=42,
            issue_type="EXPLICIT",
            text="bad word",
            recommendation="Bleep or replace",
            confidence="confirmed",
            severity="hard",
        )
        result = _make_result(flags=[flag])
        rows = list(csv.DictReader(io.StringIO(_compliance_flags_to_csv(result).decode("utf-8-sig"))))
        lyric_rows = [r for r in rows if r["section"] == "Lyric Audit"]
        assert len(lyric_rows) == 1
        assert lyric_rows[0]["check"] == "EXPLICIT"
        assert lyric_rows[0]["timestamp_s"] == "42"

    def test_structural_checks_always_included(self) -> None:
        result = _make_result()
        rows = list(csv.DictReader(io.StringIO(_compliance_flags_to_csv(result).decode("utf-8-sig"))))
        structural = [r for r in rows if r["section"] == "Structural"]
        assert len(structural) == 3  # Sting, Energy Evolution, Intro

    def test_no_compliance_returns_empty_body(self) -> None:
        result = AnalysisResult(
            audio=AudioBuffer(raw=b"\x00" * 44, label="x"),
            compliance=None,
        )
        rows = list(csv.DictReader(io.StringIO(_compliance_flags_to_csv(result).decode("utf-8-sig"))))
        assert rows == []


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

class TestAnalysisToPdf:
    def test_returns_bytes(self) -> None:
        result = _make_result()
        out = _analysis_to_pdf(result)
        assert isinstance(out, (bytes, bytearray))

    def test_pdf_magic_bytes(self) -> None:
        result = _make_result()
        out = bytes(_analysis_to_pdf(result))
        assert out[:4] == b"%PDF"

    def test_unicode_title_does_not_crash(self) -> None:
        """Non-latin-1 chars in title/artist must be replaced, not crash."""
        result = AnalysisResult(
            audio=AudioBuffer(
                raw=b"\x00" * 44,
                label="Test",
                metadata={"title": "Sigur Rós — \u65e5\u672c\u8a9e", "artist": "Beyoncé"},
            ),
        )
        out = bytes(_analysis_to_pdf(result))
        assert out[:4] == b"%PDF"

    def test_with_flags_and_legal(self) -> None:
        flag = ComplianceFlag(
            timestamp_s=10,
            issue_type="BRAND",
            text="Nike",
            recommendation="Remove",
        )
        legal = LegalLinks(isrc="US-ABC-23-00001", pro_match="ASCAP / BMI / SESAC (US)")
        result = _make_result(flags=[flag], legal=legal)
        out = bytes(_analysis_to_pdf(result))
        assert out[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# DAW marker export helpers (#152)
# ---------------------------------------------------------------------------

def _make_cut(
    duration_s: int = 30,
    start_s: float = 48.0,
    end_s: float = 78.0,
    confidence: float = 1.0,
    note: str = "Chorus at 0:48",
) -> SyncCut:
    return SyncCut(
        duration_s=duration_s,
        start_s=start_s,
        end_s=end_s,
        actual_duration_s=end_s - start_s,
        confidence=confidence,
        note=note,
    )


class TestSecondsToTimecode:
    def test_zero(self) -> None:
        assert _seconds_to_timecode(0.0, 24.0) == "00:00:00:00"

    def test_one_minute_thirty(self) -> None:
        # 90s @ 24fps → 00:01:30:00
        assert _seconds_to_timecode(90.0, 24.0) == "00:01:30:00"

    def test_frame_count(self) -> None:
        # 90.5s @ 24fps → 90*24 + 12 frames → 00:01:30:12
        assert _seconds_to_timecode(90.5, 24.0) == "00:01:30:12"

    def test_25fps(self) -> None:
        # 1s @ 25fps → 00:00:01:00
        assert _seconds_to_timecode(1.0, 25.0) == "00:00:01:00"

    def test_non_integer_fps(self) -> None:
        # At 29.97fps: int(1.0 * 29.97)=29 frames; fps=round(29.97)=30
        # 29 frames = 0 full seconds + 29 frame remainder → NDF 00:00:00:29
        tc = _seconds_to_timecode(1.0, 29.97)
        assert tc == "00:00:00:29"

    def test_large_value(self) -> None:
        # 3661s → 1h 1m 1s
        assert _seconds_to_timecode(3661.0, 24.0) == "01:01:01:00"


class TestBuildPremiereXml:
    def test_empty_cuts_produces_valid_xml(self) -> None:
        xml = _build_premiere_xml([], 24.0)
        assert xml.startswith("<markers>")
        assert xml.endswith("</markers>")
        assert "<marker>" not in xml

    def test_single_cut_present(self) -> None:
        xml = _build_premiere_xml([_make_cut()], 24.0)
        assert "<marker>" in xml
        assert "</marker>" in xml

    def test_in_out_timecodes_present(self) -> None:
        xml = _build_premiere_xml([_make_cut(start_s=0.0, end_s=30.0)], 24.0)
        assert "<in>00:00:00:00</in>" in xml
        assert "<out>00:00:30:00</out>" in xml

    def test_xml_special_chars_escaped(self) -> None:
        cut = _make_cut(note="Chorus & Bridge <loud>")
        xml = _build_premiere_xml([cut], 24.0)
        assert "&amp;" in xml
        assert "&lt;" in xml
        assert "<loud>" not in xml

    def test_confidence_in_comment(self) -> None:
        xml = _build_premiere_xml([_make_cut(confidence=0.8)], 24.0)
        assert "80%" in xml

    def test_multiple_cuts(self) -> None:
        cuts = [_make_cut(duration_s=d) for d in (15, 30, 60)]
        xml = _build_premiere_xml(cuts, 24.0)
        assert xml.count("<marker>") == 3


class TestBuildDavinciDrt:
    def test_empty_cuts_has_header_only(self) -> None:
        drt = _build_davinci_drt([], 24.0)
        lines = drt.strip().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("Timecode")

    def test_header_columns(self) -> None:
        drt = _build_davinci_drt([], 24.0)
        assert "Timecode\tLabel\tColor\tDuration\tNote" == drt.strip()

    def test_single_cut_row(self) -> None:
        drt = _build_davinci_drt([_make_cut()], 24.0)
        lines = drt.strip().splitlines()
        assert len(lines) == 2  # header + 1 data row
        cols = lines[1].split("\t")
        assert len(cols) == 5

    def test_color_is_valid_davinci_color(self) -> None:
        drt = _build_davinci_drt([_make_cut()], 24.0)
        row = drt.strip().splitlines()[1]
        color = row.split("\t")[2]
        assert color == "Cyan"

    def test_label_contains_duration(self) -> None:
        drt = _build_davinci_drt([_make_cut(duration_s=30)], 24.0)
        row = drt.strip().splitlines()[1]
        assert "30s Cut" in row

    def test_multiple_cuts(self) -> None:
        cuts = [_make_cut(duration_s=d) for d in (15, 30, 60)]
        drt = _build_davinci_drt(cuts, 24.0)
        assert len(drt.strip().splitlines()) == 4  # header + 3 rows
