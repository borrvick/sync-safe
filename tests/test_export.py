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
)
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
