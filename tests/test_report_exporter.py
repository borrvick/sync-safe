"""
tests/test_report_exporter.py

Unit tests for core/report.py and services/report_exporter.py.

Covers:
  - _track_id          — idempotency, empty-string inputs, case normalisation
  - _dumps             — empty list, list of Pydantic models, plain strings
  - ReportExporter.build — full result, minimal result (all stages None)
  - ReportExporter.to_csv — CSV round-trip, header/value parity

No GPU, no audio, no network — all tests complete in < 1 second.
"""
from __future__ import annotations

import csv
import io
import json

import pytest

from core.models import (
    AiSegment,
    AnalysisResult,
    AudioBuffer,
    AudioQualityResult,
    AuthorshipResult,
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    ForensicsResult,
    IntroResult,
    LegalLinks,
    PopularityResult,
    Section,
    StingResult,
    StructureResult,
    TrackCandidate,
    TranscriptSegment,
)
from core.report import TrackReport, _dumps, _track_id
from services.report_exporter import ReportExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_audio() -> AudioBuffer:
    return AudioBuffer(
        raw=b"\x00" * 44,
        sample_rate=22_050,
        label="test",
        metadata={"title": "Test Track", "artist": "Test Artist"},
        source="file",
    )


def _minimal_result() -> AnalysisResult:
    """AnalysisResult with every Optional stage set to None."""
    return AnalysisResult(audio=_minimal_audio())


def _full_result() -> AnalysisResult:
    """AnalysisResult with every pipeline stage populated."""
    audio = AudioBuffer(
        raw=b"\x00" * 44,
        sample_rate=22_050,
        label="full",
        metadata={"title": "Full Track", "artist": "Full Artist"},
        engagement={"view_count": 1_000_000, "like_count": 50_000},
        source="youtube",
    )
    structure = StructureResult(
        bpm=120.0,
        key="C Major",
        sections=[
            Section(label="intro",  start=0.0,  end=8.0),
            Section(label="verse",  start=8.0,  end=24.0),
            Section(label="chorus", start=24.0, end=40.0),
            Section(label="bridge", start=40.0, end=48.0),
            Section(label="chorus", start=48.0, end=64.0),
        ],
        beats=[float(i) for i in range(128)],
    )
    forensics = ForensicsResult(
        c2pa_flag=True,
        c2pa_origin="ai",
        ibi_variance=0.0002,
        loop_score=0.95,
        ai_probability=0.82,
        verdict="Likely AI",
        flags=["Perfect Quantization", "High Loop Correlation"],
        forensic_notes=["Rhythm is suspiciously perfect"],
        ai_segments=[AiSegment(start_s=0.0, end_s=10.0, probability=0.9)],
        is_vocal=True,
        harmonic_ratio_score=0.88,
        noise_floor_ratio=0.001,
    )
    compliance = ComplianceReport(
        flags=[
            ComplianceFlag(
                timestamp_s=30,
                issue_type="EXPLICIT",
                text="damn",
                recommendation="Bleep or remove.",
                confidence="confirmed",
                severity="hard",
            ),
            ComplianceFlag(
                timestamp_s=60,
                issue_type="BRAND",
                text="Rolex",
                recommendation="Director's call.",
                confidence="potential",
                severity="soft",
            ),
        ],
        sting=StingResult(ending_type="sting", sync_ready=True, final_energy_ratio=0.04, flag=False),
        evolution=EnergyEvolutionResult(stagnant_windows=1, total_windows=8, flag=True, detail="1 of 8"),
        intro=IntroResult(intro_seconds=8.0, flag=False, source="allin1"),
        grade="C",
    )
    authorship = AuthorshipResult(
        verdict="Likely Human",
        signal_count=1,
        roberta_score=0.12,
        feature_notes=["Low burstiness"],
        scores={
            "burstiness": 0.18,
            "unique_word_ratio": 0.52,
            "rhyme_density": 0.65,
            "repetition_score": 0.30,
        },
    )
    quality = AudioQualityResult(
        integrated_lufs=-14.5,
        true_peak_dbfs=-0.8,
        loudness_range_lu=6.2,
        delta_spotify=-0.5,
        delta_apple_music=-2.5,
        delta_youtube=-0.5,
        delta_broadcast=-8.5,
        true_peak_warning=False,
        dialogue_score=0.71,
        dialogue_label="Dialogue-Ready",
    )
    popularity = PopularityResult(
        listeners=250_000,
        playcount=5_000_000,
        spotify_score=72,
        platform_metrics={"view_count": 1_000_000, "like_count": 50_000},
        popularity_score=68,
        tier="Mainstream",
        sync_cost_low=15_000,
        sync_cost_high=100_000,
    )
    legal = LegalLinks(
        ascap="https://ascap.com/search?q=Full+Track",
        bmi="https://bmi.com/search?q=Full+Track",
        sesac="https://sesac.com/search?q=Full+Track",
        isrc="US-AB1-23-00001",
        pro_match="ASCAP/BMI (US)",
    )
    return AnalysisResult(
        audio=audio,
        structure=structure,
        forensics=forensics,
        transcript=[TranscriptSegment(start=8.0, end=12.0, text="hello world")],
        compliance=compliance,
        authorship=authorship,
        audio_quality=quality,
        popularity=popularity,
        legal=legal,
        similar_tracks=[TrackCandidate(title="Alt Track", artist="Alt Artist", similarity=0.85)],
    )


# ---------------------------------------------------------------------------
# _track_id
# ---------------------------------------------------------------------------

class TestTrackId:
    def test_returns_12_hex_chars(self):
        tid = _track_id("Title", "Artist", 180.0)
        assert len(tid) == 12
        assert all(c in "0123456789abcdef" for c in tid)

    def test_idempotent(self):
        assert _track_id("Title", "Artist", 180.0) == _track_id("Title", "Artist", 180.0)

    def test_case_insensitive(self):
        assert _track_id("TITLE", "ARTIST", 180.0) == _track_id("title", "artist", 180.0)

    def test_different_titles_differ(self):
        assert _track_id("Title A", "Artist", 180.0) != _track_id("Title B", "Artist", 180.0)

    def test_empty_strings_do_not_raise(self):
        tid = _track_id("", "", 0.0)
        assert len(tid) == 12

    def test_duration_rounding_groups_similar_durations(self):
        # :.1f rounds at 0.05 — values in the same 0.1s bucket produce the same ID
        assert _track_id("T", "A", 180.01) == _track_id("T", "A", 180.04)

    def test_different_durations_differ(self):
        assert _track_id("T", "A", 180.0) != _track_id("T", "A", 240.0)


# ---------------------------------------------------------------------------
# _dumps
# ---------------------------------------------------------------------------

class TestDumps:
    def test_empty_list_returns_empty_json_array(self):
        assert _dumps([]) == "[]"

    def test_plain_strings(self):
        result = _dumps(["a", "b"])
        parsed = json.loads(result)
        assert parsed == ["a", "b"]

    def test_pydantic_models(self):
        segs = [TranscriptSegment(start=0.0, end=5.0, text="hi")]
        result = _dumps(segs)
        parsed = json.loads(result)
        assert parsed[0]["text"] == "hi"
        assert parsed[0]["start"] == 0.0

    def test_compact_separators(self):
        result = _dumps(["x"])
        assert " " not in result  # compact JSON, no spaces after delimiters


# ---------------------------------------------------------------------------
# ReportExporter.build — minimal result
# ---------------------------------------------------------------------------

class TestBuildMinimal:
    def setup_method(self):
        self.exporter = ReportExporter()
        self.report = self.exporter.build(_minimal_result())

    def test_returns_track_report(self):
        assert isinstance(self.report, TrackReport)

    def test_track_id_is_12_chars(self):
        assert len(self.report.track_id) == 12

    def test_scan_timestamp_is_iso(self):
        ts = self.report.scan_timestamp
        # ISO 8601 with timezone offset
        assert "T" in ts and (ts.endswith("Z") or "+" in ts or "-" in ts[10:])

    def test_title_and_artist_from_metadata(self):
        assert self.report.title == "Test Track"
        assert self.report.artist == "Test Artist"

    def test_no_forensics_sets_defaults(self):
        assert self.report.forensic_verdict == ""
        assert self.report.ai_probability == 0.0
        assert self.report.c2pa_flag is False

    def test_no_compliance_sets_defaults(self):
        assert self.report.compliance_grade == ""
        assert self.report.total_flag_count == 0
        assert self.report.sting_flag is None

    def test_no_quality_sets_none(self):
        assert self.report.integrated_lufs is None
        assert self.report.dialogue_score is None

    def test_no_popularity_sets_none(self):
        assert self.report.popularity_score is None
        assert self.report.lastfm_listeners == 0

    def test_json_blobs_are_empty_arrays(self):
        assert self.report.compliance_flags_json == "[]"
        assert self.report.ai_segments_json == "[]"
        assert self.report.sections_json == "[]"
        assert self.report.transcript_json == "[]"
        assert self.report.similar_tracks_json == "[]"

    def test_duration_zero_when_no_structure(self):
        assert self.report.duration_s == 0.0


# ---------------------------------------------------------------------------
# ReportExporter.build — full result
# ---------------------------------------------------------------------------

class TestBuildFull:
    def setup_method(self):
        self.exporter = ReportExporter()
        self.result = _full_result()
        self.report = self.exporter.build(self.result)

    def test_title_artist_source(self):
        assert self.report.title == "Full Track"
        assert self.report.artist == "Full Artist"
        assert self.report.source == "youtube"

    def test_engagement_counts(self):
        assert self.report.yt_view_count == 1_000_000
        assert self.report.yt_like_count == 50_000

    def test_structure_counts(self):
        assert self.report.bpm == pytest.approx(120.0)
        assert self.report.key == "C Major"
        assert self.report.section_count == 5
        assert self.report.beat_count == 128
        assert self.report.intro_s == pytest.approx(8.0)
        assert self.report.verse_count == 1
        assert self.report.chorus_count == 2
        assert self.report.bridge_count == 1

    def test_duration_from_last_beat(self):
        # beats = [0, 1, ..., 127] → last beat = 127.0
        assert self.report.duration_s == pytest.approx(127.0)

    def test_forensics_verdict_and_probability(self):
        assert self.report.forensic_verdict == "Likely AI"
        assert self.report.ai_probability == pytest.approx(0.82)
        assert self.report.c2pa_flag is True
        assert self.report.c2pa_origin == "ai"
        assert self.report.is_vocal is True
        assert self.report.forensic_flag_count == 2

    def test_forensics_raw_signals(self):
        assert self.report.ibi_variance == pytest.approx(0.0002)
        assert self.report.loop_score == pytest.approx(0.95)
        assert self.report.harmonic_ratio_score == pytest.approx(0.88)
        assert self.report.noise_floor_ratio == pytest.approx(0.001)

    def test_compliance_grade_and_counts(self):
        assert self.report.compliance_grade == "C"
        assert self.report.total_flag_count == 2
        assert self.report.confirmed_flag_count == 1
        assert self.report.potential_flag_count == 1
        assert self.report.hard_flag_count == 1
        assert self.report.soft_flag_count == 0  # confirmed soft = 0 (the confirmed one is hard)

    def test_compliance_sub_results(self):
        assert self.report.sting_ending_type == "sting"
        assert self.report.sting_flag is False
        assert self.report.energy_evolution_flag is True
        assert self.report.stagnant_windows == 1
        assert self.report.total_windows == 8
        assert self.report.intro_flag is False
        assert self.report.intro_seconds == pytest.approx(8.0)
        assert self.report.intro_source == "allin1"

    def test_authorship_fields(self):
        assert self.report.authorship_verdict == "Likely Human"
        assert self.report.authorship_signal_count == 1
        assert self.report.roberta_score == pytest.approx(0.12)
        assert self.report.burstiness_score == pytest.approx(0.18)
        assert self.report.unique_word_ratio == pytest.approx(0.52)
        assert self.report.rhyme_density == pytest.approx(0.65)
        assert self.report.repetition_score == pytest.approx(0.30)

    def test_audio_quality_fields(self):
        assert self.report.integrated_lufs == pytest.approx(-14.5)
        assert self.report.true_peak_dbfs == pytest.approx(-0.8)
        assert self.report.true_peak_warning is False
        assert self.report.dialogue_label == "Dialogue-Ready"

    def test_popularity_fields(self):
        assert self.report.popularity_score == 68
        assert self.report.popularity_tier == "Mainstream"
        assert self.report.lastfm_listeners == 250_000
        assert self.report.spotify_score == 72
        assert self.report.sync_cost_low == 15_000
        assert self.report.sync_cost_high == 100_000

    def test_legal_fields(self):
        assert self.report.isrc == "US-AB1-23-00001"
        assert self.report.pro_match == "ASCAP/BMI (US)"

    def test_compliance_flags_json(self):
        parsed = json.loads(self.report.compliance_flags_json)
        assert len(parsed) == 2
        assert parsed[0]["issue_type"] == "EXPLICIT"

    def test_ai_segments_json(self):
        parsed = json.loads(self.report.ai_segments_json)
        assert len(parsed) == 1
        assert parsed[0]["probability"] == pytest.approx(0.9)

    def test_sections_json(self):
        parsed = json.loads(self.report.sections_json)
        assert len(parsed) == 5
        assert parsed[0]["label"] == "intro"

    def test_transcript_json(self):
        parsed = json.loads(self.report.transcript_json)
        assert len(parsed) == 1
        assert parsed[0]["text"] == "hello world"

    def test_similar_tracks_json(self):
        parsed = json.loads(self.report.similar_tracks_json)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Alt Track"


# ---------------------------------------------------------------------------
# ReportExporter.to_csv
# ---------------------------------------------------------------------------

class TestToCsv:
    def setup_method(self):
        self.exporter = ReportExporter()
        self.report = self.exporter.build(_minimal_result())
        self.csv_bytes = self.exporter.to_csv(self.report)

    def test_returns_bytes(self):
        assert isinstance(self.csv_bytes, bytes)

    def test_utf8_decodable(self):
        text = self.csv_bytes.decode("utf-8")
        assert len(text) > 0

    def test_has_header_and_data_row(self):
        reader = csv.DictReader(io.StringIO(self.csv_bytes.decode("utf-8")))
        rows = list(reader)
        assert len(rows) == 1

    def test_header_contains_key_fields(self):
        reader = csv.DictReader(io.StringIO(self.csv_bytes.decode("utf-8")))
        fieldnames = reader.fieldnames or []
        for field in ("track_id", "scan_timestamp", "title", "artist",
                      "ai_probability", "compliance_grade", "integrated_lufs"):
            assert field in fieldnames, f"Missing field: {field}"

    def test_title_in_row(self):
        reader = csv.DictReader(io.StringIO(self.csv_bytes.decode("utf-8")))
        row = next(reader)
        assert row["title"] == "Test Track"
        assert row["artist"] == "Test Artist"

    def test_json_blob_column_is_valid_json(self):
        reader = csv.DictReader(io.StringIO(self.csv_bytes.decode("utf-8")))
        row = next(reader)
        parsed = json.loads(row["compliance_flags_json"])
        assert parsed == []

    def test_full_result_csv_round_trip(self):
        full_report = self.exporter.build(_full_result())
        csv_bytes = self.exporter.to_csv(full_report)
        reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
        row = next(reader)
        assert row["forensic_verdict"] == "Likely AI"
        assert row["compliance_grade"] == "C"
        assert float(row["ai_probability"]) == pytest.approx(0.82)

    def test_column_count_matches_model_fields(self):
        reader = csv.DictReader(io.StringIO(self.csv_bytes.decode("utf-8")))
        expected_count = len(TrackReport.model_fields)
        assert len(reader.fieldnames or []) == expected_count
