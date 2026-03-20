"""
tests/test_logging.py
Unit tests for core/logging.py — LogCleaner, PipelineLogger, _classify_source.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from core.logging import LogCleaner, PipelineLogger, _classify_source


# ---------------------------------------------------------------------------
# _classify_source
# ---------------------------------------------------------------------------

class TestClassifySource:
    def test_youtube_com(self):
        assert _classify_source("https://www.youtube.com/watch?v=abc") == "youtube"

    def test_youtu_be(self):
        assert _classify_source("https://youtu.be/abc123") == "youtube"

    def test_youtube_case_insensitive(self):
        assert _classify_source("HTTPS://WWW.YOUTUBE.COM/watch?v=abc") == "youtube"

    def test_generic_url(self):
        assert _classify_source("https://soundcloud.com/track") == "url"

    def test_non_string_upload(self):
        # Simulates a Streamlit UploadedFile object
        assert _classify_source(object()) == "upload"

    def test_none_is_upload(self):
        assert _classify_source(None) == "upload"

    def test_integer_is_upload(self):
        assert _classify_source(42) == "upload"


# ---------------------------------------------------------------------------
# LogCleaner
# ---------------------------------------------------------------------------

class TestLogCleaner:
    def test_does_nothing_when_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        LogCleaner(missing).clean()   # must not raise

    def test_deletes_stale_log(self, tmp_path: Path) -> None:
        stale = tmp_path / "sync-safe-2000-01-01.log"
        stale.write_text("old\n")
        LogCleaner(tmp_path).clean()
        assert not stale.exists()

    def test_preserves_todays_log(self, tmp_path: Path) -> None:
        today_file = tmp_path / f"sync-safe-{date.today().isoformat()}.log"
        today_file.write_text("keep\n")
        LogCleaner(tmp_path).clean()
        assert today_file.exists()

    def test_preserves_unrelated_files(self, tmp_path: Path) -> None:
        other = tmp_path / "other-file.txt"
        other.write_text("data")
        LogCleaner(tmp_path).clean()
        assert other.exists()

    def test_deletes_multiple_stale_logs(self, tmp_path: Path) -> None:
        stale1 = tmp_path / "sync-safe-2000-01-01.log"
        stale2 = tmp_path / "sync-safe-2000-01-02.log"
        stale1.write_text("a")
        stale2.write_text("b")
        LogCleaner(tmp_path).clean()
        assert not stale1.exists()
        assert not stale2.exists()

    def test_does_not_raise_on_unlink_error(self, tmp_path: Path, monkeypatch) -> None:
        stale = tmp_path / "sync-safe-2000-01-01.log"
        stale.write_text("old")

        monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: (_ for _ in ()).throw(OSError("permission denied")))
        LogCleaner(tmp_path).clean()   # must not raise


# ---------------------------------------------------------------------------
# PipelineLogger
# ---------------------------------------------------------------------------

class TestPipelineLogger:
    def _read_entries(self, tmp_path: Path) -> list[dict]:
        log_file = tmp_path / f"sync-safe-{date.today().isoformat()}.log"
        return [json.loads(line) for line in log_file.read_text().splitlines()]

    def test_creates_log_dir(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "nested" / "logs"
        PipelineLogger(log_dir).pipeline_start(source="https://youtu.be/abc")
        assert log_dir.exists()

    def test_pipeline_start_writes_entry(self, tmp_path: Path) -> None:
        PipelineLogger(tmp_path).pipeline_start(source="https://youtu.be/abc")
        entries = self._read_entries(tmp_path)
        assert len(entries) == 1
        assert entries[0]["event"] == "pipeline_start"
        assert entries[0]["source_type"] == "youtube"

    def test_pipeline_start_classifies_upload(self, tmp_path: Path) -> None:
        PipelineLogger(tmp_path).pipeline_start(source=str(object()))
        entries = self._read_entries(tmp_path)
        assert entries[0]["source_type"] == "url"

    def test_step_end_records_duration(self, tmp_path: Path) -> None:
        log = PipelineLogger(tmp_path)
        log.step_end("transcription", duration_s=47.2)
        entries = self._read_entries(tmp_path)
        assert entries[0]["event"] == "step_end"
        assert entries[0]["step"] == "transcription"
        assert entries[0]["duration_s"] == 47.2

    def test_step_error_records_error(self, tmp_path: Path) -> None:
        PipelineLogger(tmp_path).step_error("structure", error="allin1 failed")
        entries = self._read_entries(tmp_path)
        assert entries[0]["event"] == "step_error"
        assert entries[0]["error"] == "allin1 failed"

    def test_pipeline_error(self, tmp_path: Path) -> None:
        PipelineLogger(tmp_path).pipeline_error(error="ingestion failed")
        entries = self._read_entries(tmp_path)
        assert entries[0]["event"] == "pipeline_error"

    def test_pipeline_end_records_duration(self, tmp_path: Path) -> None:
        PipelineLogger(tmp_path).pipeline_end(duration_s=132.5)
        entries = self._read_entries(tmp_path)
        assert entries[0]["duration_s"] == 132.5

    def test_all_entries_have_ts_field(self, tmp_path: Path) -> None:
        log = PipelineLogger(tmp_path)
        log.step_start("ingestion")
        log.step_end("ingestion", duration_s=1.0)
        for entry in self._read_entries(tmp_path):
            assert "ts" in entry
            # Timezone-aware ISO format ends with +00:00
            assert entry["ts"].endswith("+00:00")

    def test_multiple_entries_appended(self, tmp_path: Path) -> None:
        log = PipelineLogger(tmp_path)
        log.step_start("ingestion")
        log.step_end("ingestion", duration_s=8.1)
        assert len(self._read_entries(tmp_path)) == 2

    def test_duration_rounded_to_two_decimal_places(self, tmp_path: Path) -> None:
        PipelineLogger(tmp_path).step_end("forensics", duration_s=20.123456)
        entries = self._read_entries(tmp_path)
        assert entries[0]["duration_s"] == 20.12

    def test_does_not_raise_on_write_error(self, tmp_path: Path, monkeypatch) -> None:
        log = PipelineLogger(tmp_path)
        # Force _get_handle to raise
        monkeypatch.setattr(log, "_get_handle", lambda: (_ for _ in ()).throw(OSError("disk full")))
        log.step_start("ingestion")   # must not raise
