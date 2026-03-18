"""
core/logging.py
Structured daily logging for the Sync-Safe pipeline.

Two classes, two responsibilities:

  LogCleaner     — deletes any log files that are not today's file.
                   Call once at app startup. Never raises: failures are
                   printed to stderr so they don't crash the app.

  PipelineLogger — writes newline-delimited JSON entries to today's log
                   file (logs/sync-safe-YYYY-MM-DD.log). Thread-safe via
                   a per-instance lock. Keeps the file handle open for
                   the lifetime of the instance to avoid repeated open/close
                   overhead. Never raises: I/O errors are printed to stderr
                   and silently swallowed so a logging failure never
                   interrupts the pipeline.

Usage:
    from core.logging import LogCleaner, PipelineLogger
    from core.config import get_settings

    settings = get_settings()
    LogCleaner(settings.log_dir).clean()
    log = PipelineLogger(settings.log_dir)

    log.pipeline_start(source="https://youtube.com/...")
    log.step_start("transcription")
    log.step_end("transcription", duration_s=47.2)
    log.step_error("structure", error="allin1 analysis failed.")
    log.pipeline_end(duration_s=132.5)
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import IO, Any

# Absolute path to the project root (parent of core/).
_PROJECT_ROOT = Path(__file__).parent.parent

# Default log directory — absolute so it resolves correctly regardless of cwd.
DEFAULT_LOG_DIR = _PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# LogCleaner
# ---------------------------------------------------------------------------

class LogCleaner:
    """
    Deletes log files in log_dir that do not belong to today.

    Matches files of the form sync-safe-YYYY-MM-DD.log — any file
    matching that pattern whose date differs from today is removed.
    Non-matching files are left untouched.

    Implements: no protocol — purely a startup side-effect.
    """

    _PATTERN = "sync-safe-*.log"

    def __init__(self, log_dir: str | Path = DEFAULT_LOG_DIR) -> None:
        self._dir = Path(log_dir)

    def clean(self) -> None:
        """Remove stale log files. Silently skips on any error."""
        try:
            if not self._dir.exists():
                return
            today = date.today().isoformat()          # "2026-03-18"
            for path in self._dir.glob(self._PATTERN):
                # Filename: sync-safe-YYYY-MM-DD.log → extract date segment
                stem = path.stem                      # "sync-safe-2026-03-17"
                file_date = stem.removeprefix("sync-safe-")
                if file_date != today:
                    try:
                        path.unlink()
                    except OSError as exc:
                        print(f"[LogCleaner] Could not delete {path}: {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[LogCleaner] Cleanup failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# PipelineLogger
# ---------------------------------------------------------------------------

class PipelineLogger:
    """
    Writes structured JSON log entries to today's daily log file.

    File path: <log_dir>/sync-safe-YYYY-MM-DD.log
    Format:    one JSON object per line (newline-delimited JSON / NDJSON)

    Each entry always contains:
        ts    — ISO-8601 UTC timestamp (timezone-aware)
        event — snake_case event name

    Additional fields are passed as keyword arguments and merged in.

    The file handle is opened once on first write and kept open for the
    lifetime of the instance — avoiding repeated open/close cycles across
    the ~16 log calls per pipeline run.

    Constructor injection: pass log_dir to override the default.
    """

    def __init__(self, log_dir: str | Path = DEFAULT_LOG_DIR) -> None:
        self._dir = Path(log_dir)
        self._lock = threading.Lock()
        self._fh: IO[str] | None = None
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------

    def pipeline_start(self, source: str) -> None:
        """Log the start of a full pipeline run."""
        # Redact full URLs to just the domain/type to avoid storing user data.
        source_type = _classify_source(source)
        self._write(event="pipeline_start", source_type=source_type)

    def pipeline_end(self, duration_s: float) -> None:
        """Log successful completion of the full pipeline."""
        self._write(event="pipeline_end", duration_s=round(duration_s, 2))

    def pipeline_error(self, error: str) -> None:
        """Log a fatal pipeline error (e.g. ingestion failure)."""
        self._write(event="pipeline_error", error=error)

    def step_start(self, step: str) -> None:
        """Log the start of a named pipeline step."""
        self._write(event="step_start", step=step)

    def step_end(self, step: str, duration_s: float) -> None:
        """Log successful completion of a pipeline step."""
        self._write(event="step_end", step=step, duration_s=round(duration_s, 2))

    def step_error(self, step: str, error: str) -> None:
        """Log a non-fatal step failure (step degraded to None)."""
        self._write(event="step_error", step=step, error=error)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _log_path(self) -> Path:
        return self._dir / f"sync-safe-{date.today().isoformat()}.log"

    def _get_handle(self) -> IO[str]:
        """Return the open file handle, opening it if necessary."""
        if self._fh is None or self._fh.closed:
            self._fh = self._log_path().open("a", encoding="utf-8")
        return self._fh

    def _write(self, **fields: Any) -> None:
        """Serialise fields to JSON and append to today's log file."""
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry = {"ts": ts, **fields}
        line = json.dumps(entry, ensure_ascii=False)
        try:
            with self._lock:
                fh = self._get_handle()
                fh.write(line + "\n")
                fh.flush()
        except Exception as exc:  # noqa: BLE001
            print(f"[PipelineLogger] Write failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def _classify_source(source: object) -> str:
    """
    Return a safe label for the ingestion source without storing user data.

    Pure function — no I/O, independently testable.
    """
    if isinstance(source, str):
        lower = source.lower()
        if "youtube.com" in lower or "youtu.be" in lower:
            return "youtube"
        return "url"
    # Streamlit UploadedFile or similar file-like object
    return "upload"
