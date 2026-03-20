"""
pipeline.py
Single-entry-point orchestrator for the Sync-Safe analysis pipeline.

Dependency order:
  1. Ingestion       — AudioBuffer (required; all other steps depend on it)
  2. Structure       — StructureResult          (title/artist needed for lyrics lookup)
  3. Transcription   — list[TranscriptSegment]  (uses title/artist from step 2)
  4. Forensics       — ForensicsResult          (independent)
  5. Compliance      — ComplianceReport         (needs audio + transcript + structure)
  6. Authorship      — AuthorshipResult         (needs transcript)
  7. Discovery       — list[TrackCandidate]     (needs structure metadata)
  8. Legal           — LegalLinks               (needs structure metadata)

Note: Structure runs before Transcription (reversed from historical order) so
that title + artist metadata are available for the LRCLib lyrics lookup.

Design rules:
- Pipeline accepts all services via constructor injection so tests can swap
  any implementation without touching env vars or globals.
- Default constructors create the production service instances; callers that
  only care about one step can still construct the full Pipeline — unused
  services are never called unless their step is reached.
- Each step is timed and logged via PipelineLogger. A failing step logs the
  error and leaves the corresponding field as None/empty — a transcript failure
  does not block forensics, for example.
- No Streamlit imports — this module must be importable outside a UI context.
"""
from __future__ import annotations

import time
import traceback
from typing import Union

import streamlit as st  # UploadedFile type hint only — no UI calls here

from core.config import get_settings
from core.exceptions import SyncSafeError
from core.logging import PipelineLogger
from core.models import AnalysisResult, AudioBuffer
from services.analysis import Analysis
from services.authorship import Authorship
from services.compliance import Compliance
from services.discovery import Discovery
from services.forensics import Forensics
from services.ingestion import Ingestion
from services.legal import Legal
from services.transcription import LyricsOrchestrator, Transcription


class Pipeline:
    """
    Orchestrates the full Sync-Safe analysis flow.

    All service dependencies are constructor-injected. Passing None for any
    service uses the production default. This design allows tests to inject
    stubs without monkey-patching.

    Usage:
        result = Pipeline().run(source)
        # or with custom services:
        result = Pipeline(transcription=MyFastTranscriber()).run(source)
    """

    def __init__(
        self,
        ingestion:     Ingestion              | None = None,
        transcription: LyricsOrchestrator     | None = None,
        structure:     Analysis               | None = None,
        forensics:     Forensics              | None = None,
        compliance:    Compliance             | None = None,
        authorship:    Authorship             | None = None,
        discovery:     Discovery              | None = None,
        legal:         Legal                  | None = None,
        log:           PipelineLogger         | None = None,
    ) -> None:
        self._ingestion     = ingestion     or Ingestion()
        self._transcription = transcription or LyricsOrchestrator()
        self._structure     = structure     or Analysis()
        self._forensics     = forensics     or Forensics()
        self._compliance    = compliance    or Compliance()
        self._authorship    = authorship    or Authorship()
        self._discovery     = discovery     or Discovery()
        self._legal         = legal         or Legal()
        self._log           = log           or PipelineLogger(get_settings().log_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        source: Union[str, "st.runtime.uploaded_file_manager.UploadedFile"],
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline on a YouTube URL or uploaded file.

        Args:
            source: A YouTube URL string or a Streamlit UploadedFile object.

        Returns:
            AnalysisResult with every field populated, or None/empty for any
            step that failed.

        Raises:
            AudioSourceError:   if audio ingestion fails (nothing to analyse).
            ValidationError:    if the source URL is malformed.
            ConfigurationError: if a required binary or API key is missing.
            All other SyncSafeError subtypes propagate from ingestion only —
            downstream step failures are captured into errors[], not raised.
        """
        self._log.pipeline_start(source=str(source))
        t0 = time.monotonic()
        try:
            audio: AudioBuffer = self._ingestion.load(source)
            result = self.run_analysis(audio)
            self._log.pipeline_end(duration_s=round(time.monotonic() - t0, 2))
            return result
        except SyncSafeError as exc:
            self._log.pipeline_error(error=str(exc))
            raise

    def run_analysis(self, audio: AudioBuffer) -> AnalysisResult:
        """
        Run all analysis steps on an already-ingested AudioBuffer.

        Call this when audio is fetched separately (e.g. on the landing page)
        so ingestion is not duplicated on the report page.

        Raises:
            Never — all step failures are captured and logged; missing
            steps produce None/empty fields in the result.
        """
        # Step 2: Structure first — title/artist metadata needed for lyrics lookup
        structure = self._run_step("structure", lambda: self._structure.analyze(audio))

        # Prefer embedded-tag metadata from structure analysis; fall back to
        # yt-dlp metadata stored on the AudioBuffer at ingestion time.
        title  = (structure.metadata.get("title", "") if structure else "") or audio.metadata.get("title", "")
        artist = (structure.metadata.get("artist", "") if structure else "") or audio.metadata.get("artist", "")

        # Step 3: Transcription — passes title/artist to enable LRCLib lookup
        transcript = self._run_step(
            "transcription",
            lambda: self._transcription.transcribe(audio, title=title, artist=artist),
        )

        # Step 4: Forensics — independent of transcript and structure
        forensics = self._run_step("forensics", lambda: self._forensics.analyze(audio))

        # Step 5: Compliance — needs audio + transcript + structure
        sections = structure.sections if structure else []
        beats    = structure.beats    if structure else []
        compliance = self._run_step(
            "compliance",
            lambda: self._compliance.check(audio, transcript or [], sections, beats),
        )

        # Step 6: Authorship — needs transcript
        authorship = self._run_step(
            "authorship",
            lambda: self._authorship.analyze(transcript or []),
        )

        # Steps 7–8: Discovery and legal — need title/artist from structure
        similar_tracks_result = self._run_step(
            "discovery",
            lambda: self._discovery.find_similar(title, artist),
        )
        legal = self._run_step(
            "legal",
            lambda: self._legal.get_links(title, artist),
        )

        return AnalysisResult(
            audio=audio,
            structure=structure,
            forensics=forensics,
            transcript=transcript or [],
            compliance=compliance,
            authorship=authorship,
            similar_tracks=similar_tracks_result or [],
            legal=legal,
        )

    # ------------------------------------------------------------------
    # Private: fault-isolated step runner
    # ------------------------------------------------------------------

    def _run_step(self, name: str, fn):
        """
        Execute fn(), timing the call and logging start/end/error.

        On any exception, logs a structured error entry and returns None so
        downstream steps can continue. SyncSafeError context dicts are
        included in the log entry for diagnostics.
        """
        self._log.step_start(name)
        t0 = time.monotonic()
        try:
            result = fn()
            self._log.step_end(name, duration_s=round(time.monotonic() - t0, 2))
            return result
        except SyncSafeError as exc:
            self._log.step_error(name, error=str(exc))
            _print_step_error(name, exc, context=exc.context)
            return None
        except Exception as exc:  # noqa: BLE001
            self._log.step_error(name, error=str(exc))
            _print_step_error(name, exc)
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _print_step_error(name: str, exc: Exception, context: dict | None = None) -> None:
    """Print a structured error line for a failed pipeline step."""
    ctx = f" | context={context}" if context else ""
    print(f"[pipeline] {name} failed: {exc}{ctx}")
    if not isinstance(exc, SyncSafeError):
        traceback.print_exc()
