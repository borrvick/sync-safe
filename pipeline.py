"""
pipeline.py
Single-entry-point orchestrator for the Sync-Safe analysis pipeline.

Dependency order:
  1. Ingestion       — AudioBuffer (required; all other steps depend on it)
  2. Transcription   — list[TranscriptSegment]  (needed by compliance + authorship)
  3. Structure       — StructureResult          (needed by compliance)
  4. Forensics       — ForensicsResult          (independent of 2 & 3)
  5. Compliance      — ComplianceReport         (needs 1 + 2 + 3)
  6. Authorship      — AuthorshipResult         (needs 2)
  7. Discovery       — list[TrackCandidate]     (needs structure metadata)
  8. Legal           — LegalLinks               (needs structure metadata)

Design rules:
- Pipeline accepts all services via constructor injection so tests can swap
  any implementation without touching env vars or globals.
- Default constructors create the production service instances; callers that
  only care about one step can still construct the full Pipeline — unused
  services are never called unless their step is reached.
- Each step is wrapped in try/except. A failing step logs the error to
  AnalysisResult.errors and leaves the corresponding field as None/empty.
  This means a transcript failure doesn't block forensics, for example.
- No Streamlit imports — this module must be importable outside a UI context.
"""
from __future__ import annotations

import traceback
from typing import Union

import streamlit as st  # UploadedFile type hint only — no UI calls here

from core.exceptions import SyncSafeError
from core.models import AnalysisResult, AudioBuffer
from services.analysis import Analysis
from services.authorship import Authorship
from services.compliance import Compliance
from services.discovery import Discovery
from services.forensics import Forensics
from services.ingestion import Ingestion
from services.legal import Legal
from services.transcription import Transcription


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
        ingestion:     Ingestion     | None = None,
        transcription: Transcription | None = None,
        structure:     Analysis      | None = None,
        forensics:     Forensics     | None = None,
        compliance:    Compliance    | None = None,
        authorship:    Authorship    | None = None,
        discovery:     Discovery     | None = None,
        legal:         Legal         | None = None,
    ) -> None:
        self._ingestion     = ingestion     or Ingestion()
        self._transcription = transcription or Transcription()
        self._structure     = structure     or Analysis()
        self._forensics     = forensics     or Forensics()
        self._compliance    = compliance    or Compliance()
        self._authorship    = authorship    or Authorship()
        self._discovery     = discovery     or Discovery()
        self._legal         = legal         or Legal()

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
        audio: AudioBuffer = self._ingestion.load(source)
        return self.run_analysis(audio)

    def run_analysis(self, audio: AudioBuffer) -> AnalysisResult:
        """
        Run all analysis steps on an already-ingested AudioBuffer.

        Call this when audio is fetched separately (e.g. on the landing page)
        so ingestion is not duplicated on the report page.

        Raises:
            Never — all step failures are captured and logged; missing
            steps produce None/empty fields in the result.
        """
        # Steps 2–8 — each wrapped so a partial failure doesn't abort the rest
        transcript = self._run_step("transcription", lambda: self._transcription.transcribe(audio))
        structure  = self._run_step("structure",     lambda: self._structure.analyze(audio))
        forensics  = self._run_step("forensics",     lambda: self._forensics.analyze(audio))

        sections = structure.sections if structure else []
        beats    = structure.beats    if structure else []
        compliance = self._run_step(
            "compliance",
            lambda: self._compliance.check(audio, transcript or [], sections, beats),
        )

        authorship = self._run_step(
            "authorship",
            lambda: self._authorship.analyze(transcript or []),
        )

        title  = structure.metadata.get("title", "")  if structure else ""
        artist = structure.metadata.get("artist", "") if structure else ""

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
        Execute fn() and return its result.

        On any exception, print a structured error message and return None.
        SyncSafeError subclasses include a context dict for diagnostics.
        """
        try:
            return fn()
        except SyncSafeError as exc:
            _log_step_error(name, exc, context=exc.context)
            return None
        except Exception as exc:  # noqa: BLE001
            _log_step_error(name, exc)
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _log_step_error(name: str, exc: Exception, context: dict | None = None) -> None:
    """Print a structured error line for a failed pipeline step."""
    ctx = f" | context={context}" if context else ""
    print(f"[pipeline] {name} failed: {exc}{ctx}")
    if not isinstance(exc, SyncSafeError):
        traceback.print_exc()
