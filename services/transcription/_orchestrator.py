"""
services/transcription/_orchestrator.py
Lyrics transcription — two implementations of TranscriptionProvider.

LyricsOrchestrator (primary — use this in production):
    1. Try LRCLib (free synced lyrics API) — returns timestamped segments
       immediately if the track is known. No GPU, no audio processing.
    2. On miss: transcribe the full mix with Whisper (GPU).

Transcription (Whisper-only — used as the inner fallback by LyricsOrchestrator):
    Runs OpenAI Whisper on a raw audio buffer.
    Writes bytes to a NamedTemporaryFile, transcribes, deletes in finally.
"""
from __future__ import annotations

import os
import tempfile

from core.config import ModelParams, get_settings
from core.exceptions import ConfigurationError, ModelInferenceError
from core.logging import PipelineLogger
from core.models import AudioBuffer, TranscriptSegment
from core.protocols import LyricsProvider
from services.lyrics import LRCLibClient

from ._pure import _parse_segments

try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn


class Transcription:
    """
    Transcribes audio to time-stamped lyric segments using OpenAI Whisper.

    Implements: TranscriptionProvider protocol (core/protocols.py)

    Constructor injection: pass a ModelParams instance to control model size
    and inference settings without touching environment variables.

    Usage:
        service  = Transcription()
        segments = service.transcribe(audio_buffer)
        for seg in segments:
            print(seg.start, seg.text)
    """

    def __init__(self, params: ModelParams | None = None) -> None:
        self._params = params or ModelParams(
            whisper_model=get_settings().whisper_model
        )

    # ------------------------------------------------------------------
    # Public interface (TranscriptionProvider protocol)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: AudioBuffer,
        title: str = "",
        artist: str = "",
    ) -> list[TranscriptSegment]:
        """
        Transcribe an AudioBuffer to a list of time-stamped segments.

        title and artist are accepted for protocol compatibility with
        LyricsOrchestrator but are not used — Whisper works from audio only.

        Args:
            audio:  In-memory audio buffer from Ingestion.
            title:  Unused — accepted for TranscriptionProvider compatibility.
            artist: Unused — accepted for TranscriptionProvider compatibility.

        Returns:
            Ordered list of TranscriptSegment objects (may be empty for
            instrumental tracks or very short audio).

        Raises:
            ConfigurationError:  openai-whisper is not installed.
            ModelInferenceError: Whisper model load or inference failed
                                 (OOM, corrupt audio, unexpected output).
        """
        return self._run_whisper(audio.raw)

    # ------------------------------------------------------------------
    # Private: Whisper inference  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _run_whisper(self, raw: bytes) -> list[TranscriptSegment]:
        """
        Write raw bytes to a temp WAV, run Whisper, delete the file.

        Raises:
            ConfigurationError:  whisper package not importable.
            ModelInferenceError: model.transcribe() or load_model() failed.
        """
        try:
            import whisper
        except ImportError as exc:
            raise ConfigurationError(
                "openai-whisper is not installed.",
                context={
                    "suggestion": "pip install openai-whisper",
                    "original_error": str(exc),
                },
            ) from exc

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            model = whisper.load_model(
                self._params.whisper_model,
                # fp16=False keeps inference safe on CPU and MPS
            )
            transcribe_kwargs: dict = dict(
                task="transcribe",
                language=self._params.whisper_language,
                fp16=self._params.whisper_fp16,
                temperature=self._params.whisper_temperature,
                # Prevents repetition loops common when transcribing music.
                condition_on_previous_text=self._params.whisper_condition_on_previous_text,
                no_speech_threshold=self._params.whisper_no_speech_threshold,
                compression_ratio_threshold=self._params.whisper_compression_ratio_threshold,
                logprob_threshold=self._params.whisper_logprob_threshold,
            )
            # Only pass initial_prompt when non-empty — an empty prompt still
            # activates Whisper's prompt-completion path on silent windows,
            # generating meta-text rather than transcribing audio.
            if self._params.whisper_initial_prompt:
                transcribe_kwargs["initial_prompt"] = self._params.whisper_initial_prompt

            result = model.transcribe(tmp_path, **transcribe_kwargs)

            raw_segments = result.get("segments", [])
            return _parse_segments(raw_segments)

        except (ConfigurationError, ModelInferenceError):
            raise
        except Exception as exc:
            raise ModelInferenceError(
                "Whisper transcription failed.",
                context={
                    "model": self._params.whisper_model,
                    "original_error": str(exc),
                },
            ) from exc

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


# ---------------------------------------------------------------------------
# LyricsOrchestrator — primary TranscriptionProvider for production
# ---------------------------------------------------------------------------

class LyricsOrchestrator:
    """
    Orchestrates lyrics acquisition with a tiered fallback strategy.

    Tier 1 — LRCLib (no GPU, no audio processing):
        Queries lrclib.net for synced lyrics by title + artist. Returns
        timestamped segments immediately on hit. This is fast and accurate
        because the lyrics come from a human-curated database.

    Tier 2 — Whisper on the full mix (GPU):
        On LRCLib miss, runs Whisper directly on the full audio buffer.
        Demucs vocal isolation was tested and removed — it destroyed vocal
        content in chorus-first arrangements and degraded Whisper accuracy.

    Implements: TranscriptionProvider protocol (core/protocols.py)

    Constructor injection:
        lyrics_provider — any LyricsProvider (default: LRCLibClient).
        whisper         — Transcription instance used for the Whisper fallback.
        params          — ModelParams controlling Whisper settings.

    Usage:
        service  = LyricsOrchestrator()
        segments = service.transcribe(audio, title="Shape of You", artist="Ed Sheeran")
    """

    def __init__(
        self,
        lyrics_provider: LyricsProvider | None = None,
        whisper: Transcription | None = None,
        params: ModelParams | None = None,
    ) -> None:
        self._lyrics = lyrics_provider or LRCLibClient()
        self._whisper = whisper or Transcription()
        self._params = params or ModelParams()
        self._log = PipelineLogger(get_settings().log_dir)

    # ------------------------------------------------------------------
    # Public interface (TranscriptionProvider protocol)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: AudioBuffer,
        title: str = "",
        artist: str = "",
    ) -> list[TranscriptSegment]:
        """
        Return lyrics segments, preferring LRCLib over audio-based transcription.

        Args:
            audio:  In-memory audio buffer.
            title:  Track title for LRCLib lookup.
            artist: Artist name for LRCLib lookup.

        Returns:
            Ordered list of TranscriptSegment objects.

        Raises:
            ModelInferenceError: if both LRCLib and the Whisper fallback fail.
        """
        # Tier 1: synced lyrics from LRCLib (fast, exact — no audio processing)
        segments = self._lyrics.get_lyrics(title, artist)
        if segments is not None:
            return segments

        # Tier 2: Whisper on the full mix.
        # NOTE: Demucs vocal isolation was tested and found to hurt accuracy —
        # it destroys vocal content in chorus-first arrangements and introduces
        # artifacts that degrade Whisper recognition. Full-mix input is better.
        return self._whisper.transcribe(audio)
