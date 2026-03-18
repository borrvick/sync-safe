"""
services/nlp.py
Lyrics transcription via OpenAI Whisper — implements TranscriptionProvider.

Whisper requires a file path, so audio bytes are written to a NamedTemporaryFile
and deleted in a finally block. This is the only intentional disk write here.

Design notes:
- Transcription.transcribe() is the single public entry point.
- Model size comes from ModelParams.whisper_model (constructor-injected),
  not os.environ — callers control the model without touching env vars.
- A missing whisper library raises ConfigurationError (setup problem),
  not ModelInferenceError (runtime problem).
- Whisper inference failures raise ModelInferenceError with the original
  exception in context — never returned as a string in the transcript.
- _parse_segments is a pure module-level function for independent testing.
"""
from __future__ import annotations

import os
import tempfile

from core.config import ModelParams, get_settings
from core.exceptions import ConfigurationError, ModelInferenceError
from core.models import AudioBuffer, TranscriptSegment

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

    def transcribe(self, audio: AudioBuffer) -> list[TranscriptSegment]:
        """
        Transcribe an AudioBuffer to a list of time-stamped segments.

        Args:
            audio: In-memory audio buffer from Ingestion.

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
            result = model.transcribe(
                tmp_path,
                task="transcribe",
                fp16=self._params.whisper_fp16,
                temperature=self._params.whisper_temperature,
            )

            return _parse_segments(result.get("segments", []))

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
# Module-level pure function — independently testable
# ---------------------------------------------------------------------------

def _parse_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    """
    Convert Whisper's raw segment dicts into typed TranscriptSegment objects.

    Pure function — no I/O, no model calls. Handles missing or malformed
    keys gracefully so a partial Whisper result still produces valid output.
    """
    segments: list[TranscriptSegment] = []
    for i, seg in enumerate(raw_segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(TranscriptSegment(
            start=round(float(seg.get("start", 0.0)), 2),
            end=round(float(seg.get("end", 0.0)), 2),
            text=text,
        ))
    return segments
