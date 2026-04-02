"""
services/transcription.py
Lyrics transcription — two implementations of TranscriptionProvider.

LyricsOrchestrator (primary — use this in production):
    1. Try LRCLib (free synced lyrics API) — returns timestamped segments
       immediately if the track is known. No GPU, no audio processing.
    2. On miss: isolate vocals with Demucs (GPU), then transcribe the
       vocal stem with Whisper (GPU). Two sequential GPU calls, each
       self-contained as required by HF ZeroGPU.

Transcription (Whisper-only — used as the inner fallback by LyricsOrchestrator):
    Runs OpenAI Whisper on a raw audio buffer (or pre-isolated vocal buffer).
    Writes bytes to a NamedTemporaryFile, transcribes, deletes in finally.

Design notes:
- LyricsOrchestrator is constructor-injected with a LyricsProvider and a
  Transcription instance — both are swappable without changing this file.
- ModelParams controls both the Whisper model and the Demucs model name
  used to locate the separation output path.
- _parse_segments is a pure module-level function for independent testing.
- All GPU functions import their heavy deps (whisper, demucs) inside the
  function body so the module is importable without those packages installed.
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
# Module-level pure function — independently testable
# ---------------------------------------------------------------------------

def _parse_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    """
    Convert Whisper's raw segment dicts into typed TranscriptSegment objects,
    then strip hallucinated repetition runs.

    Pure function — no I/O, no model calls. Handles missing or malformed
    keys gracefully so a partial Whisper result still produces valid output.
    """
    segments: list[TranscriptSegment] = []
    for seg in raw_segments:
        text = _collapse_intra_repetitions(seg.get("text", "").strip())
        if not text:
            continue
        segments.append(TranscriptSegment(
            start=round(float(seg.get("start", 0.0)), 2),
            end=round(float(seg.get("end", 0.0)), 2),
            text=text,
        ))
    return _strip_repetition_runs(segments)


def _collapse_intra_repetitions(text: str, max_keep: int = 2) -> str:
    """
    Collapse repeated phrases within a single Whisper segment.

    Whisper sometimes loops *within* a single decoding window, e.g.:
      "we are not, we are not, we are not, ..." × 55  (comma-separated)
      "we are not we are not we are not ..."           (space-separated)

    Uses two passes:
      1. Comma-split: detects runs in comma-delimited clauses.
      2. Regex: detects n-gram repetitions (1–6 words) with any separator.

    Natural commas ("See you can't sleep, baby, I know") are unaffected
    because no clause there appears more than max_keep times in a row.

    Pure function — no I/O.
    """
    import re

    # Pass 1 — comma-separated runs
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= max_keep + 2:
            result: list[str] = []
            i = 0
            while i < len(parts):
                phrase = parts[i].lower()
                run_end = i + 1
                while run_end < len(parts) and parts[run_end].lower() == phrase:
                    run_end += 1
                keep = min(run_end - i, max_keep)
                result.extend(parts[i : i + keep])
                i = run_end
            collapsed = ", ".join(result)
            if len(collapsed) < len(text):
                text = collapsed

    # Pass 2 — regex n-gram repetition (catches space-only separators)
    # Matches a phrase of 1–6 words repeated 4+ times (with any separator).
    pattern = re.compile(
        r'\b((?:\w[\w\']*(?:\s+|,\s*))){1,6}(?=(?:\1){3,})',
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        phrase = match.group(0).strip(", ")
        # Replace the entire repetitive block with max_keep copies
        repetition_block = re.compile(
            r'(?:' + re.escape(phrase) + r'(?:[,\s]+|$)){4,}',
            re.IGNORECASE,
        )
        replacement = (", ".join([phrase] * max_keep))
        text = repetition_block.sub(replacement, text)

    return text.strip(", ")


def _strip_repetition_runs(
    segments: list[TranscriptSegment],
    max_run: int = 3,
    uniform_gap_tolerance: float = 0.15,
) -> list[TranscriptSegment]:
    """
    Remove hallucinated repetition runs from Whisper output.

    When Whisper transcribes near-silence (e.g. an instrumental intro after
    vocal isolation), it hallucinates the same phrase repeatedly at fixed
    intervals — e.g. "I can't relate" × 14, each exactly 2.0s apart.

    Real repeated lyrics (a chorus, a hook) also repeat, but their inter-segment
    gaps vary naturally. This function only drops a run when BOTH conditions hold:
      1. The run length exceeds max_run (> 3 identical consecutive segments)
      2. The gaps between segments in the run are suspiciously uniform
         (all within uniform_gap_tolerance seconds of each other)

    This prevents false-positives where a real lyric phrase like "I can't relate"
    happens to appear early and gets mistakenly stripped along with the
    hallucinated run that preceded it.

    Pure function — no I/O.
    """
    if not segments:
        return segments

    result: list[TranscriptSegment] = []
    i = 0
    while i < len(segments):
        run_text = segments[i].text.lower()
        run_end  = i + 1
        while run_end < len(segments) and segments[run_end].text.lower() == run_text:
            run_end += 1
        run_length = run_end - i

        # Drop if EITHER: very long run (≥5 = always hallucination, real
        # choruses don't repeat the same phrase identically 5+ times in a row)
        # OR shorter run with machine-regular gaps (uniform-interval
        # hallucination on silence).
        is_hallucination = (
            run_length >= 5
            or (run_length > max_run and _gaps_are_uniform(segments[i:run_end], uniform_gap_tolerance))
        )
        if is_hallucination:
            pass  # drop
        else:
            result.extend(segments[i:run_end])

        i = run_end

    return result


def _gaps_are_uniform(
    run: list[TranscriptSegment],
    tolerance: float,
) -> bool:
    """
    Return True if the gaps between consecutive segments in a run are all
    within `tolerance` seconds of the median gap.

    Whisper hallucination runs land at perfectly even intervals (e.g. every
    2.000s). Real repeated lyrics have natural variation in spacing.

    Pure function — no I/O.
    """
    if len(run) < 2:
        return False
    gaps = [run[k + 1].start - run[k].start for k in range(len(run) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]
    return all(abs(g - median_gap) <= tolerance for g in gaps)


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

