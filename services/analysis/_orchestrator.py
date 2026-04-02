"""
services/analysis.py
Structure analysis — implements the StructureAnalyzer protocol.

Responsibilities (single class, three private concerns):
  1. _analyse_structure  — allin1: BPM, beat grid, structural sections
  2. _detect_key         — librosa chroma + Krumhansl-Schmuckler key estimation
  3. _extract_metadata   — mutagen: title/artist from embedded tags (best-effort)

allin1 requires a file path, so audio is written to a NamedTemporaryFile inside
a dedicated temp directory (demix/ and spec/ caches go there too) and the whole
directory is removed in a finally block.

Apple Silicon note: allin1 is dispatched with device='mps' when available. The
natten and madmom compatibility patches live in the .venv — see run.sh for the
DYLD_LIBRARY_PATH required by torchcodec.
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile

import numpy as np

from core.config import CONSTANTS, ModelParams
from core.exceptions import ModelInferenceError
from core.models import AudioBuffer, Section, StructureResult

from ._allin1 import _extract_allin1_results, _patch_allin1_metrical
from ._pure import _merge_consecutive_sections, _parse_sections

try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn


# Krumhansl-Schmuckler key profiles (Krumhansl, 1990).
# Defined at module level so they are allocated once, not on every call.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

# ID3 / Vorbis / MP4 tag keys mapped to our metadata field names.
_TAG_MAP: dict[str, str] = {
    "TIT2": "title",    # ID3 (MP3)
    "TPE1": "artist",
    "title": "title",   # Vorbis (FLAC, OGG)
    "artist": "artist",
    "\xa9nam": "title", # MP4/AAC
    "\xa9ART": "artist",
}


class Analysis:
    """
    Detects tempo, musical key, structural sections, beat grid, and metadata.

    Implements: StructureAnalyzer protocol (core/protocols.py)

    Constructor injection: pass a ModelParams instance to override device
    selection or the demucs checkpoint, e.g. in tests or a paid tier.

    Usage:
        service = Analysis()
        result  = service.analyze(audio_buffer)
        print(result.bpm, result.key, result.sections)
    """

    def __init__(self, params: ModelParams | None = None) -> None:
        self._params = params or ModelParams()

    # ------------------------------------------------------------------
    # Public interface (StructureAnalyzer protocol)
    # ------------------------------------------------------------------

    def analyze(self, audio: AudioBuffer) -> StructureResult:
        """
        Run full structure analysis on an AudioBuffer.

        Args:
            audio: In-memory audio buffer from Ingestion.

        Returns:
            StructureResult with bpm, key, sections, beats, and metadata.

        Raises:
            ModelInferenceError: if allin1 or librosa raises an unrecoverable
                                 error. Metadata extraction never raises —
                                 it degrades to empty strings.
        """
        raw = audio.raw
        metadata = self._extract_metadata(raw)   # best-effort, never raises
        bpm, sections, beats = self._analyse_structure(raw)
        key = self._detect_key(raw)

        return StructureResult(
            bpm=bpm,
            key=key,
            sections=sections,
            beats=beats,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private: allin1 structure analysis  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_structure(
        self, raw: bytes
    ) -> tuple[float | str, list[Section], list[float]]:
        """
        Write audio to a temp directory, run allin1.analyze(), clean up.

        Returns:
            (bpm, sections, beats) — bpm is a float on success, a str on error.

        Raises:
            ModelInferenceError: on allin1 import failure or analysis error.
        """
        try:
            import allin1
        except ImportError as exc:
            raise ModelInferenceError(
                "allin1 is not installed.",
                context={"suggestion": "pip install allin1", "original_error": str(exc)},
            ) from exc

        tmp_dir: str | None = None
        try:
            import torch

            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, "audio.wav")
            with open(tmp_path, "wb") as fh:
                fh.write(raw)

            device = self._resolve_device()
            _patch_allin1_metrical()
            result = allin1.analyze(
                tmp_path,
                model=self._params.allin1_model,
                device=device,
                demix_dir=os.path.join(tmp_dir, "demix"),
                spec_dir=os.path.join(tmp_dir, "spec"),
                # multiprocess=True (allin1 default) spawns worker processes via
                # Python's 'spawn' start method on macOS. Each worker re-executes
                # app.py as __main__ to initialise itself, which crashes because
                # there is no Streamlit session state in the subprocess.
                multiprocess=False,
            )

            bpm, sections, beats = _extract_allin1_results(result, raw)
            return bpm, sections, beats

        except ModelInferenceError:
            raise
        except Exception as exc:
            # MPS can leave dirty state between runs on Apple Silicon.
            # If the chosen device was MPS, clear the cache and retry once on CPU
            # before giving up — this fixes the intermittent allin1 failure pattern.
            if device == "mps":
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass
                try:
                    result = allin1.analyze(
                        tmp_path,
                        model=self._params.allin1_model,
                        device="cpu",
                        demix_dir=os.path.join(tmp_dir, "demix"),
                        spec_dir=os.path.join(tmp_dir, "spec"),
                        multiprocess=False,
                    )
                    bpm, sections, beats = _extract_allin1_results(result, raw)
                    return bpm, sections, beats
                except Exception as cpu_exc:
                    raise ModelInferenceError(
                        "allin1 analysis failed.",
                        context={"original_error": str(exc), "cpu_retry_error": str(cpu_exc)},
                    ) from cpu_exc
            raise ModelInferenceError(
                "allin1 analysis failed.",
                context={"original_error": str(exc)},
            ) from exc

        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Private: key detection via librosa  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _detect_key(self, raw: bytes) -> str:
        """
        Estimate musical key using Krumhansl-Schmuckler chroma correlation.

        Returns:
            A string like "C# Major" or "A Minor", or "Unknown" on failure.

        Raises:
            ModelInferenceError: on librosa load or feature extraction failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw),
                sr=CONSTANTS.SAMPLE_RATE,
                mono=True,
            )
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            chroma_mean = chroma.mean(axis=1)

            best_score = -np.inf
            best_key = "Unknown"

            for tonic in range(len(_NOTE_NAMES)):
                score_major = float(
                    np.corrcoef(chroma_mean, np.roll(_MAJOR_PROFILE, tonic))[0, 1]
                )
                score_minor = float(
                    np.corrcoef(chroma_mean, np.roll(_MINOR_PROFILE, tonic))[0, 1]
                )
                if score_major > best_score:
                    best_score = score_major
                    best_key = f"{_NOTE_NAMES[tonic]} Major"
                if score_minor > best_score:
                    best_score = score_minor
                    best_key = f"{_NOTE_NAMES[tonic]} Minor"

            return best_key

        except Exception as exc:
            raise ModelInferenceError(
                "Key detection failed.",
                context={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Private: metadata extraction via mutagen  (CPU, best-effort)
    # ------------------------------------------------------------------

    def _extract_metadata(self, raw: bytes) -> dict[str, str]:
        """
        Extract title and artist from embedded audio tags.

        Never raises — missing or unreadable tags degrade to empty strings.
        This is intentional: metadata is supplementary information. A track
        with no tags should still be analysed.
        """
        try:
            from mutagen import File as MutagenFile

            audio_file = MutagenFile(io.BytesIO(raw))
            if audio_file is None:
                return {"title": "", "artist": ""}

            metadata: dict[str, str] = {"title": "", "artist": ""}
            for tag_key, field in _TAG_MAP.items():
                if tag_key not in audio_file:
                    continue
                value = audio_file[tag_key]
                if hasattr(value, "text"):
                    metadata[field] = str(value.text[0])
                elif isinstance(value, list) and value:
                    metadata[field] = str(value[0])
                if metadata["title"] and metadata["artist"]:
                    break

            return metadata

        except Exception:  # noqa: BLE001 — metadata is always best-effort
            return {"title": "", "artist": ""}

    # ------------------------------------------------------------------
    # Private: device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        """
        Return the best available torch device string.

        Respects ModelParams.demucs_device when explicitly set (e.g. in tests).
        Auto-detection order: MPS (Apple Silicon) → CPU.
        CUDA is handled by the @spaces.GPU decorator on HF Spaces.
        """
        if self._params.demucs_device is not None:
            return self._params.demucs_device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


