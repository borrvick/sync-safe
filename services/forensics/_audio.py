"""
services/forensics/_audio.py
Audio carrier dataclass — decoded once, shared across all analyzers.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import AudioBuffer


@dataclass(frozen=True)
class _AudioData:
    """
    Decoded audio carrier.  Loaded once per analyze() call.

    Attributes:
        y:   Mono waveform at CONSTANTS.SAMPLE_RATE.
        sr:  Sample rate (== CONSTANTS.SAMPLE_RATE).
        raw: Original bytes — retained for analyzers that need a different
             sample rate or channel count (SynthID at 44.1 kHz, phase
             coherence stereo).  Treated as read-only.
    """
    y: np.ndarray
    sr: int
    raw: bytes


def load_audio(audio: AudioBuffer) -> _AudioData:
    """
    Decode *audio* into an _AudioData carrier.

    Loads mono at CONSTANTS.SAMPLE_RATE.  All standard-path analyzers use
    the pre-decoded arrays; special-case analyzers (SynthID, phase coherence,
    ultrasonic) re-decode from raw bytes at their required sample rate.

    Raises:
        ModelInferenceError: if librosa cannot decode the buffer.
    """
    try:
        import librosa

        y, sr = librosa.load(
            io.BytesIO(audio.raw),
            sr=CONSTANTS.SAMPLE_RATE,
            mono=True,
        )
        return _AudioData(y=y, sr=sr, raw=audio.raw)

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Audio decode failed.",
            context={"original_error": str(exc)},
        ) from exc
