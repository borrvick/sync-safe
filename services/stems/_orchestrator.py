"""
services/stem_validator.py
Stem and alternate mix validator — mono compatibility and phase alignment checks.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import StemValidationResult

_log = logging.getLogger(__name__)


def compute_phase_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation between L/R channels. Returns -1.0 on error."""
    try:
        if len(left) == 0 or len(left) != len(right):
            return -1.0
        corr_matrix = np.corrcoef(left, right)
        return float(corr_matrix[0, 1])
    except Exception as exc:
        _log.debug("phase_correlation failed: %s", exc)
        return -1.0


def compute_cancellation_db(left: np.ndarray, right: np.ndarray) -> float:
    """dB difference between mono sum and stereo RMS. Negative = cancellation."""
    try:
        stereo_rms = float(np.sqrt(np.mean(0.5 * (left ** 2 + right ** 2))))
        mono_rms   = float(np.sqrt(np.mean((0.5 * (left + right)) ** 2)))
        if stereo_rms <= 0.0:
            return 0.0
        return float(20.0 * np.log10(max(mono_rms, 1e-12) / stereo_rms))
    except Exception as exc:
        _log.debug("cancellation_db failed: %s", exc)
        return 0.0


def compute_mid_side_ratio(left: np.ndarray, right: np.ndarray) -> float:
    """Side/Mid energy ratio. Returns -1.0 for mono or on error."""
    try:
        mid  = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_rms  = float(np.sqrt(np.mean(mid ** 2)))
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        if mid_rms <= 0.0:
            return -1.0
        return round(float(side_rms / mid_rms), 4)
    except Exception as exc:
        _log.debug("mid_side_ratio failed: %s", exc)
        return -1.0


def validate_stem(audio_bytes: bytes, sr: Optional[int] = None) -> StemValidationResult:
    """
    Load stereo audio and compute mono compatibility metrics.

    Args:
        audio_bytes: Raw audio file bytes.
        sr:          Target sample rate (None = preserve original).

    Returns:
        StemValidationResult with phase_correlation, cancellation_db, mid_side_ratio, flags.

    Raises:
        ModelInferenceError: on unexpected load or computation failure.
    """
    try:
        import librosa
        y, actual_sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=False)
    except Exception as exc:
        raise ModelInferenceError(
            "stem_validator: audio load failed",
            context={"error": str(exc)},
        ) from exc

    try:
        # Mono input: librosa returns shape (n,) when mono=False and audio is mono
        if y.ndim == 1:
            return StemValidationResult(
                mono_compatible=True,
                phase_correlation=1.0,
                cancellation_db=0.0,
                mid_side_ratio=0.0,
                flags=["Track is mono — no stereo phase analysis needed"],
            )

        left, right = y[0], y[1]

        phase_corr      = compute_phase_correlation(left, right)
        cancellation_db = compute_cancellation_db(left, right)
        ms_ratio        = compute_mid_side_ratio(left, right)

        flags: list[str] = []
        if phase_corr < CONSTANTS.PHASE_CORRELATION_FAIL:
            flags.append(
                f"Anti-phase channels detected — mono sum will cancel significantly "
                f"(correlation {phase_corr:.3f})"
            )
        elif phase_corr < CONSTANTS.PHASE_CORRELATION_WARN:
            flags.append(
                f"Phase correlation low — possible stereo imaging issue "
                f"(correlation {phase_corr:.3f})"
            )

        if cancellation_db < CONSTANTS.MONO_CANCELLATION_DB_FAIL:
            flags.append(
                f"Significant mono cancellation: {cancellation_db:.1f} dB loss"
            )
        elif cancellation_db < CONSTANTS.MONO_CANCELLATION_DB_WARN:
            flags.append(
                f"Moderate mono cancellation: {cancellation_db:.1f} dB loss"
            )

        return StemValidationResult(
            mono_compatible=cancellation_db >= CONSTANTS.MONO_CANCELLATION_DB_WARN,
            phase_correlation=round(phase_corr, 4),
            cancellation_db=round(cancellation_db, 2),
            mid_side_ratio=ms_ratio,
            flags=flags,
        )

    except ModelInferenceError:
        raise
    except Exception as exc:
        raise ModelInferenceError(
            "stem_validator: computation failed",
            context={"error": str(exc)},
        ) from exc
