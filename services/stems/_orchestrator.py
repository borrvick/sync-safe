"""
services/stems/_orchestrator.py
Stem and alternate mix validator — mono compatibility and phase alignment checks.

I/O entry point: validate_stem(audio_bytes) → StemValidationResult

Delegates all signal computation to pure functions in _pure.py.
"""
from __future__ import annotations

import io
from typing import Optional

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import StemValidationResult

from ._pure import compute_cancellation_db, compute_mid_side_ratio, compute_phase_correlation


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
