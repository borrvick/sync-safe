"""
services/loudness/_pure.py
Pure loudness measurement and dialogue-readiness functions — no I/O, no Streamlit.
"""
from __future__ import annotations

import logging

import librosa
import numpy as np
import pyloudnorm as pyln

from core.config import CONSTANTS

_log = logging.getLogger(__name__)


def _measure_loudness(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    """
    Compute integrated LUFS, true peak (dBFS), and loudness range (LU).

    Uses pyloudnorm BS.1770-4 meter.  pyloudnorm requires float64 and
    a 2-D array for multi-channel; for mono we pass a (N, 1) array.

    Returns:
        (integrated_lufs, true_peak_dbfs, loudness_range_lu)
        All values are rounded to 1 decimal place.
    """
    data = y.astype(np.float64).reshape(-1, 1)  # (N, 1) mono

    meter    = pyln.Meter(sr)  # BS.1770-4
    loudness = meter.integrated_loudness(data)

    true_peak_linear = float(np.max(np.abs(y)))
    true_peak_dbfs   = float(20 * np.log10(true_peak_linear + 1e-12))

    try:
        lra = float(meter.loudness_range(data))
    except ValueError as exc:
        _log.debug("LRA not available for this track: %s", exc)
        lra = 0.0

    return round(float(loudness), 1), round(true_peak_dbfs, 1), round(lra, 1)


def _dialogue_score(y: np.ndarray, sr: int) -> float:
    """
    Compute a 0.0–1.0 dialogue-readiness score.

    Score = fraction of total spectral energy that falls OUTSIDE the
    300–3000 Hz voiceover competition band.  Pure function — no I/O.
    """
    stft  = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    dialogue_mask   = (freqs >= 300) & (freqs <= 3000)
    total_energy    = float(stft.mean())
    dialogue_energy = float(stft[dialogue_mask].mean())

    if total_energy < 1e-10:
        return 0.5  # silent track — neutral score

    competition_ratio = dialogue_energy / total_energy
    return round(float(max(0.0, min(1.0, 1.0 - competition_ratio))), 3)


def _classify_dialogue(score: float) -> str:
    """
    Map a dialogue score to a human-readable label using CONSTANTS thresholds.

    Pure function — no I/O.
    """
    if score >= CONSTANTS.DIALOGUE_READY_HIGH:
        return "Dialogue-Ready"
    if score >= CONSTANTS.DIALOGUE_READY_LOW:
        return "Mixed"
    return "Dialogue-Heavy"
