"""
services/stems/_pure.py
Pure stereo analysis functions — no I/O, no librosa load, deterministic output.

Used by _orchestrator.validate_stem after audio is loaded.
"""
from __future__ import annotations

import logging

import numpy as np

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
