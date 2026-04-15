"""
services/loudness/_pure.py
Pure loudness measurement and dialogue-readiness functions — no I/O, no Streamlit.
"""
from __future__ import annotations

import logging
import math

import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly

from core.config import CONSTANTS
from core.models import Section

_log = logging.getLogger(__name__)


def _measure_true_peak(y: np.ndarray) -> float:
    """
    Inter-sample true peak via oversampling (ITU-R BS.1770-4).

    Upsamples by TRUE_PEAK_OVERSAMPLE (4×) using a polyphase anti-aliasing
    filter, then finds the maximum absolute value and converts to dBFS.
    Inter-sample peak is always ≥ sample peak by definition.

    Pure function — no I/O.
    """
    factor = CONSTANTS.TRUE_PEAK_OVERSAMPLE
    upsampled = resample_poly(y.astype(np.float64), up=factor, down=1)
    peak_linear = float(np.max(np.abs(upsampled)))
    return float(20.0 * np.log10(max(peak_linear, 1e-9)))


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

    true_peak_dbfs = _measure_true_peak(y)

    try:
        lra = float(meter.loudness_range(data))
    except ValueError as exc:
        _log.debug("LRA not available for this track: %s", exc)
        lra = 0.0

    return round(float(loudness), 1), round(true_peak_dbfs, 1), round(lra, 1)


def _measure_section_loudness(
    y: np.ndarray,
    sr: int,
    sections: list[Section],
) -> list[dict[str, str | float | bool]]:
    """
    Compute integrated LUFS and LRA for each structural section (#96).

    Sections shorter than CONSTANTS.DIALOGUE_MIN_SECTION_DUR_S are skipped
    (pyloudnorm's gating algorithm requires several 400ms blocks to converge).
    Skipped sections carry nan values so the UI can render "—" instead of crashing.
    The loudest valid section is annotated with is_peak=True.

    Pure function — no I/O.
    """
    meter   = pyln.Meter(sr)
    results: list[dict[str, str | float | bool]] = []

    for sec in sections:
        start_sample = int(sec.start * sr)
        end_sample   = min(int(sec.end * sr), len(y))
        slice_       = y[start_sample:end_sample]
        duration     = len(slice_) / sr

        if duration < CONSTANTS.DIALOGUE_MIN_SECTION_DUR_S:
            results.append({
                "label":            sec.label,
                "start_s":          round(sec.start, 1),
                "end_s":            round(sec.end, 1),
                "integrated_lufs":  float("nan"),
                "lra_lu":           float("nan"),
            })
            continue

        data = slice_.astype(np.float64).reshape(-1, 1)
        try:
            lufs = float(meter.integrated_loudness(data))
        except ValueError:
            lufs = float("nan")

        try:
            lra = float(meter.loudness_range(data))
        except ValueError:
            lra = float("nan")

        results.append({
            "label":            sec.label,
            "start_s":          round(sec.start, 1),
            "end_s":            round(sec.end, 1),
            "integrated_lufs":  round(lufs, 1) if not math.isnan(lufs) else float("nan"),
            "lra_lu":           round(lra, 1)  if not math.isnan(lra)  else float("nan"),
        })

    # Mark the loudest valid section
    valid = [r for r in results if not math.isnan(float(r["integrated_lufs"]))]
    if valid:
        peak = max(valid, key=lambda r: float(r["integrated_lufs"]))
        peak["is_peak"] = True

    return results


def _dialogue_score(y: np.ndarray, sr: int) -> float:
    """
    Compute a 0.0–1.0 dialogue-readiness score.

    Score = fraction of total spectral energy that falls OUTSIDE the
    300–3000 Hz voiceover competition band.  Pure function — no I/O.
    """
    stft  = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    dialogue_mask   = (freqs >= 300) & (freqs <= 3000)
    total_energy    = float(stft.sum())
    dialogue_energy = float(stft[dialogue_mask].sum())

    if total_energy < 1e-10:
        return 0.5  # silent track — neutral score

    competition_ratio = dialogue_energy / total_energy
    return round(float(max(0.0, min(1.0, 1.0 - competition_ratio))), 3)


def _classify_loudness(
    integrated_lufs: float,
    true_peak_warning: bool,
    delta_broadcast: float,
) -> str:
    """
    Return a top-line loudness verdict string (#95).

    Priority order: clipping risk > broadcast-ready > streaming-hot > needs mastering > streaming-ready.
    Pure function — no I/O.
    """
    if true_peak_warning:
        return "Clipping risk"
    if abs(delta_broadcast) <= CONSTANTS.LOUDNESS_BROADCAST_DELTA_MAX:
        return "Broadcast-ready"
    if integrated_lufs > CONSTANTS.LOUDNESS_STREAMING_HOT_MIN:
        return "Streaming-hot"
    if integrated_lufs < CONSTANTS.LOUDNESS_NEEDS_MASTERING_MAX:
        return "Needs mastering"
    return "Streaming-ready"


def _compute_vo_headroom(dialogue_score: float, max_db: float) -> float:
    """
    Estimate available VO headroom in dB from the dialogue-readiness score (#92).

    Maps a 0.0–1.0 dialogue score linearly onto a 0–max_db range.
    A fully dialogue-ready track (score=1.0) offers max_db of headroom; a
    dialogue-heavy track (score=0.0) offers none.

    Pure function — no I/O.
    """
    clamped = max(0.0, min(1.0, dialogue_score))
    return round(clamped * max_db, 1)


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
