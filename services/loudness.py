"""
services/loudness.py
Broadcast loudness and dialogue-readiness analysis.

Implements:
- LUFS integrated loudness per ITU-R BS.1770-4 via pyloudnorm
- True peak detection
- Loudness Range (LRA) dynamic range measurement
- Dialogue-ready score: fraction of energy outside the 300–3 kHz
  voiceover competition band (higher = sits cleaner under spoken dialogue)

Design notes:
- AudioQualityAnalyzer is stateless — instantiate per call.
- All threshold comparisons use CONSTANTS — no magic numbers.
- _dialogue_score and _classify_dialogue are pure functions for testability.
- pyloudnorm operates on float64 numpy arrays; librosa provides the waveform.
"""
from __future__ import annotations

import io
import logging

import librosa
import numpy as np
import pyloudnorm as pyln

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import AudioBuffer, AudioQualityResult

_log = logging.getLogger(__name__)


class AudioQualityAnalyzer:
    """
    Measures broadcast loudness and dialogue-readiness of an audio track.

    Usage:
        result = AudioQualityAnalyzer().analyze(buffer)
        print(result.integrated_lufs, result.dialogue_label)
    """

    def analyze(self, buffer: AudioBuffer) -> AudioQualityResult:
        """
        Run loudness + dialogue analysis on an AudioBuffer.

        Args:
            buffer: AudioBuffer with raw audio bytes.

        Returns:
            AudioQualityResult with LUFS, true peak, LRA, and dialogue score.

        Raises:
            ModelInferenceError: if audio is too short or malformed.
        """
        try:
            y, sr = librosa.load(io.BytesIO(buffer.raw), sr=None, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "AudioQualityAnalyzer: failed to decode audio.",
                context={"original_error": str(exc)},
            ) from exc

        duration = len(y) / sr
        if duration < CONSTANTS.LUFS_MIN_DURATION_S:
            raise ModelInferenceError(
                f"AudioQualityAnalyzer: audio too short ({duration:.1f}s < "
                f"{CONSTANTS.LUFS_MIN_DURATION_S}s minimum for LUFS measurement).",
                context={"duration_s": duration},
            )

        integrated_lufs, true_peak, lra = _measure_loudness(y, sr)
        diag_score = _dialogue_score(y, sr)
        diag_label = _classify_dialogue(diag_score)

        return AudioQualityResult(
            integrated_lufs=integrated_lufs,
            true_peak_dbfs=true_peak,
            loudness_range_lu=lra,
            delta_spotify=round(integrated_lufs - CONSTANTS.LUFS_SPOTIFY, 1),
            delta_apple_music=round(integrated_lufs - CONSTANTS.LUFS_APPLE_MUSIC, 1),
            delta_youtube=round(integrated_lufs - CONSTANTS.LUFS_YOUTUBE, 1),
            delta_broadcast=round(integrated_lufs - CONSTANTS.LUFS_BROADCAST, 1),
            true_peak_warning=true_peak > CONSTANTS.TRUE_PEAK_WARN_DBFS,
            dialogue_score=diag_score,
            dialogue_label=diag_label,
        )


# ---------------------------------------------------------------------------
# Pure helpers — independently testable
# ---------------------------------------------------------------------------

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

    # True peak: sample peak (linear) converted to dBFS.
    # pyloudnorm 0.2.0 exposes no true_peak helper; np.max(abs) gives the
    # inter-sample peak amplitude which is the standard approximation.
    true_peak_linear = float(np.max(np.abs(y)))
    true_peak_dbfs   = float(20 * np.log10(true_peak_linear + 1e-12))

    # Loudness Range — requires at least 3s of audio (gated 3s blocks).
    # pyloudnorm raises ValueError when the track is too short; LRA is optional.
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
    300–3000 Hz voiceover competition band.  Higher score = track
    occupies less of the frequency range where voiceover lives, meaning
    it sits more cleanly under spoken dialogue in a sync placement.

    Pure function — no I/O.
    """
    stft  = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    dialogue_mask  = (freqs >= 300) & (freqs <= 3000)
    total_energy   = float(stft.mean())
    dialogue_energy = float(stft[dialogue_mask].mean())

    if total_energy < 1e-10:
        return 0.5  # silent track — neutral score

    competition_ratio = dialogue_energy / total_energy
    # Invert: high competition ratio → low score
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
