"""
services/forensics/detectors/rhythm.py
RhythmAnalyzer — IBI groove, loop cross-correlation, loop autocorrelation,
and sub-beat grid deviation.

All signals are computed from the pre-decoded standard-SR array in _AudioData.
Beat tracking is shared between IBI and spectral slop to avoid redundant work.
"""
from __future__ import annotations

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError

from .._audio import _AudioData
from .._bundle import _SignalBundle
from .._pure import (
    _check_spectral_slop,
    _cross_correlate,
    _spectral_fingerprint,
    compute_loop_autocorrelation_score,
    compute_subbeat_grid_deviation,
)


class RhythmAnalyzer:
    """
    Computes rhythm and loop structure signals.

    Signals populated:
        ibi_variance, spectral_slop, loop_score,
        loop_autocorr_score, subbeat_grid_deviation
    """

    def process(self, data: _AudioData, bundle: _SignalBundle) -> None:
        import librosa
        # Beat tracking is expensive — run once and share between _groove and _loops.
        tempo, beat_frames = librosa.beat.beat_track(y=data.y, sr=data.sr)
        bundle.ibi_variance, bundle.spectral_slop = self._groove(data.y, data.sr, beat_frames)
        bundle.loop_score                          = self._loops(data.y, data.sr, tempo)
        bundle.loop_autocorr_score                 = compute_loop_autocorrelation_score(
            data.y, data.sr,
            CONSTANTS.LOOP_PEAK_COUNT_THRESHOLD,
            CONSTANTS.LOOP_PEAK_SPACING_MAX,
        )
        bundle.subbeat_grid_deviation = compute_subbeat_grid_deviation(data.y, data.sr)

    # ------------------------------------------------------------------
    # IBI + spectral slop (shares beat-tracker result)
    # ------------------------------------------------------------------

    def _groove(self, y: np.ndarray, sr: int, beat_frames: np.ndarray) -> tuple[float, float]:
        """
        Compute inter-beat interval variance and spectral slop ratio.

        Args:
            beat_frames: Pre-computed beat frames from process() — avoids a
                         second beat_track call.

        Returns:
            (ibi_variance, spectral_slop)
            ibi_variance = -1.0 when the track is too short to analyse.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        try:
            import librosa

            beat_times_ms   = librosa.frames_to_time(beat_frames, sr=sr) * 1000.0

            if len(beat_times_ms) < 2:
                return -1.0, 0.0

            ibi          = np.diff(beat_times_ms)
            ibi_variance = float(np.var(ibi))
            spectral_slop = _check_spectral_slop(
                y, sr,
                CONSTANTS.SPECTRAL_SLOP_HZ,
                CONSTANTS.SPECTRAL_SLOP_RATIO,
            )
            return ibi_variance, spectral_slop

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Groove analysis failed.",
                context={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Loop cross-correlation
    # ------------------------------------------------------------------

    def _loops(self, y: np.ndarray, sr: int, tempo: np.ndarray) -> float:
        """
        Cross-correlate 4-bar spectral fingerprints across the track.

        Args:
            tempo: Pre-computed tempo from process() — avoids a second
                   beat_track call.

        Returns:
            Maximum cosine similarity (0.0–1.0). 0.0 when track is too short.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        try:
            bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

            if not (CONSTANTS.LOOP_BPM_MIN <= bpm <= CONSTANTS.LOOP_BPM_MAX):
                return 0.0

            segment_len = int((60.0 / bpm) * CONSTANTS.BEATS_PER_WINDOW * sr)
            if segment_len > len(y):
                return 0.0

            segments     = [y[i : i + segment_len] for i in range(0, len(y) - segment_len, segment_len)]
            if len(segments) < 2:
                return 0.0

            fingerprints = [_spectral_fingerprint(s) for s in segments]

            max_score = 0.0
            for i in range(len(fingerprints)):
                for j in range(i + 1, len(fingerprints)):
                    score = _cross_correlate(fingerprints[i], fingerprints[j])
                    if score > max_score:
                        max_score = score

            return max_score

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Loop detection failed.",
                context={"original_error": str(exc)},
            ) from exc
