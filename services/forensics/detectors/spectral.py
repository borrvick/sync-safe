"""
services/forensics/detectors/spectral.py
SpectralAnalyzer — frequency-domain AI signals computed from the standard-SR array.

Signals:
  - centroid_instability_score  (formant drift within sustained notes)
  - spectral_centroid_mean      (low-mid energy concentration)
  - spectral_flatness_var       (uniform synthesis texture)
  - noise_floor_ratio           (digital silence between notes)
  - self_similarity_entropy     (blocky AI attention structure)
"""
from __future__ import annotations

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError

from .._audio import _AudioData
from .._bundle import _SignalBundle
from .._pure import (
    compute_centroid_instability_score,
    compute_noise_floor_ratio,
    compute_self_similarity_entropy,
    compute_spectral_flatness_variance,
)


class SpectralAnalyzer:
    """
    Computes spectral texture and structure signals.

    Signals populated:
        centroid_instability_score, spectral_centroid_mean,
        spectral_flatness_var, noise_floor_ratio, self_similarity_entropy
    """

    def process(self, data: _AudioData, bundle: _SignalBundle) -> None:
        bundle.centroid_instability_score = compute_centroid_instability_score(
            data.y, data.sr,
            CONSTANTS.CENTROID_TOP_DB,
            CONSTANTS.CENTROID_MIN_INTERVAL_S,
        )
        bundle.spectral_centroid_mean     = self._centroid_mean(data.y, data.sr)
        bundle.spectral_flatness_var      = compute_spectral_flatness_variance(data.y)
        bundle.noise_floor_ratio          = compute_noise_floor_ratio(data.y)
        bundle.self_similarity_entropy    = compute_self_similarity_entropy(data.y, data.sr)

    # ------------------------------------------------------------------

    def _centroid_mean(self, y: np.ndarray, sr: int) -> float:
        """
        Compute the mean spectral centroid of the track in Hz.

        Returns:
            Mean spectral centroid > 0, or -1.0 on failure.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        try:
            import librosa

            centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            return float(np.mean(centroid_frames))

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Spectral centroid mean: computation failed.",
                context={"original_error": str(exc)},
            ) from exc
