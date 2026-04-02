"""
services/forensics/analyzers/monitoring.py
MonitoringAnalyzer — pending signals awaiting cross-dataset calibration.

Signals in this group return their sentinel -1.0 / 0.0 values when gated
(compressed source, insufficient SR, etc.) and are excluded from verdict
logic until their CONSTANTS thresholds are set above 0.0.

Signals:
  - kurtosis_variability        (mel-band codec artifacts — pending FMC calibration)
  - decoder_peak_score          (transposed-convolution comb — pending)
  - ultrasonic_noise_ratio      (diffusion residue 20–22 kHz — upload-only)
  - infrasonic_energy_ratio     (sub-20 Hz math drift — pending calibration)
  - phase_coherence_differential (LF vs HF stereo phase — upload-only, stereo)
"""
from __future__ import annotations

import io

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError

from .._audio import _AudioData
from .._bundle import _SignalBundle
from .._pure import (
    _SYNTHID_LOAD_SR,
    compute_decoder_peak_score,
    compute_infrasonic_energy_ratio,
    compute_kurtosis_variability,
    compute_phase_coherence_differential,
    compute_ultrasonic_noise_ratio,
)


class MonitoringAnalyzer:
    """
    Staging ground for pending signals.

    Once a signal is calibrated and moved into the verdict logic, move its
    computation call to the appropriate domain analyzer and remove it here.

    Signals populated:
        kurtosis_variability, decoder_peak_score, ultrasonic_noise_ratio,
        infrasonic_energy_ratio, phase_coherence_differential
    """

    def process(self, data: _AudioData, bundle: _SignalBundle) -> None:
        bundle.kurtosis_variability   = compute_kurtosis_variability(
            data.y, data.sr, CONSTANTS.KURTOSIS_N_MELS
        )
        bundle.decoder_peak_score     = compute_decoder_peak_score(
            *self._load_44k(data.raw),
            CONSTANTS.DECODER_PEAK_WINDOW_HZ,
            CONSTANTS.DECODER_PEAK_PROMINENCE_DB,
            CONSTANTS.DECODER_PEAK_REGULARITY_MAX,
            CONSTANTS.DECODER_PEAK_MIN_COUNT,
        )
        bundle.ultrasonic_noise_ratio      = self._ultrasonic(data.raw, bundle.compressed_source)
        bundle.infrasonic_energy_ratio     = compute_infrasonic_energy_ratio(data.y, data.sr)
        bundle.phase_coherence_differential = self._phase_coherence(data.raw, bundle.compressed_source)

    # ------------------------------------------------------------------
    # Helpers that need non-standard loads
    # ------------------------------------------------------------------

    def _load_44k(self, raw: bytes) -> tuple[np.ndarray, int]:
        """Load audio at 44.1 kHz mono for decoder peak / ultrasonic analysis."""
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(raw), sr=_SYNTHID_LOAD_SR, mono=True)
            return audio, sr
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "MonitoringAnalyzer: 44.1 kHz load failed.",
                context={"original_error": str(exc)},
            ) from exc

    def _ultrasonic(self, raw: bytes, compressed_source: bool) -> float:
        """
        Ultrasonic noise plateau (20–22 kHz).

        Gated on upload-only and native SR ≥ 40 kHz.
        Resampling a 16 kHz file cannot create real content above its 8 kHz Nyquist.
        """
        if compressed_source:
            return -1.0

        try:
            import librosa

            _, native_sr = librosa.load(io.BytesIO(raw), sr=None, mono=True, duration=1.0)
            if native_sr < 40_000:
                return -1.0

            audio, sr = librosa.load(io.BytesIO(raw), sr=_SYNTHID_LOAD_SR, mono=True)
            return compute_ultrasonic_noise_ratio(audio, sr)

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Ultrasonic noise analysis failed.",
                context={"original_error": str(exc)},
            ) from exc

    def _phase_coherence(self, raw: bytes, compressed_source: bool) -> float:
        """
        Inter-channel LF vs HF phase coherence differential.

        Gated on stereo upload-only sources.  Stereo compression artifacts
        (YouTube AAC) decorrelate HF identically to the AI pattern — running
        on compressed sources would produce false positives.
        """
        if compressed_source:
            return -1.0

        try:
            import librosa

            # mono=False returns (channels, samples) for stereo, or (samples,) for mono.
            audio, sr = librosa.load(io.BytesIO(raw), sr=_SYNTHID_LOAD_SR, mono=False)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Phase coherence analysis: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        if audio.ndim < 2 or audio.shape[0] < 2:
            return -1.0

        return compute_phase_coherence_differential(audio[0], audio[1], sr)
