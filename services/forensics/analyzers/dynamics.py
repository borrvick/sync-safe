"""
services/forensics/analyzers/dynamics.py
DynamicsAnalyzer — performance dynamics and vocal-path signals.

Signals:
  - onset_strength_cv      (uniform hit strength = no dynamics = AI)
  - harmonic_ratio_score   (unnaturally clean harmonics in sustained notes)
  - pitch_quantization_score (perfect 12-TET alignment = AI)
  - plr_std                (frozen loudness density)
  - is_vocal               (pyin vocal detection — routes vocal scoring path)
  - voiced_noise_floor     (clean between-harmonic space in vocal frames)

Vocal presence is computed first so the voiced_noise_floor gate can read
bundle.is_vocal and bundle.compressed_source before deciding whether to run.
"""
from __future__ import annotations

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError

from .._audio import _AudioData
from .._bundle import _SignalBundle
from .._pure import (
    _detect_is_vocal,
    compute_harmonic_ratio_score,
    compute_onset_strength_cv,
    compute_pitch_quantization_score,
    compute_plr_std,
    compute_voiced_noise_floor,
)


class DynamicsAnalyzer:
    """
    Computes performance-dynamics and vocal-path signals.

    Signals populated:
        onset_strength_cv, harmonic_ratio_score, pitch_quantization_score,
        plr_std, is_vocal, voiced_noise_floor
    """

    def process(self, data: _AudioData, bundle: _SignalBundle) -> None:
        bundle.onset_strength_cv       = compute_onset_strength_cv(data.y, data.sr)
        bundle.harmonic_ratio_score    = compute_harmonic_ratio_score(
            data.y, data.sr,
            CONSTANTS.CENTROID_TOP_DB,
            CONSTANTS.CENTROID_MIN_INTERVAL_S,
        )
        bundle.pitch_quantization_score = compute_pitch_quantization_score(data.y, data.sr)
        bundle.plr_std                  = compute_plr_std(data.y, data.sr)

        # Vocal presence must be set before voiced_noise_floor is computed.
        bundle.is_vocal = _detect_is_vocal(data.y, data.sr)

        # voiced_noise_floor is gated on upload-only + vocal tracks.
        # Compressed sources (YouTube AAC/Opus) add codec noise that floods the
        # between-harmonic band, making AI and human indistinguishable.
        if bundle.is_vocal and not bundle.compressed_source:
            bundle.voiced_noise_floor = compute_voiced_noise_floor(data.y, data.sr)
        else:
            bundle.voiced_noise_floor = -1.0
