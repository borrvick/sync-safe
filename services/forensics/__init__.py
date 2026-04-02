"""
services/forensics/__init__.py
Forensic AI-origin detection — implements the ForensicsAnalyzer protocol.

Public entry point: Forensics.analyze(audio) → ForensicsResult

Active detection signals:
  1.  C2PA manifest check            — cryptographic provenance (no librosa needed)
  2.  IBI groove analysis            — inter-beat interval variance via beat tracking
  3.  Loop detection (cross-corr)    — 4-bar spectral fingerprint cross-correlation
  4.  Loop autocorrelation           — onset-envelope periodicity independent of BPM
  5.  Spectral centroid instability  — within-note formant drift (AI vocoder pattern)
  6.  Harmonic ratio (HNR)           — unnaturally clean harmonics in sustained notes
  7.  SynthID-style scan             — phase coherence in 18–22 kHz band (44.1 kHz)
  8.  Spectral slop                  — anomalous HF energy above 16 kHz
  9.  Spectral centroid mean         — AI energy concentrated in low-mid range
  10. Noise floor ratio              — near-zero quiet-frame RMS (VST render, no room noise)
  11. Self-similarity entropy        — blocky AI attention structure in spectrogram
  12. Onset strength CV              — uniform hit strength = no dynamics = AI
  13. Spectral flatness variance     — uniform synthesis texture
  14. Subbeat grid deviation         — perfect sub-beat quantization = AI
  15. Pitch quantization score       — perfect 12-TET alignment = AI
  16. PLR flatness                   — frozen loudness density (pending calibration)
  17. Kurtosis variability           — mel-band codec artifacts (pending FMC calibration)
  18. Decoder peak score             — periodic deconvolution comb (pending)
  19. Ultrasonic noise ratio         — diffusion residue 20–22 kHz (upload-only, pending)
  20. Infrasonic energy ratio        — sub-20 Hz math drift (pending calibration)
  21. Phase coherence differential   — LF vs HF stereo phase (upload-only, pending)

Verdict tiers (four-tier system):
  "AI"           — hard evidence: C2PA born-AI assertion or high-confidence SynthID
  "Likely AI"    — probability score ≥ PROB_VERDICT_HYBRID (0.45); no embedded proof
  "Likely Not AI"— default; no significant AI indicators, or organic sampled production
  "Not AI"       — reserved; requires a verifiable born-human proof standard (none exists yet)
"""
from __future__ import annotations

import logging

from core.exceptions import ModelInferenceError
from core.models import AiSegment, AudioBuffer, ForensicsResult

from ._aggregators import (
    _build_flags,
    _build_forensic_notes,
    _classify,
    _compute_ai_probability,
    _compute_verdict,
    _synthid_confidence,
)
from core.config import CONSTANTS

from ._audio import _AudioData, load_audio
from ._bundle import _SignalBundle
from ._pure import (
    _check_spectral_slop,
    _classify_c2pa_origin,
    _cross_correlate,
    _spectral_fingerprint,
    compute_infrasonic_energy_ratio,
    compute_phase_coherence_differential,
    compute_plr_std,
    compute_ultrasonic_noise_ratio,
    compute_voiced_noise_floor,
    segment_ai_probabilities,
)
from .analyzers import (
    DynamicsAnalyzer,
    MetadataAnalyzer,
    MonitoringAnalyzer,
    RhythmAnalyzer,
    SpectralAnalyzer,
)

try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn


_log = logging.getLogger(__name__)


class Forensics:
    """
    Detects signals suggesting AI-generated or stock audio.

    Implements: ForensicsAnalyzer protocol (core/protocols.py)

    Usage:
        service = Forensics()
        result  = service.analyze(audio_buffer)
        print(result.verdict, result.flags)
    """

    def __init__(self) -> None:
        self._analyzers = [
            MetadataAnalyzer(),
            RhythmAnalyzer(),
            SpectralAnalyzer(),
            DynamicsAnalyzer(),
            MonitoringAnalyzer(),
        ]

    @spaces.GPU
    def analyze(self, audio: AudioBuffer) -> ForensicsResult:
        """
        Run all forensic checks and aggregate into a ForensicsResult.

        Args:
            audio: In-memory audio buffer from Ingestion.

        Returns:
            ForensicsResult with per-signal scores, human-readable flags,
            and an overall verdict.

        Raises:
            ModelInferenceError: if audio decoding fails unrecoverably.
            C2PA errors degrade gracefully — they never propagate.
        """
        data   = load_audio(audio)
        bundle = _SignalBundle(compressed_source=(audio.source == "youtube"))

        for analyzer in self._analyzers:
            try:
                analyzer.process(data, bundle)
            except ModelInferenceError as exc:
                _log.warning(
                    "forensics: %s skipped — %s",
                    type(analyzer).__name__,
                    exc,
                )

        ml_prob        = _classify(bundle)
        ai_probability = ml_prob if ml_prob is not None else _compute_ai_probability(bundle)
        flags          = _build_flags(bundle)
        verdict        = _compute_verdict(bundle, ml_prob)
        forensic_notes = _build_forensic_notes(bundle, verdict)
        ai_segments    = self._compute_ai_segments(data)

        return ForensicsResult(
            c2pa_flag=bundle.c2pa_flag,
            c2pa_origin=bundle.c2pa_label,
            ibi_variance=bundle.ibi_variance,
            loop_score=bundle.loop_score,
            loop_autocorr_score=bundle.loop_autocorr_score,
            spectral_slop=bundle.spectral_slop,
            synthid_score=float(bundle.synthid_bins),
            centroid_instability_score=bundle.centroid_instability_score,
            harmonic_ratio_score=bundle.harmonic_ratio_score,
            kurtosis_variability=bundle.kurtosis_variability,
            decoder_peak_score=bundle.decoder_peak_score,
            spectral_centroid_mean=bundle.spectral_centroid_mean,
            ai_segments=ai_segments,
            ai_probability=ai_probability,
            flags=flags,
            forensic_notes=forensic_notes,
            verdict=verdict,
            self_similarity_entropy=bundle.self_similarity_entropy,
            noise_floor_ratio=bundle.noise_floor_ratio,
            onset_strength_cv=bundle.onset_strength_cv,
            spectral_flatness_var=bundle.spectral_flatness_var,
            subbeat_grid_deviation=bundle.subbeat_grid_deviation,
            pitch_quantization_score=bundle.pitch_quantization_score,
            ultrasonic_noise_ratio=bundle.ultrasonic_noise_ratio,
            infrasonic_energy_ratio=bundle.infrasonic_energy_ratio,
            phase_coherence_differential=bundle.phase_coherence_differential,
            plr_std=bundle.plr_std,
            voiced_noise_floor=bundle.voiced_noise_floor,
            is_vocal=bundle.is_vocal,
        )

    # ------------------------------------------------------------------
    # Private: per-segment AI probability (heatmap)
    # ------------------------------------------------------------------

    def _compute_ai_segments(self, data: _AudioData) -> list[AiSegment]:
        """
        Delegate to the pure segment_ai_probabilities function.

        Returns empty list on any error — heatmap is non-fatal.
        """
        try:
            return segment_ai_probabilities(
                y=data.y,
                sr=data.sr,
                window_s=CONSTANTS.AI_HEATMAP_WINDOW_S,
                hop_s=CONSTANTS.AI_HEATMAP_HOP_S,
            )
        except (ModelInferenceError, OSError, RuntimeError, ValueError) as exc:
            _log.warning("ai_segments: skipped — %s", exc)
            return []


# ---------------------------------------------------------------------------
# Backward-compatibility re-exports
# Tests import these directly from services.forensics
# ---------------------------------------------------------------------------

__all__ = [
    # Public class
    "Forensics",
    # Bundle
    "_SignalBundle",
    # Aggregators
    "_build_flags",
    "_compute_ai_probability",
    "_compute_verdict",
    "_synthid_confidence",
    # Pure helpers
    "_check_spectral_slop",
    "_classify_c2pa_origin",
    "_cross_correlate",
    "_spectral_fingerprint",
    "compute_infrasonic_energy_ratio",
    "compute_phase_coherence_differential",
    "compute_plr_std",
    "compute_ultrasonic_noise_ratio",
    "compute_voiced_noise_floor",
    "segment_ai_probabilities",
]
