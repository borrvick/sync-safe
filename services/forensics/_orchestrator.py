"""
services/forensics/_orchestrator.py
Forensics class — orchestrates all AI-origin detection analyzers.
"""
from __future__ import annotations

import logging

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import AiSegment, AudioBuffer, ForensicsResult

from ._aggregators import (
    _build_flags,
    _build_forensic_notes,
    _classify,
    _compute_ai_probability,
    _compute_verdict,
)
from ._audio import _AudioData, load_audio
from ._bundle import _SignalBundle
from ._pure import segment_ai_probabilities
from .detectors import (
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
