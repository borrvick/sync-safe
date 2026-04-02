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

from ._aggregators import (
    _build_flags,
    _build_forensic_notes,
    _classify,
    _compute_ai_probability,
    _compute_verdict,
    _synthid_confidence,
)
from ._bundle import _SignalBundle
from ._orchestrator import Forensics
from ._pure import (
    _check_spectral_slop,
    _classify_c2pa_origin,
    _cross_correlate,
    _spectral_fingerprint,
    compute_centroid_instability_score,
    compute_decoder_peak_score,
    compute_harmonic_ratio_score,
    compute_infrasonic_energy_ratio,
    compute_kurtosis_variability,
    compute_loop_autocorrelation_score,
    compute_noise_floor_ratio,
    compute_onset_strength_cv,
    compute_phase_coherence_differential,
    compute_pitch_quantization_score,
    compute_plr_std,
    compute_self_similarity_entropy,
    compute_spectral_flatness_variance,
    compute_subbeat_grid_deviation,
    compute_ultrasonic_noise_ratio,
    compute_voiced_noise_floor,
    segment_ai_probabilities,
)

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
    # compute_* signal functions (all 16)
    "compute_centroid_instability_score",
    "compute_decoder_peak_score",
    "compute_harmonic_ratio_score",
    "compute_infrasonic_energy_ratio",
    "compute_kurtosis_variability",
    "compute_loop_autocorrelation_score",
    "compute_noise_floor_ratio",
    "compute_onset_strength_cv",
    "compute_phase_coherence_differential",
    "compute_pitch_quantization_score",
    "compute_plr_std",
    "compute_self_similarity_entropy",
    "compute_spectral_flatness_variance",
    "compute_subbeat_grid_deviation",
    "compute_ultrasonic_noise_ratio",
    "compute_voiced_noise_floor",
    "segment_ai_probabilities",
]
