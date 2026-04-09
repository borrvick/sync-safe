"""
services/forensics/_bundle.py
Shared signal carrier dataclass — all per-signal scores in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.models import SectionRepetition


@dataclass
class _SignalBundle:
    """All per-signal scores in one place, eliminating 15-param signatures."""
    c2pa_flag: bool = False
    c2pa_label: str = "No C2PA Manifest"
    compressed_source: bool = True   # True = YouTube/MP3; False = direct file upload
    ibi_variance: float = -1.0
    loop_score: float = 0.0
    loop_autocorr_score: float = 0.0
    centroid_instability_score: float = -1.0
    harmonic_ratio_score: float = -1.0
    synthid_bins: int = 0
    spectral_slop: float = 0.0
    kurtosis_variability: float = -1.0
    decoder_peak_score: float = 0.0
    spectral_centroid_mean: float = -1.0
    self_similarity_entropy: float = -1.0
    noise_floor_ratio: float = -1.0
    onset_strength_cv: float = -1.0
    spectral_flatness_var: float = -1.0
    subbeat_grid_deviation: float = -1.0
    pitch_quantization_score: float = -1.0
    ultrasonic_noise_ratio: float = -1.0
    infrasonic_energy_ratio: float = -1.0
    phase_coherence_differential: float = -1.0
    plr_std: float = -1.0
    voiced_noise_floor: float = -1.0  # mean spectral flatness in voiced 4–12 kHz frames
    is_vocal: bool = False            # True → pyin-detected vocal content

    # Section-aware repetition (#142, #143, #145)
    loop_window_scores: list[tuple[float, float]]       = field(default_factory=list)
    section_similarities: dict[str, SectionRepetition]          = field(default_factory=dict)
    section_internal_repetition: dict[str, SectionRepetition]   = field(default_factory=dict)
