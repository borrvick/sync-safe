"""
core/models/forensics.py
AI-humanity forensics output models.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._types import ForensicVerdict


@dataclasses.dataclass(frozen=True)
class SectionRepetition:
    """
    Per-label repetition score from section-aware loop analysis (#143, #145).

    max_similarity:  highest pairwise cosine score within this label group.
    mean_similarity: mean across all pairs — low pair_count = low confidence.
    pair_count:      number of same-label pairs compared.
    """
    max_similarity:  float
    mean_similarity: float
    pair_count:      int


class AiSegment(BaseModel):
    """One time-windowed AI-probability estimate within a track."""

    model_config = ConfigDict(frozen=True)

    start_s: float      # window start (seconds from track start)
    end_s: float        # window end
    probability: float  # [0.0, 1.0]; higher = more AI-like

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ForensicsResult(BaseModel):
    """Output of the AI-humanity forensics stage."""

    model_config = ConfigDict(frozen=True)

    # Individual signal scores (0.0–1.0 where 1.0 = most AI-like)
    c2pa_flag: bool         = False     # True → born-AI assertion found in manifest
    ibi_variance: float     = 1.0       # inter-beat interval variance
    loop_score: float       = 0.0       # highest cross-correlation across 4-bar windows
    loop_autocorr_score: float = 0.0    # onset autocorrelation loop repetition score
    spectral_slop: float    = 0.0       # anomalous energy above SPECTRAL_SLOP_HZ
    synthid_score: float    = 0.0       # phase coherence in 18–22 kHz band
    centroid_instability_score: float = -1.0  # mean within-interval centroid CV; -1 = not computed
    harmonic_ratio_score: float = -1.0        # mean HNR within sustained intervals; -1 = not computed
    # New signals (2026-03-21) — calibrated from ISMIR TISMIR 2025 + arXiv 2506.19108
    kurtosis_variability: float = -1.0        # variance of per-frame mel-band kurtosis; -1 = not computed
    decoder_peak_score: float = 0.0           # periodic deconvolution peak density in 1–16 kHz band
    spectral_centroid_mean: float = -1.0      # mean spectral centroid in Hz across the track
    ai_probability: float = 0.0               # weighted probability score [0.0–1.0] used for verdict
    # Structural / instrumental signals (2026-03-21) — pending calibration, weights=0 until thresholds set
    self_similarity_entropy: float = -1.0     # Shannon entropy of chroma recurrence matrix; low = repetitive AI structure
    noise_floor_ratio: float = -1.0           # quiet-frame RMS / mean RMS; near-zero = VST render (no room noise)
    onset_strength_cv: float = -1.0           # CV of onset strength envelope; low = uniform AI dynamics
    spectral_flatness_var: float = -1.0       # variance of Wiener entropy over time; low = AI synth uniformity
    subbeat_grid_deviation: float = -1.0      # variance of onset-to-nearest-16th-note offset; low = on-grid
    pitch_quantization_score: float = -1.0    # mean abs cents deviation from 12-TET; near-zero = AI pitch-perfect
    ultrasonic_noise_ratio: float = -1.0      # energy ratio in 20–22 kHz band; elevated = diffusion residue (-1 = not computed)
    infrasonic_energy_ratio: float = -1.0     # energy ratio in 1–20 Hz band; elevated = AI math drift / DC bias (-1 = not computed)
    phase_coherence_differential: float = -1.0  # LF coherence − HF coherence; positive = AI phase pattern (-1 = mono/not computed)
    plr_std: float = -1.0                        # std of per-window peak-to-loudness ratio; low = frozen density (AI) (-1 = too short)
    voiced_noise_floor: float = -1.0             # mean spectral flatness in voiced 4–12 kHz frames; low = AI clean synthesis (-1 = non-vocal/not computed)
    is_vocal: bool = False                       # True → pyin detected vocal content; routes vocal scoring path
    c2pa_origin: str = ""                        # "ai" | "daw" | "unknown" | "" (no manifest)

    # Blended Repetition Index: 0.6 * loop_score + 0.4 * loop_autocorr_score (see #144)
    repetition_index: Optional[float] = None   # None → forensics skipped or pre-#144 result

    # Per-4-bar-window scores for heatmap rendering (#142)
    loop_window_scores: list[tuple[float, float]] = Field(default_factory=list)
    # (start_s, max_similarity) per window

    # Section-aware repetition: inter-section + intra-section (#143, #145)
    section_similarities: dict[str, SectionRepetition]          = Field(default_factory=dict)
    section_internal_repetition: dict[str, SectionRepetition]   = Field(default_factory=dict)

    ai_segments: list[AiSegment] = Field(default_factory=list)  # per-window heatmap data

    flags: list[str]        = Field(default_factory=list)  # human-readable flag labels
    forensic_notes: list[str] = Field(default_factory=list)  # secondary context shown below verdict
    verdict: ForensicVerdict = "Likely Not AI"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
