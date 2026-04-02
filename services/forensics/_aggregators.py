"""
services/forensics/_aggregators.py
Policy functions — classify signals, compute verdicts, build flags.
No I/O, no librosa.  All functions accept _SignalBundle and return plain types.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import ForensicVerdict

from ._bundle import _SignalBundle
from ._pure import _synthid_confidence

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ML classifier — loaded once, read-only after that
# ---------------------------------------------------------------------------

# Acceptable global: read-only pkl cache.  Concurrent reads are safe without
# locking.  A global is justified here because pkl files are large artefacts
# that are expensive to reload on every call.
_CLASSIFIERS: dict[str, object] | None = None


def _load_classifiers() -> dict[str, object] | None:
    """
    Load youtube + upload classifier pkls from models/.

    Returns None if either file is absent or joblib is unavailable.
    """
    try:
        import joblib  # type: ignore[import-untyped]
    except ImportError:
        return None

    _models_dir = Path(__file__).parent.parent.parent / "models"
    yt_path = _models_dir / "youtube_classifier.pkl"
    up_path = _models_dir / "upload_classifier.pkl"
    if not yt_path.exists() or not up_path.exists():
        return None
    try:
        return {
            "youtube": joblib.load(yt_path),
            "upload":  joblib.load(up_path),
        }
    except (OSError, ValueError, RuntimeError) as exc:
        _log.warning("Failed to load classifier pkl files: %s", exc)
        return None


def _classify(bundle: _SignalBundle) -> float | None:
    """
    Return P(AI) ∈ [0.0, 1.0] from the trained logistic regression classifier.

    Returns None when pkl files are absent or features contain sentinels.
    Caller falls back to _compute_ai_probability() on None.
    """
    global _CLASSIFIERS
    if _CLASSIFIERS is None:
        _CLASSIFIERS = _load_classifiers()
    if _CLASSIFIERS is None:
        return None

    key        = "youtube" if bundle.compressed_source else "upload"
    model_dict = _CLASSIFIERS[key]
    if not isinstance(model_dict, dict) or "pipeline" not in model_dict:
        return None

    feature_names: list[str] = model_dict["features"]
    pipeline = model_dict["pipeline"]

    _feature_map: dict[str, float] = {
        "ci":   bundle.centroid_instability_score,
        "hnr":  bundle.harmonic_ratio_score,
        "plr":  bundle.plr_std,
        "ibi":  bundle.ibi_variance,
        "cent": bundle.spectral_centroid_mean,
        "vnf":  bundle.voiced_noise_floor,
    }
    values = [_feature_map[f] for f in feature_names]
    if any(v < 0.0 for v in values):
        return None

    X = np.array(values, dtype=float).reshape(1, -1)
    try:
        return float(pipeline.predict_proba(X)[0, 1])
    except (ValueError, RuntimeError, AttributeError) as exc:
        _log.warning("_classify: predict_proba failed — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Weighted probability scoring
# ---------------------------------------------------------------------------

def _score_organic_signals(bundle: _SignalBundle) -> float:
    """
    Weighted sum of dampable AI probability signals.

    Signals that can be explained by organic production (pitch stacking,
    heavy DSP) are grouped here so _compute_ai_probability can apply
    damping only to this subtotal — not to hardware-evidence signals.

    Pure function — no I/O, no side effects, deterministic.
    """
    score = 0.0

    centroid_flagged = (
        bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN
        and bundle.centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    )

    if centroid_flagged:
        if bundle.is_vocal:
            score += CONSTANTS.PROB_WEIGHT_CENTROID_VOCAL
        else:
            score += CONSTANTS.PROB_WEIGHT_CENTROID  # 0.0 — disabled for instrumental

    if 0.0 <= bundle.ibi_variance < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        score += CONSTANTS.PROB_WEIGHT_IBI_QUANTIZED

    if bundle.loop_score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        score += CONSTANTS.PROB_WEIGHT_LOOP_CROSS_CORR

    if centroid_flagged and bundle.loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_VERDICT_THRESHOLD:
        score += CONSTANTS.PROB_WEIGHT_AUTOCORR_CENTROID

    harmonic_ai_min = (
        CONSTANTS.HARMONIC_RATIO_AI_MIN_VOCAL if bundle.is_vocal
        else CONSTANTS.HARMONIC_RATIO_AI_MIN
    )
    harmonic_weight = (
        CONSTANTS.PROB_WEIGHT_HARMONIC_RATIO_VOCAL if bundle.is_vocal
        else CONSTANTS.PROB_WEIGHT_HARMONIC_RATIO
    )
    if bundle.harmonic_ratio_score >= harmonic_ai_min:
        score += harmonic_weight

    synthid_conf = _synthid_confidence(bundle.synthid_bins)
    if synthid_conf == "medium":
        score += CONSTANTS.PROB_WEIGHT_SYNTHID_MEDIUM
    elif synthid_conf == "low":
        score += CONSTANTS.PROB_WEIGHT_SYNTHID_LOW

    if bundle.spectral_slop > CONSTANTS.SPECTRAL_SLOP_RATIO:
        score += CONSTANTS.PROB_WEIGHT_SPECTRAL_SLOP

    if not bundle.compressed_source and bundle.kurtosis_variability >= CONSTANTS.KURTOSIS_VARIABILITY_AI_MIN:
        score += CONSTANTS.PROB_WEIGHT_KURTOSIS

    if not bundle.compressed_source and bundle.decoder_peak_score >= CONSTANTS.DECODER_PEAK_SCORE_MIN:
        score += CONSTANTS.PROB_WEIGHT_DECODER_PEAK

    if 0.0 < bundle.spectral_centroid_mean <= CONSTANTS.SPECTRAL_CENTROID_MEAN_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_CENTROID_MEAN

    if 0.0 <= bundle.self_similarity_entropy < CONSTANTS.SELF_SIMILARITY_ENTROPY_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_SELF_SIMILARITY

    if 0.0 <= bundle.onset_strength_cv < CONSTANTS.ONSET_STRENGTH_CV_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_ONSET_STRENGTH

    if 0.0 <= bundle.spectral_flatness_var < CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_SPECTRAL_FLATNESS

    if 0.0 <= bundle.subbeat_grid_deviation < CONSTANTS.SUBBEAT_DEVIATION_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_SUBBEAT_GRID

    if 0.0 <= bundle.pitch_quantization_score < CONSTANTS.PITCH_QUANTIZATION_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_PITCH_QUANTIZATION

    if (
        not bundle.compressed_source
        and CONSTANTS.ULTRASONIC_ENERGY_RATIO_AI_MIN > 0.0
        and bundle.ultrasonic_noise_ratio >= CONSTANTS.ULTRASONIC_ENERGY_RATIO_AI_MIN
    ):
        score += CONSTANTS.PROB_WEIGHT_ULTRASONIC

    if (
        not bundle.compressed_source
        and CONSTANTS.INFRASONIC_ENERGY_RATIO_AI_MIN > 0.0
        and bundle.infrasonic_energy_ratio >= CONSTANTS.INFRASONIC_ENERGY_RATIO_AI_MIN
    ):
        score += CONSTANTS.PROB_WEIGHT_INFRASONIC

    if (
        not bundle.compressed_source
        and CONSTANTS.PHASE_COHERENCE_DIFFERENTIAL_AI_MIN > 0.0
        and bundle.phase_coherence_differential >= CONSTANTS.PHASE_COHERENCE_DIFFERENTIAL_AI_MIN
    ):
        score += CONSTANTS.PROB_WEIGHT_PHASE_COHERENCE

    if CONSTANTS.PLR_STD_AI_MAX > 0.0 and 0.0 <= bundle.plr_std <= CONSTANTS.PLR_STD_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_PLR_FLATNESS

    if (
        not bundle.compressed_source
        and bundle.is_vocal
        and CONSTANTS.VOICED_NOISE_FLOOR_AI_MAX > 0.0
        and 0.0 <= bundle.voiced_noise_floor <= CONSTANTS.VOICED_NOISE_FLOOR_AI_MAX
    ):
        score += CONSTANTS.PROB_WEIGHT_VOICED_NOISE_FLOOR

    return score


def _compute_ai_probability(bundle: _SignalBundle) -> float:
    """
    Compute a weighted AI probability score in [0.0, 1.0].

    Organic production damping halves the organic signal subtotal when
    loop autocorr is below PROB_AUTOCORR_ORGANIC_THRESHOLD — covers
    experimental productions (pitch stacking, heavy DSP, non-repetitive).

    Hardware-evidence signals (noise_floor_ratio) bypass damping.

    Pure function — no I/O, no side effects, deterministic.
    """
    score = _score_organic_signals(bundle)

    if (
        bundle.loop_autocorr_score < CONSTANTS.PROB_AUTOCORR_ORGANIC_THRESHOLD
        and bundle.centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    ):
        score *= CONSTANTS.PROB_ORGANIC_DAMPING_FACTOR

    hardware_score = 0.0
    if 0.0 <= bundle.noise_floor_ratio < CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX:
        hardware_score += CONSTANTS.PROB_WEIGHT_NOISE_FLOOR

    return float(min(score + hardware_score, 1.0))


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------

def _compute_verdict(
    bundle: _SignalBundle,
    ml_prob: float | None = None,
) -> ForensicVerdict:
    """
    Aggregate numeric forensic scores into a four-tier verdict.

    Tiers (ordered by certainty):
      "AI"            — cryptographic proof: C2PA born-AI or high SynthID.
      "Likely AI"     — strong algorithmic signals, no embedded proof.
      "Likely Not AI" — default; no significant AI indicators.
      "Not AI"        — reserved; no verifiable born-human proof standard yet.
    """
    if bundle.c2pa_flag:
        return "AI"
    if _synthid_confidence(bundle.synthid_bins) == "high":
        return "AI"

    centroid_flagged = (
        bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN
        and bundle.centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    )

    hnr_flagged         = bundle.harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN
    in_vocoder          = bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    hnr_blocks_override = bundle.harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_SAMPLE_OVERRIDE_BLOCK

    no_ai_signals    = not centroid_flagged and not hnr_flagged
    strong_human_timing = (
        bundle.ibi_variance > CONSTANTS.IBI_SAMPLE_LOOP_HUMAN_MIN_WITH_AI_SIGNALS
        and not hnr_blocks_override
    )

    if (
        bundle.loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_SAMPLE_VERDICT_THRESHOLD
        and bundle.ibi_variance > CONSTANTS.IBI_ERRATIC_MIN
        and bundle.synthid_bins == 0
        and bundle.spectral_slop <= CONSTANTS.SPECTRAL_SLOP_RATIO
        and not in_vocoder
        and (no_ai_signals or strong_human_timing)
    ):
        return "Likely Not AI"

    if ml_prob is not None:
        return "Likely AI" if ml_prob >= 0.5 else "Likely Not AI"

    prob = _compute_ai_probability(bundle)
    return "Likely AI" if prob >= CONSTANTS.PROB_VERDICT_HYBRID else "Likely Not AI"


# ---------------------------------------------------------------------------
# Flag and note builders
# ---------------------------------------------------------------------------

def _build_flags(bundle: _SignalBundle) -> list[str]:
    """
    Build the list of human-readable flag strings for the UI.

    Pure function — separated from verdict logic so each can be tested
    independently.
    """
    flags: list[str] = []

    if "Born-AI" in bundle.c2pa_label:
        flags.append(bundle.c2pa_label)

    if bundle.ibi_variance < 0:
        flags.append("Insufficient data for groove analysis")
    elif bundle.ibi_variance < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        flags.append("Perfect Quantization (AI signal)")
    elif bundle.ibi_variance > CONSTANTS.IBI_ERRATIC_MIN:
        flags.append("Human-Feel Timing (Organic)")

    if bundle.spectral_slop > CONSTANTS.SPECTRAL_SLOP_RATIO:
        flags.append(f"Spectral Slop detected ({bundle.spectral_slop:.1%} HF energy)")

    if bundle.loop_score > CONSTANTS.LOOP_SCORE_CEILING:
        flags.append(f"Highly Repetitive Structure (score {bundle.loop_score:.3f})")
    elif bundle.loop_score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        flags.append(f"Moderately Repetitive Structure (score {bundle.loop_score:.3f})")

    if bundle.loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_VERDICT_THRESHOLD:
        flags.append(
            f"Sample-Heavy / Loop-Based (Organic Production) "
            f"(autocorr score {bundle.loop_autocorr_score:.3f})"
        )

    if bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN:
        flags.append(
            f"Extreme Spectral Processing Detected (centroid CV {bundle.centroid_instability_score:.3f}) — "
            f"total formant replacement; consistent with vocoder, talkbox, or heavy DSP. "
            f"Not typical of AI voice generators (AI range: 0.32–0.38)"
        )
    elif bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN:
        if bundle.loop_autocorr_score < CONSTANTS.PROB_AUTOCORR_ORGANIC_THRESHOLD:
            flags.append(
                f"Spectral Processing Detected (centroid CV {bundle.centroid_instability_score:.3f}) — "
                f"elevated formant drift, but non-repetitive song structure (autocorr "
                f"{bundle.loop_autocorr_score:.3f}) suggests pitch correction / vocal stacking "
                f"rather than AI generation. Probability weight reduced."
            )
        else:
            flags.append(
                f"Formant Drift Detected (centroid CV {bundle.centroid_instability_score:.3f}) — "
                f"erratic within-note spectral shift (AI signal)"
            )

    if bundle.harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN:
        flags.append(
            f"Unnaturally Clean Harmonics (HNR {bundle.harmonic_ratio_score:.3f}) — "
            f"no breath/noise artifacts detected within sustained notes (AI signal)"
        )

    conf = _synthid_confidence(bundle.synthid_bins)
    if conf == "high":
        flags.append(f"High-confidence AI watermark ({bundle.synthid_bins} coherent bins in 18–22 kHz)")
    elif conf == "medium":
        flags.append(f"Medium-confidence AI watermark ({bundle.synthid_bins} coherent bins in 18–22 kHz)")
    elif conf == "low":
        flags.append(f"Low-confidence HF phase anomaly ({bundle.synthid_bins} bin{'s' if bundle.synthid_bins > 1 else ''}) — monitor")

    if not bundle.compressed_source and bundle.kurtosis_variability >= CONSTANTS.KURTOSIS_VARIABILITY_AI_MIN:
        flags.append(
            f"Neural Codec Artifacts Detected (mel-kurtosis variance {bundle.kurtosis_variability:.1f}) — "
            f"checkerboard mel-band spikes consistent with EnCodec/DAC decoder synthesis"
        )

    if not bundle.compressed_source and bundle.decoder_peak_score >= CONSTANTS.DECODER_PEAK_SCORE_MIN:
        flags.append(
            f"Decoder Spectral Fingerprint Detected (score {bundle.decoder_peak_score:.3f}) — "
            f"periodic peaks in 1–16 kHz consistent with transposed convolution strides (AI vocoder)"
        )

    if 0.0 < bundle.spectral_centroid_mean <= CONSTANTS.SPECTRAL_CENTROID_MEAN_AI_MAX:
        flags.append(
            f"Low Mean Spectral Centroid ({bundle.spectral_centroid_mean:.0f} Hz) — "
            f"energy concentrated in lower frequencies (AI generators ~1091 Hz, human recordings ~1501 Hz)"
        )

    if CONSTANTS.SELF_SIMILARITY_ENTROPY_AI_MAX > 0.0 and 0.0 <= bundle.self_similarity_entropy < CONSTANTS.SELF_SIMILARITY_ENTROPY_AI_MAX:
        flags.append(
            f"Low Structural Entropy (self-similarity {bundle.self_similarity_entropy:.2f} bits) — "
            f"blocky, repetitive structure consistent with AI attention patterns"
        )

    if CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX > 0.0 and 0.0 <= bundle.noise_floor_ratio < CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX:
        flags.append(
            f"Digital Silence Detected (noise floor ratio {bundle.noise_floor_ratio:.4f}) — "
            f"near-zero energy between notes; consistent with VST render (no room/mic noise)"
        )

    if CONSTANTS.ONSET_STRENGTH_CV_AI_MAX > 0.0 and 0.0 <= bundle.onset_strength_cv < CONSTANTS.ONSET_STRENGTH_CV_AI_MAX:
        flags.append(
            f"Uniform Onset Dynamics (CV {bundle.onset_strength_cv:.3f}) — "
            f"abnormally consistent hit strength; no performance dynamics variation (AI signal)"
        )

    if CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX > 0.0 and 0.0 <= bundle.spectral_flatness_var < CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX:
        flags.append(
            f"Uniform Spectral Flatness (var {bundle.spectral_flatness_var:.6f}) — "
            f"no physical noise source detected between notes (AI synthesizer signal)"
        )

    if CONSTANTS.SUBBEAT_DEVIATION_AI_MAX > 0.0 and 0.0 <= bundle.subbeat_grid_deviation < CONSTANTS.SUBBEAT_DEVIATION_AI_MAX:
        flags.append(
            f"On-Grid Timing (16th-note deviation var {bundle.subbeat_grid_deviation:.4f}) — "
            f"onsets land precisely on the grid (AI or heavily quantized production)"
        )

    if (
        not bundle.compressed_source
        and CONSTANTS.ULTRASONIC_ENERGY_RATIO_AI_MIN > 0.0
        and bundle.ultrasonic_noise_ratio >= CONSTANTS.ULTRASONIC_ENERGY_RATIO_AI_MIN
    ):
        flags.append(
            f"Ultrasonic Noise Plateau ({bundle.ultrasonic_noise_ratio:.5f} energy ratio) — "
            f"elevated energy in 20–22 kHz band; consistent with diffusion model denoising residue "
            f"(human masters are shelf-filtered above 18–20 kHz)"
        )

    if CONSTANTS.PLR_STD_AI_MAX > 0.0 and 0.0 <= bundle.plr_std <= CONSTANTS.PLR_STD_AI_MAX:
        flags.append(
            f"Frozen Loudness Density (PLR std {bundle.plr_std:.2f} dB) — "
            f"crest factor is abnormally consistent across the track; "
            f"human masters vary dynamically — consistent with AI look-ahead limiting"
        )

    if (
        not bundle.compressed_source
        and CONSTANTS.PHASE_COHERENCE_DIFFERENTIAL_AI_MIN > 0.0
        and bundle.phase_coherence_differential >= CONSTANTS.PHASE_COHERENCE_DIFFERENTIAL_AI_MIN
    ):
        flags.append(
            f"Phase Coherence Anomaly (LF−HF differential {bundle.phase_coherence_differential:.3f}) — "
            f"low-frequency stereo phase is stable while high-frequency phase is incoherent; "
            f"consistent with AI diffusion generating LF and HF as separate statistical events"
        )

    if (
        not bundle.compressed_source
        and CONSTANTS.INFRASONIC_ENERGY_RATIO_AI_MIN > 0.0
        and bundle.infrasonic_energy_ratio >= CONSTANTS.INFRASONIC_ENERGY_RATIO_AI_MIN
    ):
        flags.append(
            f"Infrasonic Rumble ({bundle.infrasonic_energy_ratio:.6f} energy ratio) — "
            f"elevated energy below 20 Hz; real microphones cannot capture sub-sonic content "
            f"(consistent with AI diffusion math drift)"
        )

    if CONSTANTS.PITCH_QUANTIZATION_AI_MAX > 0.0 and 0.0 <= bundle.pitch_quantization_score < CONSTANTS.PITCH_QUANTIZATION_AI_MAX:
        flags.append(
            f"Perfect Pitch Quantization ({bundle.pitch_quantization_score:.1f} cents deviation) — "
            f"pitch centers align precisely to equal temperament; "
            f"real instruments typically drift 10–30 cents (AI signal)"
        )

    if (
        not bundle.compressed_source
        and bundle.is_vocal
        and CONSTANTS.VOICED_NOISE_FLOOR_AI_MAX > 0.0
        and 0.0 <= bundle.voiced_noise_floor <= CONSTANTS.VOICED_NOISE_FLOOR_AI_MAX
    ):
        flags.append(
            f"Unnaturally Clean Vocal Synthesis (voiced noise floor {bundle.voiced_noise_floor:.4f} flatness) — "
            f"spectral content between harmonics in voiced frames is near-silent; "
            f"real vocal recordings have continuous mic/room noise even during sustained notes (AI signal)"
        )

    return flags


def _build_forensic_notes(bundle: _SignalBundle, verdict: ForensicVerdict) -> list[str]:
    """
    Build secondary context notes shown below the verdict badge.

    Pure function — no I/O, no side effects, deterministic.
    """
    notes: list[str] = []

    if bundle.loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_SAMPLE_VERDICT_THRESHOLD:
        notes.append(
            "This track has a highly regular rhythmic structure consistent with loop-based "
            "production — common in modern pop and hip-hop. This alone does not indicate AI generation."
        )

    if bundle.compressed_source:
        notes.append(
            "Detection limited for streaming audio — YouTube and streaming sources are "
            "AAC/Opus transcoded before analysis, which masks several AI signals. "
            "Upload the original file for more reliable results."
        )

    return notes
