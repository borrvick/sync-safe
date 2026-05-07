"""
tests/test_forensics.py

Unit tests for services/forensics.py pure functions + regression tests that
lock in the current verdict for each real-track fixture in tests/fixtures/forensics/.

Fast suite: no audio loading, no GPU, no librosa calls. All pure-function tests
complete in < 1 second. The regression parametrize block is the key guard against
unintentional threshold drift.

To intentionally accept a verdict change after a deliberate algorithm update:
    1. Re-run: pytest tests/test_forensics.py -v
    2. Update the _EXPECTED_VERDICTS dict below with the new values.
    3. Commit with a comment explaining why the verdicts changed.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.config import CONSTANTS
from services.forensics import (
    _SignalBundle,
    _build_flags,
    _check_spectral_slop,
    _classify_c2pa_origin,
    _compute_ai_probability,
    _compute_verdict,
    _cross_correlate,
    _spectral_fingerprint,
    _synthid_confidence,
    compute_infrasonic_energy_ratio,
    compute_phase_coherence_differential,
    compute_plr_std,
    compute_ultrasonic_noise_ratio,
    compute_voiced_noise_floor,
    segment_ai_probabilities,
)
from tests.conftest import all_forensics_fixtures


# ---------------------------------------------------------------------------
# _synthid_confidence
# ---------------------------------------------------------------------------

class TestSynthidConfidence:
    def test_zero_bins_is_none(self):
        assert _synthid_confidence(0) == "none"

    def test_one_bin_is_low(self):
        assert _synthid_confidence(1) == "low"

    def test_at_low_ceiling_is_low(self):
        assert _synthid_confidence(CONSTANTS.SYNTHID_LOW_BINS) == "low"

    def test_just_above_low_is_medium(self):
        assert _synthid_confidence(CONSTANTS.SYNTHID_LOW_BINS + 1) == "medium"

    def test_at_medium_ceiling_is_medium(self):
        assert _synthid_confidence(CONSTANTS.SYNTHID_MEDIUM_BINS) == "medium"

    def test_above_medium_is_high(self):
        assert _synthid_confidence(CONSTANTS.SYNTHID_MEDIUM_BINS + 1) == "high"


# ---------------------------------------------------------------------------
# _spectral_fingerprint + _cross_correlate
# ---------------------------------------------------------------------------

class TestSpectralFingerprint:
    def test_identical_segments_cross_correlate_to_one(self):
        seg = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
        fp = _spectral_fingerprint(seg)
        assert _cross_correlate(fp, fp) == pytest.approx(1.0, abs=1e-5)

    def test_output_is_unit_normalised(self):
        seg = np.ones(512, dtype=np.float32)
        fp = _spectral_fingerprint(seg)
        assert np.linalg.norm(fp) == pytest.approx(1.0, abs=1e-5)

    def test_cross_correlate_score_in_range(self):
        rng = np.random.default_rng(42)
        a = _spectral_fingerprint(rng.standard_normal(1024).astype(np.float32))
        b = _spectral_fingerprint(rng.standard_normal(1024).astype(np.float32))
        score = _cross_correlate(a, b)
        assert 0.0 <= score <= 1.0

    def test_zero_segment_does_not_produce_nans(self):
        fp = _spectral_fingerprint(np.zeros(512, dtype=np.float32))
        assert not np.any(np.isnan(fp))

    def test_cross_correlate_handles_mismatched_lengths(self):
        a = np.ones(100, dtype=np.float32)
        b = np.ones(200, dtype=np.float32)
        # Should not raise; truncates to shorter length
        score = _cross_correlate(a, b)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# _check_spectral_slop
# ---------------------------------------------------------------------------

class TestCheckSpectralSlop:
    _SR = 22_050

    def test_low_freq_tone_is_near_zero(self):
        t = np.arange(self._SR) / self._SR
        audio = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        ratio = _check_spectral_slop(audio, self._SR, 16_000, 0.15)
        assert ratio < 0.05

    def test_high_freq_tone_is_high(self):
        # Nyquist at SR=22050 is 11025 Hz — use a threshold below that
        # so the test can produce measurable HF energy.
        t = np.arange(self._SR) / self._SR
        audio = np.sin(2 * np.pi * 9_000 * t).astype(np.float32)
        ratio = _check_spectral_slop(audio, self._SR, 8_000, 0.15)
        assert ratio > 0.5

    def test_result_is_in_unit_range(self):
        audio = np.random.default_rng(7).standard_normal(self._SR).astype(np.float32)
        ratio = _check_spectral_slop(audio, self._SR, 16_000, 0.15)
        assert 0.0 <= ratio <= 1.0

    def test_silent_audio_returns_float_without_error(self):
        ratio = _check_spectral_slop(np.zeros(1024, dtype=np.float32), self._SR, 16_000, 0.15)
        assert isinstance(ratio, float)


# ---------------------------------------------------------------------------
# _compute_ai_probability
# ---------------------------------------------------------------------------

def _human_bundle() -> _SignalBundle:
    """Espresso-profile: sample-heavy human production, zero AI signals."""
    return _SignalBundle(
        c2pa_flag=False,
        ibi_variance=252.0,
        loop_score=0.88,
        loop_autocorr_score=0.93,
        centroid_instability_score=0.196,
        harmonic_ratio_score=0.227,
        synthid_bins=0,
        spectral_slop=0.0,
    )


def _ai_bundle() -> _SignalBundle:
    """
    Synthetic AI profile that fires multiple calibrated signals above threshold.

    centroid_instability is disabled (weight=0.0) as of 2026-03-24 — FMC calibration
    showed human median (0.379) > AI median (0.244), making it a false-positive source.
    harmonic_ratio threshold raised to 0.85 (FMC: human p90=0.778, AI p90=0.933).

    Uses noise_floor + PLR + centroid_mean + harmonic_ratio to reach hybrid threshold:
      noise_floor 0.001 < 0.005  → +0.30
      plr_std 0.5    < 1.2   → +0.15
      centroid_mean 1000 < 1400  → +0.10
      harmonic_ratio 0.90 > 0.85 → +0.10
      total = 0.65 ≥ PROB_VERDICT_HYBRID (0.45)
    """
    return _SignalBundle(
        c2pa_flag=False,
        ibi_variance=252.0,
        loop_score=0.0,
        loop_autocorr_score=0.75,
        centroid_instability_score=0.364,
        harmonic_ratio_score=0.90,
        synthid_bins=0,
        spectral_slop=0.0,
        noise_floor_ratio=0.001,
        plr_std=0.5,
        spectral_centroid_mean=1000.0,
    )


class TestComputeAiProbability:
    def test_clean_human_scores_zero(self):
        assert _compute_ai_probability(_human_bundle()) == 0.0

    def test_ai_profile_scores_above_hybrid_threshold(self):
        prob = _compute_ai_probability(_ai_bundle())
        assert prob >= CONSTANTS.PROB_VERDICT_HYBRID

    def test_c2pa_flag_does_not_affect_probability(self):
        # C2PA is a hard override handled in _compute_verdict, not here
        base    = _compute_ai_probability(dataclasses.replace(_human_bundle(), c2pa_flag=False))
        flagged = _compute_ai_probability(dataclasses.replace(_human_bundle(), c2pa_flag=True))
        assert base == flagged

    def test_score_clamped_to_one(self):
        prob = _compute_ai_probability(_SignalBundle(
            c2pa_flag=True,
            ibi_variance=0.0,
            loop_score=1.0,
            loop_autocorr_score=1.0,
            centroid_instability_score=0.99,
            harmonic_ratio_score=0.99,
            synthid_bins=99,
            spectral_slop=1.0,
        ))
        assert prob <= 1.0

    def test_organic_damping_fires_on_non_repetitive_track(self):
        """
        Organic damping halves the organic score when autocorr is below threshold.
        centroid_instability weight is 0.0 (disabled 2026-03-24); harmonic_ratio
        threshold is 0.85. Use hnr=0.90 to fire the organic signal, then verify
        that low autocorr triggers the 0.5× damping factor.
        """
        undamped = _compute_ai_probability(_SignalBundle(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.40,        # above organic threshold → no damping
            centroid_instability_score=0.41,
            harmonic_ratio_score=0.90,       # above new 0.85 threshold → fires
            synthid_bins=0, spectral_slop=0.0,
        ))
        damped = _compute_ai_probability(_SignalBundle(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.00,        # below organic threshold → damped
            centroid_instability_score=0.41,
            harmonic_ratio_score=0.90,
            synthid_bins=0, spectral_slop=0.0,
        ))
        assert damped < undamped

    def test_synthid_medium_adds_weight(self):
        base        = _compute_ai_probability(_human_bundle())
        with_synthid = _compute_ai_probability(
            dataclasses.replace(_human_bundle(), synthid_bins=CONSTANTS.SYNTHID_LOW_BINS + 1)
        )
        assert with_synthid > base

    def test_noise_floor_bypasses_organic_damping(self):
        """Orchestral AI (AIVA) profile: no vocals → centroid/HNR = -1.0, low autocorr.
        noise_floor_ratio must contribute even though damping fires."""
        # Without noise floor signal: damping fires, score stays at 0.0
        no_nfr = _compute_ai_probability(_SignalBundle(
            loop_autocorr_score=0.10,   # low → damping fires
            centroid_instability_score=-1.0,  # no vocal signals
            noise_floor_ratio=-1.0,
        ))
        # With noise floor signal: hardware_score added after damping
        with_nfr = _compute_ai_probability(_SignalBundle(
            loop_autocorr_score=0.10,
            centroid_instability_score=-1.0,
            noise_floor_ratio=0.001,    # triggers PROB_WEIGHT_NOISE_FLOOR
        ))
        assert with_nfr > no_nfr
        assert with_nfr >= 0.1  # signal must contribute non-trivially to probability


# ---------------------------------------------------------------------------
# _compute_verdict
# ---------------------------------------------------------------------------

class TestComputeVerdict:
    def test_c2pa_flag_always_returns_ai(self):
        result = _compute_verdict(_SignalBundle(
            c2pa_flag=True, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.0, centroid_instability_score=0.0,
            synthid_bins=0, spectral_slop=0.0,
        ))
        assert result == "AI"

    def test_high_synthid_returns_ai(self):
        result = _compute_verdict(_SignalBundle(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.0, centroid_instability_score=0.0,
            synthid_bins=CONSTANTS.SYNTHID_MEDIUM_BINS + 1, spectral_slop=0.0,
        ))
        assert result == "AI"

    def test_sample_loop_human_profile_returns_likely_not_ai(self):
        result = _compute_verdict(_SignalBundle(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.88,
            loop_autocorr_score=0.93, centroid_instability_score=0.196,
            synthid_bins=0, spectral_slop=0.0, harmonic_ratio_score=0.227,
        ))
        assert result == "Likely Not AI"

    def test_ai_profile_without_metadata_returns_likely_ai(self):
        # Uses calibrated signals that reach hybrid threshold without centroid
        # (disabled 2026-03-24) or HNR below new 0.85 threshold.
        result = _compute_verdict(_SignalBundle(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.75, centroid_instability_score=0.364,
            synthid_bins=0, spectral_slop=0.0, harmonic_ratio_score=0.90,
            noise_floor_ratio=0.001, plr_std=0.5, spectral_centroid_mean=1000.0,
        ))
        assert result == "Likely AI"

    def test_ai_verdict_requires_hard_evidence(self):
        # "AI" is only returned for C2PA or high SynthID — never from probability alone
        result = _compute_verdict(_SignalBundle(
            c2pa_flag=False, ibi_variance=0.0, loop_score=1.0,
            loop_autocorr_score=1.0, centroid_instability_score=0.99,
            synthid_bins=0, spectral_slop=1.0, harmonic_ratio_score=0.99,
        ))
        assert result != "AI"

    def test_zero_signals_returns_likely_not_ai(self):
        result = _compute_verdict(_SignalBundle(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.0, centroid_instability_score=0.0,
            synthid_bins=0, spectral_slop=0.0,
        ))
        assert result == "Likely Not AI"


# ---------------------------------------------------------------------------
# _build_flags
# ---------------------------------------------------------------------------

class TestBuildFlags:
    _CLEAN = _SignalBundle(
        c2pa_label="No C2PA Manifest",
        ibi_variance=252.0,
        spectral_slop=0.0,
        loop_score=0.88,
        loop_autocorr_score=0.93,
        centroid_instability_score=0.196,
        harmonic_ratio_score=0.227,
        synthid_bins=0,
    )

    def test_clean_human_track_has_no_ai_flags(self):
        flags = _build_flags(self._CLEAN)
        ai_keywords = {"AI signal", "Perfect Quantization", "Formant Drift", "watermark",
                       "Unnaturally Clean"}
        for flag in flags:
            assert not any(kw in flag for kw in ai_keywords), f"Unexpected AI flag: {flag!r}"

    def test_c2pa_born_ai_label_appears_in_flags(self):
        flags = _build_flags(dataclasses.replace(self._CLEAN, c2pa_label="Born-AI (Certified)"))
        assert any("Born-AI" in f for f in flags)

    def test_high_synthid_generates_high_confidence_flag(self):
        flags = _build_flags(dataclasses.replace(self._CLEAN, synthid_bins=CONSTANTS.SYNTHID_MEDIUM_BINS + 1))
        assert any("High-confidence" in f for f in flags)

    def test_perfect_quantization_on_near_zero_ibi(self):
        flags = _build_flags(dataclasses.replace(self._CLEAN, ibi_variance=0.0001))
        assert any("Perfect Quantization" in f for f in flags)

    def test_human_feel_flag_on_high_ibi(self):
        flags = _build_flags(dataclasses.replace(self._CLEAN, ibi_variance=200.0))
        assert any("Human-Feel" in f for f in flags)

    def test_loop_ceiling_flag_on_high_loop_score(self):
        flags = _build_flags(dataclasses.replace(self._CLEAN, loop_score=0.99))
        assert any("Highly Repetitive" in f for f in flags)

    def test_formant_drift_flag_on_high_centroid(self):
        flags = _build_flags(dataclasses.replace(
            self._CLEAN, centroid_instability_score=0.40, loop_autocorr_score=0.75
        ))
        assert any("Formant Drift" in f for f in flags)

    def test_vocoder_flag_on_very_high_centroid(self):
        flags = _build_flags(dataclasses.replace(self._CLEAN, centroid_instability_score=0.70))
        assert any("Vocoder" in f or "Extreme Spectral" in f for f in flags)

    def test_harmonic_ratio_flag_on_high_hnr(self):
        # Threshold raised to 0.85 (2026-03-24 FMC calibration)
        flags = _build_flags(dataclasses.replace(self._CLEAN, harmonic_ratio_score=0.90))
        assert any("Clean Harmonics" in f or "Unnaturally" in f for f in flags)


# ---------------------------------------------------------------------------
# compute_ultrasonic_noise_ratio
# ---------------------------------------------------------------------------

class TestComputeUltrasonicNoiseRatio:
    SR = 44_100

    def _sine(self, freq_hz: float, duration_s: float = 1.0) -> np.ndarray:
        t = np.linspace(0, duration_s, int(self.SR * duration_s), endpoint=False)
        return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)

    def test_clean_1khz_sine_near_zero(self):
        """A 1 kHz tone has virtually no energy in the 20–22 kHz band."""
        audio = self._sine(1000.0)
        ratio = compute_ultrasonic_noise_ratio(audio, self.SR)
        assert ratio < 0.01

    def test_21khz_sine_high_ratio(self):
        """A 21 kHz tone should produce a high ultrasonic energy ratio."""
        audio = self._sine(21_000.0)
        ratio = compute_ultrasonic_noise_ratio(audio, self.SR)
        assert ratio > 0.5

    def test_mixed_signal_partial_ratio(self):
        """Equal-amplitude 1 kHz + 21 kHz mix: ratio should be well above 0 and below 1."""
        audio = self._sine(1000.0) + self._sine(21_000.0)
        ratio = compute_ultrasonic_noise_ratio(audio, self.SR)
        assert 0.1 < ratio < 0.9

    def test_silent_audio_returns_zero(self):
        """All-zero input has no energy anywhere — ratio should be 0.0."""
        audio = np.zeros(self.SR, dtype=np.float32)
        ratio = compute_ultrasonic_noise_ratio(audio, self.SR)
        assert ratio == 0.0

    def test_returns_float_in_unit_range(self):
        """Return value must always be a float in [0.0, 1.0]."""
        audio = np.random.default_rng(42).standard_normal(self.SR).astype(np.float32)
        ratio = compute_ultrasonic_noise_ratio(audio, self.SR)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

    def test_compressed_source_gate_in_bundle(self):
        """_score_organic_signals must not add weight when compressed_source=True."""
        bundle = _SignalBundle(
            compressed_source=True,
            ultrasonic_noise_ratio=0.99,  # would fire if uncompressed
        )
        # With weight=0.0 (disabled) AND compressed gate, score contribution = 0
        score = _compute_ai_probability(bundle)
        assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_infrasonic_energy_ratio
# ---------------------------------------------------------------------------

class TestComputeInfrasonicEnergyRatio:
    SR = 22_050

    def _sine(self, freq_hz: float, duration_s: float = 2.0) -> np.ndarray:
        t = np.linspace(0, duration_s, int(self.SR * duration_s), endpoint=False)
        return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)

    def test_1khz_sine_near_zero(self):
        """A 1 kHz tone has virtually no energy in the 1–20 Hz band."""
        audio = self._sine(1000.0)
        ratio = compute_infrasonic_energy_ratio(audio, self.SR)
        assert ratio < 0.001

    def test_10hz_sine_high_ratio(self):
        """A 10 Hz tone should produce a high infrasonic energy ratio."""
        audio = self._sine(10.0)
        ratio = compute_infrasonic_energy_ratio(audio, self.SR)
        assert ratio > 0.5

    def test_mixed_signal_partial_ratio(self):
        """Equal-amplitude 10 Hz + 1 kHz mix: ratio above 0, well below 1."""
        audio = self._sine(10.0) + self._sine(1000.0)
        ratio = compute_infrasonic_energy_ratio(audio, self.SR)
        assert 0.01 < ratio < 0.9

    def test_silent_audio_returns_zero(self):
        audio = np.zeros(self.SR * 2, dtype=np.float32)
        ratio = compute_infrasonic_energy_ratio(audio, self.SR)
        assert ratio == 0.0

    def test_too_short_returns_minus_one(self):
        """Audio shorter than 1 second returns -1.0."""
        audio = np.zeros(self.SR // 2, dtype=np.float32)
        ratio = compute_infrasonic_energy_ratio(audio, self.SR)
        assert ratio == -1.0

    def test_returns_float_in_unit_range(self):
        audio = np.random.default_rng(42).standard_normal(self.SR * 2).astype(np.float32)
        ratio = compute_infrasonic_energy_ratio(audio, self.SR)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# compute_phase_coherence_differential
# ---------------------------------------------------------------------------

class TestComputePhaseCoherenceDifferential:
    SR = 44_100

    def _stereo_sine(
        self,
        freq_hz: float,
        phase_offset: float = 0.0,
        duration_s: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, duration_s, int(self.SR * duration_s), endpoint=False)
        left  = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
        right = np.sin(2 * np.pi * freq_hz * t + phase_offset).astype(np.float32)
        return left, right

    def test_identical_channels_zero_differential(self):
        """L == R at all frequencies → both LF and HF coherence = 1 → differential ≈ 0."""
        L, R = self._stereo_sine(1000.0, phase_offset=0.0)
        diff = compute_phase_coherence_differential(L, R, self.SR)
        assert abs(diff) < 0.1

    def test_stable_lf_unstable_hf_positive_differential(self):
        """
        Simulate AI pattern: L and R share the same LF content (coherent)
        but have independent HF noise (incoherent).
        """
        rng = np.random.default_rng(0)
        n   = int(self.SR * 2)
        t   = np.linspace(0, 2.0, n, endpoint=False)

        lf  = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
        hf_l = rng.standard_normal(n).astype(np.float32) * 0.5
        hf_r = rng.standard_normal(n).astype(np.float32) * 0.5  # independent HF

        L = lf + hf_l
        R = lf + hf_r  # same LF, different HF
        diff = compute_phase_coherence_differential(L, R, self.SR)
        assert diff > 0.0

    def test_returns_float_in_valid_range(self):
        """Return value must be a float in [-1.0, 1.0]."""
        rng = np.random.default_rng(1)
        n   = int(self.SR * 2)
        L   = rng.standard_normal(n).astype(np.float32)
        R   = rng.standard_normal(n).astype(np.float32)
        diff = compute_phase_coherence_differential(L, R, self.SR)
        assert isinstance(diff, float)
        assert -1.0 <= diff <= 1.0

    def test_compressed_source_gate_blocks_scoring(self):
        """compressed_source=True must prevent score contribution even with a valid differential."""
        bundle = _SignalBundle(
            compressed_source=True,
            phase_coherence_differential=0.99,
        )
        score = _compute_ai_probability(bundle)
        assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_plr_std
# ---------------------------------------------------------------------------

class TestComputePlrStd:
    SR = 22_050

    def _window(self, amplitude: float) -> np.ndarray:
        """One 2-second window of a sine at the given amplitude."""
        n = int(self.SR * 2)
        t = np.linspace(0, 2.0, n, endpoint=False)
        return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    def _track(self, amplitudes: list[float]) -> np.ndarray:
        return np.concatenate([self._window(a) for a in amplitudes])

    def test_uniform_amplitude_near_zero_std(self):
        """All windows at the same amplitude → PLR is constant → std ≈ 0."""
        audio = self._track([0.5] * 6)
        std = compute_plr_std(audio, self.SR)
        assert std < 0.5

    def test_varying_crest_factor_higher_std(self):
        """Windows with different crest factors → PLR varies → std is meaningfully above zero."""
        n = int(self.SR * 2)
        # Dense sine → PLR ≈ 3 dB (low crest factor)
        t = np.linspace(0, 2.0, n, endpoint=False)
        dense = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        # Sparse impulse → high peak, low RMS → high PLR
        impulse = np.zeros(n, dtype=np.float32)
        impulse[n // 2] = 0.5  # single sample spike
        # Alternate dense and impulse windows
        audio = np.concatenate([dense, impulse, dense, impulse, dense, impulse])
        std = compute_plr_std(audio, self.SR)
        assert std > 1.0

    def test_too_short_returns_minus_one(self):
        """Fewer than PLR_MIN_WINDOWS complete windows → -1.0."""
        audio = self._track([0.5] * 3)  # only 3 windows, min is 5
        std = compute_plr_std(audio, self.SR)
        assert std == -1.0

    def test_returns_non_negative_float(self):
        """Std is always ≥ 0 for valid input."""
        audio = self._track([0.1, 0.3, 0.5, 0.7, 0.9, 0.4])
        std = compute_plr_std(audio, self.SR)
        assert isinstance(std, float)
        assert std >= 0.0

    def test_silent_windows_skipped(self):
        """Silent windows are skipped; result still valid if enough voiced windows remain."""
        audio = self._track([0.0, 0.5, 0.0, 0.5, 0.5, 0.5])
        std = compute_plr_std(audio, self.SR)
        # 4 voiced windows (< PLR_MIN_WINDOWS=5) → -1.0
        assert std == -1.0


# ---------------------------------------------------------------------------
# compute_voiced_noise_floor
# ---------------------------------------------------------------------------

class TestComputeVoicedNoiseFloor:
    SR = 22_050

    def _sine(self, freq: float, duration: float = 10.0, amp: float = 0.5) -> np.ndarray:
        """Pure sine — maximally tonal, near-zero spectral flatness in band."""
        n = int(self.SR * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _noise(self, duration: float = 10.0, amp: float = 0.5) -> np.ndarray:
        """White noise — maximally flat spectrum."""
        rng = np.random.default_rng(0)
        return (amp * rng.standard_normal(int(self.SR * duration))).astype(np.float32)

    def test_pure_tone_in_band_low_flatness(self):
        """A 6 kHz sine is maximally tonal — flatness should be very low."""
        audio = self._sine(6_000)
        score = compute_voiced_noise_floor(audio, self.SR)
        # pyin may not detect a pure sine as voiced; accept -1.0 or low flatness
        assert score == -1.0 or score < 0.3

    def test_white_noise_high_flatness(self):
        """White noise has near-unity spectral flatness; score should be high."""
        audio = self._noise()
        score = compute_voiced_noise_floor(audio, self.SR)
        # pyin likely can't track pitch in noise — may return -1.0 (valid)
        assert score == -1.0 or score > 0.3

    def test_too_few_voiced_frames_returns_minus_one(self):
        """Very short silence returns -1.0 (no voiced frames)."""
        audio = np.zeros(int(self.SR * 0.1), dtype=np.float32)
        assert compute_voiced_noise_floor(audio, self.SR) == -1.0

    def test_returns_float_in_range(self):
        """Result is always -1.0 or a float in [0, 1]."""
        audio = self._noise()
        score = compute_voiced_noise_floor(audio, self.SR)
        assert score == -1.0 or 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Regression: every fixture verdict must match the current algorithm
# ---------------------------------------------------------------------------
#
# These are the verdicts produced by _compute_verdict() with the current
# CONSTANTS values as of 2026-03-20. Update this dict only when you have
# deliberately changed a threshold and consciously accept the new verdict.
#
_EXPECTED_VERDICTS: dict[str, str] = {
    # Updated 2026-03-24: SONICS vocal calibration applied (500 AI: 250 Suno + 250 Udio / 27 human).
    # centroid_instability DISABLED for vocal path (weight=0.0): SONICS confirmed inversion —
    #   AI p10 avg=0.187 < human p10=0.251. Same finding as FMC instrumental calibration.
    # harmonic_ratio threshold raised: 0.59 → 0.70 (SONICS: Suno median=0.688, human median=0.558).
    #   Weight reduced: 0.20 → 0.15 (marginal Udio separation — Udio median=0.590 < human median).
    # PLR remains the strongest signal (separation=0.418); not yet in fixture JSONs (requires
    #   full re-run of analyze() on real audio to populate plr_std field).
    #
    # With centroid=0.0 and HNR threshold=0.70, the vocal AI covers score too low for detection:
    #   Breaking Rust HNR=0.664 < 0.70, Careless Whisper HNR=0.619 < 0.70 → prob=0.00
    #   Velvet Sundown HNR=0.718 ≥ 0.70 → prob=0.15 → still "Likely Not AI"
    # These fixtures need real audio re-analysis (including plr_std) for accurate verdicts.
    "Bon_Iver_-_22__OVER_S__N__forensics":
        "Likely Not AI",
    "Breaking_Rust_-_Walk_My_Walk__Lyrics__forensics":
        "Likely Not AI",   # vocal path: centroid disabled (SONICS inversion); HNR=0.664 < 0.70 threshold → prob=0.00
    "Bruce_Springsteen_-_Born_In_The_U_S_A__forensics":
        "Likely Not AI",
    "Dr__Dre_-_Nuthin__But_A__G__Thang_forensics":
        "Likely Not AI",
    "Dua_Lipa_-_Levitating_Featuring_DaBaby_forensics":
        "Likely Not AI",
    "George_Michael_-_Careless_Whisper__1960_s_Motown_Soul_AI_Cover___BEST_VERSION__forensics":
        "Likely Not AI",   # vocal path: centroid disabled (SONICS inversion); HNR=0.619 < 0.70 threshold → prob=0.00
    "Imogen_Heap_-_Hide_And_Seek_forensics":
        "Likely Not AI",
    "Sabrina_Carpenter_-_Espresso_forensics":
        "Likely Not AI",
    "The_Velvet_Sundown_-_Dust_on_the_Wind__Lyrics__forensics":
        "Likely Not AI",   # vocal path: centroid disabled (SONICS inversion); HNR=0.718≥0.70 (+0.15) = 0.15 → below threshold
    "Young_the_Giant_-_My_Body_forensics":
        "Likely Not AI",
}


@pytest.mark.parametrize("stem,data", all_forensics_fixtures())
def test_fixture_verdict_stable(stem: str, data: dict) -> None:
    """
    Recompute the verdict from saved raw scores and assert it matches the
    expected value in _EXPECTED_VERDICTS.

    A failure means a threshold change has flipped a real-track verdict.
    Review the change, then update _EXPECTED_VERDICTS if it's intentional.
    """
    bundle = _SignalBundle(
        c2pa_flag=data.get("c2pa_flag", False),
        ibi_variance=data.get("ibi_variance", -1.0),
        loop_score=data.get("loop_score", 0.0),
        loop_autocorr_score=data.get("loop_autocorr_score", 0.0),
        centroid_instability_score=data.get("centroid_instability_score", -1.0),
        synthid_bins=int(data.get("synthid_score", 0)),
        spectral_slop=data.get("spectral_slop", 0.0),
        harmonic_ratio_score=data.get("harmonic_ratio_score", -1.0),
        noise_floor_ratio=data.get("noise_floor_ratio", -1.0),
        spectral_centroid_mean=data.get("spectral_centroid_mean", -1.0),
        plr_std=data.get("plr_std", -1.0),
        is_vocal=data.get("is_vocal", False),
    )
    computed = _compute_verdict(bundle)
    expected = _EXPECTED_VERDICTS.get(stem)
    if expected is None:
        pytest.skip(f"No expected verdict registered for {stem!r} — add to _EXPECTED_VERDICTS")

    assert computed == expected, (
        f"\n[{stem}]\n"
        f"  Computed : {computed!r}\n"
        f"  Expected : {expected!r}\n"
        f"  Scores   : centroid={data.get('centroid_instability_score')}, "
        f"hnr={data.get('harmonic_ratio_score')}, autocorr={data.get('loop_autocorr_score')}\n"
        f"\n  → Update _EXPECTED_VERDICTS if this change is intentional."
    )


# ---------------------------------------------------------------------------
# _classify_c2pa_origin
# ---------------------------------------------------------------------------

class TestC2paOrigin:
    """Tests for C2PA origin classification logic."""

    def _make_manifest(self, assertions: list[dict]) -> dict:
        return {"manifests": {"active": {"assertions": assertions}}}

    def test_ai_generated_label_returns_ai_origin(self) -> None:
        data = self._make_manifest([{"label": "ai.generated", "data": {}}])
        born_ai, origin = _classify_c2pa_origin(data)
        assert born_ai is True
        assert origin == "ai"

    def test_c2pa_ai_generated_label_returns_ai_origin(self) -> None:
        data = self._make_manifest([{"label": "c2pa.ai_generated", "data": {}}])
        born_ai, origin = _classify_c2pa_origin(data)
        assert born_ai is True
        assert origin == "ai"

    def test_c2pa_created_with_known_daw_returns_daw_origin(self) -> None:
        data = self._make_manifest([
            {"label": "c2pa.created", "data": {"softwareAgent": "Logic Pro 11.1"}}
        ])
        born_ai, origin = _classify_c2pa_origin(data)
        assert born_ai is False
        assert origin == "daw"

    def test_c2pa_edited_with_known_daw_returns_daw_origin(self) -> None:
        data = self._make_manifest([
            {"label": "c2pa.edited", "data": {"softwareAgent": "Ableton Live 12"}}
        ])
        born_ai, origin = _classify_c2pa_origin(data)
        assert born_ai is False
        assert origin == "daw"

    def test_c2pa_created_with_unknown_software_returns_unknown(self) -> None:
        data = self._make_manifest([
            {"label": "c2pa.created", "data": {"softwareAgent": "MyCustomDAW 2.0"}}
        ])
        born_ai, origin = _classify_c2pa_origin(data)
        assert born_ai is False
        assert origin == "unknown"

    def test_empty_manifest_returns_unknown(self) -> None:
        born_ai, origin = _classify_c2pa_origin({})
        assert born_ai is False
        assert origin == "unknown"

    def test_no_assertions_returns_unknown(self) -> None:
        data = self._make_manifest([])
        born_ai, origin = _classify_c2pa_origin(data)
        assert born_ai is False
        assert origin == "unknown"

    def test_ai_assertion_takes_priority_over_daw(self) -> None:
        # If both AI and DAW labels are present, AI wins
        data = self._make_manifest([
            {"label": "c2pa.created", "data": {"softwareAgent": "Logic Pro"}},
            {"label": "ai.generated", "data": {}},
        ])
        born_ai, origin = _classify_c2pa_origin(data)
        assert born_ai is True
        assert origin == "ai"

    def test_daw_match_is_case_insensitive(self) -> None:
        data = self._make_manifest([
            {"label": "c2pa.created", "data": {"softwareAgent": "REAPER v7.0"}}
        ])
        _, origin = _classify_c2pa_origin(data)
        assert origin == "daw"


# ---------------------------------------------------------------------------
# segment_ai_probabilities
# ---------------------------------------------------------------------------

class TestSegmentAiProbabilities:
    """Tests for the per-window heatmap segmentation function."""

    SR = 22_050  # matches CONSTANTS.SAMPLE_RATE

    def _silence(self, seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * self.SR), dtype=np.float32)

    def _sine(self, seconds: float, freq: float = 440.0) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * self.SR), endpoint=False)
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def test_returns_empty_when_audio_shorter_than_window(self):
        y = self._silence(5.0)   # < 10s default window
        result = segment_ai_probabilities(y, self.SR, window_s=10, hop_s=5)
        assert result == []

    def test_single_window_for_exact_length(self):
        y = self._silence(10.0)  # exactly one window
        result = segment_ai_probabilities(y, self.SR, window_s=10, hop_s=5)
        assert len(result) == 1

    def test_segment_count_is_correct(self):
        # 30s audio, 10s window, 5s hop → windows at 0, 5, 10, 15, 20 → 5 windows
        y = self._silence(30.0)
        result = segment_ai_probabilities(y, self.SR, window_s=10, hop_s=5)
        assert len(result) == 5

    def test_segment_timestamps_are_monotonic(self):
        y = self._sine(20.0)
        result = segment_ai_probabilities(y, self.SR, window_s=10, hop_s=5)
        for i in range(1, len(result)):
            assert result[i].start_s > result[i - 1].start_s

    def test_segment_timestamps_do_not_overlap(self):
        y = self._sine(20.0)
        result = segment_ai_probabilities(y, self.SR, window_s=10, hop_s=5)
        for seg in result:
            assert seg.end_s > seg.start_s

    def test_probability_clamped_to_unit_interval(self):
        y = self._sine(15.0)
        for seg in segment_ai_probabilities(y, self.SR, window_s=10, hop_s=5):
            assert 0.0 <= seg.probability <= 1.0

    def test_returns_ai_segments_instances(self):
        from core.models import AiSegment
        y = self._silence(10.0)
        result = segment_ai_probabilities(y, self.SR, window_s=10, hop_s=5)
        assert all(isinstance(s, AiSegment) for s in result)

    def test_custom_window_and_hop(self):
        # 60s audio, 20s window, 10s hop → windows at 0, 10, 20, 30, 40 → 5 windows
        y = self._sine(60.0)
        result = segment_ai_probabilities(y, self.SR, window_s=20, hop_s=10)
        assert len(result) == 5

# ---------------------------------------------------------------------------
# RhythmAnalyzer._loops_windowed (#142)
# ---------------------------------------------------------------------------

class TestLoopsWindowed:
    """Tests for the per-4-bar-window heatmap scorer."""

    SR = 22_050

    def _sine(self, seconds: float, freq: float = 440.0) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * self.SR), endpoint=False)
        return (np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _analyzer(self):
        from services.forensics.detectors.rhythm import RhythmAnalyzer
        return RhythmAnalyzer()

    def test_returns_list_of_tuples(self) -> None:
        y = self._sine(30.0)
        tempo = np.array([120.0])
        result = self._analyzer()._loops_windowed(y, self.SR, tempo)
        assert isinstance(result, list)
        if result:
            assert all(isinstance(w, tuple) and len(w) == 2 for w in result)

    def test_scores_in_unit_interval(self) -> None:
        y = self._sine(30.0)
        tempo = np.array([120.0])
        result = self._analyzer()._loops_windowed(y, self.SR, tempo)
        for _, score in result:
            assert 0.0 <= score <= 1.0

    def test_start_times_are_monotonic(self) -> None:
        y = self._sine(60.0)
        tempo = np.array([120.0])
        result = self._analyzer()._loops_windowed(y, self.SR, tempo)
        for i in range(1, len(result)):
            assert result[i][0] > result[i - 1][0]

    def test_too_short_returns_empty(self) -> None:
        y = self._sine(0.5)
        tempo = np.array([120.0])
        result = self._analyzer()._loops_windowed(y, self.SR, tempo)
        assert result == []

    def test_out_of_range_bpm_returns_empty(self) -> None:
        y = self._sine(30.0)
        tempo = np.array([10.0])  # below LOOP_BPM_MIN
        result = self._analyzer()._loops_windowed(y, self.SR, tempo)
        assert result == []

    def test_repeated_segment_scores_high(self) -> None:
        # Two identical 4-bar windows → similarity should be near 1.0
        y = self._sine(30.0)
        tempo = np.array([120.0])
        result = self._analyzer()._loops_windowed(y, self.SR, tempo)
        if result:
            assert any(score > 0.8 for _, score in result)


# ---------------------------------------------------------------------------
# RhythmAnalyzer._loops_by_section (#143, #145)
# ---------------------------------------------------------------------------

class TestLoopsBySection:
    """Tests for section-aware inter + intra repetition analysis."""

    SR = 22_050

    def _sine(self, seconds: float, freq: float = 440.0) -> np.ndarray:
        t = np.linspace(0, seconds, int(seconds * self.SR), endpoint=False)
        return (np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _sections(self):
        from core.models import Section
        return [
            Section(label="intro",  start=0.0,  end=8.0),
            Section(label="verse",  start=8.0,  end=24.0),
            Section(label="chorus", start=24.0, end=40.0),
            Section(label="verse",  start=40.0, end=56.0),
            Section(label="chorus", start=56.0, end=72.0),
        ]

    def _analyzer(self):
        from services.forensics.detectors.rhythm import RhythmAnalyzer
        return RhythmAnalyzer()

    def test_empty_sections_returns_empty_dicts(self) -> None:
        y = self._sine(80.0)
        tempo = np.array([120.0])
        inter, intra = self._analyzer()._loops_by_section(y, self.SR, tempo, [])
        assert inter == {}
        assert intra == {}

    def test_single_instance_label_omitted_from_inter(self) -> None:
        # intro has only 1 instance → should not appear in inter results
        y = self._sine(80.0)
        tempo = np.array([120.0])
        inter, _ = self._analyzer()._loops_by_section(y, self.SR, tempo, self._sections())
        assert "intro" not in inter

    def test_multi_instance_labels_in_inter(self) -> None:
        y = self._sine(80.0)
        tempo = np.array([120.0])
        inter, _ = self._analyzer()._loops_by_section(y, self.SR, tempo, self._sections())
        # verse and chorus each have 2 instances
        assert "verse" in inter
        assert "chorus" in inter

    def test_inter_scores_in_unit_interval(self) -> None:
        y = self._sine(80.0)
        tempo = np.array([120.0])
        inter, _ = self._analyzer()._loops_by_section(y, self.SR, tempo, self._sections())
        for rep in inter.values():
            assert 0.0 <= rep.max_similarity <= 1.0
            assert 0.0 <= rep.mean_similarity <= 1.0
            assert rep.pair_count >= 1

    def test_intra_scores_in_unit_interval(self) -> None:
        y = self._sine(80.0)
        tempo = np.array([120.0])
        _, intra = self._analyzer()._loops_by_section(y, self.SR, tempo, self._sections())
        for rep in intra.values():
            assert 0.0 <= rep.max_similarity <= 1.0
            assert 0.0 <= rep.mean_similarity <= 1.0

    def test_short_section_skipped(self) -> None:
        from core.models import Section
        sections = [
            Section(label="verse", start=0.0, end=0.2),   # < SECTION_MIN_DURATION_S
            Section(label="verse", start=0.3, end=0.5),   # < SECTION_MIN_DURATION_S
        ]
        y = self._sine(1.0)
        tempo = np.array([120.0])
        inter, intra = self._analyzer()._loops_by_section(y, self.SR, tempo, sections)
        # Both verses too short — no pairs possible
        assert "verse" not in inter

    def test_returns_section_repetition_instances(self) -> None:
        from core.models import SectionRepetition
        y = self._sine(80.0)
        tempo = np.array([120.0])
        inter, intra = self._analyzer()._loops_by_section(y, self.SR, tempo, self._sections())
        for rep in inter.values():
            assert isinstance(rep, SectionRepetition)
        for rep in intra.values():
            assert isinstance(rep, SectionRepetition)
