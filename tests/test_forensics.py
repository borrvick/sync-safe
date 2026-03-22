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

import numpy as np
import pytest

from core.config import CONSTANTS
from services.forensics import (
    _build_flags,
    _check_spectral_slop,
    _compute_ai_probability,
    _compute_verdict,
    _cross_correlate,
    _spectral_fingerprint,
    _synthid_confidence,
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

def _human_scores() -> dict:
    """Espresso-profile: sample-heavy human production, zero AI signals."""
    return dict(
        c2pa_flag=False,
        ibi_variance=252.0,
        loop_score=0.88,
        loop_autocorr_score=0.93,
        centroid_instability_score=0.196,
        harmonic_ratio_score=0.227,
        synthid_bins=0,
        spectral_slop=0.0,
    )


def _ai_scores() -> dict:
    """Careless Whisper AI cover profile: centroid + HNR both above threshold."""
    return dict(
        c2pa_flag=False,
        ibi_variance=252.0,
        loop_score=0.88,
        loop_autocorr_score=0.75,
        centroid_instability_score=0.364,
        harmonic_ratio_score=0.619,
        synthid_bins=0,
        spectral_slop=0.0,
    )


class TestComputeAiProbability:
    def test_clean_human_scores_zero(self):
        assert _compute_ai_probability(**_human_scores()) == 0.0

    def test_ai_profile_scores_above_hybrid_threshold(self):
        prob = _compute_ai_probability(**_ai_scores())
        assert prob >= CONSTANTS.PROB_VERDICT_HYBRID

    def test_c2pa_flag_does_not_affect_probability(self):
        # C2PA is a hard override handled in _compute_verdict, not here
        base = _compute_ai_probability(**{**_human_scores(), "c2pa_flag": False})
        flagged = _compute_ai_probability(**{**_human_scores(), "c2pa_flag": True})
        assert base == flagged

    def test_score_clamped_to_one(self):
        prob = _compute_ai_probability(
            c2pa_flag=True,
            ibi_variance=0.0,
            loop_score=1.0,
            loop_autocorr_score=1.0,
            centroid_instability_score=0.99,
            harmonic_ratio_score=0.99,
            synthid_bins=99,
            spectral_slop=1.0,
        )
        assert prob <= 1.0

    def test_organic_damping_fires_on_non_repetitive_track(self):
        """Bon Iver profile: elevated centroid but near-zero autocorr should be damped."""
        undamped = _compute_ai_probability(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.40,        # above organic threshold → no damping
            centroid_instability_score=0.41,
            harmonic_ratio_score=0.79,
            synthid_bins=0, spectral_slop=0.0,
        )
        damped = _compute_ai_probability(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.00,        # below organic threshold → damped
            centroid_instability_score=0.41,
            harmonic_ratio_score=0.79,
            synthid_bins=0, spectral_slop=0.0,
        )
        assert damped < undamped

    def test_synthid_medium_adds_weight(self):
        base = _compute_ai_probability(**{**_human_scores(), "synthid_bins": 0})
        with_synthid = _compute_ai_probability(
            **{**_human_scores(), "synthid_bins": CONSTANTS.SYNTHID_LOW_BINS + 1}
        )
        assert with_synthid > base


# ---------------------------------------------------------------------------
# _compute_verdict
# ---------------------------------------------------------------------------

class TestComputeVerdict:
    def test_c2pa_flag_always_returns_ai(self):
        result = _compute_verdict(
            c2pa_flag=True, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.0, centroid_instability_score=0.0,
            synthid_bins=0, spectral_slop=0.0,
        )
        assert result == "AI"

    def test_high_synthid_returns_ai(self):
        result = _compute_verdict(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.0, centroid_instability_score=0.0,
            synthid_bins=CONSTANTS.SYNTHID_MEDIUM_BINS + 1, spectral_slop=0.0,
        )
        assert result == "AI"

    def test_sample_loop_human_profile_returns_human_sample_loop(self):
        result = _compute_verdict(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.88,
            loop_autocorr_score=0.93, centroid_instability_score=0.196,
            synthid_bins=0, spectral_slop=0.0, harmonic_ratio_score=0.227,
        )
        assert result == "Human (Sample/Loop)"

    def test_ai_profile_without_metadata_returns_likely_ai(self):
        result = _compute_verdict(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.88,
            loop_autocorr_score=0.75, centroid_instability_score=0.364,
            synthid_bins=0, spectral_slop=0.0, harmonic_ratio_score=0.619,
        )
        assert result == "Likely AI"

    def test_ai_verdict_requires_hard_evidence(self):
        # "AI" is only returned for C2PA or high SynthID — never from probability alone
        result = _compute_verdict(
            c2pa_flag=False, ibi_variance=0.0, loop_score=1.0,
            loop_autocorr_score=1.0, centroid_instability_score=0.99,
            synthid_bins=0, spectral_slop=1.0, harmonic_ratio_score=0.99,
        )
        assert result != "AI"

    def test_zero_signals_returns_human(self):
        result = _compute_verdict(
            c2pa_flag=False, ibi_variance=252.0, loop_score=0.0,
            loop_autocorr_score=0.0, centroid_instability_score=0.0,
            synthid_bins=0, spectral_slop=0.0,
        )
        assert result == "Human"


# ---------------------------------------------------------------------------
# _build_flags
# ---------------------------------------------------------------------------

class TestBuildFlags:
    _CLEAN = dict(
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
        flags = _build_flags(**self._CLEAN)
        ai_keywords = {"AI signal", "Perfect Quantization", "Formant Drift", "watermark",
                       "Unnaturally Clean"}
        for flag in flags:
            assert not any(kw in flag for kw in ai_keywords), f"Unexpected AI flag: {flag!r}"

    def test_c2pa_born_ai_label_appears_in_flags(self):
        flags = _build_flags(**{**self._CLEAN, "c2pa_label": "Born-AI (Certified)"})
        assert any("Born-AI" in f for f in flags)

    def test_high_synthid_generates_high_confidence_flag(self):
        flags = _build_flags(**{**self._CLEAN, "synthid_bins": CONSTANTS.SYNTHID_MEDIUM_BINS + 1})
        assert any("High-confidence" in f for f in flags)

    def test_perfect_quantization_on_near_zero_ibi(self):
        flags = _build_flags(**{**self._CLEAN, "ibi_variance": 0.0001})
        assert any("Perfect Quantization" in f for f in flags)

    def test_human_feel_flag_on_high_ibi(self):
        flags = _build_flags(**{**self._CLEAN, "ibi_variance": 200.0})
        assert any("Human-Feel" in f for f in flags)

    def test_loop_ceiling_flag_on_high_loop_score(self):
        flags = _build_flags(**{**self._CLEAN, "loop_score": 0.99})
        assert any("Stock Loop" in f or "Likely Stock" in f for f in flags)

    def test_formant_drift_flag_on_high_centroid(self):
        flags = _build_flags(**{**self._CLEAN, "centroid_instability_score": 0.40,
                                "loop_autocorr_score": 0.75})
        assert any("Formant Drift" in f for f in flags)

    def test_vocoder_flag_on_very_high_centroid(self):
        flags = _build_flags(**{**self._CLEAN, "centroid_instability_score": 0.70})
        assert any("Vocoder" in f or "Extreme Spectral" in f for f in flags)

    def test_harmonic_ratio_flag_on_high_hnr(self):
        flags = _build_flags(**{**self._CLEAN, "harmonic_ratio_score": 0.70})
        assert any("Clean Harmonics" in f or "Unnaturally" in f for f in flags)


# ---------------------------------------------------------------------------
# Regression: every fixture verdict must match the current algorithm
# ---------------------------------------------------------------------------
#
# These are the verdicts produced by _compute_verdict() with the current
# CONSTANTS values as of 2026-03-20. Update this dict only when you have
# deliberately changed a threshold and consciously accept the new verdict.
#
_EXPECTED_VERDICTS: dict[str, str] = {
    "Bon_Iver_-_22__OVER_S__N__forensics":
        "Uncertain",
    "Breaking_Rust_-_Walk_My_Walk__Lyrics__forensics":
        "Likely AI",
    "Bruce_Springsteen_-_Born_In_The_U_S_A__forensics":
        "Human",
    "Dr__Dre_-_Nuthin__But_A__G__Thang_forensics":
        "Human (Sample/Loop)",
    "Dua_Lipa_-_Levitating_Featuring_DaBaby_forensics":
        "Human (Sample/Loop)",
    "George_Michael_-_Careless_Whisper__1960_s_Motown_Soul_AI_Cover___BEST_VERSION__forensics":
        "Likely AI",
    # Intentionally updated 2026-03-21: centroid=0.677 is in vocoder territory
    # (>= CENTROID_INSTABILITY_VOCODER_MIN 0.50) — extreme DSP processing, not AI.
    # Centroid no longer contributes to ai_probability above VOCODER_MIN.
    # "Human" is correct: Hide and Seek is a famously processed but fully human track.
    "Imogen_Heap_-_Hide_And_Seek_forensics":
        "Human",
    "Sabrina_Carpenter_-_Espresso_forensics":
        "Human (Sample/Loop)",
    "The_Velvet_Sundown_-_Dust_on_the_Wind__Lyrics__forensics":
        "Likely AI",
    "Young_the_Giant_-_My_Body_forensics":
        "Human (Sample/Loop)",
}


@pytest.mark.parametrize("stem,data", all_forensics_fixtures())
def test_fixture_verdict_stable(stem: str, data: dict) -> None:
    """
    Recompute the verdict from saved raw scores and assert it matches the
    expected value in _EXPECTED_VERDICTS.

    A failure means a threshold change has flipped a real-track verdict.
    Review the change, then update _EXPECTED_VERDICTS if it's intentional.
    """
    computed = _compute_verdict(
        c2pa_flag=data.get("c2pa_flag", False),
        ibi_variance=data.get("ibi_variance", -1.0),
        loop_score=data.get("loop_score", 0.0),
        loop_autocorr_score=data.get("loop_autocorr_score", 0.0),
        centroid_instability_score=data.get("centroid_instability_score", -1.0),
        synthid_bins=int(data.get("synthid_score", 0)),
        spectral_slop=data.get("spectral_slop", 0.0),
        harmonic_ratio_score=data.get("harmonic_ratio_score", -1.0),
    )
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
