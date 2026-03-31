"""tests/test_stem_validator.py — unit tests for services/stem_validator.py pure functions."""
from __future__ import annotations

import numpy as np
import pytest

from services.stem_validator import (
    compute_cancellation_db,
    compute_mid_side_ratio,
    compute_phase_correlation,
)


class TestPhaseCorrelation:
    def test_identical_channels_is_one(self):
        x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        assert abs(compute_phase_correlation(x, x) - 1.0) < 1e-6

    def test_inverted_channels_is_minus_one(self):
        x = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
        assert abs(compute_phase_correlation(x, -x) - (-1.0)) < 1e-6

    def test_empty_returns_sentinel(self):
        assert compute_phase_correlation(np.array([]), np.array([])) == -1.0

    def test_mismatched_length_returns_sentinel(self):
        a = np.ones(100, dtype=np.float32)
        b = np.ones(200, dtype=np.float32)
        assert compute_phase_correlation(a, b) == -1.0

    def test_uncorrelated_near_zero(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(10000).astype(np.float32)
        b = rng.standard_normal(10000).astype(np.float32)
        assert abs(compute_phase_correlation(a, b)) < 0.1


class TestCancellationDb:
    def test_identical_channels_zero_cancellation(self):
        x = np.ones(1000, dtype=np.float32) * 0.5
        result = compute_cancellation_db(x, x)
        assert abs(result) < 0.1  # ≈ 0 dB

    def test_anti_phase_large_negative(self):
        x = np.ones(1000, dtype=np.float32) * 0.5
        result = compute_cancellation_db(x, -x)
        assert result < -60.0  # near -inf

    def test_silent_returns_zero(self):
        z = np.zeros(1000, dtype=np.float32)
        assert compute_cancellation_db(z, z) == 0.0


class TestMidSideRatio:
    def test_mono_mix_zero_side(self):
        x = np.ones(1000, dtype=np.float32)
        assert abs(compute_mid_side_ratio(x, x)) < 0.01

    def test_anti_phase_returns_sentinel(self):
        # Fully anti-phase: mid = (x - x)/2 = 0, mid_rms = 0 → sentinel -1.0
        x = np.ones(1000, dtype=np.float32)
        ratio = compute_mid_side_ratio(x, -x)
        assert ratio == -1.0

    def test_silence_returns_sentinel(self):
        z = np.zeros(1000, dtype=np.float32)
        assert compute_mid_side_ratio(z, z) == -1.0
