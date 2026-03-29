"""
tests/test_loudness.py
Unit tests for services/loudness.py — pure helper functions only.
AudioQualityAnalyzer.analyze() requires audio I/O and is tested manually.
"""
from __future__ import annotations

import numpy as np

from core.config import CONSTANTS
from services.loudness import _classify_dialogue, _dialogue_score


# ---------------------------------------------------------------------------
# _classify_dialogue (pure — no I/O)
# ---------------------------------------------------------------------------

class TestClassifyDialogue:
    def test_dialogue_ready(self) -> None:
        assert _classify_dialogue(CONSTANTS.DIALOGUE_READY_HIGH) == "Dialogue-Ready"

    def test_dialogue_ready_above_threshold(self) -> None:
        assert _classify_dialogue(1.0) == "Dialogue-Ready"

    def test_mixed_at_lower_boundary(self) -> None:
        assert _classify_dialogue(CONSTANTS.DIALOGUE_READY_LOW) == "Mixed"

    def test_mixed_midpoint(self) -> None:
        mid = (CONSTANTS.DIALOGUE_READY_HIGH + CONSTANTS.DIALOGUE_READY_LOW) / 2
        assert _classify_dialogue(mid) == "Mixed"

    def test_dialogue_heavy_below_threshold(self) -> None:
        assert _classify_dialogue(CONSTANTS.DIALOGUE_READY_LOW - 0.01) == "Dialogue-Heavy"

    def test_dialogue_heavy_at_zero(self) -> None:
        assert _classify_dialogue(0.0) == "Dialogue-Heavy"


# ---------------------------------------------------------------------------
# _dialogue_score (pure — uses numpy, no file I/O)
# ---------------------------------------------------------------------------

class TestDialogueScore:
    _sr = 22050

    def test_score_in_range(self) -> None:
        """Score must always be between 0 and 1."""
        rng = np.random.default_rng(42)
        y   = rng.standard_normal(self._sr * 5).astype(np.float32)
        score = _dialogue_score(y, self._sr)
        assert 0.0 <= score <= 1.0

    def test_silent_track_returns_neutral(self) -> None:
        """Silent audio should return the neutral score (0.5)."""
        y = np.zeros(self._sr * 3, dtype=np.float32)
        assert _dialogue_score(y, self._sr) == 0.5

    def test_sine_440hz_has_low_score(self) -> None:
        """440 Hz sine wave is inside the dialogue band → low score."""
        t = np.linspace(0, 3, self._sr * 3)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        score = _dialogue_score(y, self._sr)
        # 440 Hz sits squarely in the dialogue band — score should be low
        assert score < 0.5

    def test_high_freq_sine_has_high_score(self) -> None:
        """8 kHz sine wave is outside the dialogue band → high score."""
        t = np.linspace(0, 3, self._sr * 3)
        y = np.sin(2 * np.pi * 8000 * t).astype(np.float32)
        score = _dialogue_score(y, self._sr)
        # 8 kHz is well outside the 300–3000 Hz competition band
        assert score > 0.5
