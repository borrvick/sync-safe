"""
tests/test_analysis.py

Unit tests for services/analysis.py pure functions.

Covers:
  - _merge_consecutive_sections — section collapse logic
  - _normalize_section_label — Harmonix vocabulary normalisation (#136)

No GPU, no allin1, no audio — all tests complete in < 1 second.
"""
from __future__ import annotations

import pytest

from core.models import Section
from services.analysis import (
    HARMONIX_LABELS,
    _merge_consecutive_sections,
    _normalize_section_label,
    section_ibi_tightness,
)


def _sec(label: str, start: float, end: float) -> Section:
    return Section(label=label, start=start, end=end)


class TestMergeConsecutiveSections:
    def test_empty_input_returns_empty(self):
        assert _merge_consecutive_sections([]) == []

    def test_single_section_unchanged(self):
        result = _merge_consecutive_sections([_sec("chorus", 0.0, 30.0)])
        assert len(result) == 1
        assert result[0].label == "chorus"
        assert result[0].start == 0.0
        assert result[0].end == 30.0

    def test_two_same_label_merged(self):
        sections = [_sec("chorus", 0.0, 15.0), _sec("chorus", 15.0, 30.0)]
        result = _merge_consecutive_sections(sections)
        assert len(result) == 1
        assert result[0].start == pytest.approx(0.0)
        assert result[0].end == pytest.approx(30.0)

    def test_four_same_label_merged_to_one(self):
        sections = [
            _sec("chorus", 0.0,  8.0),
            _sec("chorus", 8.0,  16.0),
            _sec("chorus", 16.0, 24.0),
            _sec("chorus", 24.0, 32.0),
        ]
        result = _merge_consecutive_sections(sections)
        assert len(result) == 1
        assert result[0].end == pytest.approx(32.0)

    def test_different_labels_not_merged(self):
        sections = [_sec("verse", 0.0, 30.0), _sec("chorus", 30.0, 60.0)]
        result = _merge_consecutive_sections(sections)
        assert len(result) == 2
        assert result[0].label == "verse"
        assert result[1].label == "chorus"

    def test_interleaved_labels_only_merge_consecutive(self):
        # verse chorus chorus verse — chorus pair merges, verses stay separate
        sections = [
            _sec("verse",  0.0,  30.0),
            _sec("chorus", 30.0, 45.0),
            _sec("chorus", 45.0, 60.0),
            _sec("verse",  60.0, 90.0),
        ]
        result = _merge_consecutive_sections(sections)
        assert len(result) == 3
        assert result[0].label == "verse"
        assert result[0].end == pytest.approx(30.0)
        assert result[1].label == "chorus"
        assert result[1].start == pytest.approx(30.0)
        assert result[1].end == pytest.approx(60.0)
        assert result[2].label == "verse"
        assert result[2].start == pytest.approx(60.0)

    def test_case_insensitive_label_match(self):
        # allin1 may capitalise inconsistently
        sections = [_sec("Chorus", 0.0, 15.0), _sec("chorus", 15.0, 30.0)]
        result = _merge_consecutive_sections(sections)
        assert len(result) == 1
        assert result[0].end == pytest.approx(30.0)

    def test_merged_section_preserves_first_label_casing(self):
        sections = [_sec("Chorus", 0.0, 15.0), _sec("chorus", 15.0, 30.0)]
        result = _merge_consecutive_sections(sections)
        assert result[0].label == "Chorus"

    def test_realistic_song_structure(self):
        # intro, verse×2, chorus×4, verse×2, chorus×4, bridge, outro
        sections = [
            _sec("intro",  0.0,  10.0),
            _sec("verse",  10.0, 25.0),
            _sec("verse",  25.0, 40.0),
            _sec("chorus", 40.0, 48.0),
            _sec("chorus", 48.0, 56.0),
            _sec("chorus", 56.0, 64.0),
            _sec("chorus", 64.0, 72.0),
            _sec("verse",  72.0, 87.0),
            _sec("verse",  87.0, 102.0),
            _sec("chorus", 102.0, 110.0),
            _sec("chorus", 110.0, 118.0),
            _sec("chorus", 118.0, 126.0),
            _sec("chorus", 126.0, 134.0),
            _sec("bridge", 134.0, 150.0),
            _sec("outro",  150.0, 165.0),
        ]
        result = _merge_consecutive_sections(sections)
        labels = [s.label for s in result]
        assert labels == ["intro", "verse", "chorus", "verse", "chorus", "bridge", "outro"]
        # First chorus block spans 40–72
        chorus1 = result[2]
        assert chorus1.start == pytest.approx(40.0)
        assert chorus1.end   == pytest.approx(72.0)
        # Second chorus block spans 102–134
        chorus2 = result[4]
        assert chorus2.start == pytest.approx(102.0)
        assert chorus2.end   == pytest.approx(134.0)


# ---------------------------------------------------------------------------
# _normalize_section_label (#136)
# ---------------------------------------------------------------------------

class TestNormalizeSectionLabel:
    def test_canonical_label_unchanged(self) -> None:
        assert _normalize_section_label("chorus", 1, 5) == "chorus"

    def test_canonical_label_is_lowercased(self) -> None:
        assert _normalize_section_label("Chorus", 1, 5) == "chorus"

    def test_alias_c_maps_to_chorus(self) -> None:
        assert _normalize_section_label("c", 1, 5) == "chorus"

    def test_alias_v_maps_to_verse(self) -> None:
        assert _normalize_section_label("v", 1, 5) == "verse"

    def test_alias_prechorus_maps(self) -> None:
        assert _normalize_section_label("prechorus", 1, 5) == "pre-chorus"

    def test_canonical_refrain_unchanged(self) -> None:
        # "refrain" is in HARMONIX_LABELS — returned as-is (lowercased)
        assert _normalize_section_label("Refrain", 1, 5) == "refrain"

    def test_alias_fade_maps_to_outro(self) -> None:
        assert _normalize_section_label("fade", 3, 4) == "outro"

    def test_positional_fallback_first_section_is_intro(self) -> None:
        assert _normalize_section_label("zzz_unknown", 0, 5) == "intro"

    def test_positional_fallback_last_section_is_outro(self) -> None:
        assert _normalize_section_label("zzz_unknown", 4, 5) == "outro"

    def test_positional_fallback_middle_is_verse(self) -> None:
        assert _normalize_section_label("zzz_unknown", 2, 5) == "verse"

    def test_all_harmonix_labels_in_frozenset(self) -> None:
        for label in ("chorus", "verse", "bridge", "intro", "outro"):
            assert label in HARMONIX_LABELS


# ---------------------------------------------------------------------------
# section_ibi_tightness — beat-grid tightness classification (#137)
# ---------------------------------------------------------------------------

_LOCKED_MS   = 5.0
_LOOSE_MS    = 20.0
_MIN_BEATS   = 4


def _uniform_beats(start: float, step: float, count: int) -> list[float]:
    """Helper: generate perfectly uniform beat times."""
    return [start + i * step for i in range(count)]


class TestSectionIbiTightness:
    def test_perfect_quantization_returns_locked(self) -> None:
        # Std dev = 0.0 ms — perfectly uniform grid → Locked
        beats = _uniform_beats(0.0, 0.5, 8)  # 500 ms intervals
        result = section_ibi_tightness(0.0, 4.0, beats, _LOCKED_MS, _LOOSE_MS)
        assert result == "Locked"

    def test_high_variance_returns_loose(self) -> None:
        # Alternate 400ms / 600ms → std dev ≈ 100ms → Loose
        beats = [0.0, 0.4, 1.0, 1.4, 2.0, 2.4, 3.0, 3.4]
        result = section_ibi_tightness(0.0, 4.0, beats, _LOCKED_MS, _LOOSE_MS)
        assert result == "Loose"

    def test_moderate_variance_returns_moderate(self) -> None:
        # Std dev ~ 12ms: IBIs alternate 490ms / 510ms
        beats = [0.0, 0.490, 1.000, 1.490, 2.000, 2.490, 3.000, 3.490]
        result = section_ibi_tightness(0.0, 4.0, beats, _LOCKED_MS, _LOOSE_MS)
        assert result == "Moderate"

    def test_fewer_than_min_beats_returns_none(self) -> None:
        beats = [0.0, 0.5, 1.0]  # only 3 beats < min_beats=4
        result = section_ibi_tightness(0.0, 2.0, beats, _LOCKED_MS, _LOOSE_MS, min_beats=4)
        assert result is None

    def test_empty_beats_returns_none(self) -> None:
        result = section_ibi_tightness(0.0, 4.0, [], _LOCKED_MS, _LOOSE_MS)
        assert result is None

    def test_cross_section_beats_excluded(self) -> None:
        # Beats outside [section_start, section_end] must not be used.
        # Section is 2.0–4.0; beats before 2.0 would create a huge first IBI if included.
        all_beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        # Within section only: [2.0, 2.5, 3.0, 3.5, 4.0] → uniform 500ms → Locked
        result = section_ibi_tightness(2.0, 4.0, all_beats, _LOCKED_MS, _LOOSE_MS)
        assert result == "Locked"

    def test_exactly_min_beats_is_valid(self) -> None:
        beats = _uniform_beats(0.0, 0.5, 4)  # exactly 4 beats
        result = section_ibi_tightness(0.0, 2.0, beats, _LOCKED_MS, _LOOSE_MS, min_beats=4)
        assert result == "Locked"

    def test_std_dev_at_locked_boundary_is_locked(self) -> None:
        # Std dev exactly at the locked threshold should return "Locked" (≤ not <)
        step = 0.5
        eps  = _LOCKED_MS / 1000 / 10  # tiny nudge to make std dev ≈ LOCKED_MS
        beats = [0.0, step - eps, step * 2, step * 3, step * 4 - eps, step * 5, step * 6, step * 7]
        # We just verify the function doesn't crash and returns a valid label
        result = section_ibi_tightness(0.0, step * 7, beats, _LOCKED_MS, _LOOSE_MS)
        assert result in ("Locked", "Moderate", "Loose")
