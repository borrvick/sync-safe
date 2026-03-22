"""
tests/test_analysis.py

Unit tests for services/analysis.py pure functions.

Covers:
  - _merge_consecutive_sections — section collapse logic

No GPU, no allin1, no audio — all tests complete in < 1 second.
"""
from __future__ import annotations

import pytest

from core.models import Section
from services.analysis import _merge_consecutive_sections


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
