"""
tests/test_repetition_ui_helpers.py

Unit tests for the three pure UI helpers introduced in issues #134, #141, #144:
  - _repetition_index_label(ri)   → (label, color)
  - _sync_editability_badge(ri)   → (label, detail, color) | None
  - _build_section_stats(sections) → str
"""
from __future__ import annotations

import pytest

from core.config import CONSTANTS
from core.models import Section
from ui.pages.report import (
    _build_section_stats,
    _repetition_index_label,
    _sync_editability_badge,
)

# ---------------------------------------------------------------------------
# _repetition_index_label
# ---------------------------------------------------------------------------

class TestRepetitionIndexLabel:
    def test_none_returns_dash_and_dim(self):
        label, color = _repetition_index_label(None)
        assert label == "—"
        assert color == "var(--dim)"

    def test_high_threshold_exact(self):
        label, color = _repetition_index_label(CONSTANTS.REPETITION_INDEX_HIGH)
        assert label == "High"
        assert color == "var(--danger)"

    def test_above_high(self):
        label, color = _repetition_index_label(1.0)
        assert label == "High"

    def test_moderate_threshold_exact(self):
        label, color = _repetition_index_label(CONSTANTS.REPETITION_INDEX_MODERATE)
        assert label == "Moderate"
        assert color == "var(--grade-c)"

    def test_between_moderate_and_high(self):
        mid = (CONSTANTS.REPETITION_INDEX_MODERATE + CONSTANTS.REPETITION_INDEX_HIGH) / 2
        label, color = _repetition_index_label(mid)
        assert label == "Moderate"

    def test_below_moderate(self):
        label, color = _repetition_index_label(CONSTANTS.REPETITION_INDEX_MODERATE - 0.01)
        assert label == "Low"
        assert color == "var(--grade-b)"

    def test_zero(self):
        label, _ = _repetition_index_label(0.0)
        assert label == "Low"


# ---------------------------------------------------------------------------
# _sync_editability_badge
# ---------------------------------------------------------------------------

class TestSyncEditabilityBadge:
    def test_none_returns_none(self):
        assert _sync_editability_badge(None) is None

    def test_high_is_loop_friendly(self):
        result = _sync_editability_badge(CONSTANTS.REPETITION_INDEX_HIGH)
        assert result is not None
        label, detail, color = result
        assert label == "Loop-friendly"
        assert "ad spots" in detail
        assert color == "var(--accent)"

    def test_above_high(self):
        result = _sync_editability_badge(1.0)
        assert result is not None
        assert result[0] == "Loop-friendly"

    def test_moderate_threshold_exact(self):
        result = _sync_editability_badge(CONSTANTS.REPETITION_INDEX_MODERATE)
        assert result is not None
        label, detail, color = result
        assert label == "Moderate Loop Structure"
        assert color == "var(--grade-c)"

    def test_between_moderate_and_high(self):
        mid = (CONSTANTS.REPETITION_INDEX_MODERATE + CONSTANTS.REPETITION_INDEX_HIGH) / 2
        result = _sync_editability_badge(mid)
        assert result is not None
        assert result[0] == "Moderate Loop Structure"

    def test_low(self):
        result = _sync_editability_badge(CONSTANTS.REPETITION_INDEX_MODERATE - 0.01)
        assert result is not None
        label, detail, color = result
        assert label == "Organic Flow"
        assert "narrative" in detail
        assert color == "var(--grade-b)"

    def test_zero(self):
        result = _sync_editability_badge(0.0)
        assert result is not None
        assert result[0] == "Organic Flow"

    def test_returns_three_tuple(self):
        result = _sync_editability_badge(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _build_section_stats
# ---------------------------------------------------------------------------

def _sec(label: str, start: float, end: float) -> Section:
    return Section(label=label, start=start, end=end)


class TestBuildSectionStats:
    def test_empty_returns_empty_string(self):
        assert _build_section_stats([]) == ""

    def test_zero_duration_returns_empty_string(self):
        # start == end → zero-duration section
        assert _build_section_stats([_sec("intro", 0.0, 0.0)]) == ""

    def test_section_count_singular(self):
        stats = _build_section_stats([_sec("verse", 0.0, 30.0)])
        assert "1 section" in stats

    def test_section_count_plural(self):
        sections = [_sec("verse", 0.0, 30.0), _sec("chorus", 30.0, 60.0)]
        stats = _build_section_stats(sections)
        assert "2 sections" in stats

    def test_chorus_percentage_included(self):
        sections = [
            _sec("verse",  0.0,  60.0),   # 60s
            _sec("chorus", 60.0, 120.0),  # 60s → 50 % chorus
        ]
        stats = _build_section_stats(sections)
        assert "50% chorus" in stats

    def test_hook_counts_as_chorus(self):
        sections = [_sec("hook", 0.0, 40.0), _sec("verse", 40.0, 60.0)]
        stats = _build_section_stats(sections)
        assert "chorus" in stats

    def test_instrumental_percentage_included(self):
        sections = [
            _sec("verse",        0.0,  60.0),
            _sec("instrumental", 60.0, 90.0),  # 30s of 90s total → 33%
        ]
        stats = _build_section_stats(sections)
        assert "instrumental" in stats

    def test_intro_counts_as_instrumental(self):
        sections = [_sec("intro", 0.0, 30.0), _sec("verse", 30.0, 60.0)]
        stats = _build_section_stats(sections)
        assert "instrumental" in stats

    def test_avg_section_length_included(self):
        # Two 30-second sections → avg 30s
        sections = [_sec("verse", 0.0, 30.0), _sec("chorus", 30.0, 60.0)]
        stats = _build_section_stats(sections)
        assert "Avg 30s/section" in stats

    def test_no_chorus_omits_chorus_part(self):
        sections = [_sec("verse", 0.0, 30.0), _sec("bridge", 30.0, 60.0)]
        stats = _build_section_stats(sections)
        assert "chorus" not in stats

    def test_separator_between_parts(self):
        sections = [
            _sec("chorus", 0.0, 60.0),
            _sec("verse",  60.0, 120.0),
        ]
        stats = _build_section_stats(sections)
        assert " · " in stats
