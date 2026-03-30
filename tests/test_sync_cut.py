"""
tests/test_sync_cut.py
Unit tests for services/sync_cut.py.

All tests use synthetic beat grids and sections — no audio I/O.
"""
from __future__ import annotations

import pytest

from core.models import Section, SyncCut
from services.sync_cut import (
    SyncCutAnalyzer,
    _build_note,
    _contains_chorus,
    _intro_end,
    _near_boundary,
    _score_window,
    _section_at,
    _snap_to_bar,
    suggest_sync_cuts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _beats(n: int, bpm: float = 120.0) -> list[float]:
    """Generate n evenly-spaced beats at *bpm*."""
    interval = 60.0 / bpm
    return [round(i * interval, 4) for i in range(n)]


def _sections() -> list[Section]:
    return [
        Section(label="intro",  start=0.0,  end=8.0),
        Section(label="verse",  start=8.0,  end=24.0),
        Section(label="chorus", start=24.0, end=40.0),
        Section(label="verse",  start=40.0, end=56.0),
        Section(label="chorus", start=56.0, end=72.0),
        Section(label="outro",  start=72.0, end=80.0),
    ]


# ---------------------------------------------------------------------------
# _section_at
# ---------------------------------------------------------------------------

class TestSectionAt:
    def test_returns_label_for_contained_time(self) -> None:
        assert _section_at(10.0, _sections()) == "verse"

    def test_returns_label_at_section_start(self) -> None:
        assert _section_at(24.0, _sections()) == "chorus"

    def test_returns_none_past_end(self) -> None:
        assert _section_at(999.0, _sections()) is None

    def test_returns_none_empty_sections(self) -> None:
        assert _section_at(5.0, []) is None


# ---------------------------------------------------------------------------
# _intro_end
# ---------------------------------------------------------------------------

class TestIntroEnd:
    def test_returns_intro_end_time(self) -> None:
        assert _intro_end(_sections()) == 8.0

    def test_returns_zero_when_no_intro(self) -> None:
        secs = [Section(label="verse", start=0.0, end=16.0)]
        assert _intro_end(secs) == 0.0

    def test_returns_last_intro_end_for_multiple(self) -> None:
        secs = [
            Section(label="intro", start=0.0,  end=4.0),
            Section(label="intro", start=4.0,  end=10.0),
            Section(label="verse", start=10.0, end=30.0),
        ]
        assert _intro_end(secs) == 10.0


# ---------------------------------------------------------------------------
# _near_boundary
# ---------------------------------------------------------------------------

class TestNearBoundary:
    def test_at_section_start(self) -> None:
        assert _near_boundary(8.0, _sections(), tolerance=0.5) is True

    def test_at_section_end(self) -> None:
        assert _near_boundary(24.0, _sections(), tolerance=0.5) is True

    def test_within_tolerance(self) -> None:
        assert _near_boundary(8.3, _sections(), tolerance=0.5) is True

    def test_outside_tolerance(self) -> None:
        assert _near_boundary(10.0, _sections(), tolerance=0.5) is False

    def test_empty_sections(self) -> None:
        assert _near_boundary(5.0, [], tolerance=1.0) is False


# ---------------------------------------------------------------------------
# _contains_chorus
# ---------------------------------------------------------------------------

class TestContainsChorus:
    def test_window_fully_within_chorus(self) -> None:
        assert _contains_chorus(25.0, 35.0, _sections()) is True

    def test_window_overlapping_chorus(self) -> None:
        assert _contains_chorus(20.0, 28.0, _sections()) is True

    def test_window_without_chorus(self) -> None:
        assert _contains_chorus(8.0, 20.0, _sections()) is False

    def test_empty_sections(self) -> None:
        assert _contains_chorus(0.0, 40.0, []) is False


# ---------------------------------------------------------------------------
# _snap_to_bar
# ---------------------------------------------------------------------------

class TestSnapToBar:
    def test_already_on_bar(self) -> None:
        beats = list(range(20))
        assert _snap_to_bar(8, beats, snap_bars=4) == 8

    def test_snaps_forward(self) -> None:
        beats = list(range(20))
        # beat 9 is 1 ahead of bar (bar at 8 and 12); 12 is closer than 8
        assert _snap_to_bar(9, beats, snap_bars=4) == 8

    def test_snaps_backward(self) -> None:
        beats = list(range(20))
        # beat 11 is 1 before bar at 12; prefer 12 (forward) — equidistant picks fwd
        result = _snap_to_bar(11, beats, snap_bars=4)
        assert result in (8, 12)

    def test_clamps_to_zero(self) -> None:
        beats = [0.0, 0.5, 1.0, 1.5]
        assert _snap_to_bar(0, beats, snap_bars=4) == 0

    def test_clamps_to_last(self) -> None:
        beats = list(range(5))
        result = _snap_to_bar(4, beats, snap_bars=4)
        assert result <= len(beats) - 1


# ---------------------------------------------------------------------------
# _score_window
# ---------------------------------------------------------------------------

class TestScoreWindow:
    def test_perfect_window_scores_high(self) -> None:
        # Start at chorus (post-intro + at boundary), end at chorus end (boundary)
        # chorus: 24.0 – 40.0; 30s window → 24.0 – 54.0 (crosses boundary at 40.0)
        beats = _beats(200)
        score = _score_window(
            start_s=24.0, end_s=54.0,
            sections=_sections(), beats=beats,
            snap_bars=4, boundary_tolerance=0.5, intro_end_s=8.0,
        )
        assert score >= 0.6

    def test_intro_start_penalised(self) -> None:
        beats = _beats(200)
        score = _score_window(
            start_s=2.0, end_s=32.0,
            sections=_sections(), beats=beats,
            snap_bars=4, boundary_tolerance=0.5, intro_end_s=8.0,
        )
        # Should lose the +0.20 for post-intro start
        assert score <= 0.8

    def test_score_zero_to_one_range(self) -> None:
        beats = _beats(200)
        score = _score_window(
            start_s=0.0, end_s=15.0,
            sections=_sections(), beats=beats,
            snap_bars=4, boundary_tolerance=0.5, intro_end_s=8.0,
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# suggest_sync_cuts — pure function
# ---------------------------------------------------------------------------

class TestSuggestSyncCuts:
    def test_returns_cut_for_each_valid_duration(self) -> None:
        beats = _beats(300)  # 150 seconds at 120 BPM
        cuts  = suggest_sync_cuts(
            sections=_sections(), beats=beats,
            target_durations=[15, 30, 60],
            snap_bars=4, boundary_tolerance=0.5, duration_tolerance=3.0,
        )
        durations = {c.duration_s for c in cuts}
        assert 15 in durations
        assert 30 in durations
        assert 60 in durations

    def test_skips_duration_longer_than_track(self) -> None:
        beats = _beats(40)   # ~20 seconds
        cuts  = suggest_sync_cuts(
            sections=[], beats=beats,
            target_durations=[30, 60],
            snap_bars=4, boundary_tolerance=0.5, duration_tolerance=3.0,
        )
        assert all(c.duration_s < 60 for c in cuts)

    def test_empty_beats_returns_empty(self) -> None:
        cuts = suggest_sync_cuts(
            sections=_sections(), beats=[],
            target_durations=[15, 30, 60],
            snap_bars=4, boundary_tolerance=0.5, duration_tolerance=3.0,
        )
        assert cuts == []

    def test_cut_fields_are_populated(self) -> None:
        beats = _beats(200)
        cuts  = suggest_sync_cuts(
            sections=_sections(), beats=beats,
            target_durations=[30],
            snap_bars=4, boundary_tolerance=0.5, duration_tolerance=3.0,
        )
        assert len(cuts) == 1
        cut = cuts[0]
        assert cut.duration_s == 30
        assert cut.start_s >= 0.0
        assert cut.end_s > cut.start_s
        assert 0.0 <= cut.confidence <= 1.0
        assert cut.note != ""

    def test_actual_duration_within_tolerance(self) -> None:
        beats = _beats(200)
        cuts  = suggest_sync_cuts(
            sections=_sections(), beats=beats,
            target_durations=[30],
            snap_bars=4, boundary_tolerance=0.5, duration_tolerance=3.0,
        )
        for cut in cuts:
            assert abs(cut.actual_duration_s - cut.duration_s) <= 3.0


# ---------------------------------------------------------------------------
# SyncCutAnalyzer (integration smoke test)
# ---------------------------------------------------------------------------

class TestSyncCutAnalyzer:
    def test_returns_list_of_sync_cuts(self) -> None:
        beats = _beats(300)
        cuts  = SyncCutAnalyzer().suggest(
            sections=_sections(), beats=beats,
            target_durations=[15, 30, 60],
        )
        assert isinstance(cuts, list)
        assert all(isinstance(c, SyncCut) for c in cuts)

    def test_uses_constants_target_durations(self) -> None:
        from core.config import CONSTANTS
        beats = _beats(300)
        cuts  = SyncCutAnalyzer().suggest(
            sections=_sections(), beats=beats,
            target_durations=list(CONSTANTS.SYNC_CUT_TARGET_DURATIONS),
        )
        returned_durations = {c.duration_s for c in cuts}
        for t in CONSTANTS.SYNC_CUT_TARGET_DURATIONS:
            if beats[-1] >= t - CONSTANTS.SYNC_CUT_DURATION_TOLERANCE_S:
                assert t in returned_durations
