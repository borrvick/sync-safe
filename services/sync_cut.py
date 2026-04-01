"""
services/sync_cut.py
Suggest beat-aligned edit windows for standard ad/TV format durations.

Algorithm:
  For each target duration T (e.g. 15, 30, 60 seconds):
  1. Slide a window of ≈T seconds across the beat grid.
  2. Score each candidate window based on:
       - Starts after the intro (avoids cold-open cuts)
       - Starts near a section boundary (clean entry)
       - Ends near a section boundary (clean exit)
       - End point lands on a bar boundary (snap_bars alignment)
       - Window contains at least one chorus section (emotional peak)
  3. Return the highest-scoring SyncCut per target duration.
  4. Skip a target if the track is shorter than that duration.

All scoring is pure (no I/O).  SyncCutAnalyzer is a thin wrapper that
reads constants from Settings and delegates to the pure layer.

Implements: SyncCutProvider protocol (core/protocols.py)
"""
from __future__ import annotations

import logging
from typing import Optional

from core.config import CONSTANTS
from core.models import Section, SyncCut

_log = logging.getLogger(__name__)

# Label strings used for section-type detection (allin1 output).
_CHORUS_LABELS: frozenset[str] = frozenset({"chorus", "refrain", "hook"})
_INTRO_LABELS:  frozenset[str] = frozenset({"intro", "introduction", "intro_"})


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _section_at(t: float, sections: list[Section]) -> Optional[str]:
    """Return the label of the section that contains time t, or None."""
    for sec in sections:
        if sec.start <= t < sec.end:
            return sec.label
    return None


def _intro_end(sections: list[Section]) -> float:
    """Return the end time (seconds) of the last intro section, or 0.0."""
    end = 0.0
    for sec in sections:
        if any(sec.label.lower().startswith(lbl) for lbl in _INTRO_LABELS):
            end = max(end, sec.end)
    return end


def _near_boundary(
    t: float,
    sections: list[Section],
    tolerance: float,
) -> bool:
    """Return True if t is within *tolerance* seconds of any section boundary."""
    for sec in sections:
        if abs(t - sec.start) <= tolerance or abs(t - sec.end) <= tolerance:
            return True
    return False


def _contains_chorus(
    start_s: float,
    end_s: float,
    sections: list[Section],
) -> bool:
    """Return True if any chorus section overlaps the [start_s, end_s] window."""
    for sec in sections:
        if any(sec.label.lower().startswith(lbl) for lbl in _CHORUS_LABELS):
            # overlaps if sec.start < end_s and sec.end > start_s
            if sec.start < end_s and sec.end > start_s:
                return True
    return False


def _snap_to_bar(
    beat_idx: int,
    beats: list[float],
    snap_bars: int,
) -> int:
    """
    Return the beat index closest to beat_idx that falls on a bar boundary
    (i.e. beat_idx % snap_bars == 0).  Clamps to [0, len(beats)-1].
    """
    remainder = beat_idx % snap_bars
    if remainder == 0:
        return beat_idx
    # Try snapping forward or backward; pick the closer one.
    back = beat_idx - remainder
    fwd  = back + snap_bars
    back = max(0, back)
    fwd  = min(len(beats) - 1, fwd)
    if abs(beats[fwd] - beats[beat_idx]) <= abs(beats[beat_idx] - beats[back]):
        return fwd
    return back


def _build_note(
    start_s: float,
    end_s: float,
    sections: list[Section],
) -> str:
    """Compose a human-readable note string for a SyncCut."""
    def _fmt(t: float) -> str:
        m, s = divmod(int(t), 60)
        return f"{m}:{s:02d}"

    start_label = _section_at(start_s, sections) or "track"
    end_label   = _section_at(end_s,   sections) or "track end"
    return f"{start_label.capitalize()} at {_fmt(start_s)} → {end_label.capitalize()} at {_fmt(end_s)}"


def _score_window(
    start_s: float,
    end_s: float,
    sections: list[Section],
    beats: list[float],
    snap_bars: int,
    boundary_tolerance: float,
    intro_end_s: float,
) -> float:
    """
    Score a candidate [start_s, end_s] edit window.  Returns 0.0–1.0.

    Scoring criteria (equal weight, additive):
      +0.20  starts after the intro
      +0.20  start is near a section boundary
      +0.20  end is near a section boundary
      +0.20  contains a chorus (or hook/refrain)
      +0.20  end lands on a bar boundary (snap_bars alignment)
    """
    score = 0.0

    if start_s >= intro_end_s:
        score += 0.20

    if _near_boundary(start_s, sections, boundary_tolerance):
        score += 0.20

    if _near_boundary(end_s, sections, boundary_tolerance):
        score += 0.20

    if _contains_chorus(start_s, end_s, sections):
        score += 0.20

    # Check whether the beat nearest to end_s falls on a bar boundary.
    if beats:
        nearest_end_idx = min(
            range(len(beats)),
            key=lambda i: abs(beats[i] - end_s),
        )
        if nearest_end_idx % snap_bars == 0:
            score += 0.20

    return round(score, 4)


def suggest_sync_cuts(
    sections: list[Section],
    beats: list[float],
    target_durations: list[int],
    snap_bars: int,
    boundary_tolerance: float,
    duration_tolerance: float,
) -> list[SyncCut]:
    """
    Pure function.  Return the best SyncCut for each target duration.

    Args:
        sections:           allin1 structural sections.
        beats:              Beat grid as seconds-from-track-start.
        target_durations:   Format lengths to target (e.g. [15, 30, 60]).
        snap_bars:          Bar size in beats (4 for 4/4 time).
        boundary_tolerance: Max distance (s) from a section boundary to count as aligned.
        duration_tolerance: Allowed deviation (s) from each target duration.

    Returns:
        One SyncCut per target duration where the track is long enough.
        Shorter tracks yield no SyncCut for that duration.
    """
    if not beats:
        return []

    track_end    = beats[-1]
    intro_end_s  = _intro_end(sections)
    cuts: list[SyncCut] = []

    for target in target_durations:
        if track_end < target - duration_tolerance:
            _log.debug("sync_cut: track too short for %ds cut (%.1fs)", target, track_end)
            continue

        best_score: float   = -1.0
        best_cut: Optional[SyncCut] = None

        for i, beat_start in enumerate(beats):
            # Only consider start points that begin after the intro.
            if beat_start < intro_end_s and intro_end_s > 0.0:
                continue

            # Find the beat index whose timestamp is closest to beat_start + target.
            target_end = beat_start + target
            if target_end > track_end + duration_tolerance:
                break  # all remaining windows are also too long

            # Snap the end point to the nearest bar boundary.
            raw_end_idx = min(
                range(len(beats)),
                key=lambda j: abs(beats[j] - target_end),
            )
            end_idx = _snap_to_bar(raw_end_idx, beats, snap_bars)
            end_s   = beats[end_idx]

            # Reject windows outside the duration tolerance window.
            actual = end_s - beat_start
            if abs(actual - target) > duration_tolerance:
                continue

            score = _score_window(
                beat_start, end_s,
                sections, beats,
                snap_bars, boundary_tolerance,
                intro_end_s,
            )

            if score > best_score:
                best_score = score
                best_cut = SyncCut(
                    duration_s        = target,
                    start_s           = round(beat_start, 3),
                    end_s             = round(end_s, 3),
                    actual_duration_s = round(actual, 3),
                    confidence        = score,
                    note              = _build_note(beat_start, end_s, sections),
                )

        if best_cut is not None:
            cuts.append(best_cut)
            _log.debug(
                "sync_cut: %ds → start=%.2f end=%.2f conf=%.2f note=%s",
                target, best_cut.start_s, best_cut.end_s,
                best_cut.confidence, best_cut.note,
            )

    return cuts


# ---------------------------------------------------------------------------
# Public service class
# ---------------------------------------------------------------------------

class SyncCutAnalyzer:
    """
    Reads constants from Settings and delegates to suggest_sync_cuts().

    Implements: SyncCutProvider protocol (core/protocols.py)
    """

    def suggest(
        self,
        sections: list[Section],
        beats: list[float],
        target_durations: list[int],
    ) -> list[SyncCut]:
        """
        Suggest beat-aligned edit points for each target duration.

        Args:
            sections:         allin1 structural sections (label, start, end).
            beats:            Beat grid as seconds-from-track-start.
            target_durations: Format lengths to target (e.g. [15, 30, 60]).

        Returns:
            One SyncCut per target duration the track is long enough for.
        """
        return suggest_sync_cuts(
            sections           = sections,
            beats              = beats,
            target_durations   = target_durations,
            snap_bars          = CONSTANTS.SYNC_CUT_SNAP_BARS,
            boundary_tolerance = CONSTANTS.SYNC_CUT_BOUNDARY_TOLERANCE_S,
            duration_tolerance = CONSTANTS.SYNC_CUT_DURATION_TOLERANCE_S,
        )
