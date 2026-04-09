"""
services/sync_cut/_pure.py
Pure scoring helpers for sync edit point suggestion — no I/O, no class dependencies.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.config import CONSTANTS
from core.models import Section, SyncCut

_log = logging.getLogger(__name__)

_CHORUS_LABELS: frozenset[str] = frozenset({"chorus", "refrain", "hook"})
_INTRO_LABELS:  frozenset[str] = frozenset({"intro", "introduction", "intro_"})


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
    """
    Compose a human-readable note string for a SyncCut (#154).

    Lists every section that overlaps the window with its clipped timestamps.
    Appends a bar-count annotation when the window exits mid-section.
    Single-section windows are prefixed with 'Full' when the section is fully spanned.
    """
    def _fmt(t: float) -> str:
        m, s = divmod(int(t), 60)
        return f"{m}:{s:02d}"

    overlapping = [
        sec for sec in sections
        if sec.start < end_s and sec.end > start_s
    ]
    if not overlapping:
        return f"Track ({_fmt(start_s)}–{_fmt(end_s)})"

    parts: list[str] = []
    for sec in overlapping:
        clip_start = max(sec.start, start_s)
        clip_end   = min(sec.end,   end_s)
        label      = sec.label.capitalize()

        # A window exits mid-section when it ends more than one tolerance before the section end
        cuts_before_end = clip_end < sec.end - CONSTANTS.SYNC_CUT_BOUNDARY_TOLERANCE_S
        if cuts_before_end:
            sec_dur  = sec.end - sec.start
            cut_dur  = clip_end - sec.start
            # Rough bar length — assume 4 beats, use section duration / 4 as a proxy
            bar_s    = max(1.0, round(sec_dur / 4))
            bars_in  = max(1, round(cut_dur / bar_s))
            bar_word = "bar" if bars_in == 1 else "bars"
            parts.append(
                f"{label} ({_fmt(clip_start)}–{_fmt(clip_end)}) — cuts {bars_in} {bar_word} in"
            )
        else:
            # Full span: window covers this section from its clip_start to its end
            full_span = clip_start <= sec.start + CONSTANTS.SYNC_CUT_BOUNDARY_TOLERANCE_S
            prefix = "Full " if (full_span and len(overlapping) == 1) else ""
            parts.append(f"{prefix}{label} ({_fmt(clip_start)}–{_fmt(clip_end)})")

    return " · ".join(parts)


def _score_window(
    start_s: float,
    end_s: float,
    sections: list[Section],
    beats: list[float],
    snap_bars: int,
    boundary_tolerance: float,
    intro_end_s: float,
) -> tuple[float, dict[str, bool]]:
    """
    Score a candidate [start_s, end_s] edit window.

    Scoring criteria (equal weight, additive):
      +0.20  starts after the intro
      +0.20  start is near a section boundary
      +0.20  end is near a section boundary
      +0.20  contains a chorus (or hook/refrain)
      +0.20  end lands on a bar boundary (snap_bars alignment)

    Returns:
        (score, breakdown) — score is 0.0–1.0; breakdown is a dict of
        criterion name → bool for use in JSON export (#155).
    """
    post_intro           = start_s >= intro_end_s
    start_near_boundary  = _near_boundary(start_s, sections, boundary_tolerance)
    end_near_boundary    = _near_boundary(end_s,   sections, boundary_tolerance)
    contains_chorus      = _contains_chorus(start_s, end_s, sections)

    bar_aligned_end = False
    if beats:
        nearest_end_idx = min(
            range(len(beats)),
            key=lambda i: abs(beats[i] - end_s),
        )
        bar_aligned_end = nearest_end_idx % snap_bars == 0

    breakdown: dict[str, bool] = {
        "post_intro":           post_intro,
        "start_near_boundary":  start_near_boundary,
        "end_near_boundary":    end_near_boundary,
        "contains_chorus":      contains_chorus,
        "bar_aligned_end":      bar_aligned_end,
    }
    score = round(sum(0.20 for v in breakdown.values() if v), 4)
    return score, breakdown


def suggest_sync_cuts(
    sections: list[Section],
    beats: list[float],
    target_durations: list[int],
    snap_bars: int,
    boundary_tolerance: float,
    duration_tolerance: float,
    top_n: int = 3,
) -> list[SyncCut]:
    """
    Pure function.  Return the top-N SyncCuts for each target duration (#148).

    Args:
        sections:           allin1 structural sections.
        beats:              Beat grid as seconds-from-track-start.
        target_durations:   Format lengths to target (e.g. [15, 30, 60]).
        snap_bars:          Bar size in beats (4 for 4/4 time).
        boundary_tolerance: Max distance (s) from a section boundary to count as aligned.
        duration_tolerance: Allowed deviation (s) from each target duration.
        top_n:              Maximum candidates to return per target duration.

    Returns:
        Up to *top_n* SyncCuts per target duration, sorted by rank (1 = best).
        Fewer are returned when the track is too short or when fewer distinct
        windows are found.
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

        # Collect all valid candidates for this target duration
        candidates: list[tuple[float, dict[str, bool], float, float, float]] = []

        for i, beat_start in enumerate(beats):
            if beat_start < intro_end_s and intro_end_s > 0.0:
                continue

            target_end = beat_start + target
            if target_end > track_end + duration_tolerance:
                break

            raw_end_idx = min(
                range(len(beats)),
                key=lambda j: abs(beats[j] - target_end),
            )
            end_idx = _snap_to_bar(raw_end_idx, beats, snap_bars)
            end_s   = beats[end_idx]

            actual = end_s - beat_start
            if abs(actual - target) > duration_tolerance:
                continue

            score, breakdown = _score_window(
                beat_start, end_s,
                sections, beats,
                snap_bars, boundary_tolerance,
                intro_end_s,
            )
            candidates.append((score, breakdown, beat_start, end_s, actual))

        # Sort descending by score; take top_n with unique start positions
        candidates.sort(key=lambda c: c[0], reverse=True)
        seen_starts: set[float] = set()
        ranked: list[tuple[float, dict[str, bool], float, float, float]] = []
        for cand in candidates:
            if cand[2] not in seen_starts:
                seen_starts.add(cand[2])
                ranked.append(cand)
            if len(ranked) >= top_n:
                break

        for rank_idx, (score, breakdown, beat_start, end_s, actual) in enumerate(ranked, 1):
            cut = SyncCut(
                duration_s        = target,
                start_s           = round(beat_start, 3),
                end_s             = round(end_s, 3),
                actual_duration_s = round(actual, 3),
                confidence        = score,
                note              = _build_note(beat_start, end_s, sections),
                rank              = rank_idx,
                score_breakdown   = breakdown,
            )
            cuts.append(cut)
            _log.debug(
                "sync_cut: %ds rank=%d → start=%.2f end=%.2f conf=%.2f note=%s",
                target, rank_idx, cut.start_s, cut.end_s, cut.confidence, cut.note,
            )

    return cuts
