"""
services/sync_cut/_orchestrator.py
SyncCutAnalyzer — reads CONSTANTS and delegates to suggest_sync_cuts().
"""
from __future__ import annotations

from core.config import CONSTANTS
from core.models import Section, SyncCut

from ._pure import suggest_sync_cuts


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
        loop_score: float = 0.0,
    ) -> list[SyncCut]:
        """
        Suggest beat-aligned edit points for each target duration.

        Args:
            sections:         allin1 structural sections (label, start, end).
            beats:            Beat grid as seconds-from-track-start.
            target_durations: Format lengths to target (e.g. [15, 30, 60]).
            loop_score:       Track-level loop repetition score from forensics;
                              used to apply a small bonus for moderately loopy tracks (#151).

        Returns:
            Up to SYNC_CUT_TOP_N SyncCuts per target duration, ranked 1–N.
        """
        return suggest_sync_cuts(
            sections           = sections,
            beats              = beats,
            target_durations   = target_durations,
            snap_bars          = CONSTANTS.SYNC_CUT_SNAP_BARS,
            boundary_tolerance = CONSTANTS.SYNC_CUT_BOUNDARY_TOLERANCE_S,
            duration_tolerance = CONSTANTS.SYNC_CUT_DURATION_TOLERANCE_S,
            top_n              = CONSTANTS.SYNC_CUT_TOP_N,
            loop_score         = loop_score,
        )
