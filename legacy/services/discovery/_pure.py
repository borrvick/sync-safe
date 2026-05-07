"""
services/discovery/_pure.py
Pure normalisation helpers and track-list parser — no HTTP, no I/O.
"""
from __future__ import annotations

from core.config import CONSTANTS


def _parse_track_list(tracks: list) -> list[tuple[str, str, int]]:
    """
    Extract (artist, title, listeners) triples from a Last.fm track list payload.

    `listeners` is the raw Last.fm listener count for the similar track, used to
    classify a popularity tier without an additional API call (#124).
    Returns 0 when the field is absent (e.g. fallback artist-similar paths).

    Pure function — no I/O.
    """
    results: list[tuple[str, str, int]] = []
    for t in tracks[:CONSTANTS.MAX_SIMILAR_TRACKS]:
        title       = t.get("name", "")
        artist_data = t.get("artist", {})
        artist      = (
            artist_data.get("name", "")
            if isinstance(artist_data, dict)
            else str(artist_data)
        )
        if title and artist:
            try:
                listeners = int(t.get("listeners", 0) or 0)
            except (ValueError, TypeError):
                listeners = 0
            results.append((artist, title, listeners))
    return results


def _listeners_to_tier(listeners: int) -> str:
    """
    Map a raw Last.fm listener count to a popularity tier label.

    Uses the same LASTFM_LISTENERS_* ceiling constants as the full
    popularity pipeline so tier vocabulary is consistent (#124).

    Pure function — no I/O.
    """
    if listeners >= CONSTANTS.LASTFM_LISTENERS_GLOBAL:
        return "Global"
    if listeners >= CONSTANTS.LASTFM_LISTENERS_MAINSTREAM:
        return "Mainstream"
    if listeners >= CONSTANTS.LASTFM_LISTENERS_REGIONAL:
        return "Regional"
    return "Emerging"


def _piecewise_score(value: int, low: int, mid: int, high: int) -> int:
    """
    Map a raw count to a 0–100 score using three tier-boundary breakpoints.

    Pure function — no I/O.
    """
    if value <= 0:
        return 0
    if value >= high:
        extra = min(value - high, high) / high * 25
        return min(100, 75 + int(extra))
    if value >= mid:
        return 50 + int((value - mid) / (high - mid) * 25)
    if value >= low:
        return 25 + int((value - low) / (mid - low) * 25)
    return int(value / low * 25)


def _normalise_lastfm(listeners: int, constants: object) -> int:
    """
    Normalise a Last.fm listener count to a 0–100 score.

    Breakpoints: LASTFM_LISTENERS_REGIONAL / _MAINSTREAM / _GLOBAL → 25/50/75.

    Pure function — no I/O.
    """
    cfg = constants
    return _piecewise_score(
        listeners,
        cfg.LASTFM_LISTENERS_REGIONAL,
        cfg.LASTFM_LISTENERS_MAINSTREAM,
        cfg.LASTFM_LISTENERS_GLOBAL,
    )


def _normalise_views(view_count: int, constants: object) -> int:
    """
    Normalise a platform view count to a 0–100 score.

    Breakpoints: PLATFORM_VIEWS_REGIONAL / _MAINSTREAM / _GLOBAL → 25/50/75.

    Pure function — no I/O.
    """
    cfg = constants
    return _piecewise_score(
        view_count,
        cfg.PLATFORM_VIEWS_REGIONAL,
        cfg.PLATFORM_VIEWS_MAINSTREAM,
        cfg.PLATFORM_VIEWS_GLOBAL,
    )
