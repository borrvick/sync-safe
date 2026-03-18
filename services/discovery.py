"""
services/discovery.py
Similar-track discovery via Last.fm + yt-dlp — implements TrackDiscovery.

Strategy:
  1. Last.fm track.getSimilar — direct similar tracks for the input song
  2. Last.fm artist.getSimilar + artist.getTopTracks — fallback when step 1
     returns nothing (e.g. obscure track, typo in title)
  3. yt-dlp ytsearch1: — resolve a YouTube URL for each candidate
     (metadata-only fetch — no audio downloaded)

Design notes:
- Discovery.find_similar() is the single public entry point.
- API key is sourced from get_settings().lastfm_api_key — never os.environ
  directly — so the same service works in Streamlit, FastAPI, and tests.
- Missing API key → ConfigurationError (setup problem, not a runtime error).
- Last.fm / yt-dlp failures → AudioSourceError with context dict.
- YouTube URL resolution failure is non-fatal — TrackCandidate.youtube_url
  is Optional[str]; None means the track was found but URL lookup failed.
- _find_binary, _parse_track_list are pure module-level functions for
  independent unit testing.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests

from core.config import CONSTANTS, get_settings
from core.exceptions import AudioSourceError, ConfigurationError
from core.models import TrackCandidate


# ---------------------------------------------------------------------------
# Last.fm API base URL (module-level constant — not a business threshold)
# ---------------------------------------------------------------------------

_LASTFM_BASE: str = "http://ws.audioscrobbler.com/2.0/"


class Discovery:
    """
    Finds commercially similar tracks for sync licensing reference.

    Implements: TrackDiscovery protocol (core/protocols.py)

    Usage:
        service  = Discovery()
        tracks   = service.find_similar("Blinding Lights", "The Weeknd")
        for t in tracks:
            print(t.title, t.artist, t.youtube_url)
    """

    # ------------------------------------------------------------------
    # Public interface (TrackDiscovery protocol)
    # ------------------------------------------------------------------

    def find_similar(self, title: str, artist: str) -> list[TrackCandidate]:
        """
        Return up to CONSTANTS.MAX_SIMILAR_TRACKS similar tracks with
        resolved YouTube URLs.

        Args:
            title:  Track title string.
            artist: Artist name string.

        Returns:
            List of TrackCandidate objects ordered by similarity descending.
            youtube_url may be None when yt-dlp lookup fails for a candidate.

        Raises:
            ConfigurationError: if lastfm_api_key is missing.
            AudioSourceError:   on Last.fm API failure (network error, 4xx/5xx).
        """
        api_key = get_settings().lastfm_api_key
        if not api_key:
            raise ConfigurationError(
                "Last.fm API key is not configured.",
                context={"suggestion": "Set LASTFM_API_KEY in .env or environment."},
            )

        if not title and not artist:
            return []

        pairs = _fetch_similar(title, artist, api_key)

        candidates: list[TrackCandidate] = []
        for i, (sim_artist, sim_title) in enumerate(pairs):
            yt_url = _resolve_youtube_url(sim_artist, sim_title)
            # similarity is rank-based (1.0 → 1/n range) since Last.fm doesn't
            # expose a numeric score in the getSimilar response we consume
            similarity = round(1.0 - (i / max(len(pairs), 1)) * 0.5, 3)
            candidates.append(TrackCandidate(
                title=sim_title,
                artist=sim_artist,
                youtube_url=yt_url,
                similarity=similarity,
            ))

        return candidates


# ---------------------------------------------------------------------------
# Last.fm API helpers — pure functions, independently testable
# ---------------------------------------------------------------------------

def _fetch_similar(title: str, artist: str, api_key: str) -> list[tuple[str, str]]:
    """
    Query Last.fm for similar tracks; fall back to artist.getSimilar.

    Returns:
        List of (artist, title) tuples, up to CONSTANTS.MAX_SIMILAR_TRACKS.

    Raises:
        AudioSourceError: on network failure or non-2xx HTTP status.
    """
    params = {
        "method":      "track.getSimilar",
        "track":       title,
        "artist":      artist,
        "api_key":     api_key,
        "format":      "json",
        "limit":       CONSTANTS.MAX_SIMILAR_TRACKS,
        "autocorrect": 1,
    }

    try:
        resp = requests.get(_LASTFM_BASE, params=params, timeout=10)
        resp.raise_for_status()
        tracks = resp.json().get("similartracks", {}).get("track", [])
        if tracks:
            return _parse_track_list(tracks)
    except (AudioSourceError, ConfigurationError):
        raise
    except Exception as exc:
        raise AudioSourceError(
            "Last.fm track.getSimilar request failed.",
            context={"original_error": str(exc)},
        ) from exc

    # Fallback: similar artists → their top tracks
    if artist:
        return _fetch_artist_similar(artist, api_key)

    return []


def _fetch_artist_similar(artist: str, api_key: str) -> list[tuple[str, str]]:
    """
    Fallback strategy: fetch similar artists then each artist's top track.

    Returns:
        List of (artist, title) tuples; empty list on any failure.
    """
    params = {
        "method":      "artist.getSimilar",
        "artist":      artist,
        "api_key":     api_key,
        "format":      "json",
        "limit":       CONSTANTS.MAX_SIMILAR_TRACKS,
        "autocorrect": 1,
    }

    try:
        resp = requests.get(_LASTFM_BASE, params=params, timeout=10)
        resp.raise_for_status()
        similar_artists = resp.json().get("similarartists", {}).get("artist", [])

        results: list[tuple[str, str]] = []
        for entry in similar_artists[:CONSTANTS.MAX_SIMILAR_TRACKS]:
            sim_artist = entry.get("name", "")
            top_track  = _fetch_top_track(sim_artist, api_key)
            if top_track:
                results.append((sim_artist, top_track))

        return results

    except Exception:  # noqa: BLE001 — fallback; any failure → empty
        return []


def _fetch_top_track(artist: str, api_key: str) -> Optional[str]:
    """Return the name of an artist's #1 Last.fm top track, or None."""
    params = {
        "method":      "artist.getTopTracks",
        "artist":      artist,
        "api_key":     api_key,
        "format":      "json",
        "limit":       1,
        "autocorrect": 1,
    }

    try:
        resp   = requests.get(_LASTFM_BASE, params=params, timeout=8)
        resp.raise_for_status()
        tracks = resp.json().get("toptracks", {}).get("track", [])
        if tracks:
            return tracks[0].get("name")
    except Exception:  # noqa: BLE001
        pass

    return None


def _parse_track_list(tracks: list) -> list[tuple[str, str]]:
    """
    Extract (artist, title) pairs from a Last.fm track list payload.

    Pure function — no I/O.
    """
    results: list[tuple[str, str]] = []
    for t in tracks[:CONSTANTS.MAX_SIMILAR_TRACKS]:
        title       = t.get("name", "")
        artist_data = t.get("artist", {})
        artist      = (
            artist_data.get("name", "")
            if isinstance(artist_data, dict)
            else str(artist_data)
        )
        if title and artist:
            results.append((artist, title))
    return results


# ---------------------------------------------------------------------------
# YouTube URL resolution — pure function, independently testable
# ---------------------------------------------------------------------------

def _resolve_youtube_url(artist: str, title: str) -> Optional[str]:
    """
    Use yt-dlp ytsearch1: to resolve a YouTube watch URL.

    No audio is downloaded — only metadata is fetched (--no-download).
    Returns None on any failure so a missing URL never blocks the caller.
    """
    ytdlp = _find_binary("yt-dlp")
    if ytdlp is None:
        return None

    query = f"ytsearch1:{artist} - {title}"
    cmd   = [
        ytdlp,
        "--quiet",
        "--no-warnings",
        "--no-download",
        "--print", "webpage_url",
        query,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        url    = result.stdout.strip()
        if url.startswith("https://"):
            return url
    except Exception:  # noqa: BLE001
        pass

    return None


def _find_binary(name: str) -> Optional[str]:
    """
    Locate a CLI binary on PATH or in the user site-packages bin directory.

    Returns the full path string, or None if not found (never raises).
    """
    found = shutil.which(name)
    if found:
        return found
    try:
        import site
        user_bin = Path(site.getuserbase()) / "bin" / name
        if user_bin.exists():
            return str(user_bin)
    except Exception:  # noqa: BLE001
        pass
    return None
