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

import base64
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests

from core.config import CONSTANTS, get_settings
from core.exceptions import AudioSourceError, ConfigurationError
from core.models import PopularityResult, TrackCandidate
from core.protocols import YtDlpProvider
from services.ingestion._ytdlp import YtDlpClient

from ._pure import _listeners_to_tier, _normalise_lastfm, _normalise_views, _parse_track_list


# ---------------------------------------------------------------------------
# Last.fm API base URL (module-level constant — not a business threshold)
# ---------------------------------------------------------------------------

_LASTFM_BASE: str = "http://ws.audioscrobbler.com/2.0/"


class Discovery:
    """
    Finds commercially similar tracks for sync licensing reference.

    Implements: TrackDiscovery protocol (core/protocols.py)

    Constructor injection: pass a YtDlpProvider to swap the YouTube URL
    resolution backend — e.g. for a paid service or in integration tests
    that stub subprocess calls.

    Usage:
        service  = Discovery()
        tracks   = service.find_similar("Blinding Lights", "The Weeknd")
        for t in tracks:
            print(t.title, t.artist, t.youtube_url)
    """

    def __init__(self, ytdlp_client: YtDlpProvider | None = None) -> None:
        self._ytdlp = ytdlp_client or YtDlpClient()

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

        settings = get_settings()
        lastfm_triples = _fetch_similar(title, artist, api_key)

        # ── Spotify recommendations (#126) ────────────────────────────────────
        # When Spotify credentials are available, fetch recommendations seeded by
        # the input track and interleave them with the Last.fm results.
        spotify_pairs: list[tuple[str, str]] = []
        if settings.spotify_client_id and settings.spotify_client_secret:
            token = _get_spotify_token(
                settings.spotify_client_id, settings.spotify_client_secret
            )
            if token:
                pop_result = _fetch_spotify_popularity(
                    title, artist,
                    settings.spotify_client_id,
                    settings.spotify_client_secret,
                    token=token,  # reuse already-fetched token — avoids a second round-trip
                )
                if pop_result is not None:
                    _, track_id = pop_result
                    spotify_pairs = _fetch_spotify_recommendations(
                        track_id, token, limit=CONSTANTS.MAX_SIMILAR_TRACKS
                    )

        triples = _merge_candidates(
            lastfm_triples, spotify_pairs, max_results=CONSTANTS.MAX_SIMILAR_TRACKS
        )

        # Build Last.fm key set once — used per-candidate to detect Spotify-sourced entries.
        lfm_keys = {
            (a.lower().strip(), ttl.lower().strip()) for a, ttl, _ in lastfm_triples
        }

        candidates: list[TrackCandidate] = []
        for i, (sim_artist, sim_title, listeners) in enumerate(triples):
            yt_url = self._ytdlp.search_url(sim_artist, sim_title)
            # similarity is rank-based (1.0 → 1/n range) since Last.fm doesn't
            # expose a numeric score in the getSimilar response we consume
            similarity = round(1.0 - (i / max(len(triples), 1)) * 0.5, 3)
            # Tier from Last.fm listeners already in the similar-tracks response (#124).
            # None when listeners=0 (fallback path or Spotify-sourced entry).
            tier = _listeners_to_tier(listeners) if listeners > 0 else None
            # Source: if this came from the Spotify-only portion of the merge,
            # listeners will be 0 and the artist+title won't be in lastfm_triples.
            source = (
                "lastfm"
                if (sim_artist.lower().strip(), sim_title.lower().strip()) in lfm_keys
                else "spotify"
            )
            candidates.append(TrackCandidate(
                title=sim_title,
                artist=sim_artist,
                youtube_url=yt_url,
                similarity=similarity,
                popularity_tier=tier,
                source=source,
            ))

        return candidates

    def get_track_popularity(
        self,
        title: str,
        artist: str,
        platform_metrics: Optional[dict[str, int]] = None,
    ) -> Optional[PopularityResult]:
        """
        Fetch multi-signal popularity data and return a blended PopularityResult.

        Signals gathered (each independently best-effort, never raises):
          - Last.fm listeners + playcount via track.getInfo (autocorrect=1)
          - Spotify popularity score (0–100) via client credentials OAuth
          - Platform engagement metrics from AudioBuffer.metadata (view_count, etc.)

        The blended popularity_score is the max of all normalised per-signal
        scores, so a strong signal on any single platform cannot be suppressed
        by a weak or missing one.  This prevents a bad Last.fm lookup from
        misclassifying a mainstream track as "Emerging".

        Returns None only when no signals are available at all.
        """
        if not title and not artist:
            return None

        settings = get_settings()
        listeners, playcount = 0, 0
        spotify_score: Optional[int] = None
        metrics: dict[str, int] = platform_metrics or {}

        # ── Last.fm ──────────────────────────────────────────────────────────
        if settings.lastfm_api_key:
            try:
                params = {
                    "method":      "track.getInfo",
                    "track":       title,
                    "artist":      artist,
                    "api_key":     settings.lastfm_api_key,
                    "format":      "json",
                    "autocorrect": 1,
                }
                resp = requests.get(_LASTFM_BASE, params=params, timeout=10)
                resp.raise_for_status()
                track_data = resp.json().get("track", {})
                if track_data:
                    listeners = int(track_data.get("listeners", 0))
                    playcount = int(track_data.get("playcount", 0))

                # ── Fuzzy retry (#89) ─────────────────────────────────────
                # If the first lookup returns suspiciously few listeners
                # (e.g. Bruno Mars / 24K Magic showing 13 due to decorated
                # title mismatch), strip feat./parentheticals and retry once.
                if listeners < CONSTANTS.LASTFM_LOW_LISTENER_THRESHOLD:
                    clean_title = _clean_title_for_lastfm(title)
                    if clean_title and clean_title != title:
                        retry_params = {**params, "track": clean_title}
                        retry_resp = requests.get(
                            _LASTFM_BASE, params=retry_params, timeout=10
                        )
                        retry_resp.raise_for_status()
                        retry_data = retry_resp.json().get("track", {})
                        if retry_data:
                            retry_listeners = int(retry_data.get("listeners", 0))
                            retry_playcount = int(retry_data.get("playcount", 0))
                            # Only upgrade — never downgrade to the cleaned result
                            if retry_listeners > listeners:
                                listeners = retry_listeners
                                playcount = max(playcount, retry_playcount)
            except Exception:  # noqa: BLE001 — popularity is always best-effort
                pass

        # ── Spotify ───────────────────────────────────────────────────────────
        if settings.spotify_client_id and settings.spotify_client_secret:
            result = _fetch_spotify_popularity(
                title, artist,
                settings.spotify_client_id,
                settings.spotify_client_secret,
            )
            spotify_score = result[0] if result is not None else None

        # ── Bail if nothing at all ────────────────────────────────────────────
        if not listeners and not playcount and spotify_score is None and not metrics:
            return None

        return _classify_popularity(
            listeners=listeners,
            playcount=playcount,
            spotify_score=spotify_score,
            platform_metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Last.fm API helpers — pure functions, independently testable
# ---------------------------------------------------------------------------

def _fetch_similar(title: str, artist: str, api_key: str) -> list[tuple[str, str, int]]:
    """
    Query Last.fm for similar tracks; fall back to artist.getSimilar.

    Returns:
        List of (artist, title, listeners) triples, up to CONSTANTS.MAX_SIMILAR_TRACKS.
        `listeners` is 0 when unavailable (fallback path).

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


def _fetch_artist_similar(artist: str, api_key: str) -> list[tuple[str, str, int]]:
    """
    Fallback strategy: fetch similar artists then each artist's top track.

    Returns:
        List of (artist, title, listeners) triples; listeners=0 (unavailable on this path).
        Empty list on any failure.
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

        results: list[tuple[str, str, int]] = []
        for entry in similar_artists[:CONSTANTS.MAX_SIMILAR_TRACKS]:
            sim_artist = entry.get("name", "")
            top_track  = _fetch_top_track(sim_artist, api_key)
            if top_track:
                results.append((sim_artist, top_track, 0))  # no listener data on fallback path

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

    # Try "official audio" before "official video" — reduces live/cover hits (#128).
    # Both queries are passed as list args (not shell=True), so no quoting needed.
    queries = [
        f"ytsearch1:{artist} {title} official audio",
        f"ytsearch1:{artist} {title} official video",
    ]

    for query in queries:
        cmd = [
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


def _clean_title_for_lastfm(title: str) -> str:
    """
    Strip common title decorations that cause Last.fm lookup mismatches.

    Removes:
      - Featured artist credits: "feat. X", "ft. X", "featuring X"
      - Parenthetical / bracket suffixes: "(Official Video)", "[Lyrics]", etc.
      - Leading/trailing whitespace

    Designed for use as a fuzzy retry when the first lookup returns
    suspiciously low listener counts (< LASTFM_LOW_LISTENER_THRESHOLD).

    Pure function — no I/O.

    >>> _clean_title_for_lastfm("24K Magic (feat. Bruno Mars) [Lyrics]")
    '24K Magic'
    >>> _clean_title_for_lastfm("Blinding Lights")
    'Blinding Lights'
    """
    # Strip "feat." / "ft." / "featuring" blocks (with optional parentheses)
    cleaned = re.sub(
        r"\s*[\(\[]?\s*(?:feat\.?|ft\.?|featuring)\s+[^\)\]]+[\)\]]?",
        "",
        title,
        flags=re.IGNORECASE,
    )
    # Strip any remaining trailing parenthetical / bracket suffixes
    cleaned = re.sub(r"\s*[\(\[][^\)\]]+[\)\]]\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _classify_popularity(
    listeners: int,
    playcount: int,
    spotify_score: Optional[int] = None,
    platform_metrics: Optional[dict[str, int]] = None,
    constants: object | None = None,
) -> PopularityResult:
    """
    Derive a blended popularity_score and tier from all available signals.

    Strategy: normalise each signal independently to 0–100, take the maximum,
    then apply a minimum-signal gate for higher tiers.

    max() protects against false lows (a bad Last.fm lookup can't suppress a
    mainstream track).  The signal gate protects against false highs (a meme
    remix with 200M views but near-zero Last.fm + Spotify can't reach Global).

    Gate rules (from SystemConstants):
      Emerging / Regional: 1 signal sufficient
      Mainstream:          requires POPULARITY_MIN_SIGNALS_MAINSTREAM signals > 0
      Global:              requires POPULARITY_MIN_SIGNALS_GLOBAL signals > 0

    Signals used (each optional — omitted when unavailable):
      - Last.fm listeners (piecewise-linear normalised against tier boundaries)
      - Spotify popularity (native 0–100, passed through directly)
      - Platform view_count from platform_metrics (piecewise-linear normalised)

    Tier boundaries and cost ranges come from SystemConstants.
    Accepts an optional constants override for unit testing without env deps.

    Pure function — no I/O.
    """
    cfg = constants or CONSTANTS
    metrics = platform_metrics or {}

    scores: list[int] = []

    lastfm_score = _normalise_lastfm(listeners, cfg)
    if lastfm_score > 0:
        scores.append(lastfm_score)

    if spotify_score is not None:
        scores.append(max(0, min(100, spotify_score)))

    view_count = metrics.get("view_count", 0)
    if view_count > 0:
        scores.append(_normalise_views(view_count, cfg))

    popularity_score = max(scores) if scores else 0
    signal_count = len(scores)

    # Determine tier, applying the minimum-signal gate to higher tiers.
    # If the score qualifies for a tier but not enough signals are present,
    # cap at the next tier down.
    if (popularity_score >= cfg.POPULARITY_GLOBAL_MIN
            and signal_count >= cfg.POPULARITY_MIN_SIGNALS_GLOBAL):
        tier                = "Global"
        cost_low, cost_high = cfg.SYNC_COST_GLOBAL
    elif (popularity_score >= cfg.POPULARITY_MAINSTREAM_MIN
            and signal_count >= cfg.POPULARITY_MIN_SIGNALS_MAINSTREAM):
        tier                = "Mainstream"
        cost_low, cost_high = cfg.SYNC_COST_MAINSTREAM
    elif popularity_score >= cfg.POPULARITY_REGIONAL_MIN:
        tier                = "Regional"
        cost_low, cost_high = cfg.SYNC_COST_REGIONAL
    else:
        tier                = "Emerging"
        cost_low, cost_high = cfg.SYNC_COST_EMERGING

    return PopularityResult(
        listeners=listeners,
        playcount=playcount,
        spotify_score=spotify_score,
        platform_metrics=metrics,
        popularity_score=popularity_score,
        tier=tier,
        sync_cost_low=cost_low,
        sync_cost_high=cost_high,
    )


def _get_spotify_token(client_id: str, client_secret: str) -> Optional[str]:
    """
    Fetch a Spotify client credentials OAuth token.

    Returns the access token string, or None on any failure.
    Pure I/O boundary — no business logic.
    """
    try:
        credentials = base64.b64encode(
            f"{client_id}:{client_secret}".encode()
        ).decode()
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            headers={"Authorization": f"Basic {credentials}"},
            data={"grant_type": "client_credentials"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("access_token") or None
    except Exception:  # noqa: BLE001
        return None


def _fetch_spotify_popularity(
    title: str,
    artist: str,
    client_id: str,
    client_secret: str,
    token: Optional[str] = None,
) -> Optional[tuple[int, str]]:
    """
    Fetch Spotify popularity score (0–100) and track ID via the Web API.

    Uses client credentials OAuth — no user login required.  Returns a
    (score, track_id) tuple so the caller can reuse track_id for recommendations
    (#126) without a second search round-trip.

    Args:
        token: Optional pre-fetched OAuth token.  When provided, skips the
               internal _get_spotify_token() call to avoid a redundant round-trip.

    Returns None on any failure (missing credentials, network error, no match).

    Pure I/O boundary — no business logic.
    """
    try:
        resolved_token = token or _get_spotify_token(client_id, client_secret)
        if not resolved_token:
            return None
        token = resolved_token

        query = f"track:{title} artist:{artist}" if artist else f"track:{title}"
        search_resp = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": 1},
            timeout=10,
        )
        search_resp.raise_for_status()
        items = search_resp.json().get("tracks", {}).get("items", [])
        if not items:
            return None

        item      = items[0]
        score     = int(item.get("popularity", 0))
        track_id  = item.get("id", "")
        if not track_id:
            return None

        return (score, track_id)
    except Exception:  # noqa: BLE001 — Spotify is always best-effort
        return None


def _fetch_spotify_recommendations(
    track_id: str,
    token: str,
    limit: int,
) -> list[tuple[str, str]]:
    """
    Fetch Spotify track recommendations seeded by a single track ID (#126).

    Returns a list of (artist, title) tuples up to `limit` entries.
    Returns an empty list on any failure — recommendations are always best-effort.

    Pure I/O boundary — no business logic.
    """
    try:
        resp = requests.get(
            "https://api.spotify.com/v1/recommendations",
            headers={"Authorization": f"Bearer {token}"},
            params={"seed_tracks": track_id, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        tracks = resp.json().get("tracks", [])
        results: list[tuple[str, str]] = []
        for t in tracks:
            artists = t.get("artists", [])
            artist  = artists[0].get("name", "") if artists else ""
            title   = t.get("name", "")
            if artist and title:
                results.append((artist, title))
        return results
    except Exception:  # noqa: BLE001
        return []


def _merge_candidates(
    lastfm: list[tuple[str, str, int]],
    spotify: list[tuple[str, str]],
    max_results: int,
) -> list[tuple[str, str, int]]:
    """
    Round-robin interleave Last.fm and Spotify candidates; dedup by normalised key.

    Normalisation: lowercase, strip leading/trailing whitespace.  The first
    occurrence of each (artist, title) pair is kept; duplicates are dropped.
    Last.fm entries carry their listener counts; Spotify entries carry 0.

    Pure function — no I/O.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[tuple[str, str, int]] = []

    # Convert Spotify pairs to the same 3-tuple shape (listeners=0)
    spotify_triples: list[tuple[str, str, int]] = [
        (art, ttl, 0) for art, ttl in spotify
    ]

    # Round-robin interleave
    for lfm, spot in zip(lastfm, spotify_triples):
        for entry in (lfm, spot):
            key = (entry[0].lower().strip(), entry[1].lower().strip())
            if key not in seen:
                seen.add(key)
                merged.append(entry)
            if len(merged) >= max_results:
                return merged

    # Drain whichever list is longer
    for remainder in (lastfm[len(spotify_triples):], spotify_triples[len(lastfm):]):
        for entry in remainder:
            key = (entry[0].lower().strip(), entry[1].lower().strip())
            if key not in seen:
                seen.add(key)
                merged.append(entry)
            if len(merged) >= max_results:
                return merged

    return merged


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
