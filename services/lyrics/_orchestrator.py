"""
services/lyrics/_orchestrator.py
LRCLib lyrics lookup — implements LyricsProvider protocol.

Tries to fetch synced (LRC-format) lyrics from the free, open LRCLib API
(https://lrclib.net). Returns list[TranscriptSegment] on hit, None on miss
or any HTTP/parse error so the caller can fall back to audio-based transcription.

Design notes:
- Pure HTTP — no GPU, no audio processing, no disk writes.
- All network errors degrade to None (logged as step_error); the caller is
  responsible for the fallback strategy.
- _parse_lrc and _best_result are pure functions in _pure.py for independent
  testability.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Optional

from core.config import get_settings
from core.logging import PipelineLogger
from core.models import TranscriptSegment

from ._pure import _best_result, _parse_lrc

_LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
_REQUEST_TIMEOUT_S = 8


class LRCLibClient:
    """
    Fetches synced lyrics from LRCLib (https://lrclib.net).

    Implements: LyricsProvider protocol (core/protocols.py)

    Constructor injection:
        log — PipelineLogger for structured step logging. A default instance
              is created from Settings.log_dir when not provided.

    Usage:
        client   = LRCLibClient()
        segments = client.get_lyrics("Bohemian Rhapsody", "Queen")
        # → list[TranscriptSegment] on hit, None on miss or error
    """

    def __init__(self, log: PipelineLogger | None = None) -> None:
        self._log = log or PipelineLogger(get_settings().log_dir)

    # ------------------------------------------------------------------
    # Public interface (LyricsProvider protocol)
    # ------------------------------------------------------------------

    def get_lyrics(self, title: str, artist: str) -> Optional[list[TranscriptSegment]]:
        """
        Search LRCLib for synced lyrics and return parsed segments.

        Args:
            title:  Track title.
            artist: Artist name.

        Returns:
            Ordered list of TranscriptSegment objects if synced lyrics are
            found, or None if the track is not found, has no synced lyrics,
            the HTTP request fails, or title/artist are blank.
        """
        if not title or not artist:
            self._log.step_error("lrclib", "Skipped — title or artist missing")
            return None

        self._log.step_start("lrclib")
        t0 = time.monotonic()

        try:
            results = self._search(title, artist)
        except Exception as exc:
            self._log.step_error("lrclib", f"HTTP error: {exc}")
            return None

        best = _best_result(results)
        if best is None:
            self._log.info("lrclib_miss", title=title, artist=artist)
            return None

        synced_lrc = best.get("syncedLyrics") or ""
        if not synced_lrc:
            self._log.info("lrclib_no_synced", title=title, artist=artist)
            return None

        segments = _parse_lrc(synced_lrc)
        duration_s = round(time.monotonic() - t0, 2)

        if segments:
            self._log.step_end("lrclib", duration_s=duration_s)
            self._log.info(
                "lrclib_hit",
                title=title,
                artist=artist,
                segment_count=len(segments),
            )
        else:
            self._log.info("lrclib_parse_empty", title=title, artist=artist)

        return segments or None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _search(self, title: str, artist: str) -> list[dict]:
        """
        Call GET /api/search?q=artist+title and return the decoded JSON list.

        Raises:
            OSError:             on network failure or non-200 status.
            json.JSONDecodeError: on malformed response body.
        """
        query = urllib.parse.urlencode({"q": f"{artist} {title}"})
        url = f"{_LRCLIB_SEARCH_URL}?{query}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "sync-safe-forensic-portal/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            if resp.status != 200:
                raise OSError(f"LRCLib returned HTTP {resp.status}")
            return json.loads(resp.read().decode("utf-8"))
