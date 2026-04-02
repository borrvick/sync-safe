"""
services/lyrics_provider.py
LRCLib lyrics lookup — implements LyricsProvider protocol.

Tries to fetch synced (LRC-format) lyrics from the free, open LRCLib API
(https://lrclib.net). Returns list[TranscriptSegment] on hit, None on miss
or any HTTP/parse error so the caller can fall back to audio-based transcription.

Design notes:
- Pure HTTP — no GPU, no audio processing, no disk writes.
- All network errors degrade to None (logged as step_error); the caller is
  responsible for the fallback strategy.
- _parse_lrc and _best_result are pure module-level functions for independent
  testability.
- LRC timestamp format: [MM:SS.xx] or [MM:SS.xxx] followed by lyric text.
  Lines with no text (instrumental markers) are skipped.
- Segment end times are inferred from the start of the following line;
  the final segment gets a fixed 5-second tail.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Optional

from core.config import get_settings
from core.logging import PipelineLogger
from core.models import TranscriptSegment


_LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
_REQUEST_TIMEOUT_S = 8           # seconds before giving up on LRCLib
_FINAL_SEGMENT_TAIL_S = 5.0      # seconds added to the last segment's end time
_LRC_RE = re.compile(
    r"^\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)"
)


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
            OSError:       on network failure or non-200 status.
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


# ---------------------------------------------------------------------------
# Module-level pure functions — independently testable
# ---------------------------------------------------------------------------

def _best_result(results: list[dict]) -> dict | None:
    """
    Pick the best LRCLib result from a search response.

    Preference: first entry that has syncedLyrics (LRC format with timestamps).
    Falls back to None if no synced result exists — plain lyrics without
    timestamps are not useful for our compliance audit pipeline.

    Pure function — no I/O, no side effects.
    """
    if not results:
        return None
    for entry in results:
        if entry.get("syncedLyrics"):
            return entry
    return None


def _parse_lrc(lrc_text: str) -> list[TranscriptSegment]:
    """
    Parse LRC-format synced lyrics into TranscriptSegment objects.

    LRC format example:
        [00:12.45]First line of lyrics
        [00:17.32]Second line of lyrics
        [01:05.00]

    Rules:
    - Lines that don't match the timestamp pattern are skipped.
    - Lines with no text after the timestamp (e.g. instrumental markers)
      are skipped.
    - Segment end is inferred from the start of the next non-empty line.
    - The final segment end = its start + _FINAL_SEGMENT_TAIL_S.

    Pure function — no I/O, no side effects.

    Args:
        lrc_text: Raw LRC string from LRCLib syncedLyrics field.

    Returns:
        Ordered list of TranscriptSegment objects (may be empty).
    """
    timed: list[tuple[float, str]] = []

    for raw_line in lrc_text.splitlines():
        match = _LRC_RE.match(raw_line.strip())
        if not match:
            continue
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frac_str = match.group(3)
        # Normalise 2- or 3-digit fractional seconds
        frac = int(frac_str) / (1000 if len(frac_str) == 3 else 100)
        start = minutes * 60.0 + seconds + frac
        text = match.group(4).strip()
        if text:
            timed.append((round(start, 2), text))

    segments: list[TranscriptSegment] = []
    for i, (start, text) in enumerate(timed):
        end = (
            timed[i + 1][0]
            if i + 1 < len(timed)
            else start + _FINAL_SEGMENT_TAIL_S
        )
        segments.append(TranscriptSegment(
            start=start,
            end=round(end, 2),
            text=text,
        ))

    return segments
