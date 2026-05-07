"""
services/lyrics/_pure.py
Pure LRC parsing functions — no HTTP, no I/O.
"""
from __future__ import annotations

import re

from core.models import TranscriptSegment

_FINAL_SEGMENT_TAIL_S = 5.0
_LRC_RE = re.compile(r"^\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)")


def _best_result(results: list[dict]) -> dict | None:
    """
    Pick the best LRCLib result from a search response.

    Preference: first entry that has syncedLyrics (LRC format with timestamps).
    Falls back to None if no synced result exists.

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
