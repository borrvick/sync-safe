"""
services/pro_lookup.py
MusicBrainz-based PRO (Performing Rights Organisation) lookup.

Queries the MusicBrainz recordings API to:
1. Find a matching recording for a given title + artist.
2. Extract the ISRC if available (sound recording identifier).
3. Infer the likely PRO from the ISRC country prefix or recording area.

Design notes:
- ProLookup is stateless — instantiate per call.
- _infer_pro and _parse_first_recording are pure functions for testability.
- MusicBrainz requires a descriptive User-Agent; sourced from Settings.musicbrainz_app.
- Rate limit: MusicBrainz allows max 1 req/s — caller (loading.py) is already
  sequential; no explicit sleep required for single-call use.
- Returns (None, None) on any network failure — PRO lookup is best-effort.
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlencode

import requests

from core.config import CONSTANTS, get_settings

from ._pure import _infer_pro, _parse_first_recording

_log = logging.getLogger(__name__)

_MB_API_BASE = "https://musicbrainz.org/ws/2"


class ProLookup:
    """
    Queries MusicBrainz to infer ISRC and PRO for a given track.

    Usage:
        isrc, pro = ProLookup().lookup("Blinding Lights", "The Weeknd")
    """

    def lookup(self, title: str, artist: str) -> tuple[Optional[str], Optional[str]]:
        """
        Look up ISRC and inferred PRO affiliation via MusicBrainz.

        Args:
            title:  Track title.
            artist: Artist name.

        Returns:
            (isrc, pro_match) — both None if no confident match found or on error.
            This method never raises — PRO lookup is best-effort.
        """
        if not title.strip() or not artist.strip():
            return None, None

        settings = get_settings()
        headers  = {"User-Agent": settings.musicbrainz_app}

        # urlencode handles quoting of the full query value (including the Lucene
        # quotes and spaces). inc uses '+' as MB's documented separator — appended
        # outside urlencode so the '+' is not percent-encoded to '%2B'.
        query_str = f'recording:"{title.strip()}" AND artist:"{artist.strip()}"'
        params    = urlencode({"query": query_str, "fmt": "json", "limit": "3"})
        url       = f"{_MB_API_BASE}/recording/?{params}&inc=isrcs+releases"

        try:
            resp = requests.get(url, headers=headers, timeout=CONSTANTS.MB_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            # Network failures are non-fatal — log and return empty result
            _log.warning("ProLookup: MusicBrainz request failed: %s", exc)
            return None, None

        return _parse_first_recording(data)
