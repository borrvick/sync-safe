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

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PRO inference table — ISO-3166-1 alpha-2 country codes → PRO name
# ---------------------------------------------------------------------------

_COUNTRY_PRO_MAP: dict[str, str] = {
    "US": "ASCAP / BMI / SESAC (US)",
    "GB": "PRS for Music (UK)",
    "DE": "GEMA (Germany)",
    "FR": "SACEM (France)",
    "CA": "SOCAN (Canada)",
    "AU": "APRA AMCOS (Australia)",
    "SE": "STIM (Sweden)",
    "NO": "TONO (Norway)",
    "DK": "KODA (Denmark)",
    "FI": "Teosto (Finland)",
    "NL": "Buma/Stemra (Netherlands)",
    "BE": "SABAM (Belgium)",
    "IT": "SIAE (Italy)",
    "ES": "SGAE (Spain)",
    "BR": "ECAD (Brazil)",
    "JP": "JASRAC (Japan)",
    "KR": "KOMCA (South Korea)",
    "MX": "SACM (Mexico)",
}

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


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _parse_first_recording(
    data: dict,
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract ISRC and infer PRO from the first MusicBrainz recording result.

    Pure function — no I/O.
    """
    recordings: list[dict] = data.get("recordings", [])
    if not recordings:
        return None, None

    first = recordings[0]

    # ISRC — list[str]; take the first if present
    isrcs: list[str] = first.get("isrcs", [])
    isrc = isrcs[0] if isrcs else None

    # Infer PRO from ISRC country prefix (first 2 chars of ISRC = country)
    pro: Optional[str] = None
    if isrc and len(isrc) >= 2:
        country_code = isrc[:2].upper()
        pro = _infer_pro(country_code)

    return isrc, pro


def _infer_pro(country_code: str) -> Optional[str]:
    """
    Map an ISO-3166-1 alpha-2 country code to a PRO name string.

    Pure function — no I/O.

    Args:
        country_code: Two-letter uppercase country code (e.g. "US", "GB").

    Returns:
        Human-readable PRO name, or None if country is not in the lookup table.
    """
    return _COUNTRY_PRO_MAP.get(country_code.upper())
