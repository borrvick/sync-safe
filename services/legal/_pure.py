"""
services/legal/_pure.py
Shared pure helpers for legal and PRO lookup — no network, no I/O.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlencode

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


def _build_url(base_url: str, params: dict) -> str:
    """Append URL-encoded query parameters to a base URL string."""
    return f"{base_url}?{urlencode(params)}"


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

    isrcs: list[str] = first.get("isrcs", [])
    isrc = isrcs[0] if isrcs else None

    pro: Optional[str] = None
    if isrc and len(isrc) >= 2:
        country_code = isrc[:2].upper()
        pro = _infer_pro(country_code)

    return isrc, pro
