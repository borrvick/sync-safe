"""
services/legal/_pure.py
Shared pure helpers for legal and PRO lookup — no network, no I/O.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus, urlencode

from core.config import CONSTANTS

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


def hfa_url(title: str, artist: str) -> str:
    """HFA (Harry Fox Agency) mechanical rights search URL.

    Args:
        title:  Track title.
        artist: Artist name.

    Returns:
        URL-encoded search URL for harryfox.com.
    """
    q = quote_plus(f"{title} {artist}".strip())
    return f"https://www.harryfox.com/find_music?q={q}"


def songfile_url(title: str, artist: str) -> str:
    """Songfile (HFA mechanical licensing portal) search URL.

    Args:
        title:  Track title.
        artist: Artist name.

    Returns:
        URL-encoded search URL for songfile.com.
    """
    q = quote_plus(f"{title} {artist}".strip())
    return f"https://songfile.com/search?q={q}"


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
) -> tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Extract ISRC, infer PRO, and return MusicBrainz relevance metadata
    from the first recording result.

    Returns:
        (isrc, pro_match, mb_score, mb_artist) — all None when no match.
        mb_score: MusicBrainz relevance score 0–100 (None if absent).
        mb_artist: first credited artist name from the MB response (None if absent).

    Pure function — no I/O.
    """
    recordings: list[dict] = data.get("recordings", [])
    if not recordings:
        return None, None, None, None

    first = recordings[0]

    isrcs: list[str] = first.get("isrcs", [])
    isrc = isrcs[0] if isrcs else None

    pro: Optional[str] = None
    if isrc and len(isrc) >= 2:
        country_code = isrc[:2].upper()
        pro = _infer_pro(country_code)

    try:
        mb_score: Optional[int] = int(first.get("score", 0) or 0)
    except (ValueError, TypeError):
        mb_score = None

    mb_artist: Optional[str] = None
    credits = first.get("artist-credit", [])
    if credits and isinstance(credits[0], dict):
        artist_obj = credits[0].get("artist", {})
        mb_artist  = artist_obj.get("name") or None

    return isrc, pro, mb_score, mb_artist


def _compute_pro_confidence(
    mb_score: Optional[int],
    mb_artist: Optional[str],
    query_artist: str,
    is_country_inference: bool,
) -> str:
    """
    Return 'High', 'Medium', or 'Low' PRO confidence (#118).

    Country-only inference (no MusicBrainz hit) is always 'Low' because
    an ISRC country prefix does not guarantee PRO affiliation — a US ISRC
    may belong to a UK-based artist recorded domestically.

    Pure function — no I/O.
    """
    if is_country_inference:
        return "Low"

    score_ok = mb_score is not None and mb_score > CONSTANTS.PRO_CONFIDENCE_MB_SCORE_THRESHOLD

    overlap_ok = False
    if mb_artist and query_artist:
        mb_tokens = set(mb_artist.lower().split())
        q_tokens  = set(query_artist.lower().split())
        if mb_tokens and q_tokens:
            overlap = len(mb_tokens & q_tokens) / max(len(mb_tokens), len(q_tokens))
            overlap_ok = overlap >= CONSTANTS.PRO_CONFIDENCE_ARTIST_OVERLAP

    if score_ok and overlap_ok:
        return "High"
    if score_ok or overlap_ok:
        return "Medium"
    return "Low"
