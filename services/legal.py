"""
services/legal.py
PRO repertory search link generator — implements LegalLinksProvider.

Generates ASCAP, BMI, and SESAC search URLs for a given title and artist.
Pure URL construction — no network calls, no GPU, no external dependencies
beyond the standard library.

Design notes:
- Legal.get_links() is the single public entry point.
- _PROS is a module-level constant mapping PRO name → (base_url, param_fn).
  Adding a new PRO requires only a new entry here — no class changes.
- _build_url is a pure module-level function for independent unit testing.
- Returns LegalLinks (typed domain model) not a raw dict, so callers get
  IDE autocomplete and type-checker validation.
"""
from __future__ import annotations

from urllib.parse import urlencode

from core.models import LegalLinks


# ---------------------------------------------------------------------------
# PRO endpoint registry
# ---------------------------------------------------------------------------

_PROS: dict[str, tuple[str, dict]] = {
    "ascap": (
        "https://www.ascap.com/repertory",
        lambda title, artist: {"query": f"{artist} {title}".strip(), "type": "1"},
    ),
    "bmi": (
        "https://repertoire.bmi.com/Search/BMIRepertoireSearch",
        lambda title, artist: {
            "search-type": "all",
            "search-text": f"{artist} {title}".strip(),
        },
    ),
    "sesac": (
        "https://www.sesac.com/repertory/search",
        lambda title, artist: {
            "q": f"{artist} {title}".strip(),
            "type": "works",
        },
    ),
}


class Legal:
    """
    Generates PRO repertory search URLs for sync licensing due diligence.

    Implements: LegalLinksProvider protocol (core/protocols.py)

    Usage:
        service = Legal()
        links   = service.get_links("Blinding Lights", "The Weeknd")
        print(links.ascap, links.bmi, links.sesac)
    """

    # ------------------------------------------------------------------
    # Public interface (LegalLinksProvider protocol)
    # ------------------------------------------------------------------

    def get_links(self, title: str, artist: str) -> LegalLinks:
        """
        Build PRO repertory search URLs for a given title and artist.

        Args:
            title:  Track title (may be empty).
            artist: Artist name (may be empty).

        Returns:
            LegalLinks with ascap, bmi, and sesac URL strings.
        """
        urls = {
            pro: _build_url(base_url, param_fn(title.strip(), artist.strip()))
            for pro, (base_url, param_fn) in _PROS.items()
        }
        return LegalLinks(
            ascap=urls["ascap"],
            bmi=urls["bmi"],
            sesac=urls["sesac"],
        )


# ---------------------------------------------------------------------------
# Module-level pure function — independently testable
# ---------------------------------------------------------------------------

def _build_url(base_url: str, params: dict) -> str:
    """Append URL-encoded query parameters to a base URL string."""
    return f"{base_url}?{urlencode(params)}"
