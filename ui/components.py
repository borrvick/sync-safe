"""
ui/components.py
Shared HTML rendering helpers used across landing and report pages.

All functions return HTML strings — no Streamlit calls.
"""
from __future__ import annotations

import html as html_mod

from core.models import ComplianceFlag

# ── Issue type metadata ───────────────────────────────────────────────────────

# Named colour constants for issue types — hex required here because
# issue_pill() appends alpha suffixes for background/border transparency.
_CLR_BRAND: str    = "#F5A623"
_CLR_EXPLICIT: str = "#FF3060"
_CLR_VIOLENCE: str = "#FF6B35"
_CLR_LOCATION: str = "#60A5FA"   # blue — geographic references, not inherently harmful
_CLR_NEEDS_REVIEW: str = "#C8E86A"

ISSUE_META: dict[str, dict] = {
    "BRAND":    {"icon": "🏢", "color": _CLR_BRAND},
    "EXPLICIT": {"icon": "🔞", "color": _CLR_EXPLICIT},
    "LOCATION": {"icon": "📍", "color": _CLR_LOCATION},
    "VIOLENCE": {"icon": "⚠️",  "color": _CLR_VIOLENCE},
    "DRUGS":    {"icon": "💊", "color": _CLR_EXPLICIT},
}

# ── Authorship verdict → display color ───────────────────────────────────────

_AUTHORSHIP_COLORS: dict[str, str] = {
    "Likely AI":         "var(--danger)",
    "Uncertain":         _CLR_BRAND,
    "Likely Human":      "var(--ok)",
    "Insufficient data": "var(--dim)",
}


def authorship_color(verdict: str) -> str:
    return _AUTHORSHIP_COLORS.get(verdict, "var(--dim)")


# ── EQ bar animation HTML ─────────────────────────────────────────────────────

def eq_bars(n: int, color: str = "#F5640A", h: int = 40) -> str:
    """Return HTML for n animated EQ bars."""
    anims = ["eq-a", "eq-b", "eq-c"]
    parts = []
    for i in range(n):
        dur = f"{1.1 + (i % 3) * 0.22:.2f}s"
        dly = f"{i * 0.07:.2f}s"
        parts.append(
            f'<div style="width:3px;height:{h}px;background:{color};border-radius:2px;'
            f'transform-origin:bottom;animation:{anims[i%3]} {dur} ease-in-out {dly} infinite;'
            f'flex-shrink:0;"></div>'
        )
    return "".join(parts)


# ── Timestamp formatter ───────────────────────────────────────────────────────

def fmt_ts(secs: int) -> str:
    return f"{secs // 60}:{secs % 60:02d}"


# ── Issue pill HTML ───────────────────────────────────────────────────────────

def issue_pill(flag: ComplianceFlag, size: str = "sm") -> str:
    """Return a styled badge HTML string for a ComplianceFlag."""
    is_potential = flag.confidence == "potential"
    m     = ISSUE_META.get(flag.issue_type, {"icon": "⚠", "color": _CLR_BRAND})
    color = _CLR_NEEDS_REVIEW if is_potential else m["color"]
    icon  = "?" if is_potential else m["icon"]
    pad   = "1px 6px" if size == "sm" else "3px 10px"
    fsize = ".54rem" if size == "sm" else ".56rem"
    brad  = "3" if size == "sm" else "5"
    alpha = "18" if is_potential else ("22" if size == "sm" else "18")
    balpha = "33" if is_potential else ("44" if size == "sm" else "33")
    safe_type = html_mod.escape(flag.issue_type)
    safe_conf = html_mod.escape(str(flag.confidence))
    label = f"? {safe_type}" if is_potential else f"{icon} {safe_type}"
    return (
        f"<span data-confidence='{safe_conf}' style='font-family:\"Chakra Petch\",monospace;font-size:{fsize};"
        f"font-weight:600;padding:{pad};border-radius:{brad}px;"
        f"background:{color}{alpha};"
        f"color:{color};border:1px solid {color}{balpha};'>"
        f"{label}</span>"
    )
