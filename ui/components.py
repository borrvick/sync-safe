"""
ui/components.py
Shared HTML rendering helpers used across landing and report pages.

All functions return HTML strings — no Streamlit calls.
"""
from __future__ import annotations

from core.models import ComplianceFlag

# ── Issue type metadata ───────────────────────────────────────────────────────

ISSUE_META: dict[str, dict] = {
    "BRAND":    {"icon": "🏢", "color": "#F5A623"},
    "EXPLICIT": {"icon": "🔞", "color": "#FF3060"},
    "VIOLENCE": {"icon": "⚠️",  "color": "#FF6B35"},
    "DRUGS":    {"icon": "💊", "color": "#FF3060"},
    "LOCATION": {"icon": "📍", "color": "#4FC3F7"},
}

# ── Authorship verdict → display color ───────────────────────────────────────

_AUTHORSHIP_COLORS: dict[str, str] = {
    "Likely AI":         "#FF3060",
    "Uncertain":         "#F5A623",
    "Likely Human":      "#0DF5A0",
    "Insufficient data": "#364C5C",
}


def authorship_color(verdict: str) -> str:
    return _AUTHORSHIP_COLORS.get(verdict, "#364C5C")


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
    m     = ISSUE_META.get(flag.issue_type, {"icon": "⚠", "color": "#F5640A"})
    color = "#C8E86A" if is_potential else m["color"]
    icon  = "?" if is_potential else m["icon"]
    pad   = "1px 6px" if size == "sm" else "3px 10px"
    fsize = ".54rem" if size == "sm" else ".56rem"
    brad  = "3" if size == "sm" else "5"
    alpha = "18" if is_potential else ("22" if size == "sm" else "18")
    balpha = "33" if is_potential else ("44" if size == "sm" else "33")
    label = f"? {flag.issue_type}" if is_potential else f"{icon} {flag.issue_type}"
    return (
        f"<span style='font-family:\"Chakra Petch\",monospace;font-size:{fsize};"
        f"font-weight:600;padding:{pad};border-radius:{brad}px;"
        f"background:{color}{alpha};"
        f"color:{color};border:1px solid {color}{balpha};'>"
        f"{label}</span>"
    )
