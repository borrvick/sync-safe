"""
services/export/_schema.py
Platform CSV column schema definitions.
"""
from __future__ import annotations

# DAW-ready section marker CSV columns (#138).
# Compatible with Reaper, Logic, and most DAWs that accept timecode marker imports.
SECTION_MARKERS_COLUMNS: list[str] = [
    "marker_name", "start_timecode", "end_timecode", "duration_s", "color_hex",
]

PLATFORM_SCHEMAS: dict[str, list[str]] = {
    "generic": [
        "title", "artist", "bpm", "key", "isrc", "pro",
        "ai_probability_pct", "verdict", "grade", "flags",
        "sync_safe_cleared", "lufs_integrated",
    ],
    "disco": [
        "title", "artist", "bpm", "key", "isrc", "tags", "notes",
    ],
    "synchtank": [
        "track_title", "artist_name", "tempo", "key_signature",
        "isrc_code", "description",
    ],
}

# Verdict strings that indicate AI-generated content.
_AI_VERDICTS: frozenset[str] = frozenset({"AI", "Likely AI"})
