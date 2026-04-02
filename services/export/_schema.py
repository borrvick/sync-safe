"""
services/export/_schema.py
Platform CSV column schema definitions.
"""
from __future__ import annotations

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
