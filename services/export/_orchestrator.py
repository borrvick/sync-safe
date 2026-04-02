"""
services/platform_export.py
Platform-aware CSV export for sync licensing catalog imports.

Generates single-track CSV rows mapped to the column schema expected by
DISCO, Synchtank, and a generic platform-agnostic format.

Supported platforms and their catalog import column schemas:
- generic   — all Sync-Safe fields; useful for custom pipelines
- disco     — DISCO catalog import (title, artist, bpm, key, isrc, tags, notes)
- synchtank — Synchtank catalog manager CSV (track_title, artist_name, tempo,
              key_signature, isrc_code, description)

Design:
- All functions are pure (no I/O, no Streamlit imports).
- _extract_track_data is the single source of truth for data extraction.
- _to_platform_row maps generic keys to platform column names and formats.
- to_platform_csv is the public API — returns UTF-8-with-BOM bytes (Excel safe).
"""
from __future__ import annotations

import csv
import io

from core.models import AnalysisResult

# ---------------------------------------------------------------------------
# Platform schema definitions
# ---------------------------------------------------------------------------
# Maps platform name → ordered list of CSV column names.
# These match the documented import templates for each platform.

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

# Verdict strings from ForensicsResult that indicate AI-generated content.
# Used to compute SYNC_SAFE_CLEARED.
_AI_VERDICTS: frozenset[str] = frozenset({"AI", "Likely AI"})


# ---------------------------------------------------------------------------
# Internal data extraction — pure, no I/O
# ---------------------------------------------------------------------------

def _extract_audio_fields(result: AnalysisResult) -> dict[str, str]:
    """Extract title, artist, BPM, key, ISRC, PRO, and LUFS. Pure — no I/O."""
    title  = result.audio.metadata.get("title", "") or result.audio.label or ""
    artist = result.audio.metadata.get("artist", "") or ""
    bpm = ""
    key = ""
    if result.structure:
        bpm = str(round(float(result.structure.bpm), 1)) if isinstance(result.structure.bpm, (int, float)) else ""
        key = result.structure.key or ""
    isrc = (result.legal.isrc or "") if result.legal else ""
    pro  = (result.legal.pro_match or "") if result.legal else ""
    lufs = str(result.audio_quality.integrated_lufs) if result.audio_quality else ""
    return {"title": title, "artist": artist, "bpm": bpm, "key": key,
            "isrc": isrc, "pro": pro, "lufs_integrated": lufs}


def _extract_verdict_fields(result: AnalysisResult) -> dict[str, str]:
    """Extract AI probability, verdict, grade, flags, and cleared status. Pure — no I/O."""
    ai_prob = ""
    verdict = ""
    if result.forensics:
        ai_prob = str(round(result.forensics.ai_probability * 100.0, 1))
        verdict = result.forensics.verdict

    grade      = (result.compliance.grade or "") if result.compliance else ""
    flag_types = sorted({f.issue_type for f in result.compliance.flags}) if result.compliance else []
    cleared    = (
        grade in ("A", "B")
        and not (result.compliance and result.compliance.hard_flags)
        and verdict not in _AI_VERDICTS
    )
    return {
        "ai_probability_pct": ai_prob,
        "verdict":            verdict,
        "grade":              grade,
        "flags":              " | ".join(flag_types) if flag_types else "none",
        "sync_safe_cleared":  "true" if cleared else "false",
    }


def _extract_track_data(result: AnalysisResult) -> dict[str, str]:
    """
    Build a normalised dict of track data from an AnalysisResult.

    All values are strings for direct CSV serialisation.
    Pure function — no I/O.
    """
    return {**_extract_audio_fields(result), **_extract_verdict_fields(result)}


def _disco_row(data: dict[str, str]) -> dict[str, str]:
    """Map normalised data to DISCO catalog import columns. Pure — no I/O."""
    tags_parts = [f"GRADE:{data['grade']}"] if data["grade"] else []
    if data["verdict"]:
        tags_parts.append(f"AI-VERDICT:{data['verdict'].replace(' ', '-')}")
    if data["flags"] != "none":
        for f in data["flags"].split(" | "):
            tags_parts.append(f"FLAG:{f}")
    if data["sync_safe_cleared"] == "true":
        tags_parts.append("SYNC_SAFE_CLEARED")

    notes_parts = []
    if data["ai_probability_pct"]:
        notes_parts.append(f"AI probability: {data['ai_probability_pct']}%")
    if data["lufs_integrated"]:
        notes_parts.append(f"Integrated LUFS: {data['lufs_integrated']}")
    if data["pro"]:
        notes_parts.append(f"PRO: {data['pro']}")

    return {
        "title": data["title"], "artist": data["artist"],
        "bpm":   data["bpm"],   "key":    data["key"],
        "isrc":  data["isrc"],
        "tags":  ", ".join(tags_parts),
        "notes": " | ".join(notes_parts),
    }


def _synchtank_row(data: dict[str, str]) -> dict[str, str]:
    """Map normalised data to Synchtank catalog import columns. Pure — no I/O."""
    desc_parts = []
    if data["grade"]:
        desc_parts.append(f"Sync-Safe Grade: {data['grade']}")
    if data["ai_probability_pct"] and data["verdict"]:
        desc_parts.append(f"AI: {data['ai_probability_pct']}% ({data['verdict']})")
    if data["flags"] != "none":
        desc_parts.append(f"Flags: {data['flags']}")
    if data["sync_safe_cleared"] == "true":
        desc_parts.append("Pre-cleared: yes")
    if data["lufs_integrated"]:
        desc_parts.append(f"LUFS: {data['lufs_integrated']}")

    return {
        "track_title":   data["title"],  "artist_name":   data["artist"],
        "tempo":         data["bpm"],    "key_signature": data["key"],
        "isrc_code":     data["isrc"],
        "description":   " | ".join(desc_parts),
    }


def _to_platform_row(data: dict[str, str], platform: str) -> dict[str, str]:
    """
    Map normalised track data to a platform-specific column dict.

    Pure function — no I/O.

    Args:
        data:     Output of _extract_track_data.
        platform: One of "generic", "disco", "synchtank".

    Returns:
        Dict keyed by platform column names, values as strings.

    Raises:
        ValueError: if *platform* is not a recognised schema name.
    """
    if platform == "generic":
        return {col: data.get(col, "") for col in PLATFORM_SCHEMAS["generic"]}
    if platform == "disco":
        return _disco_row(data)
    if platform == "synchtank":
        return _synchtank_row(data)
    raise ValueError(f"Unknown platform '{platform}'. Valid: {list(PLATFORM_SCHEMAS)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_platform_csv(result: AnalysisResult, platform: str = "generic") -> bytes:
    """
    Generate a single-row platform CSV for catalog import.

    Pure function — no I/O, no Streamlit calls.

    Args:
        result:   Complete pipeline AnalysisResult.
        platform: Target platform schema. One of "generic", "disco", "synchtank".

    Returns:
        UTF-8-with-BOM encoded CSV bytes (BOM ensures Excel opens correctly).

    Raises:
        ValueError: if *platform* is not a recognised schema name.
    """
    if platform not in PLATFORM_SCHEMAS:
        raise ValueError(f"Unknown platform '{platform}'. Valid: {list(PLATFORM_SCHEMAS)}")
    columns = PLATFORM_SCHEMAS[platform]
    data    = _extract_track_data(result)
    row     = _to_platform_row(data, platform)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerow(row)

    return buf.getvalue().encode("utf-8-sig")
