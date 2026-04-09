"""
services/export/_mappers.py
Pure data-extraction and platform-mapping functions.
"""
from __future__ import annotations

from core.models import AnalysisResult, Section

from ._schema import PLATFORM_SCHEMAS, SECTION_MARKERS_COLUMNS, _AI_VERDICTS

# Section label → hex color for DAW marker import (#138).
# Colors chosen to align with the report timeline palette.
_SECTION_COLOR_MAP: dict[str, str] = {
    "chorus":       "#4A90D9",  # blue  (matches var(--accent))
    "hook":         "#4A90D9",  # blue
    "refrain":      "#4A90D9",  # blue
    "verse":        "#7ED321",  # green
    "bridge":       "#F5A623",  # amber
    "intro":        "#9B59B6",  # purple
    "outro":        "#95A5A6",  # grey
    "pre-chorus":   "#E67E22",  # orange
    "post-chorus":  "#E74C3C",  # red
    "instrumental": "#1ABC9C",  # teal
    "solo":         "#1ABC9C",  # teal
    "break":        "#BDC3C7",  # light grey
    "interlude":    "#BDC3C7",  # light grey
}
_SECTION_COLOR_DEFAULT: str = "#888888"


def _format_timecode(seconds: float) -> str:
    """Format seconds as M:SS.mmm for DAW marker import. Pure — no I/O."""
    mins = int(seconds) // 60
    secs = seconds - mins * 60
    return f"{mins}:{secs:06.3f}"


def _section_rows(sections: list[Section]) -> list[dict[str, str]]:
    """Convert Section objects to DAW marker dicts. Pure — no I/O."""
    rows: list[dict[str, str]] = []
    for idx, sec in enumerate(sections, 1):
        duration = round(sec.end - sec.start, 3)
        color    = _SECTION_COLOR_MAP.get(sec.label.lower(), _SECTION_COLOR_DEFAULT)
        rows.append({
            "marker_name":    f"{idx:02d} {sec.label.title()}",
            "start_timecode": _format_timecode(sec.start),
            "end_timecode":   _format_timecode(sec.end),
            "duration_s":     f"{duration:.3f}",
            "color_hex":      color,
        })
    return rows


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
    """Build a normalised dict of track data from an AnalysisResult. Pure — no I/O."""
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
