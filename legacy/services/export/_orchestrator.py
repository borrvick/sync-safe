"""
services/export/_orchestrator.py
Public API: to_platform_csv, to_section_markers_csv, to_analysis_json.
"""
from __future__ import annotations

import csv
import io
import json

from core.config import CONSTANTS
from core.models import AnalysisResult, Section

from ._mappers import _extract_track_data, _section_rows, _to_platform_row
from ._schema import PLATFORM_SCHEMAS, SECTION_MARKERS_COLUMNS


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


def to_section_markers_csv(sections: list[Section]) -> bytes:
    """
    Generate a DAW-ready section markers CSV from a list of Section objects (#138).

    Columns: marker_name, start_timecode, end_timecode, duration_s, color_hex.
    Timecodes use M:SS.mmm format, readable by Reaper, Logic, and most DAWs
    that accept timecode-based marker imports.

    Returns:
        UTF-8-with-BOM encoded CSV bytes.  Returns a header-only file when
        *sections* is empty so callers always receive valid CSV.
    """
    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SECTION_MARKERS_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_section_rows(sections))
    return buf.getvalue().encode("utf-8-sig")


def _build_sync_fee_section(result: AnalysisResult) -> dict:
    """
    Build the sync_fee section for the JSON export (#116).

    Reads base ranges from result.popularity (already in the model).
    multipliers_applied is null until the scenario slider UI (#110) is
    implemented — callers should treat null as 'no adjustment applied'.

    Pure function — no I/O, no Streamlit calls.
    """
    pop = result.popularity
    if pop is None:
        return {
            "fee_tier":           None,
            "tier_confidence":    None,
            "base_range_low":     None,
            "base_range_high":    None,
            "adjusted_range_low": None,
            "adjusted_range_high": None,
            "multipliers_applied": None,
            "popularity_score":   None,
        }
    return {
        "fee_tier":           pop.tier,
        "tier_confidence":    "strong" if pop.popularity_score >= CONSTANTS.SYNC_FEE_STRONG_CONFIDENCE_THRESHOLD else "moderate",
        "base_range_low":     pop.sync_cost_low,
        "base_range_high":    pop.sync_cost_high,
        "adjusted_range_low": None,   # populated when #110 scenario sliders are added
        "adjusted_range_high": None,
        "multipliers_applied": None,
        "popularity_score":   pop.popularity_score,
    }


def to_analysis_json(result: AnalysisResult) -> bytes:
    """
    Serialise the full AnalysisResult to UTF-8 JSON bytes (#140, #123, #116).

    Includes all pipeline outputs: structure + merged sections, forensics,
    compliance, legal / rights data (ISRC, PRO, confidence), popularity,
    audio quality, similar tracks, sync cuts, and sync fee breakdown.
    Raw audio bytes are excluded — they are not a domain value.
    """
    payload = json.loads(result.to_json())
    payload["sync_fee"] = _build_sync_fee_section(result)
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
