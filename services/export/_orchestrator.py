"""
services/export/_orchestrator.py
Public API: to_platform_csv, to_section_markers_csv, to_analysis_json.
"""
from __future__ import annotations

import csv
import io

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


def to_analysis_json(result: AnalysisResult) -> bytes:
    """
    Serialise the full AnalysisResult to UTF-8 JSON bytes (#140, #123).

    Includes all pipeline outputs: structure + merged sections, forensics,
    compliance, legal / rights data (ISRC, PRO, confidence), popularity,
    audio quality, similar tracks, and sync cuts.
    Raw audio bytes are excluded — they are not a domain value.
    """
    return result.to_json().encode("utf-8")
