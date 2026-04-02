"""
services/export/_orchestrator.py
Public API: to_platform_csv — generates platform-specific CSV bytes.
"""
from __future__ import annotations

import csv
import io

from core.models import AnalysisResult

from ._mappers import _extract_track_data, _to_platform_row
from ._schema import PLATFORM_SCHEMAS


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
