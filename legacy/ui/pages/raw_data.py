"""
ui/pages/raw_data.py

Raw Data page — full pipeline output as structured, downloadable JSON.

Every field produced by every pipeline stage is included, including fields
not rendered on the report page (gain recommendations, per-section loudness,
per-section authorship, section similarities, sting enrichment, etc.).

Source of truth: AnalysisResult.to_dict() — all nested model fields are
captured automatically, so new pipeline signals appear here without any
changes to this file. TrackReport contributes _meta identity/scalar fields
(track_id, scan_timestamp, derived counts) that are not part of AnalysisResult.

Use this view to:
  - Inspect raw signal values and verify computations
  - Reference complete data during future DB schema design
  - Export a full JSON payload for external tooling
"""
from __future__ import annotations

import json
from typing import Any, Optional

import streamlit as st

from core.models import AnalysisResult
from core.report import TrackReport
from services.report_exporter import ReportExporter

_EXPORTER = ReportExporter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_build_report(analysis: AnalysisResult) -> TrackReport:
    """Return cached TrackReport or build on first call."""
    if st.session_state.get("track_report") is None:
        st.session_state["track_report"] = _EXPORTER.build(analysis)
    return st.session_state["track_report"]  # type: ignore[return-value]


def _enrich_structure(
    raw_structure: Optional[dict[str, Any]],
    report: TrackReport,
) -> Optional[dict[str, Any]]:
    """Add derived scalar counts to the structure section."""
    if raw_structure is None:
        return None
    return {
        **raw_structure,
        # Derived counts not stored on StructureResult itself
        "duration_s":     report.duration_s,
        "beat_count":     report.beat_count,
        "section_count":  report.section_count,
        "intro_s":        report.intro_s,
        "verse_count":    report.verse_count,
        "chorus_count":   report.chorus_count,
        "bridge_count":   report.bridge_count,
    }


def _build_full_json(analysis: AnalysisResult, report: TrackReport) -> dict[str, Any]:
    """
    Build a complete, ordered JSON object from the analysis result.

    Uses AnalysisResult.to_dict() as the authoritative data source so that
    every field — including ones added after this file was written — is
    captured automatically without code changes here.

    Pure function — no I/O.
    """
    r = analysis.to_dict()

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "_meta": {
            "track_id":       report.track_id,
            "scan_timestamp": report.scan_timestamp,
            "schema_version": "2.0",
        },

        # ── Ingestion ─────────────────────────────────────────────────────
        "audio": r.get("audio"),

        # ── Structure + derived counts ────────────────────────────────────
        "structure": _enrich_structure(r.get("structure"), report),

        # ── AI / Humanity Forensics ───────────────────────────────────────
        "forensics": r.get("forensics"),

        # ── Sync Readiness Compliance ─────────────────────────────────────
        "compliance": r.get("compliance"),

        # ── Lyric Authorship ──────────────────────────────────────────────
        "authorship": r.get("authorship"),

        # ── Theme & Mood ──────────────────────────────────────────────────
        "theme_mood": r.get("theme_mood"),

        # ── Broadcast Loudness & Dialogue ─────────────────────────────────
        "audio_quality": r.get("audio_quality"),

        # ── Lyric Transcript ──────────────────────────────────────────────
        "transcript": r.get("transcript"),

        # ── Sync Edit Points ──────────────────────────────────────────────
        "sync_cuts": r.get("sync_cuts"),

        # ── Similar Tracks ────────────────────────────────────────────────
        "similar_tracks": r.get("similar_tracks"),

        # ── Popularity & Sync Fees ────────────────────────────────────────
        "popularity": r.get("popularity"),

        # ── PRO / Legal Links ─────────────────────────────────────────────
        "legal": r.get("legal"),

        # ── Rights Metadata Validation ────────────────────────────────────
        "metadata_validation": r.get("metadata_validation"),

        # ── Stem / Mono Compatibility ─────────────────────────────────────
        "stem_validation": r.get("stem_validation"),
    }


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_raw_data(analysis: AnalysisResult) -> None:
    """Render the raw data page."""
    report   = _get_or_build_report(analysis)
    payload  = _build_full_json(analysis, report)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("## Raw Data")
    st.caption(
        f"Track ID `{report.track_id}` · "
        f"Scanned {report.scan_timestamp[:19].replace('T', ' ')} UTC"
    )

    # ── Downloads ────────────────────────────────────────────────────────────
    col_json, col_csv, col_back = st.columns([2, 2, 3])

    json_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
    json_name  = f"sync_safe_{report.track_id}_{report.scan_timestamp[:10]}.json"
    col_json.download_button(
        label     = "Download JSON",
        data      = json_bytes,
        file_name = json_name,
        mime      = "application/json",
    )

    csv_bytes = _EXPORTER.to_csv(report)
    csv_name  = f"sync_safe_{report.track_id}_{report.scan_timestamp[:10]}.csv"
    col_csv.download_button(
        label     = "Download CSV",
        data      = csv_bytes,
        file_name = csv_name,
        mime      = "text/csv",
    )

    with col_back:
        if st.button("← Back to Report", use_container_width=True):
            st.session_state.page = "report"
            st.rerun()

    st.divider()

    # ── JSON viewer ───────────────────────────────────────────────────────────
    st.json(payload, expanded=2)
