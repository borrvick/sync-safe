"""
ui/pages/raw_data.py

Raw Data page — renders the flat TrackReport for a completed analysis.

Shows every pipeline data point in grouped tables and expanders, plus a
CSV download button. Intended for internal use / QA, not end-user facing.

TrackReport is computed lazily on first render and cached in session_state
so navigating back to the report and returning here doesn't recompute it.
"""
from __future__ import annotations

import json

import streamlit as st

from core.models import AnalysisResult
from core.report import TrackReport
from services.report_exporter import ReportExporter

_EXPORTER = ReportExporter()


# ---------------------------------------------------------------------------
# Field groups for display (ordered)
# ---------------------------------------------------------------------------

_SCALAR_GROUPS: list[tuple[str, list[str]]] = [
    ("Identity", [
        "track_id", "scan_timestamp",
    ]),
    ("Audio / Ingestion", [
        "title", "artist", "source", "sample_rate",
        "yt_view_count", "yt_like_count",
    ]),
    ("Structure", [
        "bpm", "key", "duration_s", "beat_count", "section_count",
        "intro_s", "verse_count", "chorus_count", "bridge_count",
    ]),
    ("Forensics — Verdict", [
        "forensic_verdict", "ai_probability", "forensic_flag_count",
        "c2pa_flag", "c2pa_origin", "is_vocal",
    ]),
    ("Forensics — Raw Signals", [
        "ibi_variance", "loop_score", "loop_autocorr_score",
        "spectral_slop", "synthid_score",
        "centroid_instability_score", "harmonic_ratio_score",
        "kurtosis_variability", "decoder_peak_score", "spectral_centroid_mean",
        "self_similarity_entropy", "noise_floor_ratio", "onset_strength_cv",
        "spectral_flatness_var", "subbeat_grid_deviation",
        "pitch_quantization_score", "ultrasonic_noise_ratio",
        "infrasonic_energy_ratio", "phase_coherence_differential",
        "plr_std", "voiced_noise_floor",
    ]),
    ("Audio Quality / Loudness", [
        "integrated_lufs", "true_peak_dbfs", "loudness_range_lu",
        "delta_spotify", "delta_apple_music", "delta_youtube", "delta_broadcast",
        "true_peak_warning", "dialogue_score", "dialogue_label",
    ]),
    ("Stem Validation", [
        "mono_compatible", "phase_correlation",
        "cancellation_db", "mid_side_ratio", "stem_flag_count",
    ]),
    ("Compliance", [
        "compliance_grade",
        "total_flag_count", "confirmed_flag_count", "potential_flag_count",
        "hard_flag_count", "soft_flag_count",
        "sting_flag", "sting_ending_type", "sting_final_energy_ratio",
        "energy_evolution_flag", "stagnant_windows", "total_windows",
        "intro_flag", "intro_seconds", "intro_source",
    ]),
    ("Authorship", [
        "authorship_verdict", "authorship_signal_count", "roberta_score",
        "burstiness_score", "unique_word_ratio", "rhyme_density", "repetition_score",
    ]),
    ("Theme & Mood", [
        "mood", "theme_confidence", "groq_enriched",
    ]),
    ("Popularity & Cost", [
        "popularity_score", "popularity_tier",
        "lastfm_listeners", "lastfm_playcount", "spotify_score",
        "sync_cost_low", "sync_cost_high",
    ]),
    ("Legal", [
        "isrc", "pro_match",
    ]),
    ("Metadata Validation", [
        "metadata_valid", "missing_fields_count",
        "split_total", "split_error", "isrc_valid",
    ]),
]

_JSON_BLOBS: list[tuple[str, str]] = [
    ("Compliance Flags",    "compliance_flags_json"),
    ("AI Segments",         "ai_segments_json"),
    ("Sections",            "sections_json"),
    ("Transcript",          "transcript_json"),
    ("Similar Tracks",      "similar_tracks_json"),
    ("Sync Cuts",           "sync_cuts_json"),
    ("Forensic Flags",      "forensic_flags_json"),
    ("Forensic Notes",      "forensic_notes_json"),
    ("Themes",              "themes_json"),
    ("Theme Keywords",      "theme_keywords_json"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_build_report(analysis: AnalysisResult) -> TrackReport:
    """Return cached TrackReport or build it on first call."""
    if st.session_state.get("track_report") is None:
        st.session_state["track_report"] = _EXPORTER.build(analysis)
    return st.session_state["track_report"]  # type: ignore[return-value]


def _render_scalar_group(label: str, fields: list[str], row: dict) -> None:
    with st.expander(label, expanded=True):
        pairs = [(k, row[k]) for k in fields if k in row]
        if not pairs:
            st.caption("No data.")
            return
        col_field, col_val = st.columns([2, 3])
        col_field.markdown("**Field**")
        col_val.markdown("**Value**")
        for field, value in pairs:
            col_field.text(field)
            col_val.text("—" if value is None else str(value))


def _render_json_blob(label: str, key: str, row: dict) -> None:
    raw = row.get(key, "[]")
    parse_error = False
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = raw
        parse_error = True

    count = len(parsed) if isinstance(parsed, list) else 1
    header = f"{label}  ({count} item{'s' if count != 1 else ''})"

    with st.expander(header, expanded=False):
        if parse_error:
            st.caption("⚠️ Could not parse JSON blob.")
            st.text(str(raw))
        elif not parsed or parsed == []:
            st.caption("Empty.")
        elif isinstance(parsed, list):
            st.dataframe(parsed, use_container_width=True)
        else:
            st.json(parsed)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_raw_data(analysis: AnalysisResult) -> None:
    report = _get_or_build_report(analysis)
    row    = report.to_dict()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("## Raw Data")
    st.caption(
        f"Track ID `{report.track_id}` · "
        f"Scanned {report.scan_timestamp[:19].replace('T', ' ')} UTC · "
        f"{len(row)} fields"
    )

    # ── Download ─────────────────────────────────────────────────────────────
    csv_bytes = _EXPORTER.to_csv(report)
    filename  = f"sync_safe_{report.track_id}_{report.scan_timestamp[:10]}.csv"
    st.download_button(
        label       = "Download CSV",
        data        = csv_bytes,
        file_name   = filename,
        mime        = "text/csv",
    )

    st.divider()

    # ── Scalar groups ────────────────────────────────────────────────────────
    st.markdown("### Scalar Fields")
    for group_label, fields in _SCALAR_GROUPS:
        _render_scalar_group(group_label, fields, row)

    st.divider()

    # ── JSON blobs ───────────────────────────────────────────────────────────
    st.markdown("### Collections (JSON)")
    for blob_label, blob_key in _JSON_BLOBS:
        _render_json_blob(blob_label, blob_key, row)

    # ── Back navigation ──────────────────────────────────────────────────────
    st.divider()
    if st.button("← Back to Report"):
        st.session_state.page = "report"
        st.rerun()
