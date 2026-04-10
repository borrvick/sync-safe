"""
ui/pages/report.py
Report page rendering — takes a typed AnalysisResult and renders all sections.

All functions accept typed domain models from core.models; no raw dict access.
"""
from __future__ import annotations

import csv
import html as html_mod
import io
import json as _json
import re
from collections import Counter, OrderedDict
from itertools import groupby
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

from core.config import CONSTANTS, get_settings
from core.utils import assign_sections
from core.models import (
    AiSegment,
    AnalysisResult,
    AudioBuffer,
    AuthorshipResult,
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    ForensicsResult,
    IntroResult,
    Section,
    SectionRepetition,
    StingResult,
    StructureResult,
    SyncCut,
    TranscriptSegment,
)
from services.export import (
    PLATFORM_SCHEMAS,
    _build_davinci_drt,
    _build_premiere_xml,
    to_analysis_json,
    to_platform_csv,
    to_section_markers_csv,
)
from services.content import THEME_TAXONOMY, ThemeMoodAnalyzer
from services.legal import hfa_url, songfile_url
from services.sync_cut import SyncCutAnalyzer
from services.tagging import TagInjector
from ui.components import ISSUE_META, authorship_color, eq_bars, fmt_ts, issue_pill


# AudioSource literal value for YouTube — matches core/models.py AudioSource type.
# Defined here to avoid repeating the string across multiple render functions.
_SOURCE_YOUTUBE: str = "youtube"

# Human-readable labels for platform catalog export targets.
_PLATFORM_LABELS: dict[str, str] = {
    "generic":   "Generic CSV",
    "disco":     "DISCO",
    "synchtank": "Synchtank",
}

# Extension → file suffix and MIME type maps for tagged file download.
# Defined at module level to avoid re-creation on every report render.
_TAGGED_EXT_MAP: dict[str, str] = {
    "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg", "m4a": ".m4a", "wav": ".wav",
}
_TAGGED_MIME_MAP: dict[str, str] = {
    ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg",  ".m4a": "audio/mp4", ".wav": "audio/wav",
}

# Display-only threshold for infrasonic ⚠ icon (human FMC p95 × 10).
# Does NOT affect the verdict — infrasonic is in monitoring mode.
# Verdict threshold is CONSTANTS.INFRASONIC_ENERGY_RATIO_AI_MIN (currently 0.0 / DISABLED).
_INFRASONIC_WARN_DISPLAY: float = 0.005

# Mood → CSS variable colour map for the Theme & Mood card.
# Uses existing palette variables; Romantic uses --issue-brand (warm amber #F5A623).
_MOOD_COLORS: dict[str, str] = {
    "Uplifting":   "var(--ok)",
    "Energetic":   "var(--ok)",
    "Romantic":    "var(--issue-brand)",
    "Melancholic": "var(--muted)",
    "Nostalgic":   "var(--muted)",
    "Chill":       "var(--accent)",
    "Dark":        "var(--danger)",
    "Intense":     "var(--danger)",
}

# Category → CSS variable for theme pill/bar coloring (#168)
_THEME_CATEGORY_COLORS: dict[str, str] = {
    "energy":    "var(--accent)",
    "emotional": "var(--info)",
    "seasonal":  "var(--sync-pass)",
}


def _boundary_val(v: float, fmt: str = ".5f", unavail: str = "N/A") -> str:
    """Format a spectral boundary signal score; return 'N/A' for the -1.0 sentinel."""
    return unavail if v < 0.0 else format(v, fmt)


# ---------------------------------------------------------------------------
# Downbeat-aware section seek (#135)
# ---------------------------------------------------------------------------

def _nearest_beat_in_section(
    target: float,
    beats: list[float],
    section_start: float,
    section_end: float,
) -> float:
    """
    Snap a target timestamp to the nearest beat before the section ends.

    Searches all beats with `b < section_end` — intentionally no lower bound so
    that a downbeat landing fractionally before `section_start` (the true first
    beat of the section) is still reachable.  Returns `target` unchanged when no
    candidate beats exist (beat tracking unavailable or all beats are beyond the
    section end).

    Pure function — no I/O.
    """
    candidates = [b for b in beats if b < section_end]
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))


# ---------------------------------------------------------------------------
# Chorus-adjacency highlight (#136)
# ---------------------------------------------------------------------------

def _chorus_adjacent(label: str, i: int, sections: list[Section]) -> bool:
    """
    Return True when a section should receive a chorus-adjacent highlight.

    A section is chorus-adjacent when:
    - Its normalised label contains "pre-chorus" or "build", AND
    - The nearest following chorus starts within CONSTANTS.PRE_CHORUS_ADJACENT_MAX_S
      seconds of this section's end.

    Pure function — no I/O.
    """
    lo = label.lower()
    if not any(k in lo for k in ("pre-chorus", "prechorus", "pre chorus", "build")):
        return False
    if i >= len(sections) - 1:
        return False
    this_end = sections[i].end
    for j in range(i + 1, len(sections)):
        next_label = sections[j].label.lower()
        if any(k in next_label for k in ("chorus", "hook", "refrain", "drop")):
            gap = sections[j].start - this_end
            return gap <= CONSTANTS.PRE_CHORUS_ADJACENT_MAX_S
        # stop searching once we pass the adjacent-window
        if sections[j].start - this_end > CONSTANTS.PRE_CHORUS_ADJACENT_MAX_S:
            break
    return False


# ---------------------------------------------------------------------------
# OpenGraph + JSON-LD meta tags
# ---------------------------------------------------------------------------

def _inject_og_tags(result: AnalysisResult) -> None:
    """
    Inject OpenGraph meta tags and a JSON-LD MusicRecording schema into the page.

    Note: Streamlit renders these into the document body, not <head>, so social
    crawlers (Twitter, Slack) may not pick them up. They are included as
    best-effort for sharing previews; a proper implementation would require a
    custom Streamlit index.html template.
    """
    raw_title   = result.audio.metadata.get("title", "") or result.audio.label or ""
    raw_artist  = result.audio.metadata.get("artist", "") or ""
    raw_grade   = result.compliance.grade if result.compliance else "N/A"
    raw_verdict = result.forensics.verdict if result.forensics else ""

    og_title = html_mod.escape(
        f"{raw_artist} \u2014 {raw_title} | Sync-Safe" if raw_artist else f"{raw_title} | Sync-Safe"
    )
    og_desc = html_mod.escape(
        f"Sync compliance report \u00b7 Grade: {raw_grade}"
        + (f" \u00b7 Authenticity: {raw_verdict}" if raw_verdict else "")
    )

    # JSON-LD via json.dumps — ensure_ascii=True escapes all non-ASCII.
    # Replace '</' with '<\/' to prevent </script> injection inside ld+json block.
    ld_data = {
        "@context": "https://schema.org",
        "@type": "MusicRecording",
        "name": raw_title,
        "byArtist": {"@type": "MusicGroup", "name": raw_artist},
        "additionalProperty": [
            {"@type": "PropertyValue", "name": "SyncSafeGrade", "value": raw_grade},
        ],
    }
    ld_json = _json.dumps(ld_data, ensure_ascii=True).replace("</", "<\\/")

    st.markdown(f"""
<meta property="og:title" content="{og_title}" />
<meta property="og:description" content="{og_desc}" />
<meta property="og:type" content="website" />
<meta name="twitter:card" content="summary" />
<meta name="twitter:title" content="{og_title}" />
<meta name="twitter:description" content="{og_desc}" />
<script type="application/ld+json">{ld_json}</script>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Combined authorship banner (#161)
# ---------------------------------------------------------------------------

def combined_authorship_verdict(
    forensics: Optional["ForensicsResult"],
    authorship: Optional["AuthorshipResult"],
) -> Optional[tuple[str, str]]:
    """Collapse audio-AI (forensics) and lyric-AI (authorship) into one verdict.

    Returns:
        (verdict_text, css_color) or None when neither result is available.

    Rules:
        - Either result flagged as AI → "AI-Generated Content Detected", danger color.
        - Both results available and both clear → "No AI Signals Detected", ok color.
        - Mixed or insufficient → "AI Signals Uncertain", caution color.

    Pure function — no I/O, no side effects.
    """
    if forensics is None and authorship is None:
        return None

    fv = forensics.verdict if forensics else None
    av = authorship.verdict if authorship else None

    audio_ai   = fv in ("Likely AI", "AI")
    lyric_ai   = av in ("Likely AI",)
    audio_ok   = fv in ("Likely Not AI", "Not AI")
    lyric_ok   = av in ("Likely Human",)
    audio_skip = fv is None or fv == "Insufficient data"
    lyric_skip = av is None or av in ("Insufficient data", "Uncertain")

    if audio_ai or lyric_ai:
        return "AI-Generated Content Detected", "var(--danger)"

    if audio_ok and (lyric_ok or lyric_skip):
        return "No Audio AI Signals Detected", "var(--ok)"

    if lyric_ok and audio_skip:
        return "No Lyric AI Signals Detected", "var(--ok)"

    return "AI Signals Uncertain", "var(--grade-c)"


def _render_combined_authorship_banner(result: "AnalysisResult") -> None:
    """Render a combined audio + lyric AI verdict card (#161)."""
    outcome = combined_authorship_verdict(result.forensics, result.authorship)
    if outcome is None:
        return

    verdict, color = outcome
    fv = result.forensics.verdict if result.forensics else "—"
    av = result.authorship.verdict if result.authorship else "—"

    rows_html = (
        f"<div style='display:flex;gap:24px;flex-wrap:wrap;margin-top:10px;'>"
        f"<span style='font-family:\"Figtree\",sans-serif;font-size:.74rem;color:var(--muted);'>"
        f"Audio AI: <strong style='color:var(--text);'>{html_mod.escape(fv)}</strong></span>"
        f"<span style='font-family:\"Figtree\",sans-serif;font-size:.74rem;color:var(--muted);'>"
        f"Lyric AI: <strong style='color:var(--text);'>{html_mod.escape(av)}</strong></span>"
        f"</div>"
    )

    st.markdown(f"""
    <div style="border:1px solid {color}44;border-radius:12px;background:{color}0D;
                padding:14px 20px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;
                    color:{color};flex-shrink:0;">{html_mod.escape(verdict)}</div>
        <div style="font-family:'Chakra Petch',monospace;font-size:.52rem;font-weight:600;
                    letter-spacing:.12em;text-transform:uppercase;color:var(--dim);">
          Combined AI Assessment</div>
      </div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_report(
    audio: AudioBuffer,
    result: AnalysisResult,
) -> None:
    """Render the full analysis report for a completed pipeline run."""
    _inject_og_tags(result)
    st.markdown(
        '<a href="#main-content" class="skip-link">Skip to main content</a>',
        unsafe_allow_html=True,
    )
    _render_nav(audio.label)
    st.markdown('<span id="main-content" tabindex="-1"></span>', unsafe_allow_html=True)
    _render_audio_player(audio)
    _render_sync_snapshot(result)
    _render_combined_authorship_banner(result)

    with st.expander("Track Overview", expanded=True):
        _section_tooltip(
            "Basic track metadata and detected arrangement structure. Shows title, artist, BPM, "
            "musical key, and duration alongside the section map (intro, verse, chorus, bridge, "
            "outro) detected by the allin1 structural analysis model."
        )
        c_left, c_right = st.columns([1, 1], gap="large")
        with c_left:
            _render_metadata_card(result.structure, result.audio.metadata)
        with c_right:
            _render_structure_card(result.structure)

    with st.expander("Authenticity Audit", expanded=True):
        _section_tooltip(
            "Multi-signal analysis to assess whether the track shows signs of AI generation. "
            "Checks C2PA content credentials (cryptographic DAW/AI provenance), beat-grid timing "
            "variance (perfect quantization is an AI signal), vocal centroid stability (AI vocals "
            "drift erratically mid-note), high-frequency spectral noise, and loop/sample repetition. "
            "Only available for direct file uploads — compressed streaming sources destroy the "
            "high-frequency signals this detector relies on."
        )
        _render_forensics_card(result.forensics, source=result.audio.source)

    with st.expander("Structural Repetition", expanded=True):
        _section_tooltip(
            "Measures how much the track repeats itself across its arrangement. "
            "High repetition can limit sync editability — if every section sounds identical, "
            "editors have fewer natural cut points. Scored using cross-correlation of 4-bar "
            "windows and section-to-section similarity. A lower repetition index means more "
            "dynamic range in the arrangement and more flexibility for picture editors."
        )
        _render_production_analysis_card(result.forensics)

    if result.sync_cuts:
        with st.expander("Sync Edit Points", expanded=False):
            _section_tooltip(
                "Automatically suggested edit windows for 15s, 30s, and 60s placements. "
                "Each window is scored on five criteria: starts after the intro, opens on a "
                "section boundary, closes on a section boundary, contains a chorus or hook, "
                "and ends on a bar grid. Confidence 60%+ is recommended. "
                "Click a Start timestamp to seek the audio player to that position."
            )
            _render_sync_cuts(result)

    with st.expander("Sync Readiness Checks", expanded=True):
        _section_tooltip(
            "Gallo-Method compliance checks for sync submission readiness. Covers: sting detection "
            "(sharp energy drop at track end for bumper/sting use), 4–8 bar energy evolution "
            "(tracks must build or change — flat energy fails), intro timing (intro must not "
            "exceed the configured threshold), fade detection (gradual endings can bleed into "
            "scene audio), and broadcast loudness with dialogue compatibility."
        )
        _render_sync_readiness(result.compliance)
        _render_audio_quality_card(result.audio_quality)

    with st.expander("Discovery & Licensing", expanded=True):
        _section_tooltip(
            "Popularity tier, rights data, and similar track recommendations. Popularity is a "
            "blended 0–100 score derived from Last.fm listeners, platform engagement (views, "
            "likes), and Spotify popularity — used to estimate sync fee ranges. Rights lookup "
            "identifies the likely PRO (ASCAP, BMI, SESAC) and ISRC from MusicBrainz metadata. "
            "Similar tracks are sourced from Last.fm's similarity graph with YouTube preview links."
        )
        _render_legal_and_discovery(result)

    with st.expander("Lyrics & Content Audit", expanded=True):
        _section_tooltip(
            "Full lyric transcription via Whisper with timestamped compliance flagging. "
            "Scans for explicit language (Detoxify toxicity scoring + profanity word list), "
            "brand/trademark mentions (curated keyword list), geographic references that may "
            "affect international licensing, and drug/violence references. Also detects "
            "theme and mood from lyric content. Flags are graded A–F and categorised as "
            "deal-breakers, advisory, or supervisor review."
        )
        _render_lyric_section(result)

    _render_export_buttons(result)
    _render_raw_data_link()
    _render_footer()


# ---------------------------------------------------------------------------
# Sync Snapshot — top-level verdict
# ---------------------------------------------------------------------------

_VERDICT_READY    = "Ready"
_VERDICT_CAUTION  = "Caution"
_VERDICT_NOT_READY = "Not Ready"

# (status, icon, color)
_STATUS_PASS    = ("pass",    "✓", "var(--grade-b)")
_STATUS_CAUTION = ("caution", "▲", "var(--grade-c)")
_STATUS_FAIL    = ("fail",    "✕", "var(--danger)")


def _compute_sync_verdict(
    result: AnalysisResult,
) -> tuple[str, str, list[tuple[str, str, str, str]]]:
    """
    Derive a top-level sync verdict from all pipeline outputs.

    Returns:
        (verdict, color, categories)
        verdict:    "Ready" | "Caution" | "Not Ready"
        color:      CSS color string
        categories: list of (label, icon, color, detail)

    Pure function — no side effects, no I/O.
    """
    categories: list[tuple[str, str, str, str]] = []

    # ── 1. Authenticity (AI detection) ──────────────────────────────────────
    fv = result.forensics.verdict if result.forensics else None
    if fv in ("Likely Not AI", "Not AI"):
        auth_status = _STATUS_PASS
        auth_detail = fv
    elif fv in ("Likely AI", "AI"):
        auth_status = _STATUS_FAIL
        auth_detail = fv
    else:
        auth_status = _STATUS_CAUTION
        auth_detail = (
            "Upload file for full AI detection"
            if result.audio.source == _SOURCE_YOUTUBE
            else "Scan incomplete"
        )
    categories.append(("Authenticity", auth_status[1], auth_status[2], auth_detail))

    # ── 2. Arrangement (structural fitness) ──────────────────────────────────
    comp = result.compliance
    arr_issues: list[str] = []
    if comp:
        if comp.sting.ending_type == "fade":
            arr_issues.append("Fade ending — needs clean out-point")
        if comp.intro.flag:
            arr_issues.append(f"Intro too long ({comp.intro.intro_seconds:.0f}s)")
        if comp.evolution.flag:
            arr_issues.append(
                f"{comp.evolution.stagnant_windows} stagnant energy window"
                f"{'s' if comp.evolution.stagnant_windows != 1 else ''}"
            )
    if arr_issues:
        arr_status = _STATUS_CAUTION
        arr_detail = " · ".join(arr_issues)
    else:
        arr_status = _STATUS_PASS
        arr_detail = "No structural issues"
    categories.append(("Arrangement", arr_status[1], arr_status[2], arr_detail))

    # ── 3. Content (lyric grade) ─────────────────────────────────────────────
    grade = comp.grade if comp else "N/A"
    if grade in ("A", "B", "N/A"):
        cont_status  = _STATUS_PASS
        hard_count   = len(comp.hard_flags) if comp else 0
        adv_count    = len(comp.soft_flags) + len(comp.potential_flags) if comp else 0
        if hard_count == 0 and adv_count == 0:
            cont_detail = "Clean"
        else:
            cont_detail = f"Grade {grade} — {adv_count} advisory flag{'s' if adv_count != 1 else ''}"
    elif grade == "C":
        cont_status = _STATUS_CAUTION
        hard_count  = len(comp.hard_flags) if comp else 0
        cont_detail = f"Grade {grade} — {hard_count} hard issue, clean edit required"
    else:
        cont_status = _STATUS_FAIL
        hard_count  = len(comp.hard_flags) if comp else 0
        cont_detail = f"Grade {grade} — {hard_count} hard issue{'s' if hard_count != 1 else ''}"
    categories.append(("Content", cont_status[1], cont_status[2], cont_detail))

    # ── Overall verdict ───────────────────────────────────────────────────────
    statuses = [auth_status[0], arr_status[0], cont_status[0]]
    if "fail" in statuses:
        verdict = _VERDICT_NOT_READY
        color   = "var(--danger)"
    elif "caution" in statuses:
        verdict = _VERDICT_CAUTION
        color   = "var(--grade-c)"
    else:
        verdict = _VERDICT_READY
        color   = "var(--grade-b)"

    return verdict, color, categories


def _section_tooltip(text: str) -> None:
    """Render a right-aligned ? tooltip at the top of an expander section.

    Tooltip opens DOWNWARD (top: calc(100% + 8px)) so it is never clipped
    by the expander container — expander bodies don't have infinite overflow
    upward, causing upward-opening tooltips to disappear behind the summary.
    """
    st.markdown(
        f"<div style='display:flex;justify-content:flex-end;margin:-8px 0 6px;overflow:visible;'>"
        f"<span class='tip-wrap' style='overflow:visible;'>"
        f"<span class='tip-icon' role='button' tabindex='0' aria-label='More information'>?</span>"
        f"<span class='tip-box section-tip-box' style='left:auto;right:0;bottom:auto;"
        f"top:calc(100% + 8px);'>{html_mod.escape(text)}</span>"
        f"</span></div>",
        unsafe_allow_html=True,
    )


def _render_sync_snapshot(result: AnalysisResult) -> None:
    """Render the top-level Sync Snapshot card."""
    verdict, v_color, categories = _compute_sync_verdict(result)

    rows_html = "".join(
        f"<div style='display:flex;align-items:flex-start;gap:10px;padding:7px 0;"
        f"border-bottom:1px solid var(--border-hr);'>"
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:.78rem;"
        f"font-weight:700;color:{cat_color};min-width:14px;flex-shrink:0;'>{icon}</span>"
        f"<span style='font-family:\"Chakra Petch\",monospace;font-size:.62rem;"
        f"font-weight:600;letter-spacing:.08em;text-transform:uppercase;"
        f"color:var(--text);min-width:100px;flex-shrink:0;'>{html_mod.escape(label)}</span>"
        f"<span style='font-family:\"Figtree\",sans-serif;font-size:.78rem;"
        f"color:var(--muted);'>{html_mod.escape(detail)}</span>"
        f"</div>"
        for label, icon, cat_color, detail in categories
    )

    st.markdown(f"""
    <div style="border:1px solid {v_color}44;border-radius:12px;
                background:{v_color}0D;padding:18px 22px;margin-bottom:18px;">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;
                    font-weight:700;color:{v_color};flex-shrink:0;">{verdict}</div>
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;
                      font-weight:600;letter-spacing:.18em;text-transform:uppercase;
                      color:var(--dim);">Sync Snapshot</div>
          <div style="font-family:'Figtree',sans-serif;font-size:.74rem;
                      color:var(--muted);margin-top:2px;">
            Combined verdict across authenticity, arrangement, and content.</div>
        </div>
      </div>
      <div>{rows_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

def _render_nav(source_label: str) -> None:
    _eq = eq_bars(6, color="var(--accent)", h=22)
    c_logo, c_nav = st.columns([6, 1])
    with c_logo:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:24px 0 10px;">
          <div style="display:flex;align-items:flex-end;gap:2px;height:22px;">{_eq}</div>
          <div>
            <div style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                        color:var(--accent);letter-spacing:.14em;">SYNC-SAFE</div>
            <div style="font-family:'Chakra Petch',monospace;font-size:.48rem;font-weight:500;
                        color:var(--dim);letter-spacing:.2em;text-transform:uppercase;">Forensic Portal</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c_nav:
        st.markdown("<div style='padding-top:22px;'>", unsafe_allow_html=True)
        if st.button("← New Scan", use_container_width=True):
            for key in ("page", "audio", "analysis", "start_time", "player_key"):
                st.session_state.pop(key, None)
            st.rerun()

    st.markdown(
        "<hr style='border:none;border-top:1px solid var(--border-hr);margin:2px 0 28px;'>",
        unsafe_allow_html=True,
    )

    src_html = (
        f'<div style="font-family:var(--mono,monospace);font-size:.7rem;color:var(--dim);'
        f'margin-top:6px;letter-spacing:.04em;">{html_mod.escape(source_label)}</div>'
        if source_label else ""
    )
    st.markdown(f"""
    <div style="margin-bottom:24px;">
      <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                  letter-spacing:.2em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;">
        ▶ Forensic Report
      </div>
      <div style="font-family:'Chakra Petch',monospace;font-size:2.4rem;font-weight:700;
                  color:var(--text);letter-spacing:-.03em;line-height:1;">Analysis</div>
      {src_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Audio player
# ---------------------------------------------------------------------------

def _render_audio_player(audio: AudioBuffer) -> None:
    # st.audio doesn't support `key` in Streamlit < 1.56 — use st.empty() as
    # the container instead. Replacing the slot forces a full widget re-init
    # whenever player_key increments (same effect as keyed re-mount).
    slot = st.empty()
    slot.audio(
        audio.raw,
        start_time=st.session_state.get("start_time", 0),
    )
    st.markdown("<div style='margin-bottom:28px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Track Overview: metadata + structure
# ---------------------------------------------------------------------------

def _render_metadata_card(sr: Optional[StructureResult], ingestion_meta: dict | None = None) -> None:
    ingestion_meta = ingestion_meta or {}
    meta       = sr.metadata if sr else {}
    title_str  = html_mod.escape(meta.get("title", "")  or ingestion_meta.get("title", ""))
    artist_str = html_mod.escape(meta.get("artist", "") or ingestion_meta.get("artist", ""))

    duration    = ingestion_meta.get("duration", "")
    sample_rate = ingestion_meta.get("sample_rate", "")
    bit_depth   = ingestion_meta.get("bit_depth", "")
    channels    = ingestion_meta.get("channels", "")

    title_html = (
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:1.4rem;"
        f"font-weight:700;color:var(--text);line-height:1.2;margin-bottom:5px;'>{title_str}</div>"
        if title_str else
        "<div style='font-size:.9rem;color:var(--dim);margin-bottom:5px;'>No tags found</div>"
    )
    artist_html = (
        f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.6rem;"
        f"font-weight:600;color:var(--accent);letter-spacing:.12em;text-transform:uppercase;'>{artist_str}</div>"
        if artist_str else ""
    )

    # Build tech spec pills — only include fields that are populated
    specs = [
        ("Duration",    duration),
        ("Sample Rate", sample_rate),
        ("Bit Depth",   bit_depth),
        ("Channels",    channels),
    ]
    spec_items = "".join(
        f"<div style='display:flex;flex-direction:column;gap:3px;'>"
        f"  <div style='font-family:\"Chakra Petch\",monospace;font-size:.5rem;font-weight:600;"
        f"letter-spacing:.12em;text-transform:uppercase;color:var(--dim);'>{label}</div>"
        f"  <div style='font-family:\"JetBrains Mono\",monospace;font-size:.8rem;color:var(--text);'>{value}</div>"
        f"</div>"
        for label, value in specs if value
    )
    specs_html = (
        f"<div style='display:flex;gap:24px;margin-top:12px;flex-wrap:wrap;'>{spec_items}</div>"
        if spec_items else ""
    )

    st.markdown(f"""
    <div class="sig" style="margin-bottom:14px;">
      <div class="sig-head">Track Metadata</div>
      {title_html}
      {artist_html}
      {specs_html}
    </div>
    """, unsafe_allow_html=True)


def _render_structure_card(sr: Optional[StructureResult]) -> None:
    bpm_fmt  = f"{sr.bpm:.1f}" if sr and isinstance(sr.bpm, float) else (sr.bpm if sr else "—")
    key      = sr.key if sr else "—"
    sections = sr.sections if sr else []
    beats    = sr.beats if sr else []

    def _fmt_ts(secs: float) -> str:
        m, s = divmod(int(secs), 60)
        return f"{m}:{s:02d}"

    # BPM + Key rendered as a pure HTML card (no interactive elements here)
    st.markdown(f"""
    <div class="sig">
      <div class="sig-head">Structure Analysis</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px;">
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                      letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;">Tempo</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:700;
                      color:var(--accent);line-height:1;">
            {bpm_fmt}<span style="font-size:.8rem;font-weight:400;color:var(--dim);
                                   margin-left:5px;font-family:'Chakra Petch',monospace;
                                   letter-spacing:.1em;">BPM</span>
          </div>
        </div>
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                      letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;">Key</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:700;
                      color:var(--accent);line-height:1;">{key}</div>
        </div>
      </div>
      <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                  letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:6px;">
        Detected Sections — click to seek
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Section timeline bar (#133, #136)
    if sections:
        total_dur = sum(s.end - s.start for s in sections)
        if total_dur > 0:
            blocks = "".join(
                f"<div style='flex:{max(0.01, (s.end - s.start) / total_dur):.4f};"
                f"background:{_section_label_color(s.label, idx, sections)};height:100%;border-radius:2px;'"
                f" title=\"{html_mod.escape(s.label)} ({int(s.end - s.start)}s)\"></div>"
                for idx, s in enumerate(sections)
            )
            st.markdown(
                f"<div style='display:flex;gap:2px;height:8px;border-radius:4px;"
                f"overflow:hidden;margin-bottom:14px;'"
                f" role='img' aria-label='Section arrangement timeline'>{blocks}</div>",
                unsafe_allow_html=True,
            )

    # Sections rendered as Streamlit columns so timestamps are clickable buttons
    if sections:
        for s in sections:
            ts_s = int(s.start)
            col_btn, col_label = st.columns([1.4, 3], gap="small")
            with col_btn:
                if st.button(
                    f"{_fmt_ts(s.start)} – {_fmt_ts(s.end)}",
                    key=f"sec_{ts_s}_{html_mod.escape(s.label)}",
                    help=_section_button_help(s.label, _fmt_ts(s.start)),
                    use_container_width=True,
                    type="secondary",
                ):
                    snapped = _nearest_beat_in_section(s.start, beats, s.start, s.end)
                    st.session_state.start_time = int(snapped)
                    st.session_state.player_key = st.session_state.get("player_key", 0) + 1
                    st.rerun()
            with col_label:
                st.markdown(
                    f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.76rem;"
                    f"font-weight:600;letter-spacing:.08em;text-transform:uppercase;"
                    f"color:var(--text);padding-top:6px;'>{html_mod.escape(s.label)}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            "<span style='font-family:var(--mono);font-size:.8rem;color:var(--dim);'>"
            "No section data</span>",
            unsafe_allow_html=True,
        )

    # Section stats summary row (#134)
    stats = _build_section_stats(sections)
    if stats:
        st.markdown(
            f"<div style='font-family:\"Figtree\",sans-serif;font-size:.76rem;"
            f"color:var(--dim);margin-top:10px;'>{html_mod.escape(stats)}</div>",
            unsafe_allow_html=True,
        )

    # DAW-ready section markers CSV download (#138)
    if sections:
        markers_bytes = to_section_markers_csv(sections)
        st.download_button(
            label="⬇ Export Section Markers (DAW CSV)",
            data=markers_bytes,
            file_name="section-markers.csv",
            mime="text/csv",
            use_container_width=True,
            help="Timecoded section markers for Reaper, Logic, and other DAW imports",
        )


# ---------------------------------------------------------------------------
# Clipboard copy button (#119, #125)
# ---------------------------------------------------------------------------

def _copy_button(value: str, label: str = "Copy") -> None:
    """
    Render a clipboard copy button using st.components.v1.html().

    st.markdown() HTML silently strips <script> tags (browser innerHTML
    security model). components.html() runs in a real sandboxed iframe
    where navigator.clipboard.writeText() is accessible.

    Uses JSON-encoded strings for both the value and label in JS so any
    special characters cannot break JS string literals. The visible button
    text uses html.escape() to defend against any HTML in the label arg.

    Colour note: hex values (#F5640A = --accent, #6ECC8A = --grade-b) are
    hardcoded because CSS custom properties do not cross the iframe boundary.
    Update these if the theme palette changes in styles.py.
    """
    import streamlit.components.v1 as components

    js_value   = _json.dumps(value)          # safe for embedding in JS
    js_label   = _json.dumps(label)          # safe for embedding in JS
    html_label = html_mod.escape(label)      # safe for embedding in HTML body
    components.html(
        f"""<button
            id="cb"
            onclick="navigator.clipboard.writeText({js_value}).then(function(){{
              var b=document.getElementById('cb');
              b.textContent='\u2713 Copied';
              b.style.color='#6ECC8A';
              b.style.borderColor='rgba(110,204,138,.5)';
              setTimeout(function(){{
                b.textContent={js_label};
                b.style.color='#F5640A';
                b.style.borderColor='rgba(245,100,10,.35)';
              }},1500);
            }}).catch(function(){{
              document.getElementById('cb').textContent='Copy failed';
            }})"
            style="background:transparent;border:1px solid rgba(245,100,10,.35);
                   border-radius:6px;color:#F5640A;font-family:monospace;
                   font-size:.62rem;font-weight:600;letter-spacing:.08em;
                   padding:4px 10px;cursor:pointer;white-space:nowrap;
                   text-transform:uppercase;transition:color .15s,border-color .15s;"
        >{html_label}</button>""",
        height=36,
    )


# ---------------------------------------------------------------------------
# Section timeline helpers (#133, #139)
# ---------------------------------------------------------------------------

# Sync-relevance tips shown in the seek-button tooltip per section label (#139).
# Keep tips concise — they appear in Streamlit's native help= popover.
# ORDER MATTERS: more-specific keys (pre-chorus) must precede their substrings (chorus)
# so the `key in lo` substring check hits the right entry first.
_SECTION_SYNC_TIPS: dict[str, str] = {
    "pre-chorus": "Pre-chorus — tension build; pairs well with rising action.",
    "prechorus":  "Pre-chorus — tension build; pairs well with rising action.",
    "chorus":     "Chorus — high energy, memorable; ideal for title cards and product moments.",
    "hook":       "Hook — peak recall; works well for 15–30 s ad spots.",
    "refrain":    "Refrain — emotional peak; strong for closing scenes.",
    "drop":       "Drop — maximum energy release; great for action cuts.",
    "intro":      "Intro — sets mood; good for scene-setting or cold opens.",
    "verse":      "Verse — narrative texture; works for background / montage.",
    "bridge":     "Bridge — contrast moment; useful for dramatic scene pivots.",
    "build":      "Build — rising energy; effective for trailer ramp-ups.",
    "outro":      "Outro — resolution; works for credits or fade-outs.",
    "fade":       "Fade — soft exit; good for end-of-scene transitions.",
    "coda":       "Coda — final statement; suits closing moments.",
    "instrumental": "Instrumental — no lyrics; easier dialogue clearance.",
    "break":      "Break — rhythmic pause; good for VO-over sections.",
    "solo":       "Solo — featured melody; high impact for short bursts.",
    "interlude":  "Interlude — transitional; good for scene cuts.",
}


def _section_sync_tip(label: str) -> str:
    """Return a sync-relevance tip for a section label, or empty string if unknown."""
    lo = label.lower()
    for key, tip in _SECTION_SYNC_TIPS.items():
        if key in lo:
            return tip
    return ""


def _section_button_help(label: str, start_fmt: str) -> str:
    """Combine seek action and sync tip for the section button help= text."""
    tip = _section_sync_tip(label)
    seek = f"Jump to {start_fmt} in the audio player"
    return f"{seek}\n\n{tip}" if tip else seek


def _section_label_color(
    label: str,
    i: int = 0,
    sections: Optional[list[Section]] = None,
) -> str:
    """
    Map a section label to a CSS variable for the timeline bar.

    When `sections` is provided and the section at index `i` is chorus-adjacent
    (#136), it receives a distinct pre-chorus highlight (var(--grade-c) at
    higher contrast) instead of the generic build/bridge colour.
    """
    lo = label.lower()
    if any(k in lo for k in ("chorus", "hook", "refrain", "drop")):
        return "var(--accent)"
    if "intro" in lo:
        return "var(--issue-location)"
    if any(k in lo for k in ("outro", "fade", "coda", "end")):
        return "var(--muted)"
    if any(k in lo for k in ("bridge", "pre-chorus", "prechorus", "build")):
        adjacent = (
            _chorus_adjacent(label, i, sections)
            if sections is not None
            else False
        )
        return "var(--accent)" if adjacent else "var(--grade-c)"
    if any(k in lo for k in ("instrumental", "break", "solo", "interlude")):
        return "var(--grade-b)"
    return "var(--dim)"  # verse and unrecognised labels


# ---------------------------------------------------------------------------
# Repetition Index helpers (#144, #141)
# ---------------------------------------------------------------------------

def _repetition_index_label(ri: Optional[float]) -> tuple[str, str]:
    """Return (label, color) for a repetition index value."""
    if ri is None:
        return ("—", "var(--dim)")
    if ri >= CONSTANTS.REPETITION_INDEX_HIGH:
        return ("High", "var(--danger)")
    if ri >= CONSTANTS.REPETITION_INDEX_MODERATE:
        return ("Moderate", "var(--grade-c)")
    return ("Low", "var(--grade-b)")


def _sync_editability_badge(ri: Optional[float]) -> Optional[tuple[str, str, str]]:
    """Return (label, detail, color) or None if index unavailable."""
    if ri is None:
        return None
    if ri >= CONSTANTS.REPETITION_INDEX_HIGH:
        return ("Loop-friendly", "Easy to extend or trim for ad spots", "var(--accent)")
    if ri >= CONSTANTS.REPETITION_INDEX_MODERATE:
        return ("Moderate Loop Structure", "Versatile for most placements", "var(--grade-c)")
    return ("Organic Flow", "Better suited for narrative and documentary", "var(--grade-b)")


# ---------------------------------------------------------------------------
# Section stats (#134)
# ---------------------------------------------------------------------------

def _build_section_stats(sections: list[Section]) -> str:
    """
    Build a one-line summary string for the sections list.

    Pure function — no I/O.  Returns "" when sections is empty or total
    duration is zero (prevents div-by-zero and renders nothing).
    """
    if not sections:
        return ""
    total = sum(s.end - s.start for s in sections)
    if total <= 0:
        return ""

    n       = len(sections)
    avg_s   = total / n
    chorus_dur = sum(
        s.end - s.start for s in sections
        if any(k in s.label.lower() for k in ("chorus", "hook", "refrain"))
    )
    inst_dur = sum(
        s.end - s.start for s in sections
        if any(k in s.label.lower() for k in ("instrumental", "break", "intro", "outro"))
    )
    chorus_pct = round(chorus_dur / total * 100)
    inst_pct   = round(inst_dur   / total * 100)

    parts: list[str] = [f"{n} section{'s' if n != 1 else ''}"]
    if chorus_pct > 0:
        parts.append(f"{chorus_pct}% chorus")
    if inst_pct > 0:
        parts.append(f"{inst_pct}% instrumental")
    parts.append(f"Avg {avg_s:.0f}s/section")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# AI probability heatmap helpers
# ---------------------------------------------------------------------------

# Semantic grade colors for A–F display — intentionally outside the CSS variable
# system because these map to a universal academic grading convention (green=good,
# red=fail) rather than the Sync-Safe brand palette.
_GRADE_COLORS: dict[str, str] = {
    "A": "#22c55e",
    "B": "#84cc16",
    "C": "#eab308",
    "D": "#f97316",
    "F": "#ef4444",
}


def _ai_grade(mean_prob: float) -> str:
    """Return A/B/C/D/F grade letter for a mean AI probability in [0, 1]."""
    thresholds = CONSTANTS.AI_GRADE_THRESHOLDS  # (0.20, 0.40, 0.60, 0.80)
    for threshold, letter in zip(thresholds, ("A", "B", "C", "D")):
        if mean_prob < threshold:
            return letter
    return "F"


def _render_ai_heatmap(segments: list[AiSegment]) -> None:
    """Render an AI probability timeline heatmap with an A–F grade badge."""
    if not segments:
        return

    probs     = [s.probability for s in segments]
    mean_prob = sum(probs) / len(probs)
    grade     = _ai_grade(mean_prob)
    grade_color = _GRADE_COLORS.get(grade, "#94a3b8")

    total_dur = segments[-1].end_s
    bar_parts: list[str] = []
    for seg in segments:
        width_pct = (seg.end_s - seg.start_s) / total_dur * 100 if total_dur > 0 else 0
        # HSL: hue 142 (green) at 0% AI → hue 0 (red) at 100% AI
        hue   = int(142 * (1.0 - seg.probability))
        color = f"hsl({hue},75%,45%)"
        s_m, s_s = int(seg.start_s) // 60, int(seg.start_s) % 60
        label = f"{s_m}:{s_s:02d} — {seg.probability:.0%} AI"
        bar_parts.append(
            f"<div title='{label}' "
            f"style='flex:{width_pct:.2f};background:{color};height:20px;'></div>"
        )
    bar_html = "".join(bar_parts)

    end_m, end_s = int(total_dur) // 60, int(total_dur) % 60
    st.markdown(f"""
    <div style="margin-top:20px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
        <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                    letter-spacing:.14em;text-transform:uppercase;color:var(--dim);">
          AI Probability Timeline
        </div>
        <div style="font-family:'Chakra Petch',monospace;font-size:1.1rem;font-weight:700;
                    color:{grade_color};letter-spacing:.05em;">Grade {grade}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--dim);">
          avg {mean_prob:.0%}
        </div>
      </div>
      <div role="img" aria-label="AI probability timeline: average {mean_prob:.0%}, grade {grade}"
           style="display:flex;border-radius:4px;overflow:hidden;gap:1px;background:var(--border);">
        {bar_html}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:4px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:var(--dim);">0:00</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:var(--dim);">{end_m}:{end_s:02d}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Authenticity Audit: forensics
# ---------------------------------------------------------------------------

def _render_forensics_card(fr: Optional[ForensicsResult], source: str = "file") -> None:
    if fr is None:
        msg = (
            "AI detection signals require a direct file upload. "
            "YouTube audio is re-encoded before analysis, which masks several key signals. "
            "Upload the original file for a full forensic scan."
            if source == _SOURCE_YOUTUBE
            else "Forensics analysis unavailable."
        )
        st.markdown(
            f"<div class='sig'><div class='sig-head'>Authenticity Audit</div>"
            f"<div style='color:var(--dim);font-size:.84rem;'>{msg}</div></div>",
            unsafe_allow_html=True,
        )
        return

    verdict   = fr.verdict
    v_cls     = {
        "AI":             "v-a",
        "Likely AI":      "v-a",
        "Likely Not AI":  "v-h",
        "Not AI":         "v-h",
    }.get(verdict, "v-u")

    _VERDICT_MESSAGES: dict[str, str] = {
        "AI": (
            "Verifiably 100% AI-generated — a certified AI-generation assertion was found "
            "embedded in this file's C2PA Content Credentials manifest."
        ),
        "Likely AI": (
            "Likely AI-generated — no embedded certification was found, but our analysis "
            "detected patterns strongly consistent with AI generation. We cannot confirm "
            "this with absolute certainty."
        ),
        "Likely Not AI": (
            "Likely not AI-generated — our analysis did not detect significant AI indicators. "
            "See signal notes below for additional context."
        ),
        "Not AI": (
            "Not AI-generated — confirmed by verified provenance data."
        ),
    }
    verdict_message = _VERDICT_MESSAGES.get(verdict, "")

    # C2PA
    _C2PA_ORIGIN_FMT: dict[str, str] = {
        "ai":      "⚠ Born-AI (Certified)",
        "daw":     "✓ DAW Origin (Verified)",
        "unknown": "◈ Manifest Present (Unknown Origin)",
        "":        "✓ No C2PA Manifest",
    }
    c2pa_fmt = _C2PA_ORIGIN_FMT.get(fr.c2pa_origin, "✓ No C2PA Manifest")

    # IBI / groove
    ibi       = fr.ibi_variance
    ibi_fmt   = f"{ibi:.3f}" if isinstance(ibi, float) and ibi >= 0 else "—"
    groove_flag = _groove_label(ibi)

    # Spectral slop
    slop_val  = fr.spectral_slop
    slop_fmt  = "✓ Clean" if slop_val <= CONSTANTS.SPECTRAL_SLOP_RATIO else f"⚠ {slop_val:.1%} HF energy"

    # Centroid instability
    centroid     = fr.centroid_instability_score
    centroid_fmt: str
    if centroid < 0.0:
        centroid_fmt = "— (no sustained intervals)"
    else:
        centroid_num = f"{centroid:.3f}"
        if centroid >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN:
            centroid_fmt = f"⚠ {centroid_num} — Formant Drift Detected"
        else:
            centroid_fmt = f"✓ {centroid_num} — Stable"

    # SynthID
    synthid_bins = int(fr.synthid_score)
    synthid_conf = _synthid_confidence(synthid_bins)
    _si_icons    = {"none": "✓", "low": "◈", "medium": "⚠", "high": "⚠"}
    synthid_fmt  = f"{_si_icons.get(synthid_conf, '◈')} {synthid_conf.title()} ({synthid_bins} bins)"

    st.markdown(f"""
    <div class="sig">
      <div class="sig-head">Authenticity Audit</div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.64rem;font-weight:500;
                      letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin-bottom:6px;">
            Overall Verdict
          </div>
          <span class="verd {v_cls}">{verdict}</span>
        </div>
        <div style="text-align:right;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--dim);
                      margin-bottom:3px;">6 signals analysed</div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;color:var(--dim);
                      letter-spacing:.1em;text-transform:uppercase;">C2PA · IBI · Groove · Centroid · Spectral · SynthID</div>
        </div>
      </div>
      <div style="font-family:'Figtree',sans-serif;font-size:.82rem;color:var(--text);
                  line-height:1.5;margin-bottom:20px;padding:10px 14px;
                  background:var(--surface);border-left:3px solid var(--accent);
                  border-radius:0 6px 6px 0;">
        {html_mod.escape(verdict_message)}
      </div>
      <div class="sig-row">
        <span class="sk">C2PA Manifest
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Content Credentials standard (C2PA) — a cryptographic signature embedded by some DAWs and AI tools. "Born-AI (Certified)": a hard certified signal the track was machine-generated. "DAW Origin (Verified)": manifest confirms creation in a known DAW (Logic Pro, Ableton, Pro Tools, etc.) — strong human-origin signal. "Manifest Present (Unknown Origin)": credentials exist but the software agent is unrecognised. "No C2PA Manifest": neutral — most files have none.</span>
          </span>
        </span>
        <span class="sv">{c2pa_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">IBI Variance (ms²)
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Inter-Beat Interval variance — measures millisecond-level timing drift between beats. Near-zero (&lt;0.5 ms²) = machine-quantized grid (AI/loop signal). High variance (&gt;90 ms²) = natural human feel and micro-timing variation — an organic signal, not AI.</span>
          </span>
        </span>
        <span class="sv">{ibi_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Groove Profile
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Derived from IBI variance. "Perfect Quantization" = machine-grid locked (&lt;0.5 ms²) — AI/loop signal. "Human Micro-timing" = natural drift (0.5–90 ms²). "Human-Feel Timing" = high micro-variation (&gt;90 ms²) — organic signal indicating human performance or humanized production.</span>
          </span>
        </span>
        <span class="sv">{groove_flag}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Centroid Instability
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Measures spectral centroid coefficient-of-variation within each sustained note. AI vocoders shift upper partials erratically mid-note — the source of the "glassy/hollow/formant-shifting" artifact heard in AI covers. Human vibrato modulates all partials together, keeping the centroid relatively stable. A score above 0.08 flags suspected formant drift. Scores of –1 mean no sustained intervals were found (e.g. full-instrumental or very quiet sections).</span>
          </span>
        </span>
        <span class="sv">{centroid_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Spectral Slop
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Checks for anomalous energy above 16 kHz relative to the full spectrum. AI generators often leak noise in the ultrasonic range. A high-frequency ratio &gt;15% triggers this flag.</span>
          </span>
        </span>
        <span class="sv">{slop_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">HF Phase Coherence
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Scans 18–22 kHz for phase-locked spectral bins that may indicate an AI watermark (e.g. SynthID). Confidence scales with the number of coherent bins found.</span>
          </span>
        </span>
        <span class="sv">{synthid_fmt}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if fr.flags:
        flags_html = "".join(
            f"<div style='font-family:\"Figtree\",sans-serif;font-size:.78rem;"
            f"color:var(--dim);padding:5px 0;border-bottom:1px solid var(--border);'>"
            f"◈ {html_mod.escape(flag)}</div>"
            for flag in fr.flags
        )
        st.markdown(
            f"<div style='margin-top:12px;'>"
            f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.56rem;font-weight:600;"
            f"letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;'>"
            f"Signal Notes</div>"
            f"{flags_html}</div>",
            unsafe_allow_html=True,
        )

    _render_ai_heatmap(fr.ai_segments)


# ---------------------------------------------------------------------------
# Production Analysis — sample & loop detection (separate from AI detection)
# ---------------------------------------------------------------------------

def _loop_heatmap_color(score: float) -> str:
    """Map a loop similarity score to a CSS color string."""
    if score > CONSTANTS.LOOP_SCORE_CEILING:
        return "var(--danger)"   # red — highly repetitive
    if score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        return "var(--grade-c)"  # amber — moderately repetitive
    return "var(--accent)"       # green — low repetition


def _render_loop_heatmap(window_scores: list[tuple[float, float]]) -> None:
    """
    Render a horizontal color-cell heatmap of per-4-bar-window loop scores (#142).

    Each cell's color encodes the max pairwise similarity for that window.
    A seek button row beneath lets editors jump to any window.
    """
    if not window_scores:
        return

    # Build the color bar as a flex row of divs
    cells_html = "".join(
        f"<div style='flex:1;height:18px;background:{_loop_heatmap_color(score)};"
        f"border-radius:3px;margin:0 1px;' aria-hidden='true' title='{score:.3f}'></div>"
        for _, score in window_scores
    )
    st.markdown(
        f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.54rem;font-weight:600;"
        f"letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:4px;'>"
        f"Loop Window Map</div>"
        f"<div style='display:flex;gap:0;margin-bottom:4px;'>{cells_html}</div>"
        f"<div style='display:flex;gap:0;margin-bottom:12px;'>"
        f"<span style='font-size:.58rem;color:var(--dim);'>0s</span>"
        f"<span style='flex:1;'></span>"
        f"<span style='font-size:.58rem;color:var(--dim);'>{window_scores[-1][0]:.0f}s</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    # Seek button row — one button per window
    cols = st.columns(len(window_scores))
    for col, (start_s, score) in zip(cols, window_scores):
        with col:
            label = f"{int(start_s // 60)}:{int(start_s % 60):02d}"
            if st.button(
                label,
                key=f"heatmap_seek_{start_s:.3f}",
                help=f"Jump to {label} (score {score:.3f})",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.start_time  = int(start_s)
                st.session_state.player_key  = st.session_state.get("player_key", 0) + 1
                st.rerun()


def _render_section_repetition(
    inter: dict[str, SectionRepetition],
    intra: dict[str, SectionRepetition],
) -> None:
    """
    Render per-label inter-section and intra-section repetition rows (#143, #145).

    Inter = same-label sections compared against each other.
    Intra = sub-windows within each section compared against each other.
    """
    all_labels = sorted(set(inter) | set(intra))
    if not all_labels:
        return

    _mono = "font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--muted);"
    _dim  = "font-family:'JetBrains Mono',monospace;font-size:.68rem;color:var(--dim);"
    _hdr  = (
        "font-family:'Chakra Petch',monospace;font-size:.54rem;font-weight:600;"
        "letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:6px;"
    )

    st.markdown(
        f"<div style='{_hdr}'>Section Repetition Breakdown</div>",
        unsafe_allow_html=True,
    )

    for label in all_labels:
        label_safe = html_mod.escape(label.title())
        inter_rep  = inter.get(label)
        intra_rep  = intra.get(label)

        parts: list[str] = []

        if inter_rep:
            inter_score = inter_rep.max_similarity
            inter_color = _loop_heatmap_color(inter_score)
            conf_note   = " <i>(1 pair — low confidence)</i>" if inter_rep.pair_count == 1 else ""
            flag        = " → Tight hook" if inter_score > CONSTANTS.LOOP_SCORE_CEILING else ""
            parts.append(
                f"<span style='color:var(--text);font-weight:600;'>{label_safe}</span> "
                f"<span style='color:{inter_color};'>{inter_score:.2f}</span> "
                f"<span style='color:var(--dim);'>({inter_rep.pair_count} pair{'s' if inter_rep.pair_count != 1 else ''})"
                f"{flag}{conf_note}</span>"
            )

        if intra_rep:
            intra_score = intra_rep.max_similarity
            intra_color = _loop_heatmap_color(intra_score)
            conf_note   = " <i>(2 sub-windows — low confidence)</i>" if intra_rep.pair_count <= 1 else ""
            flag        = " → Locked loop" if intra_score > CONSTANTS.LOOP_SCORE_CEILING else ""
            parts.append(
                f"<span style='color:var(--dim);'>{label_safe} (internal)</span> "
                f"<span style='color:{intra_color};'>{intra_score:.2f}</span>"
                f"<span style='color:var(--dim);'>{flag}{conf_note}</span>"
            )

        row_html = " · ".join(parts)
        st.markdown(
            f"<div style='{_mono}padding:2px 0;'>{row_html}</div>",
            unsafe_allow_html=True,
        )


def _render_production_analysis_card(fr: Optional[ForensicsResult], source: str = "file") -> None:
    if fr is None:
        st.markdown(
            "<div style='color:var(--dim);font-size:.84rem;padding:8px 0;'>"
            "Structural repetition analysis unavailable.</div>",
            unsafe_allow_html=True,
        )
        return

    loop      = fr.loop_score
    loop_num  = f"{loop:.3f}" if isinstance(loop, float) else "—"
    loop_label = (
        "Highly Repetitive" if loop > CONSTANTS.LOOP_SCORE_CEILING
        else "Moderately Repetitive" if loop > CONSTANTS.LOOP_SCORE_POSSIBLE
        else "Low Repetition"
    )
    loop_fmt = f"{loop_num} ({loop_label})"

    autocorr       = fr.loop_autocorr_score
    autocorr_num   = f"{autocorr:.3f}" if isinstance(autocorr, float) else "—"
    autocorr_label = (
        "Highly Regular" if autocorr >= CONSTANTS.LOOP_AUTOCORR_SAMPLE_VERDICT_THRESHOLD
        else "Regular" if autocorr >= CONSTANTS.LOOP_AUTOCORR_VERDICT_THRESHOLD
        else "Moderate" if autocorr >= CONSTANTS.LOOP_AUTOCORR_DISPLAY_MODERATE_MIN
        else "Low"
    )
    autocorr_fmt = f"{autocorr_num} ({autocorr_label})"

    # Blended Repetition Index headline (#144)
    ri        = fr.repetition_index
    ri_label, ri_color = _repetition_index_label(ri)
    ri_pct    = f"{ri:.0%}" if ri is not None else "—"

    st.markdown(f"""
    <div class="sig">
      <div class="sig-head">Structural Repetition</div>
      <div style="font-family:'Figtree',sans-serif;font-size:.82rem;color:var(--dim);
                  line-height:1.5;margin-bottom:18px;">
        Measures how repetitive this track's structure is. High scores indicate the
        production relies heavily on looping sections — common in modern pop and hip-hop.
        This is independent of AI detection and does not indicate AI generation on its own.
      </div>
      <div style="display:flex;align-items:center;gap:18px;margin-bottom:14px;">
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.54rem;font-weight:600;
                      letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:4px;">
            Repetition Index</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;
                      color:{ri_color};line-height:1;">{ri_pct}
            <span style="font-family:'Chakra Petch',monospace;font-size:.7rem;font-weight:500;
                         color:{ri_color};margin-left:6px;">{ri_label}</span>
          </div>
          <div style="margin-top:6px;height:4px;border-radius:2px;background:var(--border-hr);width:120px;overflow:hidden;">
            <div style="height:4px;border-radius:2px;background:{ri_color};width:120px;
                        transform:scaleX({ri or 0.0:.3f});transform-origin:left;transition:transform .3s;"></div>
          </div>
        </div>
      </div>
      <div class="sig-row">
        <span class="sk">Section Similarity
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Compares 4-bar spectral sections across the track. High score means sections sound near-identical — the production leans heavily on a repeating musical phrase.</span>
          </span>
        </span>
        <span class="sv">{loop_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Rhythmic Regularity
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Measures how regularly the beat pattern repeats. High score means the rhythm locks to a tight, consistent cycle — typical of loop-based production.</span>
          </span>
        </span>
        <span class="sv">{autocorr_fmt}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sync-editability badge (#141)
    badge = _sync_editability_badge(ri)
    if badge:
        badge_label, badge_detail, badge_color = badge
        st.markdown(
            f"<div style='border-left:3px solid {badge_color};padding:6px 12px;"
            f"margin-bottom:12px;background:{badge_color}0d;border-radius:0 6px 6px 0;'>"
            f"<span style='font-family:\"Chakra Petch\",monospace;font-size:.72rem;font-weight:600;"
            f"color:{badge_color};'>{html_mod.escape(badge_label)}</span>"
            f"<span style='font-family:\"Figtree\",sans-serif;font-size:.74rem;color:var(--dim);"
            f"margin-left:8px;'>· {html_mod.escape(badge_detail)}</span></div>",
            unsafe_allow_html=True,
        )

    # Sync context note (#147) — threshold labels derived from CONSTANTS to prevent drift
    if ri is not None:
        _ri_high = int(CONSTANTS.REPETITION_INDEX_HIGH * 100)
        _ri_mod  = int(CONSTANTS.REPETITION_INDEX_MODERATE * 100)
        with st.expander("Sync placement context", expanded=False):
            st.markdown(
                "**What does this mean for sync placements?**\n\n"
                f"The Repetition Index reflects how structurally uniform a track is across its "
                f"arrangement. A **high** index (≥ {_ri_high}%) means most sections sound nearly "
                f"identical — the track is easy to loop or extend for long-form ad spots or "
                f"background placements, but may feel monotonous over a longer scene.\n\n"
                f"A **moderate** index ({_ri_mod}–{_ri_high}%) is the most versatile range: enough "
                f"variation to support narrative pacing, while still having recognisable motifs "
                f"an editor can work around.\n\n"
                f"A **low** index (< {_ri_mod}%) signals organic, evolving arrangement — well suited "
                f"for documentary, drama, or any placement where the music needs to breathe and "
                f"change with the picture. These tracks are harder to loop cleanly but reward "
                f"editors who use the full duration."
            )

    # ── 4-bar window loop heatmap (#142) ──────────────────────────────────────
    if fr.loop_window_scores:
        _render_loop_heatmap(fr.loop_window_scores)

    # ── Section-aware repetition (#143 inter, #145 intra) ─────────────────────
    if fr.section_similarities or fr.section_internal_repetition:
        _render_section_repetition(fr.section_similarities, fr.section_internal_repetition)

    # ── Spectral boundary signals (monitoring mode — not yet in verdict) ─────
    infra  = fr.infrasonic_energy_ratio
    ultra  = fr.ultrasonic_noise_ratio
    phase  = fr.phase_coherence_differential

    # Infrasonic warning uses the FMC human-p95 (0.000467) × 10 as a display
    # floor — not the disabled CONSTANTS threshold which is 0.0.
    # Shown as informational only; does not affect verdict.
    infra_note = " ⚠" if (infra >= 0.0 and infra >= _INFRASONIC_WARN_DISPLAY) else ""
    st.markdown(f"""
    <div class="sig">
      <div class="sig-head">Spectral Boundary Signals</div>
      <div style="font-family:'Figtree',sans-serif;font-size:.82rem;color:var(--dim);
                  line-height:1.5;margin-bottom:18px;">
        Energy in normally-silent spectral zones. Real recordings have none;
        AI diffusion math can leave measurable residue. These signals are in
        <em>monitoring mode</em> — they are computed and stored but do not yet
        contribute to the verdict (pending cross-dataset calibration).
      </div>
      <div class="sig-row">
        <span class="sk">Infrasonic Ratio (&lt;20 Hz)
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Energy fraction in 1–20 Hz. Real microphones cannot capture sub-sonic frequencies. AI diffusion can leave DC bias or rumble here. Values above ~0.5% are elevated.</span>
          </span>
        </span>
        <span class="sv">{_boundary_val(infra)}{infra_note}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Ultrasonic Ratio (20–22 kHz)
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Energy fraction in 20–22 kHz band. Only computed for uploads with native SR ≥ 40 kHz. Human masters are shelf-filtered above 18–20 kHz. N/A for YouTube or low-SR files.</span>
          </span>
        </span>
        <span class="sv">{_boundary_val(ultra, ".6f")}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Phase Coherence Δ (LF−HF)
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">LF inter-channel coherence minus HF coherence. AI diffusion can generate low and high frequencies as separate events, making HF phase incoherent while LF stays stable. Positive = AI pattern; N/A for mono sources.</span>
          </span>
        </span>
        <span class="sv">{_boundary_val(phase, ".3f")}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if fr.forensic_notes:
        notes_html = "".join(
            f"<div style='font-family:\"Figtree\",sans-serif;font-size:.82rem;"
            f"color:var(--text);padding:8px 0;border-bottom:1px solid var(--border);'>"
            f"ℹ {html_mod.escape(note)}</div>"
            for note in fr.forensic_notes
        )
        st.markdown(
            f"<div style='margin-top:14px;'>"
            f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.56rem;font-weight:600;"
            f"letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;'>"
            f"Notes</div>"
            f"{notes_html}</div>",
            unsafe_allow_html=True,
        )



_DIALOGUE_LABEL_COLORS: dict[str, str] = {
    "Dialogue-Ready": "var(--ok)",
    "Mixed":          "var(--grade-c)",
    "Dialogue-Heavy": "var(--danger)",
}

_LUFS_PLATFORMS: list[tuple[str, str]] = [
    ("Spotify",      "delta_spotify"),
    ("Apple Music",  "delta_apple_music"),
    ("YouTube",      "delta_youtube"),
    ("Broadcast",    "delta_broadcast"),
]

# Verdict badge colors for each loudness verdict string (#95)
_LOUDNESS_VERDICT_COLORS: dict[str, str] = {
    "Broadcast-ready": "var(--ok)",
    "Streaming-ready": "var(--ok)",
    "Streaming-hot":   "var(--grade-c)",
    "Needs mastering": "var(--issue-location)",
    "Clipping risk":   "var(--danger)",
}

# (platform display name, gain_*_db attr, target LUFS) for gain table (#94)
_GAIN_PLATFORMS: list[tuple[str, str, float]] = [
    ("Spotify",      "gain_spotify_db",     CONSTANTS.LUFS_SPOTIFY),
    ("Apple Music",  "gain_apple_music_db", CONSTANTS.LUFS_APPLE_MUSIC),
    ("YouTube",      "gain_youtube_db",     CONSTANTS.LUFS_YOUTUBE),
    ("Broadcast",    "gain_broadcast_db",   CONSTANTS.LUFS_BROADCAST),
]


def _lufs_to_bar_pct(lufs: float, scale_min: float = -40.0, scale_max: float = 0.0) -> float:
    """Convert a LUFS value to a 0–100% position on the loudness profile bar."""
    return max(0.0, min(100.0, (lufs - scale_min) / (scale_max - scale_min) * 100.0))


def _render_loudness_profile_bar(aq: "AudioQualityResult") -> None:
    """Horizontal LUFS bar with LRA range shading and Spotify target line (#98).

    Scale: -40 LUFS (left) to 0 LUFS (right). Pure HTML/CSS — no matplotlib.
    """
    lufs        = aq.integrated_lufs
    lra         = aq.loudness_range_lu
    target_pct  = _lufs_to_bar_pct(CONSTANTS.LUFS_SPOTIFY)
    center_pct  = _lufs_to_bar_pct(lufs)
    lra_lo_pct  = _lufs_to_bar_pct(lufs - lra / 2.0)
    lra_width   = max(0.0, _lufs_to_bar_pct(lufs + lra / 2.0) - lra_lo_pct)

    verdict_color = _LOUDNESS_VERDICT_COLORS.get(aq.loudness_verdict, "var(--muted)")

    st.markdown(
        f"""<div style="margin:10px 0 4px;" aria-hidden="true">
  <div style="position:relative;height:18px;border-radius:4px;
              background:var(--border-hr);overflow:visible;">
    <div style="position:absolute;left:{lra_lo_pct:.2f}%;width:{lra_width:.2f}%;
                height:100%;background:var(--accent);opacity:.18;border-radius:3px;"></div>
    <div style="position:absolute;left:{target_pct:.2f}%;top:-3px;bottom:-3px;
                width:2px;background:var(--ok);opacity:.7;"
         title="Spotify target (–14 LUFS)"></div>
    <div style="position:absolute;left:{center_pct:.2f}%;top:50%;
                transform:translate(-50%,-50%);width:10px;height:10px;
                border-radius:50%;background:{verdict_color};
                border:2px solid var(--bg);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;
              font-family:'JetBrains Mono',monospace;font-size:.52rem;
              color:var(--dim);margin-top:3px;">
    <span>–40 LUFS</span><span>–20</span><span>–10</span><span>0 LUFS</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def _render_gain_table(aq: "AudioQualityResult") -> None:
    """Render per-platform gain adjustment recommendations (#94)."""
    rows = ""
    for name, attr, target in _GAIN_PLATFORMS:
        gain = getattr(aq, attr)
        color = "var(--ok)" if abs(gain) <= CONSTANTS.GAIN_OK_THRESHOLD_DB else ("var(--grade-c)" if abs(gain) <= CONSTANTS.GAIN_WARN_THRESHOLD_DB else "var(--danger)")
        rows += (
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:.6rem;"
            f"color:var(--dim);width:80px;flex-shrink:0;'>{html_mod.escape(name)}</div>"
            f"<div style='font-family:Chakra Petch,monospace;font-size:.75rem;"
            f"font-weight:600;color:{color};width:60px;'>{gain:+.1f} dB</div>"
            f"<div style='font-family:Figtree,sans-serif;font-size:.62rem;"
            f"color:var(--dim);'>target {target:.0f} LUFS</div>"
            f"</div>"
        )
    st.markdown(
        f"<div style='margin:8px 0 4px;'>{rows}</div>",
        unsafe_allow_html=True,
    )


def _render_audio_quality_card(aq: Optional["AudioQualityResult"]) -> None:
    """Render LUFS broadcast loudness and dialogue-readiness metrics."""
    st.markdown("""
    <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                display:flex;align-items:center;gap:10px;margin:20px 0 14px;">
      <span>◈ Loudness & Dialogue</span>
      <div style="flex:1;height:1px;background:var(--border-hr);"></div>
    </div>
    """, unsafe_allow_html=True)

    if aq is None:
        st.markdown(
            "<div style='color:var(--dim);font-size:.84rem;font-family:Figtree,sans-serif;'>"
            "Loudness data unavailable.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Verdict badge (#95) ────────────────────────────────────────────────
    verdict_color = _LOUDNESS_VERDICT_COLORS.get(aq.loudness_verdict, "var(--muted)")
    st.markdown(
        f"<div style='display:inline-block;font-family:\"Chakra Petch\",monospace;"
        f"font-size:.58rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;"
        f"color:{verdict_color};border:1px solid {verdict_color};border-radius:3px;"
        f"padding:2px 8px;margin-bottom:10px;'>{html_mod.escape(aq.loudness_verdict)}</div>",
        unsafe_allow_html=True,
    )

    # ── LUFS row ──────────────────────────────────────────────────────────
    peak_color  = "var(--danger)" if aq.true_peak_warning else "var(--ok)"
    peak_label  = f"{aq.true_peak_dbfs:+.1f} dBFS"
    peak_warn   = " ⚠ Clipping risk" if aq.true_peak_warning else ""

    platform_cells = ""
    for name, attr in _LUFS_PLATFORMS:
        delta = getattr(aq, attr)
        color = "var(--danger)" if delta > 1.0 else ("var(--ok)" if delta < -0.5 else "var(--muted)")
        sign  = "+" if delta > 0 else ""
        platform_cells += (
            f'<div style="text-align:center;min-width:70px;">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;'
            f'color:var(--dim);margin-bottom:3px;">{html_mod.escape(name)}</div>'
            f'<div style="font-family:\'Chakra Petch\',monospace;font-size:.8rem;'
            f'font-weight:600;color:{color};">{sign}{delta:.1f} LU</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div class="sig" style="padding:14px 16px;margin-bottom:10px;">
      <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:12px;">
        <div style="font-family:'Chakra Petch',monospace;font-size:1.6rem;
                    font-weight:700;color:var(--text);">{aq.integrated_lufs:.1f}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                    color:var(--dim);">LUFS integrated
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">Integrated loudness measured across the full track (ITU-R BS.1770-4). Streaming platforms automatically normalize to their target — the closer your master is to that target, the less processing is applied and the more consistent it sounds across platforms. –14 LUFS is the sweet spot for Spotify and YouTube.</span>
          </span>
        </div>
        <div style="margin-left:auto;font-family:'JetBrains Mono',monospace;
                    font-size:.65rem;color:{peak_color};">
          {html_mod.escape(peak_label)}{html_mod.escape(peak_warn)}
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">True peak level — the highest sample amplitude in the track. Anything above –1.0 dBFS risks inter-sample distortion after lossy encoding (AAC, MP3). A clipping warning means the master needs a limiter pass before streaming delivery.</span>
          </span>
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px;">{platform_cells}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:var(--dim);">
        Platform delta
        <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
          <span class="tip-box">How far your track sits from each platform's loudness target. Negative (–) = platform turns you up. Positive (+) = platform turns you down. Values within ±1 LU are negligible. A large positive value (e.g. +8 LU for Broadcast) means you need a separate quieter master for that delivery format.</span>
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Loudness profile bar (#98) ─────────────────────────────────────────
    _render_loudness_profile_bar(aq)

    # ── Gain adjustments (#94) ────────────────────────────────────────────
    with st.expander("Gain adjustments by platform", expanded=False):
        _render_gain_table(aq)

    # ── Dialogue-ready row ────────────────────────────────────────────────
    dial_color  = _DIALOGUE_LABEL_COLORS.get(aq.dialogue_label, "var(--muted)")
    dial_pct    = int(aq.dialogue_score * 100)
    vo_headroom = aq.vo_headroom_db
    vo_str      = (
        f"~{vo_headroom:g} dB of VO headroom"
        if vo_headroom is not None
        else ""
    )
    vo_html = (
        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:.66rem;"
        f"color:var(--dim);margin-top:3px;\">{html_mod.escape(vo_str)}</div>"
        if vo_str
        else ""
    )

    st.markdown(f"""
    <div class="sig" style="padding:14px 16px;display:flex;align-items:center;gap:16px;">
      <div>
        <div style="font-family:'Chakra Petch',monospace;font-size:.5rem;font-weight:600;
                    letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                    margin-bottom:4px;">Dialogue Compatibility
          <span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
            <span class="tip-box">How well this track sits under spoken dialogue or voiceover. Measured as the fraction of spectral energy sitting <em>outside</em> the 300–3000 Hz speech range. "Dialogue-Ready" (≥70%) means the track leans into bass or high-frequency texture — a VO will cut through clearly. "Dialogue-Heavy" (&lt;40%) means most energy competes directly with the human voice — expect muddiness without a stem mix or significant EQ.</span>
          </span>
        </div>
        <div style="font-family:'Chakra Petch',monospace;font-size:1.1rem;font-weight:700;
                    color:{dial_color};">{html_mod.escape(aq.dialogue_label)}</div>
        {vo_html}
      </div>
      <div style="margin-left:auto;text-align:right;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
                    font-weight:700;color:{dial_color};">{dial_pct}%</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                    color:var(--dim);">outside dialogue band</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_sync_readiness(compliance: Optional[ComplianceReport]) -> None:
    sting     = compliance.sting     if compliance else None
    evolution = compliance.evolution if compliance else None
    intro     = compliance.intro     if compliance else None

    if not (sting or evolution or intro):
        st.markdown(
            "<div style='color:var(--dim);font-size:.84rem;padding:8px 0;'>"
            "No structural data available.</div>",
            unsafe_allow_html=True,
        )
        return

    if sting:
        s_ratio   = sting.final_energy_ratio
        ratio_str = f" — tail energy {s_ratio:.1%}" if s_ratio is not None else ""

        # Enrich the display value with fade severity (#103) or cut sub-type (#104)
        if sting.ending_type == "fade" and sting.fade_severity > 0:
            sev_label = (
                "severe"   if sting.fade_severity >= 0.7
                else "moderate" if sting.fade_severity >= 0.35
                else "mild"
            )
            tail_str  = f"{sting.fade_tail_seconds:.0f}s tail"
            detail_suffix = f" ({sev_label}, {tail_str})"
        elif sting.ending_type == "cut" and sting.cut_type:
            sub = "on-beat" if sting.cut_type == "clean_cut" else "mid-phrase"
            detail_suffix = f" — {sub}"
        else:
            detail_suffix = ""

        flag_text = "Track ends with a fade — may bleed into dialogue or scene audio." if sting.flag else None
        _sync_readiness_row(
            icon="🔔", label="Sting / Ending Type",
            value=sting.ending_type.title() + detail_suffix + ratio_str,
            ok=not sting.flag,
            flag_text=flag_text,
            tip="Detects how the track ends — critical for sync placement.<br><br>"
                "<strong>Sting</strong>: sharp final hit with rapid silence after. ✓<br>"
                "<strong>Cut (on-beat)</strong>: lands on a beat boundary — clean edit point. ✓<br>"
                "<strong>Cut (mid-phrase)</strong>: abrupt off-beat stop — workable but imprecise. ✓<br>"
                "<strong>Fade (mild/moderate/severe)</strong>: gradual energy decline — can bleed into dialogue. ⚠ Flagged.",
        )

    if evolution:
        n_stag   = evolution.stagnant_windows
        e_passes = not evolution.flag
        e_value  = f"{n_stag} stagnant window{'s' if n_stag != 1 else ''}" if evolution.flag else "Energy evolves consistently"
        _sync_readiness_row(
            icon="📊", label="4-8 Bar Energy Rule",
            value=e_value,
            ok=e_passes,
            flag_text=evolution.detail if evolution.flag else None,
            tip="Splits the track into 4-bar windows and measures spectral contrast evolution. "
                "A delta below <strong>10%</strong> between windows flags stagnant energy.",
        )
        # Per-section breakdown (#106) — only render when data is available
        if evolution.section_details:
            sec_rows_html = "".join(
                f"<tr>"
                f"<td style='padding:2px 8px 2px 0;color:var(--text);'>{html_mod.escape(str(sd['label']).title())}</td>"
                f"<td style='padding:2px 8px;color:{'var(--issue-explicit)' if sd['flag'] else 'var(--pass)'};'>"
                f"{'⚠' if sd['flag'] else '✓'}"
                f"</td>"
                f"<td style='padding:2px 0;color:var(--dim);font-size:.8rem;'>{sd['note']}</td>"
                f"</tr>"
                for sd in evolution.section_details
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;margin-top:4px;'>"
                f"{sec_rows_html}</table>",
                unsafe_allow_html=True,
            )

    if intro:
        i_ok    = not intro.flag
        dur_str = f"{intro.intro_seconds:.1f}s"
        source_suffix = (
            " (allin1)" if intro.source == "allin1"
            else " (pre-vocal estimate)" if intro.source == "whisper_fallback"
            else ""
        )
        flag_text = (
            f"Intro is {intro.intro_seconds:.1f}s — exceeds the {CONSTANTS.INTRO_MAX_SECONDS}s sync readiness limit."
            if intro.flag else None
        )
        _sync_readiness_row(
            icon="⏱", label="Intro Length",
            value=dur_str + source_suffix,
            ok=i_ok,
            flag_text=flag_text,
            tip=f"Sync readiness rule: intro sections longer than <strong>{CONSTANTS.INTRO_MAX_SECONDS}s</strong> risk losing the "
                "picture editor's attention before the track establishes its feel.",
        )


def _sync_readiness_row(icon: str, label: str, value: str, ok: bool,
                        flag_text: Optional[str], tip: str) -> None:
    status_color = "var(--sync-pass)" if ok else "var(--sync-fail)"
    status_icon  = "✓" if ok else "⚠"
    st.markdown(
        f"<div style='display:flex;align-items:flex-start;gap:10px;"
        f"padding:8px 0;border-bottom:1px solid var(--border-hr);'>"
        f"<div style='font-size:.9rem;flex-shrink:0;margin-top:1px;'>{icon}</div>"
        f"<div style='flex:1;'>"
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:2px;'>"
        f"<span style='font-family:\"Chakra Petch\",monospace;font-size:.6rem;font-weight:600;"
        f"letter-spacing:.07em;color:var(--muted);text-transform:uppercase;'>{label}</span>"
        f"<span class='tip-wrap'><span class='tip-icon'>?</span>"
        f"<span class='tip-box'>{tip}</span></span>"
        f"</div>"
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:.74rem;"
        f"color:{status_color};'>{status_icon} {value}</div>"
        + (f"<div style='font-family:\"Figtree\",sans-serif;font-size:.73rem;"
           f"color:var(--sync-fail);margin-top:3px;line-height:1.45;'>{flag_text}</div>"
           if flag_text else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Discovery & Licensing
# ---------------------------------------------------------------------------

_SIMILAR_TRACK_SOURCE_COLORS: dict[str, str] = {
    "lastfm":  "var(--muted)",
    "spotify": "#1db954",        # Spotify brand green — fixed value, not a theme variable
    "audio":   "var(--info)",
}

_SIMILAR_TRACK_SOURCE_LABELS: dict[str, str] = {
    "lastfm":  "Last.fm",
    "spotify": "Spotify",
    "audio":   "Audio",
}


def _build_source_pill_html(source: str) -> str:
    """Return a small source badge pill for a similar track row (#129).

    Pure function — no I/O.
    """
    color = _SIMILAR_TRACK_SOURCE_COLORS.get(source, "var(--muted)")
    label = _SIMILAR_TRACK_SOURCE_LABELS.get(source, html_mod.escape(source))
    return (
        f"<span style='font-family:Figtree,sans-serif;font-size:.52rem;"
        f"color:{color};border:1px solid {color}66;border-radius:3px;"
        f"padding:1px 5px;white-space:nowrap;opacity:.85;'>"
        f"{label}</span>"
    )


_POPULARITY_TIER_COLORS: dict[str, str] = {
    "Emerging":   "var(--dim)",
    "Regional":   "var(--issue-location)",  # blue — defined in styles.py
    "Mainstream": "var(--grade-c)",          # amber — defined in styles.py
    "Global":     "var(--accent)",           # orange — defined in styles.py
}

# Tier score ranges used to infer position-within-tier confidence (#113).
# Derived from CONSTANTS so boundaries stay in sync with services/discovery.py.
_TIER_RANGES: dict[str, tuple[int, int]] = {
    "Emerging":   (0,                              CONSTANTS.POPULARITY_REGIONAL_MIN - 1),
    "Regional":   (CONSTANTS.POPULARITY_REGIONAL_MIN,    CONSTANTS.POPULARITY_MAINSTREAM_MIN - 1),
    "Mainstream": (CONSTANTS.POPULARITY_MAINSTREAM_MIN,  CONSTANTS.POPULARITY_GLOBAL_MIN - 1),
    "Global":     (CONSTANTS.POPULARITY_GLOBAL_MIN,      100),
}


def _tier_confidence_label(popularity_score: int, tier: str) -> Optional[str]:
    """
    Return '(strong)' / '(borderline)' sub-label based on position within the tier range.

    Top 25 % of the range → 'strong'; bottom 25 % → 'borderline'; middle 50 % → None.
    Returns None for unrecognised tiers or if score is outside the expected range.
    Pure — no I/O.
    """
    bounds = _TIER_RANGES.get(tier)
    if bounds is None:
        return None
    lo, hi = bounds
    span = hi - lo
    if span <= 0:
        return None
    # Clamp to [0, 1] — blending rounding can push scores fractionally outside bounds.
    position = max(0.0, min(1.0, (popularity_score - lo) / span))
    if position >= 0.75:
        return "(strong)"
    if position <= 0.25:
        return "(borderline)"
    return None


def _split_fee_band(low: int, high: int) -> tuple[int, int, int, int]:
    """
    Split a fee band into a narrower 'likely' range and a full 'possible' range.

    Trims 20% from each end of the band to produce the middle-60% likely range.
    Returns (likely_low, likely_high, possible_low, possible_high).
    Pure — no I/O.
    """
    span = high - low
    trim = int(span * 0.20)
    return low + trim, high - trim, low, high


def _sync_readiness_fee_modifier(
    sting_pass: bool,
    energy_pass: bool,
    intro_pass: bool,
) -> tuple[float, str]:
    """
    Return (multiplier, note_text) based on compliance check results (#112).

    All three pass  → +15% uplift (plug-and-play).
    Sting or intro fail → −20% discount (edit work expected).
    Energy fails alone → no multiplier, advisory note only.
    Pure — no I/O.
    """
    if sting_pass and energy_pass and intro_pass:
        return (CONSTANTS.SYNC_READINESS_UPLIFT, "+15% · Plug-and-play: no edit work required")
    if not sting_pass or not intro_pass:
        return (CONSTANTS.SYNC_READINESS_DISCOUNT, "−20% · Edit work expected — may reduce offer")
    # energy_pass is False but sting and intro are fine
    return (1.0, "Advisory: energy evolution may need attention")


def _popularity_signal_rows(pop: object) -> list[tuple[str, str, int, bool]]:
    """
    Build signal breakdown rows for the popularity card.

    Returns list of (label, raw_value_str, normalised_score_0_100, is_winner).
    Only includes signals that contributed a non-zero score.
    Pure — no I/O.
    """
    from core.config import CONSTANTS
    from services.discovery import _normalise_lastfm, _normalise_views

    rows: list[tuple[str, str, int, bool]] = []

    lastfm_score = _normalise_lastfm(pop.listeners, CONSTANTS) if pop.listeners else 0
    if lastfm_score > 0:
        rows.append(("Last.fm", f"{pop.listeners:,} listeners", lastfm_score, False))

    if pop.spotify_score is not None:
        rows.append(("Spotify", f"{pop.spotify_score}/100", pop.spotify_score, False))

    view_count = pop.platform_metrics.get("view_count", 0)
    if view_count > 0:
        view_score = _normalise_views(view_count, CONSTANTS)
        rows.append(("Views", _fmt_count(view_count), view_score, False))

    # Mark the winning (highest) signal
    if rows:
        max_score = max(r[2] for r in rows)
        rows = [
            (label, raw, score, score == max_score)
            for label, raw, score, _ in rows
        ]

    return rows


def _fmt_count(n: int) -> str:
    """Format a large integer as a compact string (e.g. 1.2M, 500K). Pure."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _build_clearance_template(result: AnalysisResult) -> str:
    """Build a plaintext sync clearance quote from all available scan data.

    Pure function — no I/O, no side effects.  Any missing field falls back
    to a human-readable placeholder so the template never crashes.
    """
    audio  = result.audio
    pop    = result.popularity
    comp   = result.compliance
    aq     = result.audio_quality
    legal  = result.legal

    title  = audio.metadata.get("title", "") or "Unknown Track"
    artist = audio.metadata.get("artist", "") or "Unknown Artist"

    tier_str = pop.tier if pop else "Unknown"

    if pop and pop.sync_cost_low is not None and pop.sync_cost_high is not None:
        fee_str = f"${pop.sync_cost_low:,} – ${pop.sync_cost_high:,}"
    else:
        fee_str = "not available"

    isrc = (legal.isrc or "not provided") if legal else "not provided"
    pro  = (legal.pro_match or "unknown") if legal else "unknown"

    if comp:
        sting_str  = "✓" if not comp.sting.flag    else "✗"
        energy_str = "✓" if not comp.evolution.flag else "✗"
        intro_str  = "✓" if not comp.intro.flag     else "✗"
        readiness_str = f"Sting {sting_str} · Energy {energy_str} · Intro {intro_str}"
    else:
        readiness_str = "not run"

    dialogue_str = (aq.dialogue_label or "not run") if aq else "not run"

    lines = [
        f"Quick sync quote — {title} by {artist}",
        f"Tier: {tier_str}",
        f"Estimated range: {fee_str}",
        f"ISRC: {isrc}",
        f"PRO: {pro}",
        f"Sync readiness: {readiness_str}",
        f"Dialogue compatibility: {dialogue_str}",
        "Generated by Sync-Safe",
    ]
    return "\n".join(lines)


def _render_popularity_card(result: AnalysisResult) -> None:
    """Render track popularity tier, estimated sync cost, and per-signal breakdown."""
    pop = result.popularity
    if pop is None:
        st.markdown(
            "<div style='color:var(--dim);font-size:.84rem;font-family:Figtree,sans-serif;'>"
            "Popularity data unavailable — track not found on Last.fm.</div>",
            unsafe_allow_html=True,
        )
        return

    tier_color     = _POPULARITY_TIER_COLORS.get(pop.tier, "var(--dim)")
    listeners_fmt  = f"{pop.listeners:,}"
    confidence_lbl = _tier_confidence_label(pop.popularity_score, pop.tier)
    confidence_html = (
        f"<div style=\"font-family:'JetBrains Mono',monospace;font-size:.58rem;"
        f"color:var(--dim);margin-top:2px;\">{html_mod.escape(confidence_lbl)}</div>"
        if confidence_lbl else ""
    )

    # Apply sync-readiness multiplier (#112) then split into likely/possible bands (#111)
    comp = result.compliance
    if comp:
        mult, mod_note = _sync_readiness_fee_modifier(
            sting_pass=not comp.sting.flag,
            energy_pass=not comp.evolution.flag,
            intro_pass=not comp.intro.flag,
        )
    else:
        mult, mod_note = 1.0, ""
    adj_low  = int(pop.sync_cost_low  * mult)
    adj_high = int(pop.sync_cost_high * mult)
    likely_low, likely_high, poss_low, poss_high = _split_fee_band(adj_low, adj_high)

    likely_fmt   = f"${likely_low:,} – ${likely_high:,}"
    possible_fmt = f"${poss_low:,} – ${poss_high:,}"

    # Choose note color based on modifier direction
    if mult > 1.0:
        mod_color = "var(--ok)"
    elif mult < 1.0:
        mod_color = "var(--grade-c)"
    else:
        mod_color = "var(--muted)"
    mod_html = (
        f"<div style=\"font-family:'Figtree',sans-serif;font-size:.62rem;"
        f"color:{mod_color};margin-top:3px;\">{html_mod.escape(mod_note)}</div>"
        if mod_note else ""
    )

    st.markdown(f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
      <div class="sig" style="flex:1;min-width:120px;padding:14px 16px;text-align:center;">
        <div style="font-family:'Chakra Petch',monospace;font-size:.5rem;font-weight:600;
                    letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                    margin-bottom:6px;">Popularity</div>
        <div style="font-family:'Chakra Petch',monospace;font-size:1.2rem;font-weight:700;
                    color:{tier_color};">{pop.tier}</div>
        {confidence_html}
        <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                    color:var(--muted);margin-top:4px;">{listeners_fmt} listeners</div>
      </div>
      <div class="sig" style="flex:2;min-width:180px;padding:14px 16px;">
        <div style="font-family:'Chakra Petch',monospace;font-size:.5rem;font-weight:600;
                    letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                    margin-bottom:6px;">Est. Sync Fee</div>
        <div style="font-family:'Chakra Petch',monospace;font-size:.95rem;font-weight:700;
                    color:var(--text);">{likely_fmt}</div>
        <div style="font-family:'Figtree',sans-serif;font-size:.7rem;color:var(--muted);
                    margin-top:2px;">Likely · Possible: {possible_fmt}</div>
        {mod_html}
        <div style="font-family:'Figtree',sans-serif;font-size:.62rem;color:var(--dim);
                    margin-top:2px;">Industry estimates 2024–2026.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Why the fee varies — collapsed by default (#115) ──────────────────────
    with st.expander("Why does the sync fee vary so much?", expanded=False):
        st.markdown("""
**Usage** — A 30-second TV ad commands 2–4× a documentary scene. Trailers can be 3–5×.

**Territory** — US-only is the baseline. Worldwide rights typically double the fee.

**Exclusivity** — Non-exclusive (others can license the same track) vs exclusive (you own \
the only sync) roughly doubles the rate.

**Term** — A 2-year license vs in-perpetuity can add 50–100%.

**Medium** — Broadcast TV vs online-only vs theatrical all carry different rate cards.

Ranges shown are industry averages (2024–2026) and should be treated as a starting point \
for negotiation, not a firm quote.
        """)

    # ── Sync quote template — copy-to-clipboard via st.code (#114) ────────────
    # Placed before the signal_rows guard so it always renders when pop is present.
    st.code(_build_clearance_template(result), language=None)

    # ── Signal breakdown ──────────────────────────────────────────────────────
    signal_rows = _popularity_signal_rows(pop)
    if not signal_rows:
        return

    rows_html = ""
    for label, raw, score, is_winner in signal_rows:
        bar_color  = "var(--accent)" if is_winner else "var(--dim)"
        label_color = "var(--text)" if is_winner else "var(--muted)"
        winner_mark = " ✓" if is_winner else ""
        rows_html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;
                      color:{label_color};font-weight:{'600' if is_winner else '400'};
                      width:60px;flex-shrink:0;">{html_mod.escape(label)}{winner_mark}</div>
          <div style="flex:1;height:4px;border-radius:2px;background:var(--border-hr);
                      overflow:hidden;">
            <div style="height:100%;width:{score}%;background:{bar_color};
                        border-radius:2px;transition:width .4s ease;"></div>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.56rem;
                      color:var(--muted);width:28px;text-align:right;">{score}</div>
          <div style="font-family:'Figtree',sans-serif;font-size:.62rem;
                      color:var(--dim);min-width:80px;">{html_mod.escape(raw)}</div>
        </div>"""

    st.markdown(
        "<div class='sig' style='padding:12px 14px;margin-bottom:4px;'>"
        "<div style=\"font-family:'Chakra Petch',monospace;font-size:.48rem;font-weight:600;"
        "letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-bottom:10px;\">"
        "Signal Breakdown</div>"
        + rows_html
        + "<div style=\"font-family:'Figtree',sans-serif;font-size:.6rem;color:var(--dim);"
        "margin-top:8px;\">Score 0–100 · tier uses highest signal</div>"
        "</div>",
        unsafe_allow_html=True,
    )


_PRO_CONFIDENCE_COLORS: dict[str, str] = {
    "High":   "var(--ok)",
    "Medium": "var(--grade-c)",
    "Low":    "var(--dim)",
}
_PRO_CONFIDENCE_SUFFIX: dict[str, str] = {
    "High":   "",
    "Medium": "",
    "Low":    " — inferred from ISRC",
}


def _pro_confidence_badge_html(confidence: Optional[str]) -> str:
    """Return an inline HTML badge for PRO confidence, or empty string if None (#118)."""
    if confidence is None:
        return ""
    color  = _PRO_CONFIDENCE_COLORS.get(confidence, "var(--dim)")
    suffix = _PRO_CONFIDENCE_SUFFIX.get(confidence, "")
    label  = html_mod.escape(f"{confidence} confidence{suffix}")
    return (
        f"<span style='font-family:Figtree,sans-serif;font-size:.58rem;"
        f"color:{color};border:1px solid {color};border-radius:3px;"
        f"padding:1px 6px;white-space:nowrap;'>{label}</span>"
    )


def _render_legal_and_discovery(result: AnalysisResult) -> None:
    _render_popularity_card(result)

    st.markdown("""
    <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <span>◈ Rights Lookup</span>
      <div style="flex:1;height:1px;background:var(--border-hr);"></div>
    </div>
    """, unsafe_allow_html=True)

    if result.legal:
        # ISRC + inferred PRO badge (only shown when MusicBrainz returned a hit)
        if result.legal.isrc or result.legal.pro_match:
            isrc_val  = result.legal.isrc or ""
            pro_val   = result.legal.pro_match or "Unknown"
            isrc_text = html_mod.escape(isrc_val or "—")
            pro_text  = html_mod.escape(pro_val)
            st.markdown(f"""
            <div class="sig" style="padding:14px 18px;margin-bottom:12px;display:flex;
                         flex-wrap:wrap;gap:18px;align-items:center;">
              <div>
                <div style="font-size:.58rem;font-weight:600;letter-spacing:.14em;
                            text-transform:uppercase;color:var(--dim);margin-bottom:4px;">ISRC</div>
                <div style="font-family:'Chakra Petch',monospace;font-size:.9rem;
                            color:var(--text);">{isrc_text}</div>
              </div>
              <div>
                <div style="font-size:.58rem;font-weight:600;letter-spacing:.14em;
                            text-transform:uppercase;color:var(--dim);margin-bottom:4px;">Inferred PRO</div>
                <div style="display:flex;align-items:center;gap:8px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.9rem;
                              color:var(--accent);">{pro_text}</div>
                  {_pro_confidence_badge_html(result.legal.pro_confidence)}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Copy buttons (#119) — need components.html() for JS clipboard access
            _col_isrc, _col_pro, _col_note, _ = st.columns([1, 1, 1.6, 2])
            if isrc_val:
                with _col_isrc:
                    _copy_button(isrc_val, "Copy ISRC")
            with _col_pro:
                _copy_button(pro_val, "Copy PRO")
            clearance_note = (
                f"ISRC: {isrc_val or '—'}\n"
                f"PRO: {pro_val}\n"
                f"ASCAP: {result.legal.ascap or '—'}\n"
                f"BMI: {result.legal.bmi or '—'}\n"
                f"SESAC: {result.legal.sesac or '—'}"
            )
            with _col_note:
                _copy_button(clearance_note, "Copy clearance note")

        if result.legal.isrc:
            st.caption(
                "ISRC (International Standard Recording Code) — unique identifier for this recording. "
                "Paste directly into library delivery systems, clearance forms, or CMS metadata fields."
            )

        for name, url in [("ASCAP", result.legal.ascap), ("BMI", result.legal.bmi), ("SESAC", result.legal.sesac)]:
            if url:
                st.link_button(f"Search {name} →", url, use_container_width=True)

        # HFA / Songfile mechanical rights links (#120)
        _title  = result.audio.metadata.get("title", "") or ""
        _artist = result.audio.metadata.get("artist", "") or ""
        st.link_button("HFA (Mechanical) →",  hfa_url(_title, _artist),  use_container_width=True)
        st.link_button("Songfile →",           songfile_url(_title, _artist), use_container_width=True)

    # ── Sync clearance quote template — copy via st.code built-in (#121) ──────
    st.code(_build_clearance_template(result), language=None)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <span>◈ Similar Tracks</span>
      <div style="flex:1;height:1px;background:var(--border-hr);"></div>
    </div>
    """, unsafe_allow_html=True)

    if result.similar_tracks:
        rows = ""
        for t in result.similar_tracks:
            safe_title  = html_mod.escape(t.title)
            safe_artist = html_mod.escape(t.artist)
            btn = (
                f'<a href="{html_mod.escape(t.youtube_url)}" target="_blank" rel="noopener noreferrer"'
                f' class="t-btn" aria-label="Preview {safe_title} by {safe_artist} on YouTube">▶ Preview</a>'
                if t.youtube_url
                else '<button disabled class="t-btn" style="opacity:.3;cursor:not-allowed;">No link</button>'
            )
            sim_clamped = min(1.0, max(0.0, t.similarity))
            sim_pct     = f"{sim_clamped:.0%}"
            sim_bar     = f"{sim_clamped:.3f}"
            tier_pill   = ""
            if t.popularity_tier:
                tc = _POPULARITY_TIER_COLORS.get(t.popularity_tier, "var(--dim)")
                tier_pill = (
                    f"<span style='font-family:Figtree,sans-serif;font-size:.55rem;"
                    f"color:{tc};border:1px solid {tc};border-radius:3px;"
                    f"padding:1px 5px;white-space:nowrap;'>"
                    f"{html_mod.escape(t.popularity_tier)}</span>"
                )
            source_pill = _build_source_pill_html(t.source)
            rows += f"""
            <div class="t-row">
              <div style="flex:1;min-width:0;">
                <div class="t-art">{safe_artist}</div>
                <div class="t-nm">{safe_title}</div>
                <div style="display:flex;align-items:center;gap:8px;margin-top:5px;">
                  <div style="width:72px;height:3px;border-radius:2px;background:var(--border-hr);overflow:hidden;flex-shrink:0;">
                    <div style="height:3px;border-radius:2px;background:var(--accent);width:72px;
                                transform:scaleX({sim_bar});transform-origin:left;"></div>
                  </div>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.62rem;
                               color:var(--dim);">{sim_pct}</span>
                  {tier_pill}
                  {source_pill}
                </div>
              </div>
              {btn}
            </div>"""
        st.markdown(f"<div class='sig' style='padding:18px;'>{rows}</div>", unsafe_allow_html=True)

        # Copy search list (#125) — one click copies "Title — Artist" per line
        search_list = "\n".join(
            f"{t.title} — {t.artist}" for t in result.similar_tracks
        )
        _copy_button(search_list, "Copy search list")

        if len(result.similar_tracks) < 3:
            n = len(result.similar_tracks)
            st.caption(
                f"Only {n} strong match{'es' if n != 1 else ''} found. "
                "Similar tracks data is richer for mainstream catalog — "
                "try a well-known reference track for a fuller comparison set."
            )
    else:
        st.markdown(
            "<div style='color:var(--dim);font-size:.84rem;'>No similar tracks found.</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Lyrics & Content Audit
# ---------------------------------------------------------------------------

def _render_lyric_section(result: AnalysisResult) -> None:
    segments   = result.transcript
    compliance = result.compliance
    authorship = result.authorship
    sections   = result.structure.sections if result.structure else []

    flags    = compliance.flags if compliance else []
    grade    = compliance.grade if compliance else "N/A"

    flagged_ts: set[int] = {f.timestamp_s for f in flags}
    flags_by_ts: dict[int, list[ComplianceFlag]] = {}
    for f in flags:
        flags_by_ts.setdefault(f.timestamp_s, []).append(f)

    _render_theme_mood(result)
    _render_authorship_banner(authorship)

    col_lyr, col_audit = st.columns([55, 45], gap="large")
    with col_lyr:
        _render_lyric_column(segments, sections, flagged_ts, flags_by_ts)
    with col_audit:
        _render_audit_column(flags, grade)


# ---------------------------------------------------------------------------
# Lyric section sub-renderers (extracted to keep each function under 40 lines)
# ---------------------------------------------------------------------------

def _render_theme_bars(theme_scores: dict[str, float]) -> None:
    """
    Render category-coloured pill + CSS-transform confidence bar for each theme (#167/#168).

    Uses transform:scaleX() instead of width: to avoid layout thrashing.
    Each bar has role="progressbar" + aria-valuenow for screen reader accessibility.
    Only themes at or above THEME_MIN_CONFIDENCE are rendered.
    """
    for theme, score in sorted(theme_scores.items(), key=lambda x: -x[1]):
        if score < CONSTANTS.THEME_MIN_CONFIDENCE:
            continue
        tax_entry  = THEME_TAXONOMY.get(theme, {})
        category   = tax_entry.get("category", "")
        pill_color = _THEME_CATEGORY_COLORS.get(category, "var(--surface-2)")
        bar_pct    = int(score * 100)
        scale      = round(score, 3)
        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">'
            f'<span style="display:inline-block;padding:2px 9px;border-radius:10px;'
            f'font-size:0.76rem;background:{pill_color};color:var(--text);">'
            f'{html_mod.escape(theme)}</span>'
            f'<span style="font-size:0.72rem;color:var(--dim);" aria-hidden="true">{bar_pct}%</span>'
            f'</div>'
            f'<div style="height:4px;border-radius:2px;background:var(--surface-2);overflow:hidden;"'
            f' role="progressbar" aria-valuenow="{bar_pct}" aria-valuemin="0" aria-valuemax="100"'
            f' aria-label="{html_mod.escape(theme)} confidence">'
            f'<div style="height:4px;border-radius:2px;background:{pill_color};'
            f'transform:scaleX({scale});transform-origin:left;transition:transform .3s;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_theme_mood(result: AnalysisResult) -> None:
    """Render the Theme & Mood card. Silently skips when theme_mood is None."""
    tm = result.theme_mood
    if tm is None:
        return

    mood_color = _MOOD_COLORS.get(tm.mood, "var(--accent)")

    # Section header
    st.markdown(
        '<div style="font-size:0.72rem;color:var(--dim);text-transform:uppercase;'
        'letter-spacing:.06em;margin-bottom:8px;">Theme &amp; Mood</div>',
        unsafe_allow_html=True,
    )

    # Mood label row
    enriched_badge = (
        ' <span title="Enriched by Groq LLM" style="'
        'font-size:0.65rem;background:var(--accent);color:#000;'
        'border-radius:3px;padding:1px 5px;margin-left:6px;">✦ enriched</span>'
        if tm.groq_enriched else ""
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
        f'<span style="font-size:0.95rem;font-weight:600;color:{mood_color};">'
        f'{html_mod.escape(tm.mood)}</span>'
        f'<span style="color:var(--dim);font-size:0.8rem;">{tm.confidence:.0%} confidence</span>'
        f'{enriched_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Category-coloured theme pills + per-theme confidence bars
    if tm.theme_scores:
        _render_theme_bars(tm.theme_scores)
    elif tm.themes:
        chips = "".join(
            f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:3px 10px;'
            f'border-radius:12px;font-size:0.78rem;background:var(--surface-2);'
            f'color:var(--text);border:1px solid var(--border-hr);">'
            f'{html_mod.escape(t)}</span>'
            for t in tm.themes
        )
        st.markdown(f'<div style="margin-bottom:10px;">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:var(--muted);font-size:0.82rem;margin-bottom:10px;">—</div>',
            unsafe_allow_html=True,
        )

    # Sync brief copy block
    themes_str = ", ".join(tm.themes) if tm.themes else "—"
    summary    = tm.mood_summary or f"{tm.mood} — {themes_str}"
    st.code(summary, language=None)

    # On-demand Groq enrichment toggle (hidden when key not configured)
    _render_groq_enrich_toggle(result)


def _render_groq_enrich_toggle(result: AnalysisResult) -> None:
    """Show Groq enrichment toggle only when groq_api_key is configured (#169)."""
    try:
        settings = get_settings()
        groq_key = getattr(settings, "groq_api_key", None)
    except Exception:  # noqa: BLE001
        groq_key = None
    if not groq_key:
        return

    tm = result.theme_mood
    if tm is None:
        return

    transcript_text = " ".join(
        seg.text for seg in (result.transcript or []) if seg.text.strip()
    )
    cache_key = f"groq_tm_{hash(transcript_text[:200])}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = False

    want_groq = st.toggle(
        "✦ Enrich with Groq",
        key=f"{cache_key}_toggle",
        value=st.session_state[cache_key],
        help="Send a lyric excerpt to Groq LLM for a richer mood summary.",
    )
    if want_groq and not tm.groq_enriched:
        try:
            enriched = ThemeMoodAnalyzer().enrich(tm, transcript_text)
            result.theme_mood = enriched
            st.session_state[cache_key] = True
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Groq enrichment failed: {exc}")


_AUTHORSHIP_SYNC_NOTES: dict[str, str] = {
    "Likely Human":      "Full clearance path — no additional verification needed.",
    "Uncertain":         "Recommend writer verification before submission.",
    "Likely AI":         "Check licensing terms — generated content may require separate disclosure.",
    "Insufficient data": "",
}

_AUTHORSHIP_SKIP_MESSAGES: dict[str, str] = {
    "instrumental": "Instrumental track — no lyric authorship check performed.",
    "too_short":    "Limited lyric data — results may not be representative.",
    "short":        "Note: short lyric content — treat signals as indicative only.",
}

# Real-world context tooltip per signal type (#162)
_AUTHORSHIP_SIGNAL_TOOLTIPS: dict[str, str] = {
    "burstiness": (
        "Burstiness measures variation in line length. Human writers naturally mix "
        "short punchy lines with longer ones; AI tends to produce uniform line lengths. "
        "A low CV (< 0.20) is a flag for machine-generated text."
    ),
    "vocabulary": (
        "Unique word ratio measures vocabulary breadth. AI models often recycle a "
        "limited vocabulary and produce lyrics with fewer distinct words than a human "
        "songwriter drawing on lived experience."
    ),
    "rhyme": (
        "Rhyme density measures how often consecutive line-pairs rhyme. Human lyrics "
        "balance rhyme with off-rhyme and slant-rhyme; AI systems over-rhyme, producing "
        "a sing-song feel that exceeds the threshold > 72% of consecutive pairs."
    ),
    "repetition": (
        "Repetition score measures the fraction of lines that appear more than once. "
        "High repetition is a hallmark of AI-generated lyrics — models reuse phrases "
        "to pad length rather than introducing new imagery."
    ),
    "classifier": (
        "The RoBERTa classifier was fine-tuned on human vs. AI-generated lyrics. "
        "It evaluates the full text as a sequence and outputs a probability score. "
        "Scores ≥ 70% contribute 2 AI signals; 50–69% contributes 1."
    ),
}


def _authorship_signal_tooltip(note: str) -> str:
    """Return tooltip text for an authorship signal note, or '' if unrecognised."""
    lo = note.lower()
    if "burstiness" in lo or "line length" in lo:
        return _AUTHORSHIP_SIGNAL_TOOLTIPS["burstiness"]
    if "vocabulary" in lo or "unique word" in lo:
        return _AUTHORSHIP_SIGNAL_TOOLTIPS["vocabulary"]
    if "rhyme" in lo:
        return _AUTHORSHIP_SIGNAL_TOOLTIPS["rhyme"]
    if "repetition" in lo:
        return _AUTHORSHIP_SIGNAL_TOOLTIPS["repetition"]
    if "classifier" in lo:
        return _AUTHORSHIP_SIGNAL_TOOLTIPS["classifier"]
    return ""


def _authorship_confidence_score(authorship: "AuthorshipResult") -> Optional[float]:
    """
    Return a 0-1 confidence score for the authorship verdict direction.

    For AI verdicts, high score = strong AI evidence.
    For human verdicts, score is inverted so the bar reads as verdict confidence
    (not raw AI probability) — e.g. roberta_score=0.10 → 0.90 confidence human.

    Uses roberta_score when available; falls back to signal_count ratio.
    Returns None for "Insufficient data" verdicts so the bar is hidden.
    """
    if authorship.verdict == "Insufficient data":
        return None
    is_human = authorship.verdict == "Likely Human"
    if authorship.roberta_score is not None:
        raw = authorship.roberta_score
        return (1.0 - raw) if is_human else raw
    raw = min(1.0, authorship.signal_count / CONSTANTS.AUTHORSHIP_MAX_SIGNALS)
    return (1.0 - raw) if is_human else raw


_SECTION_PILL_ICONS: dict[str, str] = {
    "Likely Human":      "✓",
    "Uncertain":         "▲",
    "Likely AI":         "⚠",
    "Insufficient data": "–",
}


def _build_section_pills_html(authorship: "AuthorshipResult") -> str:
    """Return an HTML pill row for per-section authorship verdicts (#156).

    Returns empty string when no per-section data is available.
    Pure function — no I/O.
    """
    per_section = authorship.per_section
    if not per_section:
        return ""

    pills: list[str] = []
    for label, sec in per_section.items():
        color     = authorship_color(sec.verdict)
        icon      = _SECTION_PILL_ICONS.get(sec.verdict, "–")
        aria_lbl  = html_mod.escape(f"{label}: {sec.verdict}")
        pills.append(
            f"<span aria-label='{aria_lbl}' style='display:inline-flex;align-items:center;gap:4px;"
            f"padding:2px 8px;border-radius:20px;border:1px solid {color}44;"
            f"background:{color}0F;font-family:\"Chakra Petch\",monospace;"
            f"font-size:.55rem;font-weight:600;letter-spacing:.08em;"
            f"text-transform:uppercase;color:{color};white-space:nowrap;'>"
            f"<span aria-hidden='true'>{html_mod.escape(label)}&nbsp;{icon}</span></span>"
        )

    return (
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;'>"
        + "".join(pills)
        + "</div>"
    )


def _render_authorship_banner(authorship: Optional["AuthorshipResult"]) -> None:
    if not authorship:
        return
    av        = authorship.verdict
    av_color  = authorship_color(av)
    a_notes   = authorship.feature_notes
    a_rob     = authorship.roberta_score
    rob_str   = f"Classifier: {a_rob:.0%} AI probability · " if a_rob is not None else ""
    n_sig     = authorship.signal_count
    sig_str   = f"{n_sig:.1f} lyric flag{'s' if n_sig != 1.0 else ''}"
    sync_note = _AUTHORSHIP_SYNC_NOTES.get(av, "")
    confidence = _authorship_confidence_score(authorship)

    def _note_html(note: str) -> str:
        """Render one signal bullet with an inline ? tooltip (#162)."""
        arrow      = "✓" if "✓" in note else "▲"
        note_color = "var(--muted)" if "✓" in note else "var(--text)"
        tip_text   = _authorship_signal_tooltip(note)
        tip_html   = (
            f"<span class='tip-wrap' style='vertical-align:middle;'>"
            f"<span class='tip-icon' role='button' tabindex='0' aria-label='More information'>?</span>"
            f"<span class='tip-box' style='width:220px;'>{html_mod.escape(tip_text)}</span>"
            f"</span>"
            if tip_text else ""
        )
        return (
            f"<div style='display:flex;align-items:center;gap:6px;padding:3px 0;'>"
            f"<span style='color:{av_color};font-size:.7rem;flex-shrink:0;'>{arrow}</span>"
            f"<span style='font-family:\"Figtree\",sans-serif;font-size:.76rem;"
            f"color:{note_color};'>{html_mod.escape(note)}</span>"
            f"{tip_html}</div>"
        )

    # Confidence progress bar (#159)
    conf_bar_html = ""
    if confidence is not None:
        conf_pct = f"{confidence:.0%}"
        conf_val = f"{confidence:.3f}"
        conf_bar_html = (
            f"<div style='margin-top:10px;'>"
            f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.52rem;font-weight:600;"
            f"letter-spacing:.12em;text-transform:uppercase;color:var(--dim);margin-bottom:4px;'>"
            f"Ensemble Confidence</div>"
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"<div style='flex:1;height:4px;border-radius:2px;background:var(--border-hr);overflow:hidden;'>"
            f"<div style='height:4px;border-radius:2px;background:{av_color};width:100%;"
            f"transform:scaleX({conf_val});transform-origin:left;transition:transform .3s;'></div>"
            f"</div>"
            f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:.66rem;"
            f"color:{av_color};min-width:32px;'>{conf_pct}</span>"
            f"</div></div>"
        )

    sync_suffix = (
        f"<div style='font-family:\"Figtree\",sans-serif;font-size:.72rem;"
        f"color:var(--dim);margin-top:8px;padding-top:8px;"
        f"border-top:1px solid {av_color}22;'>{html_mod.escape(sync_note)}</div>"
        if sync_note else ""
    )

    # Per-section pill row (#156) — ✓ Likely Human, ▲ Uncertain, ⚠ Likely AI
    section_pills_html = _build_section_pills_html(authorship)

    # Verdict header — always visible (#163)
    st.markdown(f"""
    <div style="border:1px solid {av_color}22;border-radius:10px;background:{av_color}08;
                padding:14px 18px;margin-bottom:6px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;
                    color:{av_color};flex-shrink:0;">{av}</div>
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                      color:{av_color};letter-spacing:.1em;text-transform:uppercase;">
            Lyric Writing — {sig_str}</div>
          <div style="font-family:'Figtree',sans-serif;font-size:.74rem;color:var(--muted);margin-top:2px;">
            {rob_str}Detects AI-written lyrics — independent of audio AI detection.</div>
        </div>
      </div>
      {conf_bar_html}
      {section_pills_html}
      {sync_suffix}
    </div>
    """, unsafe_allow_html=True)

    # Collapsible signal details — collapsed by default to keep transcript prominent (#163)
    if a_notes:
        with st.expander("Signal details", expanded=False):
            notes_html = "".join(_note_html(n) for n in a_notes)
            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:0 18px;'>"
                f"{notes_html}</div>",
                unsafe_allow_html=True,
            )

    # Tiered advisory message based on why analysis was limited (#165)
    skip_msg = _AUTHORSHIP_SKIP_MESSAGES.get(authorship.skip_reason or "", "")
    if skip_msg:
        st.caption(skip_msg)


def _render_lyric_column(
    segments: list[TranscriptSegment],
    sections: list[Section],
    flagged_ts: set[int],
    flags_by_ts: dict[int, list[ComplianceFlag]],
) -> None:
    st.markdown("""
    <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                letter-spacing:.16em;text-transform:uppercase;color:var(--dim);
                margin-bottom:10px;">Formatted Lyrics — Click timestamp to jump</div>
    """, unsafe_allow_html=True)

    if not segments:
        st.markdown(
            "<div style='color:var(--dim);font-size:.84rem;padding:20px 0;'>"
            "No lyrics detected — track may be instrumental.</div>",
            unsafe_allow_html=True,
        )
        return

    for sec_label, segs in assign_sections(segments, sections):
        st.markdown(
            f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.6rem;"
            f"font-weight:700;color:var(--accent);letter-spacing:.14em;text-transform:uppercase;"
            f"margin:16px 0 6px;padding-top:14px;border-top:1px solid var(--border-hr);'>"
            f"[ {sec_label} ]</div>",
            unsafe_allow_html=True,
        )
        for seg in segs:
            ts_s       = int(seg.start)
            text       = seg.text.strip()
            is_flagged = ts_s in flagged_ts
            if not text:
                continue
            safe_text = html_mod.escape(text)
            c_btn, c_line = st.columns([1, 6])
            with c_btn:
                btn_type = "primary" if is_flagged else "secondary"
                if st.button(fmt_ts(ts_s), key=f"lyr_{id(seg)}_{ts_s}",
                             help=f"Jump to {fmt_ts(ts_s)} in the audio player",
                             use_container_width=True, type=btn_type):
                    st.session_state.start_time = ts_s
                    st.session_state.player_key = st.session_state.get("player_key", 0) + 1
                    st.rerun()
            with c_line:
                if is_flagged:
                    ts_flags   = flags_by_ts.get(ts_s, [])
                    confirmed  = any(f.confidence == "confirmed" for f in ts_flags)
                    line_color = "var(--text)" if confirmed else "var(--muted)"
                    pills_html = " ".join(issue_pill(f) for f in ts_flags)
                    st.markdown(
                        f"<div style='padding:7px 0;font-family:\"Figtree\",sans-serif;"
                        f"font-size:.88rem;color:{line_color};line-height:1.4;'>"
                        f"{safe_text}&nbsp;&nbsp;{pills_html}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='padding:7px 0;font-family:\"Figtree\",sans-serif;"
                        f"font-size:.88rem;color:var(--muted);line-height:1.4;'>{safe_text}</div>",
                        unsafe_allow_html=True,
                    )


def _render_audit_column(flags: list[ComplianceFlag], grade: str) -> None:
    hard_flags = [f for f in flags if f.confidence == "confirmed" and f.severity == "hard"]
    soft_flags = [f for f in flags if f.confidence == "confirmed" and f.severity == "soft"]
    pot_flags  = [f for f in flags if f.confidence == "potential"]
    n_hard, n_soft, n_pot = len(hard_flags), len(soft_flags), len(pot_flags)

    if n_hard:
        issue_count_label = (
            f"{n_hard} deal-breaker{'s' if n_hard != 1 else ''}"
            + (f" · {n_soft + n_pot} advisory" if n_soft + n_pot else "")
        )
    elif n_soft or n_pot:
        issue_count_label = f"{n_soft + n_pot} advisory flag{'s' if n_soft + n_pot != 1 else ''} — director's discretion"
    else:
        issue_count_label = "All clear"

    grade_color      = _grade_color(grade)
    grade_reason_str = _grade_reason(hard_flags, soft_flags, grade)
    all_clear_html   = "<span style='font-family:\"Figtree\",sans-serif;font-size:.8rem;color:var(--sync-pass);'>✓ All clear</span>"
    grade_pills_html = (
        _deduped_pills(hard_flags) + _deduped_pills(soft_flags) + _deduped_pills(pot_flags)
    ) if flags else all_clear_html

    st.markdown(f"""
    <div class="sig" style="margin-bottom:16px;">
      <div class="sig-head">Sync Compliance Grade
        <span class="tip-wrap" style="margin-left:6px;"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>
          <span class="tip-box">Sync readiness scoring for sync licensing.<br><br>
          <strong>A</strong> — Fully clean. Clear for submission in any context.<br>
          <strong>B</strong> — Advisory flags only. Director's placement call — no hard blockers.<br>
          <strong>C</strong> — 1 hard content issue. Clean edit required for most placements.<br>
          <strong>D</strong> — 2–3 hard issues. Clean edit required.<br>
          <strong>F</strong> — 4+ hard issues or drug references. Disqualifies broadcast.<br><br>
          Grade is based on <strong>hard</strong> confirmed flags only.<br>
          Brand mentions and mild language appear as advisory — supervisor's call.</span>
        </span>
      </div>
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:4.8rem;font-weight:700;
                    color:{grade_color};line-height:1;flex-shrink:0;">{grade}</div>
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                      color:{grade_color};letter-spacing:.1em;text-transform:uppercase;
                      margin-bottom:5px;">{issue_count_label}</div>
          <div style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                      line-height:1.55;">{grade_reason_str}</div>
        </div>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;">{grade_pills_html}</div>
    </div>
    """, unsafe_allow_html=True)

    if flags:
        if hard_flags:
            st.markdown(
                "<div style='font-family:\"Chakra Petch\",monospace;font-size:.56rem;font-weight:600;"
                "letter-spacing:.16em;text-transform:uppercase;color:var(--danger);"
                "margin-bottom:10px;'>Deal-Breakers — Clean Edit Required</div>",
                unsafe_allow_html=True,
            )
            _render_flag_rows(hard_flags, "hard")
        if soft_flags:
            st.markdown(
                "<div style='font-family:\"Chakra Petch\",monospace;font-size:.56rem;font-weight:600;"
                "letter-spacing:.16em;text-transform:uppercase;color:var(--dim);"
                f"margin-top:{'14px' if hard_flags else '0'};margin-bottom:10px;'>"
                "Placement Discretion — Director's Call</div>",
                unsafe_allow_html=True,
            )
            _render_flag_rows(soft_flags, "soft")
        if pot_flags:
            st.markdown(
                "<div style='font-family:\"Chakra Petch\",monospace;font-size:.56rem;font-weight:600;"
                "letter-spacing:.16em;text-transform:uppercase;color:var(--needs-review);"
                f"margin-top:{'14px' if hard_flags or soft_flags else '0'};margin-bottom:10px;opacity:.7;'>"
                "Supervisor Review — Verify Before Submission</div>",
                unsafe_allow_html=True,
            )
            _render_flag_rows(pot_flags, "pot")
    else:
        st.markdown("""
        <div style="padding:20px;text-align:center;border:1px solid rgba(13,245,160,.15);
                    border-radius:10px;background:rgba(13,245,160,.04);margin-top:8px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--sync-pass);">
            ✓ No compliance issues detected
          </div>
        </div>
        """, unsafe_allow_html=True)


def _deduped_pills(flag_list: list[ComplianceFlag], size: str = "lg") -> str:
    """Build grade-summary pills, one per issue type with a ×N count badge."""
    counts: dict[str, int] = {}
    first:  dict[str, ComplianceFlag] = {}
    for fl in flag_list:
        t = fl.issue_type
        counts[t] = counts.get(t, 0) + 1
        if t not in first:
            first[t] = fl
    parts = []
    for t, fl in first.items():
        pill = issue_pill(fl, size)
        if counts[t] > 1:
            badge_color = "var(--needs-review)" if fl.confidence == "potential" else "var(--muted)"
            pill += (
                f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:.52rem;"
                f"font-weight:600;padding:1px 5px;border-radius:3px;"
                f"background:var(--badge-bg);color:{badge_color};"
                f"margin-left:3px;'>×{counts[t]}</span>"
            )
        parts.append(pill)
    return "".join(parts)


def _render_flag_rows(flag_list: list[ComplianceFlag], prefix: str) -> None:
    """Render a clickable timestamp + detail row for each flag."""
    grouped_rows: OrderedDict = OrderedDict()
    for fl in flag_list:
        key = (fl.text.strip().lower(), fl.timestamp_s)
        if key not in grouped_rows:
            grouped_rows[key] = {"flag": fl, "extra": []}
        else:
            grouped_rows[key]["extra"].append(fl)

    for i, ((_, ts_s), row) in enumerate(grouped_rows.items()):
        flag          = row["flag"]
        all_flags_row = [flag] + row["extra"]
        is_potential  = flag.confidence == "potential"
        text_color    = "var(--muted)" if is_potential else "var(--text)"
        text_preview  = html_mod.escape(flag.text[:52] + ("…" if len(flag.text) > 52 else ""))
        review_badge  = (
            "<span style='font-family:\"Chakra Petch\",monospace;font-size:.5rem;"
            "font-weight:600;padding:1px 5px;border-radius:3px;"
            "background:rgba(200,232,106,.09);color:var(--needs-review);"
            "border:1px solid rgba(200,232,106,.2);margin-left:6px;'>NEEDS REVIEW</span>"
            if is_potential else ""
        )
        type_pills = " ".join(issue_pill(fl) for fl in all_flags_row)
        c_ts, c_detail = st.columns([1, 3])
        with c_ts:
            if st.button(fmt_ts(ts_s),
                         key=f"{prefix}_{i}_{ts_s}_{flag.issue_type}",
                         help=f"Jump to {fmt_ts(ts_s)} — {flag.issue_type} flag",
                         use_container_width=True):
                st.session_state.start_time = ts_s
                st.session_state.player_key = st.session_state.get("player_key", 0) + 1
                st.rerun()
        with c_detail:
            st.markdown(
                f"<div style='padding:4px 0 2px;display:flex;align-items:center;gap:5px;'>"
                f"{type_pills}{review_badge}</div>"
                f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:.74rem;"
                f"color:{text_color};margin-bottom:2px;'>{text_preview}</div>"
                f"<div style='font-family:\"Figtree\",sans-serif;font-size:.75rem;"
                f"color:var(--dim);line-height:1.4;margin-bottom:6px;'>"
                f"{flag.recommendation}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<hr style='border:none;border-top:1px solid var(--border-hr);margin:2px 0 6px;'>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Export — pure serialisation helpers
# ---------------------------------------------------------------------------

def _to_latin1(text: str) -> str:
    """
    Sanitise a string for the Helvetica built-in PDF font (latin-1 only).
    Replaces characters outside latin-1 with '?' to avoid FPDFUnicodeEncodingException.
    Pure function — no I/O.
    """
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _compliance_flags_to_csv(result: AnalysisResult) -> bytes:
    """
    Serialise all compliance flags + structural checks to CSV bytes.

    Pure function — no I/O, no Streamlit calls.
    Columns: section, check, status, confidence, severity, timestamp_s, text, recommendation
    """
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["section", "check", "status", "confidence", "severity",
                    "timestamp_s", "text", "recommendation"],
        extrasaction="ignore",
    )
    writer.writeheader()

    # Compliance flags from lyric audit
    if result.compliance:
        for flag in result.compliance.flags:
            writer.writerow({
                "section":        "Lyric Audit",
                "check":          flag.issue_type,
                "status":         "FLAG",
                "confidence":     flag.confidence,
                "severity":       flag.severity,
                "timestamp_s":    flag.timestamp_s,
                "text":           flag.text,
                "recommendation": flag.recommendation,
            })
        # Structural checks
        for check, status, detail in [
            ("Sting / Ending",    "FLAG" if result.compliance.sting.flag else "PASS",
             result.compliance.sting.ending_type),
            ("Energy Evolution",  "FLAG" if result.compliance.evolution.flag else "PASS",
             result.compliance.evolution.detail),
            ("Intro Length",      "FLAG" if result.compliance.intro.flag else "PASS",
             f"{result.compliance.intro.intro_seconds:.1f}s"),
        ]:
            writer.writerow({
                "section":    "Structural",
                "check":      check,
                "status":     status,
                "confidence": "confirmed",
                "severity":   "soft",
                "text":       detail,
            })

    return buf.getvalue().encode("utf-8-sig")  # BOM for Excel compatibility


def _analysis_to_pdf(result: AnalysisResult) -> bytes:
    """
    Generate a minimal compliance certificate PDF using fpdf2.

    Pure function — no I/O, no Streamlit calls.
    Returns raw PDF bytes.
    """
    from fpdf import FPDF  # deferred — not all deployments need it

    title   = _to_latin1(result.audio.metadata.get("title", "") or result.audio.label or "Unknown Track")
    artist  = _to_latin1(result.audio.metadata.get("artist", "") or "Unknown Artist")
    grade   = result.compliance.grade if result.compliance else "N/A"
    scan_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "SYNC-SAFE COMPLIANCE CERTIFICATE", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated {scan_ts}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Track metadata
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, f"{title}  /  {artist}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(4)

    # Compliance grade
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Compliance Grade: {grade}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Forensics verdict
    if result.forensics:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI / Authenticity Verdict", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        # verdict is an ASCII Literal — _to_latin1 is a no-op but applied for consistency
        pdf.cell(0, 6, _to_latin1(result.forensics.verdict), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Compliance flags table
    _pdf_flags_table(pdf, result)

    # ISRC / PRO
    if result.legal and (result.legal.isrc or result.legal.pro_match):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Rights Information", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        if result.legal.isrc:
            pdf.cell(0, 6, _to_latin1(f"ISRC: {result.legal.isrc}"), new_x="LMARGIN", new_y="NEXT")
        if result.legal.pro_match:
            pdf.cell(0, 6, _to_latin1(f"Inferred PRO: {result.legal.pro_match}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Footer disclaimer
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 4, "This certificate is generated by Sync-Safe and is for informational "
                         "purposes only. It does not constitute legal advice. Verify all rights "
                         "information with the relevant PRO before licensing.")

    return pdf.output()


def _pdf_flags_table(pdf: "FPDF", result: AnalysisResult) -> None:
    """Render the compliance flags table into an existing FPDF document."""
    if not (result.compliance and result.compliance.flags):
        return
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Compliance Flags", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(200, 200, 200)
    col_w = [18, 28, 80, 60]
    for header, w in zip(["Time", "Type", "Excerpt", "Recommendation"], col_w):
        pdf.cell(w, 7, header, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    pdf.set_fill_color(255, 255, 255)
    for flag in result.compliance.flags:
        cells = [
            f"{flag.timestamp_s}s",
            flag.issue_type,
            _to_latin1(flag.text[:60]),
            _to_latin1(flag.recommendation[:40]),
        ]
        for text, w in zip(cells, col_w):
            pdf.cell(w, 6, text, border=1)
        pdf.ln()
    pdf.ln(4)


def _sync_cut_ts(t: float) -> str:
    """Format a float timestamp as M:SS for sync-cut display."""
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def _sync_cut_conf_bar(conf: float) -> str:
    """Return a block-character progress bar HTML span for a confidence score."""
    filled = round(conf * 10)
    bar    = "█" * filled + "░" * (10 - filled)
    pct    = int(conf * 100)
    color  = "#10B981" if conf >= 0.6 else ("#F5640A" if conf >= 0.4 else "var(--dim)")
    return (
        f'<span style="font-family:JetBrains Mono,monospace;font-size:.75rem;'
        f'color:{color};" aria-label="Confidence {pct} percent">'
        f'{bar} {pct}%</span>'
    )


@st.fragment
def _render_sync_cuts(result: AnalysisResult) -> None:
    """Render the Sync Edit Points table — top-3 candidates per target duration.

    Wrapped in @st.fragment so the custom-duration slider reruns only this
    section, not the full report page (#150).  Start timestamps seek the player.
    """
    _mono_hdr = (
        "font-family:JetBrains Mono,monospace;font-size:.6rem;font-weight:600;"
        "letter-spacing:.12em;text-transform:uppercase;color:var(--dim);"
        "padding-bottom:4px;border-bottom:1px solid var(--border);"
    )
    _conf_header = (
        f"<div style='{_mono_hdr}'>Confidence"
        f'<span class="tip-wrap"><span class="tip-icon" role="button" tabindex="0" aria-label="More information">?</span>'
        f'<span class="tip-box">How well this edit window fits sync placement conventions. '
        f'Scored on 5 equal criteria (+20% each): starts after the intro, '
        f'start lands on a section boundary, end lands on a section boundary, '
        f'contains a chorus or hook, end snaps to a bar grid. '
        f'100% = perfect on all five. 60%+ is recommended for placement.</span>'
        f'</span></div>'
    )
    _mono_cell = (
        "font-family:JetBrains Mono,monospace;font-size:.78rem;"
        "color:var(--muted);padding-top:6px;"
    )
    _mono_cell_dim = (
        "font-family:JetBrains Mono,monospace;font-size:.72rem;"
        "color:var(--dim);padding-top:4px;"
    )

    def _render_cut_row(cut: SyncCut, key_suffix: str, is_alt: bool = False) -> None:
        """Render one SyncCut as a table row. is_alt = True for rank 2/3 sub-rows."""
        cell_style = _mono_cell_dim if is_alt else _mono_cell
        ts_s = int(cut.start_s)
        c_target, c_start, c_end, c_actual, c_conf, c_note, c_copy = st.columns(
            [0.7, 0.8, 0.8, 0.7, 1.4, 2, 0.6], gap="small"
        )
        rank_label = f"#{cut.rank}" if is_alt else f"{cut.duration_s}s"
        rank_color = "var(--dim)" if is_alt else "var(--text)"
        c_target.markdown(
            f"<div style='{cell_style}font-weight:600;color:{rank_color};'>{rank_label}</div>",
            unsafe_allow_html=True,
        )
        with c_start:
            if st.button(
                _sync_cut_ts(cut.start_s),
                key=f"cut_{ts_s}_{cut.duration_s}_{key_suffix}",
                help=f"Jump to {_sync_cut_ts(cut.start_s)} in the audio player",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.start_time = ts_s
                st.session_state.player_key = st.session_state.get("player_key", 0) + 1
                st.rerun()
        c_end.markdown(
            f"<div style='{cell_style}'>{_sync_cut_ts(cut.end_s)}</div>",
            unsafe_allow_html=True,
        )
        c_actual.markdown(
            f"<div style='{cell_style}'>{cut.actual_duration_s:.1f}s</div>",
            unsafe_allow_html=True,
        )
        c_conf.markdown(
            f"<div style='padding-top:{'4' if is_alt else '6'}px;'>"
            f"{_sync_cut_conf_bar(cut.confidence)}</div>",
            unsafe_allow_html=True,
        )
        c_note.markdown(
            f"<div style='font-family:Figtree,sans-serif;font-size:"
            f"{'0.74' if is_alt else '0.78'}rem;"
            f"color:var({'--dim' if is_alt else '--muted'});padding-top:6px;'>"
            f"{html_mod.escape(cut.note)}</div>",
            unsafe_allow_html=True,
        )
        with c_copy:
            with st.popover("⧉", help="Copy timestamps or DAW marker text"):
                ts_str     = f"{fmt_ts(int(cut.start_s))}–{fmt_ts(int(cut.end_s))}"
                marker_str = (
                    f"Start: {fmt_ts(int(cut.start_s))} | "
                    f"End: {fmt_ts(int(cut.end_s))} | "
                    f"{cut.duration_s}s | {cut.note}"
                )
                st.caption("Timestamps")
                st.code(ts_str, language=None)
                st.caption("DAW marker")
                st.code(marker_str, language=None)

    # -- Build the display list: pre-computed cuts + optional custom-duration cut --
    display_cuts = list(result.sync_cuts)

    custom_s = st.slider(
        "Custom target duration (seconds)",
        min_value=CONSTANTS.SYNC_CUT_SLIDER_MIN,
        max_value=CONSTANTS.SYNC_CUT_SLIDER_MAX,
        value=30,
        step=CONSTANTS.SYNC_CUT_SLIDER_STEP,
        key="sync_cut_custom_duration",
        help="Compute edit points for a custom duration. Results are appended to the table.",
    )
    # Only recompute if the custom duration differs from all pre-computed targets
    preset_targets = {c.duration_s for c in result.sync_cuts}
    if custom_s not in preset_targets and result.structure:
        custom_cuts = SyncCutAnalyzer().suggest(
            sections         = result.structure.sections,
            beats            = result.structure.beats,
            target_durations = [custom_s],
            loop_score       = result.forensics.loop_score if result.forensics else 0.0,
        )
        display_cuts = display_cuts + custom_cuts

    # -- Header row --
    h_target, h_start, h_end, h_actual, h_conf, h_note, h_copy = st.columns(
        [0.7, 0.8, 0.8, 0.7, 1.4, 2, 0.6], gap="small"
    )
    for col, label in (
        (h_target, "Target"),
        (h_start,  "Start ▶"),
        (h_end,    "End"),
        (h_actual, "Actual"),
        (h_note,   "Note"),
        (h_copy,   "Copy"),
    ):
        col.markdown(f"<div style='{_mono_hdr}'>{label}</div>", unsafe_allow_html=True)
    h_conf.markdown(_conf_header, unsafe_allow_html=True)

    # -- Data rows: group by duration, show rank-1 full, rank 2-3 as sub-rows --
    for duration_s, group_iter in groupby(display_cuts, key=lambda c: c.duration_s):
        group = list(group_iter)
        rank1 = next((c for c in group if c.rank == 1), group[0])
        alts  = [c for c in group if c.rank != 1]
        _render_cut_row(rank1, key_suffix="r1")
        for alt in alts:
            _render_cut_row(alt, key_suffix=f"r{alt.rank}", is_alt=True)

    st.caption(
        "Edit windows are beat-aligned and scored on: post-intro start, section-boundary "
        "entry/exit, chorus presence, and bar-grid snap. Confidence = composite score (0–100%). "
        "Rank #2/#3 are alternative windows for each duration. "
        "Click a Start timestamp to seek the audio player."
    )


def _render_export_buttons(result: AnalysisResult) -> None:
    """Render CSV, platform catalog, and PDF export download buttons."""
    st.markdown("""
    <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                display:flex;align-items:center;gap:10px;margin:28px 0 14px;">
      <span>◈ Export Report</span>
      <div style="flex:1;height:1px;background:var(--border-hr);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Sanitise for safe filenames across all OS: keep alphanumeric, dash, dot only
    raw_slug   = result.audio.metadata.get("title", "") or result.audio.label or "sync-safe-report"
    track_slug = re.sub(r"[^\w\-.]", "-", raw_slug).strip("-")[:40].lower() or "sync-safe-report"

    c_csv, c_platform, c_pdf = st.columns(3, gap="medium")

    with c_csv:
        csv_bytes = _compliance_flags_to_csv(result)
        st.download_button(
            label="⬇ Compliance CSV",
            data=csv_bytes,
            file_name=f"{track_slug}-compliance.csv",
            mime="text/csv",
            use_container_width=True,
            help="Compliance flags and structural checks as a spreadsheet",
        )

    with c_platform:
        _render_platform_export(result, track_slug)

    with c_pdf:
        try:
            pdf_bytes = _analysis_to_pdf(result)
            st.download_button(
                label="⬇ Certificate PDF",
                data=bytes(pdf_bytes),
                file_name=f"{track_slug}-certificate.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Compliance certificate with grade, flags, and rights info",
            )
        except ImportError:
            st.caption("PDF export requires fpdf2 — install with `pip install fpdf2`.")
        except Exception as exc:  # noqa: BLE001 — UI boundary; PDF errors must not crash the report
            st.error(f"PDF generation failed: {exc}")

    # JSON full-data export (#140, #123) — sections, rights, all pipeline outputs
    st.download_button(
        label="⬇ Download Full Report JSON",
        data=to_analysis_json(result),
        file_name=f"{track_slug}-report.json",
        mime="application/json",
        use_container_width=False,
        help=(
            "Complete pipeline output as JSON — includes sections, ISRC, PRO, "
            "forensics, compliance flags, popularity, and sync cuts."
        ),
    )

    # DAW marker export — Premiere Pro XML + DaVinci Resolve DRT (#152)
    _render_daw_export(result, track_slug)

    # Tagged file download — only available for direct uploads (not YouTube, which
    # produces a lossy MP3 transcode that is not re-exportable as a deliverable).
    if result.audio.source != _SOURCE_YOUTUBE:
        _render_tagged_download(result, track_slug)


_DAW_FRAMERATES: list[float] = [24.0, 25.0, 29.97, 30.0]


def _render_daw_export(result: AnalysisResult, track_slug: str) -> None:
    """Render Premiere Pro XML and DaVinci Resolve DRT download buttons (#152)."""
    cuts = result.sync_cuts
    if not cuts:
        return

    if "daw_framerate" not in st.session_state:
        st.session_state["daw_framerate"] = CONSTANTS.EXPORT_FRAMERATE

    st.markdown(
        '<div style="font-size:0.72rem;color:var(--dim);text-transform:uppercase;'
        'letter-spacing:.06em;margin:18px 0 10px;">DAW Marker Export</div>',
        unsafe_allow_html=True,
    )

    col_fps, col_premiere, col_davinci = st.columns([1, 2, 2], gap="medium")

    with col_fps:
        fps = st.selectbox(
            "Frame rate",
            options=_DAW_FRAMERATES,
            index=_DAW_FRAMERATES.index(st.session_state["daw_framerate"]),
            key="daw_framerate_select",
            help="Match your editing project's frame rate setting.",
        )
        st.session_state["daw_framerate"] = fps

    with col_premiere:
        xml_str = _build_premiere_xml(cuts, fps)
        st.download_button(
            label="⬇ Premiere Pro XML",
            data=xml_str.encode("utf-8"),
            file_name=f"{track_slug}-premiere-markers.xml",
            mime="application/xml",
            use_container_width=True,
            help="Import via Markers panel → Import Markers from File",
        )

    with col_davinci:
        drt_str = _build_davinci_drt(cuts, fps)
        st.download_button(
            label="⬇ DaVinci Resolve DRT",
            data=drt_str.encode("utf-8"),
            file_name=f"{track_slug}-davinci-markers.drt",
            mime="text/plain",
            use_container_width=True,
            help="Import via Timeline → Import → Timeline Markers from EDL",
        )


def _render_platform_export(result: AnalysisResult, track_slug: str) -> None:
    """Render the 'Export for...' platform catalog dropdown + download button."""
    platform_key = st.selectbox(
        "Export for",
        options=list(PLATFORM_SCHEMAS.keys()),
        format_func=lambda k: _PLATFORM_LABELS.get(k, k),
        key="export_platform_select",
        label_visibility="collapsed",
    )
    try:
        platform_bytes = to_platform_csv(result, platform_key)
        label = _PLATFORM_LABELS.get(platform_key, platform_key)
        st.download_button(
            label=f"⬇ Export for {label}",
            data=platform_bytes,
            file_name=f"{track_slug}-{platform_key}.csv",
            mime="text/csv",
            use_container_width=True,
            help=f"Single-row catalog CSV formatted for {label} import",
        )
    except Exception as exc:  # noqa: BLE001 — UI boundary
        st.error(f"Platform export failed: {exc}")


def _render_tagged_download(result: AnalysisResult, track_slug: str) -> None:
    """
    Render a 'Download Tagged File' button that embeds audit results in ID3/Vorbis/MP4 tags.

    Only shown for direct file uploads — YouTube sources produce lossy re-encodes.
    Tag injection is performed lazily on button click (st.download_button pre-computes data).
    """
    label_ext = result.audio.label.rsplit(".", 1)[-1].lower() if "." in result.audio.label else "mp3"
    suffix    = _TAGGED_EXT_MAP.get(label_ext, ".mp3")
    mime      = _TAGGED_MIME_MAP.get(suffix, "audio/mpeg")

    try:
        # Tag injection is synchronous — writes/reads a TemporaryDirectory file.
        # For typical uploads (< 10 MB) this completes in < 200 ms.
        # If this becomes a bottleneck at higher file sizes, cache via
        # @st.cache_data(hash_funcs={bytes: id}) keyed on audio.raw identity.
        tagged_bytes = TagInjector().inject(result.audio.raw, result)
        st.download_button(
            label="⬇ Download Tagged File",
            data=tagged_bytes,
            file_name=f"{track_slug}-tagged{suffix}",
            mime=mime,
            use_container_width=False,
            help=(
                "Your audio file with Sync-Safe audit results embedded in the tags. "
                "Open in any ID3-aware player or DAW to see the compliance metadata."
            ),
        )
    except Exception as exc:  # noqa: BLE001 — UI boundary
        st.error(f"Tag injection failed: {exc}")


def _render_raw_data_link() -> None:
    """Render a navigation button to the Raw Data page."""
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("View Raw Data →", help="See every pipeline data point and download as CSV"):
        st.session_state.page = "raw_data"
        st.rerun()


def _render_footer() -> None:
    st.markdown("""
    <div style="text-align:center;margin-top:64px;margin-bottom:32px;
                border-top:1px solid var(--border-hr);padding-top:28px;">
      <span style="font-family:'Chakra Petch',monospace;font-size:.54rem;font-weight:500;
                   color:var(--dim);letter-spacing:.22em;text-transform:uppercase;">
        End of Forensic Report &nbsp;·&nbsp; Strictly Confidential &nbsp;·&nbsp; Sync-Safe v2
      </span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------

def _assign_sections(
    segments: list[TranscriptSegment],
    sections: list[Section],
) -> list[tuple[str, list[TranscriptSegment]]]:
    """Group Whisper segments into allin1 structural sections."""
    if not sections:
        return [("TRACK", segments)]
    ordered = sorted(sections, key=lambda s: s.start)
    result  = []
    captured: set[int] = set()
    for sec in ordered:
        # Use midpoint so a segment that starts just before a section boundary
        # (common with Whisper) is still assigned to the section it mostly lives in.
        grp = [
            seg for seg in segments
            if sec.start <= (seg.start + seg.end) / 2 < sec.end
        ]
        if grp:
            result.append((sec.label.upper(), grp))
            captured.update(id(seg) for seg in grp)
    leftovers = [seg for seg in segments if id(seg) not in captured]
    if leftovers:
        result.append(("OTHER", leftovers))
    return result


def _groove_label(ibi: float) -> str:
    if ibi < 0:
        return "Insufficient data"
    if ibi < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        return "Perfect Quantization (AI signal)"
    if ibi > CONSTANTS.IBI_ERRATIC_MIN:
        return "Human-Feel Timing (Organic)"
    return "Human Micro-timing"


def _synthid_confidence(bins: int) -> str:
    if bins == 0:
        return "none"
    if bins <= CONSTANTS.SYNTHID_LOW_BINS:
        return "low"
    if bins <= CONSTANTS.SYNTHID_MEDIUM_BINS:
        return "medium"
    return "high"


def _grade_color(grade: str) -> str:
    return {
        "A": "var(--ok)",
        "B": "var(--grade-b)",
        "C": "var(--grade-c)",
        "D": "var(--grade-d)",
        "F": "var(--danger)",
    }.get(grade, "var(--dim)")


def _grade_reason(
    hard_flags: list[ComplianceFlag],
    soft_flags: list[ComplianceFlag],
    grade: str,
) -> str:
    if not hard_flags and not soft_flags:
        return "No compliance issues detected. Track is clear for sync submission."
    if not hard_flags:
        n = len(soft_flags)
        return (
            f"{n} advisory flag{'s' if n != 1 else ''} — mild language or brand mentions. "
            "No hard blockers. Placement is the director's call."
        )
    hard_counts = Counter(f.issue_type for f in hard_flags)
    n_hard = len(hard_flags)
    if hard_counts.get("DRUGS", 0) > 0:
        return f"{hard_counts['DRUGS']} drug reference(s). Disqualifies broadcast and most brand placements."
    if n_hard >= 4:
        return f"{n_hard} hard content issues. Clean edit required before any sync submission."
    if hard_counts.get("EXPLICIT", 0) >= 2:
        return f"{hard_counts['EXPLICIT']} hard explicit flags. Clean edit required for most placements."
    if hard_counts.get("VIOLENCE", 0) >= 1:
        return f"{n_hard} hard content issue(s) including threatening language. Clean edit required for family and broadcast."
    return f"{n_hard} hard content issue(s). Clean edit required for network TV and family placements."
