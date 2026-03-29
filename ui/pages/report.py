"""
ui/pages/report.py
Report page rendering — takes a typed AnalysisResult and renders all sections.

All functions accept typed domain models from core.models; no raw dict access.
"""
from __future__ import annotations

import html as html_mod
from collections import Counter, OrderedDict
from typing import Optional

import streamlit as st

from core.config import CONSTANTS
from core.models import (
    AnalysisResult,
    AudioBuffer,
    AuthorshipResult,
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    ForensicsResult,
    IntroResult,
    Section,
    StingResult,
    StructureResult,
    TranscriptSegment,
)
from ui.components import ISSUE_META, authorship_color, eq_bars, fmt_ts, issue_pill


# AudioSource literal value for YouTube — matches core/models.py AudioSource type.
# Defined here to avoid repeating the string across multiple render functions.
_SOURCE_YOUTUBE: str = "youtube"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_report(
    audio: AudioBuffer,
    result: AnalysisResult,
) -> None:
    """Render the full analysis report for a completed pipeline run."""
    st.markdown(
        '<a href="#main-content" class="skip-link">Skip to main content</a>',
        unsafe_allow_html=True,
    )
    _render_nav(audio.label)
    st.markdown('<span id="main-content" tabindex="-1"></span>', unsafe_allow_html=True)
    _render_audio_player(audio)
    _render_sync_snapshot(result)

    with st.expander("Track Overview", expanded=True):
        c_left, c_right = st.columns([1, 1], gap="large")
        with c_left:
            _render_metadata_card(result.structure, result.audio.metadata)
        with c_right:
            _render_structure_card(result.structure)

    with st.expander("Authenticity Audit", expanded=True):
        _render_forensics_card(result.forensics, source=result.audio.source)

    with st.expander("Structural Repetition", expanded=True):
        _render_production_analysis_card(result.forensics)

    with st.expander("Sync Readiness Checks", expanded=True):
        _render_sync_readiness(result.compliance)
        _render_audio_quality_card(result.audio_quality)

    with st.expander("Discovery & Licensing", expanded=True):
        _render_legal_and_discovery(result)

    with st.expander("Lyrics & Content Audit", expanded=True):
        _render_lyric_section(result)

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

    def _fmt_ts(secs: float) -> str:
        m, s = divmod(int(secs), 60)
        return f"{m}:{s:02d}"

    if sections:
        rows_html = "".join(
            f"<div style='display:flex;align-items:center;gap:12px;padding:5px 0;"
            f"border-bottom:1px solid var(--border-hr);'>"
            f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:.78rem;"
            f"color:var(--dim);min-width:90px;'>{_fmt_ts(s.start)} – {_fmt_ts(s.end)}</span>"
            f"<span style='font-family:\"Chakra Petch\",monospace;font-size:.76rem;"
            f"font-weight:600;letter-spacing:.08em;text-transform:uppercase;"
            f"color:var(--text);'>{html_mod.escape(s.label)}</span>"
            f"</div>"
            for s in sections
        )
        sections_html = f"<div style='margin-top:4px;'>{rows_html}</div>"
    else:
        sections_html = "<span style='font-family:var(--mono);font-size:.8rem;color:var(--dim);'>No section data</span>"

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
                  letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:10px;">Detected Sections</div>
      {sections_html}
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
    c2pa_fmt  = (
        "⚠ Born-AI (Certified)" if fr.c2pa_flag
        else "✓ No C2PA Manifest"
    )

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
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Content Credentials standard (C2PA) — a cryptographic signature embedded by some DAWs and AI tools. A "Born-AI" assertion is a hard certified signal the track was machine-generated. "No Manifest" is neutral — most files have none.</span>
          </span>
        </span>
        <span class="sv">{c2pa_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">IBI Variance (ms²)
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Inter-Beat Interval variance — measures millisecond-level timing drift between beats. Near-zero (&lt;0.5 ms²) = machine-quantized grid (AI/loop signal). High variance (&gt;90 ms²) = natural human feel and micro-timing variation — an organic signal, not AI.</span>
          </span>
        </span>
        <span class="sv">{ibi_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Groove Profile
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Derived from IBI variance. "Perfect Quantization" = machine-grid locked (&lt;0.5 ms²) — AI/loop signal. "Human Micro-timing" = natural drift (0.5–90 ms²). "Human-Feel Timing" = high micro-variation (&gt;90 ms²) — organic signal indicating human performance or humanized production.</span>
          </span>
        </span>
        <span class="sv">{groove_flag}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Centroid Instability
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Measures spectral centroid coefficient-of-variation within each sustained note. AI vocoders shift upper partials erratically mid-note — the source of the "glassy/hollow/formant-shifting" artifact heard in AI covers. Human vibrato modulates all partials together, keeping the centroid relatively stable. A score above 0.08 flags suspected formant drift. Scores of –1 mean no sustained intervals were found (e.g. full-instrumental or very quiet sections).</span>
          </span>
        </span>
        <span class="sv">{centroid_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Spectral Slop
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Checks for anomalous energy above 16 kHz relative to the full spectrum. AI generators often leak noise in the ultrasonic range. A high-frequency ratio &gt;15% triggers this flag.</span>
          </span>
        </span>
        <span class="sv">{slop_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">HF Phase Coherence
          <span class="tip-wrap"><span class="tip-icon">?</span>
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

# ---------------------------------------------------------------------------
# Production Analysis — sample & loop detection (separate from AI detection)
# ---------------------------------------------------------------------------

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

    st.markdown(f"""
    <div class="sig">
      <div class="sig-head">Structural Repetition</div>
      <div style="font-family:'Figtree',sans-serif;font-size:.82rem;color:var(--dim);
                  line-height:1.5;margin-bottom:18px;">
        Measures how repetitive this track's structure is. High scores indicate the
        production relies heavily on looping sections — common in modern pop and hip-hop.
        This is independent of AI detection and does not indicate AI generation on its own.
      </div>
      <div class="sig-row">
        <span class="sk">Section Similarity
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Compares 4-bar spectral sections across the track. High score means sections sound near-identical — the production leans heavily on a repeating musical phrase.</span>
          </span>
        </span>
        <span class="sv">{loop_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Rhythmic Regularity
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Measures how regularly the beat pattern repeats. High score means the rhythm locks to a tight, consistent cycle — typical of loop-based production.</span>
          </span>
        </span>
        <span class="sv">{autocorr_fmt}</span>
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
                    color:var(--dim);">LUFS integrated</div>
        <div style="margin-left:auto;font-family:'JetBrains Mono',monospace;
                    font-size:.65rem;color:{peak_color};">
          {html_mod.escape(peak_label)}{html_mod.escape(peak_warn)}
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">{platform_cells}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dialogue-ready row ────────────────────────────────────────────────
    dial_color = _DIALOGUE_LABEL_COLORS.get(aq.dialogue_label, "var(--muted)")
    dial_pct   = int(aq.dialogue_score * 100)

    st.markdown(f"""
    <div class="sig" style="padding:14px 16px;display:flex;align-items:center;gap:16px;">
      <div>
        <div style="font-family:'Chakra Petch',monospace;font-size:.5rem;font-weight:600;
                    letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                    margin-bottom:4px;">Dialogue Compatibility</div>
        <div style="font-family:'Chakra Petch',monospace;font-size:1.1rem;font-weight:700;
                    color:{dial_color};">{html_mod.escape(aq.dialogue_label)}</div>
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
        flag_text = "Track ends with a fade — may bleed into dialogue or scene audio." if sting.flag else None
        _sync_readiness_row(
            icon="🔔", label="Sting / Ending Type",
            value=sting.ending_type.title() + ratio_str,
            ok=not sting.flag,
            flag_text=flag_text,
            tip="Detects how the track ends — critical for sync placement.<br><br>"
                "<strong>Sting</strong>: sharp final hit with rapid silence after. ✓<br>"
                "<strong>Cut</strong>: abrupt stop — neutral, workable. ✓<br>"
                "<strong>Fade</strong>: gradual energy decline — can bleed into dialogue. ⚠ Flagged.",
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

_POPULARITY_TIER_COLORS: dict[str, str] = {
    "Emerging":   "var(--dim)",
    "Regional":   "var(--issue-location)",  # blue — defined in styles.py
    "Mainstream": "var(--grade-c)",          # amber — defined in styles.py
    "Global":     "var(--accent)",           # orange — defined in styles.py
}


def _render_popularity_card(result: AnalysisResult) -> None:
    """Render track popularity tier and estimated sync cost from Last.fm data."""
    pop = result.popularity
    if pop is None:
        st.markdown(
            "<div style='color:var(--dim);font-size:.84rem;font-family:Figtree,sans-serif;'>"
            "Popularity data unavailable — track not found on Last.fm.</div>",
            unsafe_allow_html=True,
        )
        return

    tier_color = _POPULARITY_TIER_COLORS.get(pop.tier, "var(--dim)")
    listeners_fmt = f"{pop.listeners:,}"
    cost_fmt      = f"${pop.sync_cost_low:,} – ${pop.sync_cost_high:,}"

    st.markdown(f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
      <div class="sig" style="flex:1;min-width:120px;padding:14px 16px;text-align:center;">
        <div style="font-family:'Chakra Petch',monospace;font-size:.5rem;font-weight:600;
                    letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                    margin-bottom:6px;">Popularity</div>
        <div style="font-family:'Chakra Petch',monospace;font-size:1.2rem;font-weight:700;
                    color:{tier_color};">{pop.tier}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                    color:var(--muted);margin-top:4px;">{listeners_fmt} listeners</div>
      </div>
      <div class="sig" style="flex:2;min-width:180px;padding:14px 16px;">
        <div style="font-family:'Chakra Petch',monospace;font-size:.5rem;font-weight:600;
                    letter-spacing:.18em;text-transform:uppercase;color:var(--dim);
                    margin-bottom:6px;">Est. Sync Fee</div>
        <div style="font-family:'Chakra Petch',monospace;font-size:.95rem;font-weight:700;
                    color:var(--text);">{cost_fmt}</div>
        <div style="font-family:'Figtree',sans-serif;font-size:.7rem;color:var(--muted);
                    margin-top:4px;">Varies by usage, territory, and negotiation.
                    Source: Last.fm · Industry estimates 2024–2026.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


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
            isrc_text = html_mod.escape(result.legal.isrc or "—")
            pro_text  = html_mod.escape(result.legal.pro_match or "Unknown")
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
                <div style="font-family:'Chakra Petch',monospace;font-size:.9rem;
                            color:var(--accent);">{pro_text}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        for name, url in [("ASCAP", result.legal.ascap), ("BMI", result.legal.bmi), ("SESAC", result.legal.sesac)]:
            if url:
                st.link_button(f"Search {name} →", url, use_container_width=True)

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
            rows += f"""
            <div class="t-row">
              <div><div class="t-art">{safe_artist}</div><div class="t-nm">{safe_title}</div></div>
              {btn}
            </div>"""
        st.markdown(f"<div class='sig' style='padding:18px;'>{rows}</div>", unsafe_allow_html=True)
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

    _render_authorship_banner(authorship)

    col_lyr, col_audit = st.columns([55, 45], gap="large")
    with col_lyr:
        _render_lyric_column(segments, sections, flagged_ts, flags_by_ts)
    with col_audit:
        _render_audit_column(flags, grade)


# ---------------------------------------------------------------------------
# Lyric section sub-renderers (extracted to keep each function under 40 lines)
# ---------------------------------------------------------------------------

def _render_authorship_banner(authorship: Optional["AuthorshipResult"]) -> None:
    if not authorship:
        return
    av       = authorship.verdict
    av_color = authorship_color(av)
    a_notes  = authorship.feature_notes
    a_rob    = authorship.roberta_score
    rob_str  = f"Classifier: {a_rob:.0%} AI probability · " if a_rob is not None else ""
    n_sig    = authorship.signal_count
    sig_str  = f"{n_sig} lyric flag{'s' if n_sig != 1 else ''}"

    def _note_html(note: str) -> str:
        arrow      = "✓" if "✓" in note else "▲"
        note_color = "var(--muted)" if "✓" in note else "var(--text)"
        return (
            f"<div style='display:flex;align-items:center;gap:6px;padding:3px 0;'>"
            f"<span style='color:{av_color};font-size:.7rem;flex-shrink:0;'>{arrow}</span>"
            f"<span style='font-family:\"Figtree\",sans-serif;font-size:.76rem;"
            f"color:{note_color};'>{note}</span></div>"
        )

    notes_html = "".join(_note_html(n) for n in a_notes)
    st.markdown(f"""
    <div style="border:1px solid {av_color}22;border-radius:10px;background:{av_color}08;
                padding:14px 18px;margin-bottom:18px;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:{'10px' if a_notes else '0'};">
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
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 18px;">{notes_html}</div>
    </div>
    """, unsafe_allow_html=True)


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

    for sec_label, segs in _assign_sections(segments, sections):
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
        <span class="tip-wrap" style="margin-left:6px;"><span class="tip-icon">?</span>
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
