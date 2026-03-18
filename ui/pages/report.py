"""
ui/pages/report.py
Report page rendering — takes a typed AnalysisResult and renders all sections.

All functions accept typed domain models from core.models; no raw dict access.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Optional

import streamlit as st

from core.config import CONSTANTS
from core.models import (
    AnalysisResult,
    AudioBuffer,
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

    with st.expander("Track Overview", expanded=True):
        c_left, c_right = st.columns([1, 1], gap="large")
        with c_left:
            _render_metadata_card(result.structure)
        with c_right:
            _render_structure_card(result.structure)

    with st.expander("Authenticity Audit", expanded=True):
        _render_forensics_card(result.forensics)

    with st.expander("Sync Readiness Checks", expanded=True):
        _render_sync_readiness(result.compliance)

    with st.expander("Discovery & Licensing", expanded=True):
        _render_legal_and_discovery(result)

    with st.expander("Lyrics & Content Audit", expanded=True):
        _render_lyric_section(result)

    _render_footer()


# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

def _render_nav(source_label: str) -> None:
    _eq = eq_bars(6, color="#F5640A", h=22)
    c_logo, c_nav = st.columns([6, 1])
    with c_logo:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:24px 0 10px;">
          <div style="display:flex;align-items:flex-end;gap:2px;height:22px;">{_eq}</div>
          <div>
            <div style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                        color:#F5640A;letter-spacing:.14em;">SYNC-SAFE</div>
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
        f'margin-top:6px;letter-spacing:.04em;">{source_label}</div>'
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
    st.audio(
        audio.raw,
        start_time=st.session_state.get("start_time", 0),
    )
    st.markdown("<div style='margin-bottom:28px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Track Overview: metadata + structure
# ---------------------------------------------------------------------------

def _render_metadata_card(sr: Optional[StructureResult]) -> None:
    meta       = sr.metadata if sr else {}
    title_str  = meta.get("title", "")  or ""
    artist_str = meta.get("artist", "") or ""

    title_html = (
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:1.4rem;"
        f"font-weight:700;color:var(--text);line-height:1.2;margin-bottom:5px;'>{title_str}</div>"
        if title_str else
        "<div style='font-size:.9rem;color:var(--dim);margin-bottom:5px;'>No tags found</div>"
    )
    artist_html = (
        f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.6rem;"
        f"font-weight:600;color:#F5640A;letter-spacing:.12em;text-transform:uppercase;'>{artist_str}</div>"
        if artist_str else ""
    )

    st.markdown(f"""
    <div class="sig" style="margin-bottom:14px;">
      <div class="sig-head">Track Metadata</div>
      {title_html}
      {artist_html}
    </div>
    """, unsafe_allow_html=True)


def _render_structure_card(sr: Optional[StructureResult]) -> None:
    bpm_fmt  = f"{sr.bpm:.1f}" if sr and isinstance(sr.bpm, float) else (sr.bpm if sr else "—")
    key      = sr.key if sr else "—"
    sections = sr.sections if sr else []

    pills = "".join(
        f"<span class='s-pill'>{s.label} {s.start:.0f}s–{s.end:.0f}s</span>"
        for s in sections
    ) if sections else "<span style='font-family:var(--mono);font-size:.8rem;color:var(--dim);'>No section data</span>"

    st.markdown(f"""
    <div class="sig">
      <div class="sig-head">Structure Analysis</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:24px;">
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                      letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;">Tempo</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:700;
                      color:#F5640A;line-height:1;">
            {bpm_fmt}<span style="font-size:.8rem;font-weight:400;color:var(--dim);
                                   margin-left:5px;font-family:'Chakra Petch',monospace;
                                   letter-spacing:.1em;">BPM</span>
          </div>
        </div>
        <div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                      letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;">Key</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:2.6rem;font-weight:700;
                      color:#F5640A;line-height:1;">{key}</div>
        </div>
      </div>
      <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                  letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-bottom:10px;">Detected Sections</div>
      <div style="display:flex;flex-wrap:wrap;">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Authenticity Audit: forensics
# ---------------------------------------------------------------------------

def _render_forensics_card(fr: Optional[ForensicsResult]) -> None:
    if fr is None:
        st.markdown(
            "<div class='sig'><div class='sig-head'>Authenticity Audit</div>"
            "<div style='color:var(--dim);'>Forensics analysis unavailable.</div></div>",
            unsafe_allow_html=True,
        )
        return

    verdict   = fr.verdict
    v_cls     = {"Human": "v-h", "AI": "v-a"}.get(verdict, "v-u")

    # C2PA
    c2pa_fmt  = (
        "⚠ Born-AI (Certified)" if fr.c2pa_flag
        else "✓ No C2PA Manifest"
    )

    # IBI / groove
    ibi       = fr.ibi_variance
    ibi_fmt   = f"{ibi:.3f}" if isinstance(ibi, float) and ibi >= 0 else "—"
    groove_flag = _groove_label(ibi)

    # Loop
    loop      = fr.loop_score
    loop_num  = f"{loop:.3f}" if isinstance(loop, float) else "—"
    loop_label = (
        "Likely Stock Loop" if loop > CONSTANTS.LOOP_SCORE_CEILING
        else "Possible Repetition" if loop > CONSTANTS.LOOP_SCORE_POSSIBLE
        else "Organic"
    )
    loop_fmt  = f"{loop_num} ({loop_label})"

    # Spectral slop
    slop_val  = fr.spectral_slop
    slop_fmt  = "✓ Clean" if slop_val <= CONSTANTS.SPECTRAL_SLOP_RATIO else f"⚠ {slop_val:.1%} HF energy"

    # SynthID
    synthid_bins = int(fr.synthid_score)
    synthid_conf = _synthid_confidence(synthid_bins)
    _si_icons    = {"none": "✓", "low": "◈", "medium": "⚠", "high": "⚠"}
    synthid_fmt  = f"{_si_icons.get(synthid_conf, '◈')} {synthid_conf.title()} ({synthid_bins} bins)"

    st.markdown(f"""
    <div class="sig">
      <div class="sig-head">Authenticity Audit</div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
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
                      letter-spacing:.1em;text-transform:uppercase;">C2PA · IBI · Groove · Loop · Spectral · SynthID</div>
        </div>
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
            <span class="tip-box">Inter-Beat Interval variance — measures millisecond-level timing drift between beats. Human drummers naturally range 8–90 ms². Near-zero (&lt;0.5 ms²) = machine-quantized. Extremely high (&gt;90 ms²) = over-humanized AI timing.</span>
          </span>
        </span>
        <span class="sv">{ibi_fmt}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Groove Profile
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Derived from IBI variance. "Perfect Quantization" = machine-grid locked (&lt;0.5 ms²). "Human Micro-timing" = natural drift (0.5–90 ms²). "Erratic Humanization" = over-humanized AI pattern (&gt;90 ms²).</span>
          </span>
        </span>
        <span class="sv">{groove_flag}</span>
      </div>
      <div class="sig-row">
        <span class="sk">Loop Score
          <span class="tip-wrap"><span class="tip-icon">?</span>
            <span class="tip-box">Cross-correlation of 4-bar spectral fingerprints across the track. Score &gt;0.98 means segments are near-identical — a hallmark of stock loops or AI generation.</span>
          </span>
        </span>
        <span class="sv">{loop_fmt}</span>
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


# ---------------------------------------------------------------------------
# Sync Readiness Checks (structural placement rules)
# ---------------------------------------------------------------------------

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
    status_color = "#0DF5A0" if ok else "#FF6B35"
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
           f"color:#FF6B35;margin-top:3px;line-height:1.45;'>{flag_text}</div>"
           if flag_text else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Discovery & Licensing
# ---------------------------------------------------------------------------

def _render_legal_and_discovery(result: AnalysisResult) -> None:
    if result.legal:
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
            btn = (
                f'<a href="{t.youtube_url}" target="_blank" rel="noopener noreferrer"'
                f' class="t-btn" aria-label="Preview {t.title} by {t.artist} on YouTube">▶ Preview</a>'
                if t.youtube_url
                else '<button disabled class="t-btn" style="opacity:.3;cursor:not-allowed;">No link</button>'
            )
            rows += f"""
            <div class="t-row">
              <div><div class="t-art">{t.artist}</div><div class="t-nm">{t.title}</div></div>
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

    flags    = compliance.flags    if compliance else []
    grade    = compliance.grade    if compliance else "N/A"

    flagged_ts  = {f.timestamp_s for f in flags}
    flags_by_ts: dict[int, list[ComplianceFlag]] = {}
    for f in flags:
        flags_by_ts.setdefault(f.timestamp_s, []).append(f)

    # Lyric authorship banner
    if authorship:
        av       = authorship.verdict
        av_color = authorship_color(av)
        a_notes  = authorship.feature_notes
        a_rob    = authorship.roberta_score
        rob_str  = f"Classifier: {a_rob:.0%} AI probability · " if a_rob is not None else ""
        n_sig    = authorship.signal_count
        sig_str  = f"{n_sig} AI signal{'s' if n_sig != 1 else ''} detected"

        def _note_html(note: str) -> str:
            arrow      = "✓" if "✓" in note else "▲"
            note_color = "var(--muted)" if "✓" in note else "var(--text)"
            return (
                f"<div style='display:flex;align-items:center;gap:6px;padding:3px 0;'>"
                f"<span style='color:{av_color};font-size:.7rem;flex-shrink:0;'>{arrow}</span>"
                f"<span style='font-family:\"Figtree\",sans-serif;font-size:.76rem;"
                f"color:{note_color};'>{note}</span>"
                f"</div>"
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
                Lyric Authorship — {sig_str}
              </div>
              <div style="font-family:'Figtree',sans-serif;font-size:.74rem;color:var(--muted);margin-top:2px;">
                {rob_str}Note: short creative text is harder to classify than prose.</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 18px;">
            {notes_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

    col_lyr, col_audit = st.columns([55, 45], gap="large")

    # Left: lyrics by section
    with col_lyr:
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
        else:
            grouped = _assign_sections(segments, sections)
            for sec_label, segs in grouped:
                st.markdown(
                    f"<div style='font-family:\"Chakra Petch\",monospace;font-size:.6rem;"
                    f"font-weight:700;color:#F5640A;letter-spacing:.14em;text-transform:uppercase;"
                    f"margin:16px 0 6px;padding-top:14px;"
                    f"border-top:1px solid var(--border-hr);'>"
                    f"[ {sec_label} ]</div>",
                    unsafe_allow_html=True,
                )
                for seg in segs:
                    ts_s       = int(seg.start)
                    text       = seg.text.strip()
                    is_flagged = ts_s in flagged_ts
                    if not text:
                        continue

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
                                f"{text}&nbsp;&nbsp;{pills_html}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div style='padding:7px 0;font-family:\"Figtree\",sans-serif;"
                                f"font-size:.88rem;color:var(--muted);line-height:1.4;'>{text}</div>",
                                unsafe_allow_html=True,
                            )

    # Right: compliance grade + issue breakdown
    with col_audit:
        conf_flags = [f for f in flags if f.confidence == "confirmed"]
        pot_flags  = [f for f in flags if f.confidence == "potential"]
        n_conf, n_pot = len(conf_flags), len(pot_flags)

        if n_conf and n_pot:
            issue_count_label = f"{n_conf} confirmed · {n_pot} potential"
        elif n_conf:
            issue_count_label = f"{n_conf} confirmed issue{'s' if n_conf != 1 else ''}"
        elif n_pot:
            issue_count_label = f"{n_pot} potential flag{'s' if n_pot != 1 else ''} — review needed"
        else:
            issue_count_label = "All clear"

        grade_color  = _grade_color(grade)
        grade_reason = _grade_reason(conf_flags, grade)

        def _deduped_pills(flag_list: list[ComplianceFlag], size: str = "lg") -> str:
            counts: dict = {}
            first:  dict = {}
            for fl in flag_list:
                t = fl.issue_type
                counts[t] = counts.get(t, 0) + 1
                if t not in first:
                    first[t] = fl
            parts = []
            for t, fl in first.items():
                pill = issue_pill(fl, size)
                if counts[t] > 1:
                    badge_color = "#C8E86A" if fl.confidence == "potential" else "var(--muted)"
                    pill += (
                        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:.52rem;"
                        f"font-weight:600;padding:1px 5px;border-radius:3px;"
                        f"background:var(--badge-bg);color:{badge_color};"
                        f"margin-left:3px;'>×{counts[t]}</span>"
                    )
                parts.append(pill)
            return "".join(parts)

        _all_clear_html = "<span style='font-family:\"Figtree\",sans-serif;font-size:.8rem;color:#0DF5A0;'>✓ All clear</span>"
        grade_pills_html = (_deduped_pills(conf_flags) + _deduped_pills(pot_flags)) if flags else _all_clear_html

        st.markdown(f"""
        <div class="sig" style="margin-bottom:16px;">
          <div class="sig-head">Sync Compliance Grade
            <span class="tip-wrap" style="margin-left:6px;"><span class="tip-icon">?</span>
              <span class="tip-box">Sync readiness scoring for sync licensing.<br><br>
              <strong>A</strong> — No issues. Clear for submission.<br>
              <strong>B</strong> — Minor or potential flags only. Confirm clearances.<br>
              <strong>C</strong> — Explicit/violent content or multiple brand mentions.<br>
              <strong>D</strong> — Multiple explicit or violence flags. Clean edit required.<br>
              <strong>F</strong> — Drug references or fade ending. Likely disqualifies broadcast.<br><br>
              Grades are based on <strong>confirmed</strong> flags only.</span>
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
                          line-height:1.55;">{grade_reason}</div>
            </div>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;">
            {grade_pills_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

        if flags:
            def _render_flag_rows(flag_list: list[ComplianceFlag], prefix: str) -> None:
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
                    text_preview  = flag.text[:52] + ("…" if len(flag.text) > 52 else "")
                    review_badge  = (
                        "<span style='font-family:\"Chakra Petch\",monospace;font-size:.5rem;"
                        "font-weight:600;padding:1px 5px;border-radius:3px;"
                        "background:#C8E86A18;color:#C8E86A;border:1px solid #C8E86A33;"
                        "margin-left:6px;'>NEEDS REVIEW</span>"
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

            if conf_flags:
                st.markdown("""
                <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                            letter-spacing:.16em;text-transform:uppercase;color:var(--dim);
                            margin-bottom:10px;">Confirmed Issues — Click to Jump</div>
                """, unsafe_allow_html=True)
                _render_flag_rows(conf_flags, "conf")

            if pot_flags:
                st.markdown(
                    "<div style='font-family:\"Chakra Petch\",monospace;font-size:.56rem;font-weight:600;"
                    "letter-spacing:.16em;text-transform:uppercase;color:#C8E86A;"
                    "margin-top:14px;margin-bottom:10px;opacity:.7;'>"
                    "Potential — Supervisor Review</div>",
                    unsafe_allow_html=True,
                )
                _render_flag_rows(pot_flags, "pot")
        else:
            st.markdown("""
            <div style="padding:20px;text-align:center;border:1px solid rgba(13,245,160,.15);
                        border-radius:10px;background:rgba(13,245,160,.04);margin-top:8px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:.8rem;color:#0DF5A0;">
                ✓ No compliance issues detected
              </div>
            </div>
            """, unsafe_allow_html=True)


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
        grp = [seg for seg in segments if sec.start <= seg.start < sec.end]
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
        return "Erratic Humanization (AI signal)"
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
        "B": "#6ECC8A",
        "C": "#F5A623",
        "D": "#F57A35",
        "F": "var(--danger)",
    }.get(grade, "var(--dim)")


def _grade_reason(conf_flags: list[ComplianceFlag], grade: str) -> str:
    if not conf_flags:
        return "No confirmed compliance issues. Track is clear for sync submission."
    counts = Counter(f.issue_type for f in conf_flags)
    n      = len(conf_flags)
    if counts.get("DRUGS", 0) > 0:
        return f"{counts['DRUGS']} confirmed drug reference(s). Disqualifies broadcast and family placements."
    if counts.get("EXPLICIT", 0) >= 2:
        return f"{counts['EXPLICIT']} confirmed explicit flags. A clean edit is required for most sync placements."
    if counts.get("EXPLICIT", 0) == 1 or counts.get("VIOLENCE", 0) >= 1:
        return f"{n} confirmed issue(s) including explicit or violent content. Edits required for network TV and family placements."
    if counts.get("BRAND", 0) >= 2:
        return f"{counts['BRAND']} brand mentions detected. Explicit clearance required from each brand's rights holder."
    return f"{n} minor confirmed flag(s). No hard blockers — confirm clearances before placement."
