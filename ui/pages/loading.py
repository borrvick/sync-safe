"""
ui/pages/loading.py
Full-screen loading overlay shown while the pipeline runs.

Runs ingestion + all analysis steps sequentially, updating a live
step checklist after each step completes. Transitions to "report"
when done, or surfaces a fatal ingestion error inline without crashing.

Design rules:
- Each step is fault-isolated: non-ingestion failures degrade to None,
  never abort siblings (mirrors pipeline.py._run_step semantics).
- Ingestion failure is fatal — surfaces error and returns early.
- No disk writes: ingestion returns BytesIO-backed AudioBuffer.
- UI state is driven by four st.empty() placeholders updated in-place.
"""
from __future__ import annotations

import time
from typing import Any

import streamlit as st

from core.config import get_settings
from core.logging import PipelineLogger
from core.models import AudioBuffer
from ui.components import eq_bars

# (key, display label, estimated wall-clock seconds on ZeroGPU free tier, tooltip description)
_STEPS: list[tuple[str, str, int, str]] = [
    ("ingestion",     "Fetching Audio",      8,
     "Downloads audio from YouTube via yt-dlp, or reads your uploaded file. "
     "Audio is held as a BytesIO buffer in memory — nothing is written to disk."),
    ("transcription", "Transcribing Lyrics", 50,
     "Runs OpenAI Whisper to convert vocals into timestamped text segments. "
     "Each segment gets a start/end time so you can click any lyric line in the report to seek the player."),
    ("structure",     "Analysing Structure", 20,
     "Uses allin1 to detect BPM, musical key, and section labels (intro, verse, chorus, outro). "
     "Also reads embedded ID3 / VorbisComment metadata tags from the file."),
    ("forensics",     "Forensic Scan",       20,
     "Runs six AI-origin signals: C2PA manifest check, IBI beat-timing variance, groove profile, "
     "4-bar loop cross-correlation, spectral slop above 16 kHz, and SynthID HF phase-coherence scan."),
    ("compliance",    "Compliance Audit",    12,
     "Sync readiness checks: sting/fade ending detection, 4–8 bar energy evolution, intro length, "
     "and lyric flags for explicit content, brand names, locations, violence, and drug references."),
    ("authorship",    "Authorship Check",     8,
     "Scores lyrics on four signals — burstiness, vocabulary diversity, rhyme density, repetition — "
     "plus a RoBERTa classifier. Verdicts: Likely Human, Uncertain, or Likely AI."),
    ("discovery",     "Track Discovery",     10,
     "Queries Last.fm's similarity graph for comparable tracks, then resolves each result to a live "
     "YouTube preview URL via yt-dlp. Fully stateless — no database involved."),
    ("legal",         "Legal Links",          2,
     "Generates direct search links for ASCAP, BMI, and SESAC repertory databases "
     "so you can identify publishers and rights holders before any outreach."),
]

_TOTAL_EST: int = sum(s[2] for s in _STEPS)

# ---------------------------------------------------------------------------
# Loading page CSS — injected once at the start of render_loading.
# Makes the main block fill the full viewport and center content vertically.
# Safe to inject here: when the pipeline finishes and st.rerun() transitions
# to the report page, Streamlit re-renders from scratch so this CSS doesn't
# bleed through.
# ---------------------------------------------------------------------------

_LOADING_CSS = """
<style>
/* Full-viewport immersive loading canvas */
[data-testid="stMainBlockContainer"] {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  min-height: 100svh !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
}
.block-container {
  max-width: 540px !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  margin: 0 auto !important;
  width: 100% !important;
}
/* Slim accent progress bar */
[data-testid="stProgress"] > div {
  background: rgba(245,100,10,.12) !important;
  border-radius: 3px !important;
  height: 3px !important;
  overflow: hidden !important;
}
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, #F5640A 0%, #FF9050 100%) !important;
  border-radius: 3px !important;
  transition: width .5s ease !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _fmt(secs: float) -> str:
    """Format seconds as M:SS."""
    s = max(0, int(secs))
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Static brand header — rendered once, stays visible the whole run
# ---------------------------------------------------------------------------

def _render_brand() -> None:
    eq_html = eq_bars(22, "#F5640A", 28)
    st.markdown(f"""
    <div style="text-align:center;padding:0 0 32px;">
      <div style="display:flex;align-items:flex-end;justify-content:center;gap:3px;
                  height:28px;margin-bottom:20px;opacity:.45;">
        {eq_html}
      </div>
      <div style="font-family:'Chakra Petch',monospace;font-size:.82rem;font-weight:700;
                  color:#F5640A;letter-spacing:.18em;text-transform:uppercase;line-height:1;">
        SYNC-SAFE
      </div>
      <div style="font-family:'Chakra Petch',monospace;font-size:.44rem;font-weight:500;
                  letter-spacing:.26em;text-transform:uppercase;color:var(--dim);margin-top:4px;">
        Forensic Portal
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI renderer — called after every step change
# ---------------------------------------------------------------------------

def _draw(
    header_ph: Any,
    bar_ph: Any,
    steps_ph: Any,
    completed: int,
    current_label: str,   # empty string when all steps done
    step_durations: list[float],
) -> None:
    """Redraw headline, progress bar, and step checklist."""
    n = len(_STEPS)

    # Headline
    if current_label:
        headline = (
            f"{current_label}"
            f"<span style='color:var(--dim);margin-left:6px;font-size:1.1rem'>…</span>"
        )
        headline_color = "var(--text)"
        sub = (
            f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:.62rem;"
            f"color:var(--dim);letter-spacing:.06em;margin-top:8px;'>"
            f"Step {completed + 1} of {n}</div>"
        )
    else:
        headline = "<span style='color:var(--ok)'>Analysis Complete</span>"
        headline_color = "var(--ok)"
        sub = (
            "<div style='font-family:\"JetBrains Mono\",monospace;font-size:.62rem;"
            "color:var(--ok);letter-spacing:.06em;margin-top:8px;opacity:.7;'>"
            "Preparing report…</div>"
        )

    header_ph.html(
        f'<div style="text-align:center;padding:0 0 22px;">'
        f'<div style="font-family:\'Chakra Petch\',monospace;font-size:.5rem;font-weight:600;'
        f'letter-spacing:.28em;color:var(--dim);margin-bottom:12px;text-transform:uppercase;">'
        f'◈ INITIALIZING SCAN</div>'
        f'<div style="font-family:\'Chakra Petch\',monospace;font-size:1.55rem;font-weight:700;'
        f'color:{headline_color};letter-spacing:-.02em;line-height:1.1;">'
        f'{headline}</div>'
        f'{sub}'
        f'</div>'
    )

    bar_ph.progress(completed / n)

    # CSS for step tooltips + progress bar animation.
    # st.html() injects via innerHTML; browsers do not execute <script> tags that way.
    _TIP_CSS = (
        '<style>'
        '.sp-tip{position:relative;display:inline-flex;align-items:center;cursor:default}'
        '.sp-ti{display:inline-flex;align-items:center;justify-content:center;'
        'width:13px;height:13px;border-radius:50%;background:rgba(245,100,10,.12);'
        'color:#F5640A;font-size:.5rem;font-weight:700;font-family:monospace;'
        'border:1px solid rgba(245,100,10,.28);margin-left:7px;flex-shrink:0;'
        'line-height:1;cursor:pointer;transition:background .15s}'
        '.sp-tip:hover .sp-ti,.sp-tip:focus-within .sp-ti{background:rgba(245,100,10,.28)}'
        '.sp-tb{display:none;position:absolute;'
        'top:calc(100% + 5px);left:-8px;'
        'background:var(--tip-bg);border:1px solid rgba(245,100,10,.22);color:var(--text);'
        'font-size:.71rem;font-family:Figtree,sans-serif;font-weight:400;line-height:1.52;'
        'padding:10px 13px;border-radius:8px;width:230px;z-index:9999;'
        'box-shadow:var(--shadow-sm);pointer-events:none;white-space:normal}'
        '.sp-tip:hover .sp-tb,.sp-tip:focus-within .sp-tb{display:block}'
        '@keyframes pb-run{from{width:0%}to{width:99%}}'
        '.pb-run{animation:pb-run linear forwards}'
        '@keyframes run-pulse{0%,100%{opacity:.45}50%{opacity:1}}'
        '.run-txt{animation:run-pulse 1.4s ease-in-out infinite}'
        '.sp-ti:focus-visible{outline:2px solid #F5640A;outline-offset:1px;border-radius:50%}'
        '</style>'
    )

    rows = ""
    for i, (_, label, est, desc) in enumerate(_STEPS):
        is_last = (i == len(_STEPS) - 1)
        row_border = '' if is_last else 'border-bottom:1px solid var(--border-hr);'

        if i < completed:
            icon, icon_color    = "✓", "#10B981"
            icon_aria           = "Completed"
            label_color, weight = "var(--muted)", "400"
            pb_inner = '<div style="height:100%;width:100%;background:#10B981;border-radius:1px;"></div>'
            right = (
                f"<span style='font-family:JetBrains Mono,monospace;"
                f"font-size:.65rem;color:var(--dim);'>{_fmt(step_durations[i])}</span>"
            )
        elif label == current_label:
            icon, icon_color    = "▶", "#F5640A"
            icon_aria           = "Running"
            label_color, weight = "var(--text)", "600"
            pb_inner = (
                f'<div class="pb-run" style="height:100%;animation-duration:{est}s;'
                f'background:#F5640A;border-radius:1px;"></div>'
            )
            right = (
                "<span class='run-txt' style='font-family:JetBrains Mono,monospace;"
                "font-size:.65rem;color:#F5640A;'>running…</span>"
            )
        else:
            icon, icon_color    = "○", "var(--dim)"
            icon_aria           = "Pending"
            label_color, weight = "var(--dim)", "400"
            pb_inner            = ""
            right = (
                f"<span style='font-family:JetBrains Mono,monospace;"
                f"font-size:.65rem;color:var(--dim);'>~{_fmt(est)}</span>"
            )

        rows += (
            f'<div style="padding:10px 18px 0;{row_border}">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:7px;">'
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span aria-label="{icon_aria}" style="font-family:\'Chakra Petch\',monospace;font-size:.78rem;'
            f'color:{icon_color};font-weight:700;width:14px;flex-shrink:0;">{icon}</span>'
            f'<span style="font-family:\'Figtree\',sans-serif;font-size:.84rem;'
            f'color:{label_color};font-weight:{weight};">{label}</span>'
            f'<span class="sp-tip">'
            f'<span class="sp-ti" tabindex="0" role="button" aria-label="More information about {label}">?</span>'
            f'<span class="sp-tb" role="tooltip">{desc}</span>'
            f'</span>'
            f'</div>'
            f'{right}'
            f'</div>'
            f'<div style="height:2px;background:var(--border);border-radius:1px;margin-bottom:6px;">'
            f'{pb_inner}'
            f'</div>'
            f'</div>'
        )

    steps_ph.html(
        _TIP_CSS
        + f'<div style="border:1px solid var(--border);border-radius:12px;'
        f'margin:6px 0 18px;background:var(--s1);">'
        f'<div style="padding:9px 18px;border-bottom:1px solid var(--border-hr);">'
        f'<span style="font-family:\'Chakra Petch\',monospace;font-size:.5rem;font-weight:600;'
        f'letter-spacing:.18em;text-transform:uppercase;color:var(--dim);">'
        f'Pipeline · {completed} / {n} steps complete</span></div>'
        + rows
        + '</div>'
    )


# ---------------------------------------------------------------------------
# One-time timer — set once before pipeline starts, never replaced
# ---------------------------------------------------------------------------

def _start_timer() -> None:
    """
    Render the elapsed / ETA timer using st.components.v1.html(), which runs
    inside a real sandboxed iframe where <script> tags execute.

    st.html() injects via innerHTML and scripts are silently ignored by browsers;
    st.components.v1.html() is required for any live JS behaviour.

    Called once before the pipeline starts — never replaced, so the setInterval
    ticks every second for the full run without interruption.
    """
    import streamlit.components.v1 as components
    components.html(
        f'<div style="text-align:center;padding-bottom:4px;"'
        f' role="status" aria-live="polite" aria-label="Pipeline timer" aria-atomic="false">'
        f'<span id="t-el" style="font-family:JetBrains Mono,monospace;font-size:.68rem;'
        f'color:var(--muted);letter-spacing:.08em;">Elapsed 0:00</span>'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:.68rem;color:var(--dim);"> · ETA ~</span>'
        f'<span id="t-et" style="font-family:JetBrains Mono,monospace;font-size:.68rem;'
        f'color:var(--muted);letter-spacing:.08em;">{_fmt(_TOTAL_EST)}</span>'
        f'</div>'
        f'<script>'
        f'var el=0,et={_TOTAL_EST};'
        f'function f(s){{s=Math.max(0,s|0);return Math.floor(s/60)+":"+(s%60<10?"0":"")+s%60;}}'
        f'setInterval(function(){{'
        f'el++;et=Math.max(0,et-1);'
        f'document.getElementById("t-el").textContent="Elapsed "+f(el);'
        f'document.getElementById("t-et").textContent=f(et);'
        f'}},1000);'
        f'</script>',
        height=36,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_loading(source: Any) -> None:
    """
    Render the loading page and run the full pipeline.

    Args:
        source: YouTube URL string or Streamlit UploadedFile — passed
                directly to Ingestion().load().

    Side effects:
        On success: sets st.session_state.audio, .analysis, .page = "report"
                    then calls st.rerun().
        On fatal ingestion error: renders st.error() and returns early.
    """
    st.markdown(_LOADING_CSS, unsafe_allow_html=True)
    _render_brand()

    header_ph = st.empty()
    bar_ph    = st.empty()
    steps_ph  = st.empty()
    # Timer uses st.components.v1.html() (real iframe, JS executes).
    # Placed here in layout order before error_ph. Never updated — the
    # JS setInterval runs for the full pipeline duration on its own.
    _start_timer()
    error_ph  = st.empty()

    completed      = 0
    step_durations: list[float] = []
    log            = PipelineLogger(get_settings().log_dir)
    pipeline_start = time.time()

    log.pipeline_start(source=str(source))

    def _tick(label: str) -> None:
        _draw(header_ph, bar_ph, steps_ph, completed, label, step_durations)

    def _advance(key: str, step_start: float, error: str | None = None) -> None:
        nonlocal completed
        duration = time.time() - step_start
        step_durations.append(duration)
        completed += 1
        if error:
            log.step_error(key, error=error)
        else:
            log.step_end(key, duration_s=duration)

    # ── Step 1: Ingestion (fatal on failure) ──────────────────────────────
    _tick("Fetching Audio")
    t0 = time.time()
    log.step_start("ingestion")
    try:
        from services.ingestion import Ingestion
        audio: AudioBuffer = Ingestion().load(source)
    except Exception as exc:
        log.step_error("ingestion", error=str(exc))
        log.pipeline_error(error=str(exc))
        error_ph.error(f"Could not load audio: {exc}")
        return
    _advance("ingestion", t0)

    # ── Step 2: Transcription ─────────────────────────────────────────────
    _tick("Transcribing Lyrics")
    t0 = time.time()
    transcript = []
    log.step_start("transcription")
    try:
        from services.transcription import Transcription
        transcript = Transcription().transcribe(audio)
    except Exception as exc:
        _advance("transcription", t0, error=str(exc))
    else:
        _advance("transcription", t0)

    # ── Step 3: Structure analysis ────────────────────────────────────────
    _tick("Analysing Structure")
    t0 = time.time()
    structure = None
    log.step_start("structure")
    try:
        from services.analysis import Analysis
        structure = Analysis().analyze(audio)
    except Exception as exc:
        _advance("structure", t0, error=str(exc))
    else:
        _advance("structure", t0)

    # ── Step 4: Forensics ─────────────────────────────────────────────────
    _tick("Forensic Scan")
    t0 = time.time()
    forensics = None
    log.step_start("forensics")
    try:
        from services.forensics import Forensics
        forensics = Forensics().analyze(audio)
    except Exception as exc:
        _advance("forensics", t0, error=str(exc))
    else:
        _advance("forensics", t0)

    # ── Step 5: Compliance ────────────────────────────────────────────────
    _tick("Compliance Audit")
    t0 = time.time()
    compliance = None
    log.step_start("compliance")
    try:
        from services.compliance import Compliance
        sections = structure.sections if structure else []
        beats    = structure.beats    if structure else []
        compliance = Compliance().check(audio, transcript, sections, beats)
    except Exception as exc:
        _advance("compliance", t0, error=str(exc))
    else:
        _advance("compliance", t0)

    # ── Step 6: Authorship ────────────────────────────────────────────────
    _tick("Authorship Check")
    t0 = time.time()
    authorship = None
    log.step_start("authorship")
    try:
        from services.authorship import Authorship
        authorship = Authorship().analyze(transcript)
    except Exception as exc:
        _advance("authorship", t0, error=str(exc))
    else:
        _advance("authorship", t0)

    # ── Step 7: Track discovery ───────────────────────────────────────────
    _tick("Track Discovery")
    t0 = time.time()
    similar = []
    log.step_start("discovery")
    title  = structure.metadata.get("title", "")  if structure else ""
    artist = structure.metadata.get("artist", "") if structure else ""
    try:
        from services.discovery import Discovery
        similar = Discovery().find_similar(title, artist) or []
    except Exception as exc:
        _advance("discovery", t0, error=str(exc))
    else:
        _advance("discovery", t0)

    # ── Step 8: Legal links ───────────────────────────────────────────────
    _tick("Legal Links")
    t0 = time.time()
    legal = None
    log.step_start("legal")
    try:
        from services.legal import Legal
        legal = Legal().get_links(title, artist)
    except Exception as exc:
        _advance("legal", t0, error=str(exc))
    else:
        _advance("legal", t0)

    # ── All done — render final state then transition ─────────────────────
    log.pipeline_end(duration_s=time.time() - pipeline_start)
    _draw(header_ph, bar_ph, steps_ph, completed, "", step_durations)

    from core.models import AnalysisResult
    st.session_state.audio    = audio
    st.session_state.analysis = AnalysisResult(
        audio=audio,
        structure=structure,
        forensics=forensics,
        transcript=transcript,
        compliance=compliance,
        authorship=authorship,
        similar_tracks=similar,
        legal=legal,
    )
    st.session_state.page = "report"
    st.rerun()
