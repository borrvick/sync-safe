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

from core.models import AudioBuffer

# (key, display label, estimated wall-clock seconds on ZeroGPU free tier)
_STEPS: list[tuple[str, str, int]] = [
    ("ingestion",     "Fetching Audio",       8),
    ("transcription", "Transcribing Lyrics",  50),
    ("structure",     "Analysing Structure",  65),
    ("forensics",     "Forensic Scan",        20),
    ("compliance",    "Compliance Audit",     12),
    ("authorship",    "Authorship Check",      8),
    ("discovery",     "Track Discovery",      10),
    ("legal",         "Legal Links",           2),
]

_TOTAL_EST: int = sum(s[2] for s in _STEPS)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _fmt(secs: float) -> str:
    """Format seconds as M:SS."""
    s = max(0, int(secs))
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# UI renderer — called after every step change
# ---------------------------------------------------------------------------

def _draw(
    header_ph: Any,
    bar_ph: Any,
    steps_ph: Any,
    timer_ph: Any,
    completed: int,
    current_label: str,   # empty string when all steps done
    elapsed: float,
    step_durations: list[float],
) -> None:
    """Redraw all four placeholder slots with current pipeline state."""
    n = len(_STEPS)

    # Header
    if current_label:
        headline = (
            f"{current_label}"
            f"<span style='color:#7A95AA;margin-left:6px;font-size:1.1rem'>…</span>"
        )
        headline_color = "#D8E6F2"
    else:
        headline = "<span style='color:#3DB87A'>Analysis Complete</span>"
        headline_color = "#3DB87A"

    header_ph.markdown(f"""
    <div style="text-align:center;padding:36px 0 20px;">
      <div style="font-family:'Chakra Petch',monospace;font-size:.54rem;font-weight:600;
                  letter-spacing:.28em;color:#364C5C;margin-bottom:14px;
                  text-transform:uppercase;">◈ INITIALIZING SCAN</div>
      <div style="font-family:'Chakra Petch',monospace;font-size:1.55rem;font-weight:700;
                  color:{headline_color};letter-spacing:-.02em;min-height:2.2rem;">
        {headline}
      </div>
    </div>
    """, unsafe_allow_html=True)

    bar_ph.progress(completed / n)

    # Step checklist rows
    rows = ""
    for i, (_, label, est) in enumerate(_STEPS):
        if i < completed:
            icon, icon_color  = "✓", "#3DB87A"
            label_color, weight = "#7A95AA", "400"
            right = (
                f"<span style='font-family:JetBrains Mono,monospace;"
                f"font-size:.65rem;color:#2A3D4C;'>{_fmt(step_durations[i])}</span>"
            )
        elif label == current_label:
            icon, icon_color  = "▶", "#F5640A"
            label_color, weight = "#D8E6F2", "600"
            right = (
                "<span style='font-family:JetBrains Mono,monospace;"
                "font-size:.65rem;color:#F5640A;'>running…</span>"
            )
        else:
            icon, icon_color  = "○", "#1E2D3A"
            label_color, weight = "#364C5C", "400"
            right = (
                f"<span style='font-family:JetBrains Mono,monospace;"
                f"font-size:.65rem;color:#1A2830;'>~{_fmt(est)}</span>"
            )

        rows += f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:10px 18px;border-bottom:1px solid rgba(255,255,255,.035);">
          <div style="display:flex;align-items:center;gap:14px;">
            <span style="font-family:'Chakra Petch',monospace;font-size:.78rem;
                         color:{icon_color};font-weight:700;width:14px;">{icon}</span>
            <span style="font-family:'Figtree',sans-serif;font-size:.84rem;
                         color:{label_color};font-weight:{weight};">{label}</span>
          </div>
          {right}
        </div>"""

    steps_ph.markdown(f"""
    <div style="border:1px solid rgba(255,255,255,.06);border-radius:12px;
                overflow:hidden;margin:6px 0 18px;background:rgba(11,19,32,.5);">
      <div style="padding:9px 18px;border-bottom:1px solid rgba(255,255,255,.05);">
        <span style="font-family:'Chakra Petch',monospace;font-size:.5rem;font-weight:600;
                     letter-spacing:.18em;text-transform:uppercase;color:#364C5C;">
          Pipeline · {completed} / {n} steps complete
        </span>
      </div>
      {rows}
    </div>
    """, unsafe_allow_html=True)

    remaining_est = sum(_STEPS[i][2] for i in range(completed, n))
    timer_ph.markdown(f"""
    <div style="text-align:center;padding-bottom:16px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                   color:#364C5C;letter-spacing:.08em;">
        Elapsed {_fmt(elapsed)} · ETA ~{_fmt(remaining_est)}
      </span>
    </div>
    """, unsafe_allow_html=True)


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
    _, col, _ = st.columns([1, 2.2, 1])
    with col:
        header_ph = st.empty()
        bar_ph    = st.empty()
        steps_ph  = st.empty()
        timer_ph  = st.empty()
        error_ph  = st.empty()

        start          = time.time()
        completed      = 0
        step_durations: list[float] = []

        def _tick(label: str) -> None:
            _draw(header_ph, bar_ph, steps_ph, timer_ph,
                  completed, label, time.time() - start, step_durations)

        def _advance(step_start: float) -> None:
            nonlocal completed
            step_durations.append(time.time() - step_start)
            completed += 1

        # ── Step 1: Ingestion (fatal on failure) ──────────────────────────────
        _tick("Fetching Audio")
        t0 = time.time()
        try:
            from services.ingestion import Ingestion
            audio: AudioBuffer = Ingestion().load(source)
        except Exception as exc:
            error_ph.error(f"Could not load audio: {exc}")
            return
        _advance(t0)

        # ── Step 2: Transcription ─────────────────────────────────────────────
        _tick("Transcribing Lyrics")
        t0 = time.time()
        transcript = []
        try:
            from services.transcription import Transcription
            transcript = Transcription().transcribe(audio)
        except Exception:
            pass
        _advance(t0)

        # ── Step 3: Structure analysis ────────────────────────────────────────
        _tick("Analysing Structure")
        t0 = time.time()
        structure = None
        try:
            from services.analysis import Analysis
            structure = Analysis().analyze(audio)
        except Exception:
            pass
        _advance(t0)

        # ── Step 4: Forensics ─────────────────────────────────────────────────
        _tick("Forensic Scan")
        t0 = time.time()
        forensics = None
        try:
            from services.forensics import Forensics
            forensics = Forensics().analyze(audio)
        except Exception:
            pass
        _advance(t0)

        # ── Step 5: Compliance ────────────────────────────────────────────────
        _tick("Compliance Audit")
        t0 = time.time()
        compliance = None
        try:
            from services.compliance import Compliance
            sections = structure.sections if structure else []
            beats    = structure.beats    if structure else []
            compliance = Compliance().check(audio, transcript, sections, beats)
        except Exception:
            pass
        _advance(t0)

        # ── Step 6: Authorship ────────────────────────────────────────────────
        _tick("Authorship Check")
        t0 = time.time()
        authorship = None
        try:
            from services.authorship import Authorship
            authorship = Authorship().analyze(transcript)
        except Exception:
            pass
        _advance(t0)

        # ── Step 7: Track discovery ───────────────────────────────────────────
        _tick("Track Discovery")
        t0 = time.time()
        similar = []
        try:
            from services.discovery import Discovery
            title  = structure.metadata.get("title", "")  if structure else ""
            artist = structure.metadata.get("artist", "") if structure else ""
            similar = Discovery().find_similar(title, artist) or []
        except Exception:
            pass
        _advance(t0)

        # ── Step 8: Legal links ───────────────────────────────────────────────
        _tick("Legal Links")
        t0 = time.time()
        legal = None
        try:
            from services.legal import Legal
            title  = structure.metadata.get("title", "")  if structure else ""
            artist = structure.metadata.get("artist", "") if structure else ""
            legal = Legal().get_links(title, artist)
        except Exception:
            pass
        _advance(t0)

        # ── All done — render final state then transition ─────────────────────
        _draw(header_ph, bar_ph, steps_ph, timer_ph,
              completed, "", time.time() - start, step_durations)

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
