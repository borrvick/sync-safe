"""
Sync-Safe Forensic Portal
Hugging Face ZeroGPU — Stateless, no database.

app.py is intentionally thin: page config, session state, CSS injection,
and routing. All rendering logic lives in ui/pages/. All business logic
lives in pipeline.py and services/.
"""
import base64
from pathlib import Path

import streamlit as st

from ui.styles import STYLES

# ── Assets ────────────────────────────────────────────────────────────────────

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
_LOGO_B64  = base64.b64encode(_LOGO_PATH.read_bytes()).decode()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sync-Safe Forensic Portal",
    page_icon=str(_LOGO_PATH),
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(STYLES, unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
# audio:    AudioBuffer | None — ingested on landing page submit
# analysis: AnalysisResult | None — computed on first report page render
# page:     "landing" | "report"
# start_time, player_key — Audio State Manager (clickable timestamps)

for key, default in [
    ("page",        "landing"),
    ("audio",       None),
    ("analysis",    None),
    ("start_time",  0),
    ("player_key",  0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Routing ───────────────────────────────────────────────────────────────────

if st.session_state.page == "landing":
    from ui.pages.landing import render_landing
    render_landing(_LOGO_B64)
    st.stop()

# ── Report page ───────────────────────────────────────────────────────────────

audio = st.session_state.audio
if audio is None:
    st.session_state.page = "landing"
    st.rerun()

# Run analysis once and cache in session state
if st.session_state.analysis is None:
    from pipeline import Pipeline
    with st.spinner("Transcribing lyrics…"):
        pipeline = Pipeline()
        # Run transcription + structure first so we can show partial progress
        transcript = pipeline._run_step(
            "transcription", lambda: pipeline._transcription.transcribe(audio)
        )
    with st.spinner("Analysing structure & forensics…"):
        structure  = pipeline._run_step("structure",  lambda: pipeline._structure.analyze(audio))
        forensics  = pipeline._run_step("forensics",  lambda: pipeline._forensics.analyze(audio))
    with st.spinner("Running compliance audit & authorship check…"):
        sections   = structure.sections if structure else []
        beats      = structure.beats    if structure else []
        compliance = pipeline._run_step(
            "compliance",
            lambda: pipeline._compliance.check(audio, transcript or [], sections, beats),
        )
        authorship = pipeline._run_step(
            "authorship",
            lambda: pipeline._authorship.analyze(transcript or []),
        )
    with st.spinner("Querying Last.fm…"):
        title  = structure.metadata.get("title", "")  if structure else ""
        artist = structure.metadata.get("artist", "") if structure else ""
        similar = pipeline._run_step(
            "discovery", lambda: pipeline._discovery.find_similar(title, artist)
        )
        legal = pipeline._run_step(
            "legal", lambda: pipeline._legal.get_links(title, artist)
        )

    from core.models import AnalysisResult
    st.session_state.analysis = AnalysisResult(
        audio=audio,
        structure=structure,
        forensics=forensics,
        transcript=transcript or [],
        compliance=compliance,
        authorship=authorship,
        similar_tracks=similar or [],
        legal=legal,
    )

from ui.pages.report import render_report
render_report(_LOGO_B64, audio, st.session_state.analysis)
