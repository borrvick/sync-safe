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

from core.config import get_settings
from core.logging import DEFAULT_LOG_DIR, LogCleaner
from ui.styles import STYLES

# ── Logging setup (runs once per process) ────────────────────────────────────
# Module-level flag ensures LogCleaner runs exactly once per process, not once
# per browser session. session_state would run it once per tab, which causes
# concurrent unlink() races when multiple users open the app simultaneously.
_log_cleaner_ran = False
if not _log_cleaner_ran:
    _log_dir = get_settings().log_dir or DEFAULT_LOG_DIR
    LogCleaner(_log_dir).clean()
    _log_cleaner_ran = True

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
    ("source",      None),
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

if st.session_state.page == "loading":
    from ui.pages.loading import render_loading
    render_loading(st.session_state.source)
    st.stop()

# ── Report page ───────────────────────────────────────────────────────────────

audio    = st.session_state.audio
analysis = st.session_state.analysis

if audio is None or analysis is None:
    st.session_state.page = "landing"
    st.rerun()
    st.stop()

from ui.pages.report import render_report
render_report(_LOGO_B64, audio, analysis)
