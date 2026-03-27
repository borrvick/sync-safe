"""
ui/pages/portal.py
The Portal — the track analysis tool itself.
Separated from the landing/marketing page so the landing page can focus
on communication and the portal page can focus on the task.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from ui.components import eq_bars
from ui.nav import render_site_nav, render_site_footer

_DEBUG_DIR = Path(__file__).parent.parent.parent / "local" / "debug"
_LABELS_FILE = _DEBUG_DIR / "labels.json"
_LABEL_OPTIONS = [
    "— unrated —",
    "100% AI",
    "AI Cover",
    "Heavily Sampled/Loops",
    "May Contain One-Shot Samples",
    "Modern Production Practices",
    "RAW",
]


def render_portal() -> None:
    """Render the track submission portal."""
    st.markdown(
        '<a href="#main-content" class="skip-link">Skip to main content</a>',
        unsafe_allow_html=True,
    )
    render_site_nav("portal")

    _, col, _ = st.columns([1, 2.2, 1])
    with col:
        _eq = eq_bars(6, color="#F5640A", h=18)
        st.markdown(f"""
        <div style="padding:40px 0 32px;animation:fade-up .5s ease both;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
            <div style="display:flex;align-items:flex-end;gap:2px;height:18px;">{_eq}</div>
            <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                        letter-spacing:.2em;text-transform:uppercase;color:var(--dim);">
              Sync-Safe Forensic Portal
            </div>
          </div>
          <div style="font-family:'Chakra Petch',monospace;font-size:clamp(1.6rem,4vw,2.4rem);
                      font-weight:700;color:var(--text);line-height:1.1;letter-spacing:-.02em;">
            Run your forensic<br>
            <span style="color:var(--accent);">audit.</span>
          </div>
          <div style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                      margin-top:12px;line-height:1.55;">
            Paste a YouTube URL or upload an audio file. Analysis takes 60–90 seconds.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span id="main-content" tabindex="-1"></span>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom:22px;">
              <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                          letter-spacing:.2em;text-transform:uppercase;color:var(--dim);
                          display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span>▶</span><span>Analyse a Track</span>
                <div style="flex:1;height:1px;background:var(--border-hr);"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            mode = st.radio(
                "mode",
                ["🔗  YouTube URL", "📁  Upload File"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if mode == "🔗  YouTube URL":
                url = st.text_input(
                    "url",
                    placeholder="https://youtube.com/watch?v=...",
                    label_visibility="collapsed",
                )
                if os.getenv("DEBUG_ANALYSIS"):
                    track_label = st.selectbox(
                        "Track category",
                        _LABEL_OPTIONS,
                        key="portal_label_url",
                    )
                else:
                    track_label = None
                if st.button("Initiate Scan →", type="primary",
                             use_container_width=True, key="run_url"):
                    if url:
                        _submit_source(url, track_label)
                    else:
                        st.warning("Paste a YouTube URL first.")
            else:
                uploaded = st.file_uploader(
                    "file",
                    type=["mp3", "wav", "flac", "m4a", "ogg"],
                    label_visibility="collapsed",
                )
                if os.getenv("DEBUG_ANALYSIS"):
                    track_label = st.selectbox(
                        "Track category",
                        _LABEL_OPTIONS,
                        key="portal_label_upload",
                    )
                else:
                    track_label = None
                if st.button("Initiate Scan →", type="primary",
                             use_container_width=True, key="run_upload",
                             disabled=uploaded is None):
                    if uploaded:
                        _submit_source(uploaded, track_label)

        st.markdown("""
        <p style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:.54rem;
                  color:var(--dim);font-weight:500;letter-spacing:.12em;text-transform:uppercase;
                  margin-top:10px;">
          No audio stored · Ephemeral ZeroGPU processing
        </p>
        <br>
        """, unsafe_allow_html=True)

    render_site_footer()


def _submit_source(source: object, track_label: str | None = None) -> None:
    """Store source in session state and route to the loading page."""
    st.session_state.source        = source
    st.session_state.audio         = None
    st.session_state.analysis      = None
    st.session_state.debug_label   = track_label
    st.session_state.page          = "loading"
    st.rerun()
