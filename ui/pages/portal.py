"""
ui/pages/portal.py
The Portal — the track analysis tool itself.
Separated from the landing/marketing page so the landing page can focus
on communication and the portal page can focus on the task.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ui.components import eq_bars
from ui.nav import render_site_nav, render_site_footer

_DEBUG_DIR = Path(__file__).parent.parent.parent / "local" / "debug"


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
                if st.button("Initiate Scan →", type="primary",
                             use_container_width=True, key="run_url"):
                    if url:
                        _submit_source(url)
                    else:
                        st.warning("Paste a YouTube URL first.")
            else:
                uploaded = st.file_uploader(
                    "file",
                    type=["mp3", "wav", "flac", "m4a", "ogg"],
                    label_visibility="collapsed",
                )
                if st.button("Initiate Scan →", type="primary",
                             use_container_width=True, key="run_upload",
                             disabled=uploaded is None):
                    if uploaded:
                        _submit_source(uploaded)

            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;gap:8px;
                        margin-top:16px;padding:8px 12px;border-radius:6px;
                        background:var(--surface-raised, rgba(255,255,255,.04));
                        border:1px solid var(--border-hr);">
              <svg aria-hidden="true" width="12" height="12" viewBox="0 0 24 24" fill="none"
                   stroke="var(--dim)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
              </svg>
              <span style="font-family:'JetBrains Mono',monospace;font-size:.52rem;
                           color:var(--dim);font-weight:500;letter-spacing:.1em;
                           text-transform:uppercase;">
                Ephemeral processing &mdash; no audio stored &middot; not used for training &middot; deleted after scan
              </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    render_site_footer()


def _submit_source(source: object) -> None:
    """Store source in session state and route to the loading page."""
    st.session_state.source   = source
    st.session_state.audio    = None
    st.session_state.analysis = None
    st.session_state.page     = "loading"
    st.rerun()
