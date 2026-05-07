"""
ui/nav.py
Shared site navigation bar and footer for all public-facing pages.

Uses st.button(on_click=_go) for internal navigation — clicks travel over
the existing WebSocket connection, no page reload, no flash.

External links (ASCAP, BMI, etc.) remain as <a href target="_blank">.

CSS in styles.py targets data-nav-btn="nav" / "footer" attributes stamped
by JS in app.py to style the buttons as plain text links.
"""
from __future__ import annotations

import streamlit as st

from ui.components import eq_bars


def _go(page: str) -> None:
    """on_click callback — sets page before Streamlit's automatic rerun fires."""
    st.session_state.page = page


# ── Navigation bar ─────────────────────────────────────────────────────────────

def render_site_nav(current_page: str) -> None:
    """Render the full-width site navigation bar."""
    eq = eq_bars(6, color="var(--accent)", h=24)

    with st.container():
        st.markdown('<div class="ss-nav-marker"></div>', unsafe_allow_html=True)

        col_brand, col_hiw, col_leg, col_cta = st.columns(
            [2.8, 1.1, 0.8, 1.3], vertical_alignment="center", gap="small"
        )

        with col_brand:
            st.markdown(f"""
            <a href="/" class="ss-brand-link">
              <div class="ss-brand">
                <div class="ss-eq">{eq}</div>
                <div>
                  <div class="ss-brand-name">SYNC-SAFE™</div>
                  <div class="ss-brand-sub">Forensic Portal</div>
                </div>
              </div>
            </a>""", unsafe_allow_html=True)

        with col_hiw:
            st.button(
                "How it Works", key="nav_hiw",
                on_click=_go, args=("how_it_works",),
                disabled=(current_page == "how_it_works"),
                use_container_width=True,
            )
        with col_leg:
            st.button(
                "Legal", key="nav_leg",
                on_click=_go, args=("legal",),
                disabled=(current_page == "legal"),
                use_container_width=True,
            )
        with col_cta:
            st.button(
                "Launch Portal →", key="nav_portal",
                on_click=_go, args=("portal",),
                type="primary",
                use_container_width=True,
            )

    st.markdown(
        '<hr style="margin:0 0 8px;border:none;border-top:1px solid var(--border);">',
        unsafe_allow_html=True,
    )


# ── Footer ─────────────────────────────────────────────────────────────────────

def render_site_footer() -> None:
    """Render the full-width site footer."""
    eq = eq_bars(5, color="var(--accent)", h=16)

    st.markdown(
        '<hr style="margin:28px 0 0;border:none;border-top:1px solid var(--border);">',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="ss-footer-marker"></div>', unsafe_allow_html=True)

        col_brand, col_product, col_rights, col_legal = st.columns([1.6, 1, 1, 1])

        with col_brand:
            st.markdown(f"""
            <a href="/" class="ss-brand-link">
              <div style="display:inline-flex;align-items:flex-end;gap:6px;margin-bottom:4px;">
                <div style="display:flex;align-items:flex-end;gap:2px;height:16px;">{eq}</div>
                <div class="ss-ft-brand-name">SYNC-SAFE™</div>
              </div>
              <div class="ss-ft-brand-sub">Forensic Portal</div>
            </a>
            <div class="ss-tagline">
              Detect AI authorship · Audit sync compliance · Flag lyric risks
              before a track costs you a placement.
            </div>
            <div class="ss-badges">
              <span class="ss-badge">⬡ Hugging Face ZeroGPU</span>
              <span class="ss-badge">⬡ Stateless Architecture</span>
              <span class="ss-badge">⬡ No Audio Stored</span>
            </div>""", unsafe_allow_html=True)

        with col_product:
            st.markdown('<div class="ss-col-title">Product</div>', unsafe_allow_html=True)
            st.button("Home",         key="ft_home",   on_click=_go, args=("landing",),      use_container_width=True)
            st.button("The Portal",   key="ft_portal", on_click=_go, args=("portal",),       use_container_width=True)
            st.button("How it Works", key="ft_hiw",    on_click=_go, args=("how_it_works",), use_container_width=True)

        with col_rights:
            st.markdown("""
            <div class="ss-col-title">Rights Resources</div>
            <a class="ss-lnk ss-lnk-ext" href="https://www.ascap.com/repertory"           target="_blank" rel="noopener noreferrer">ASCAP Repertory</a>
            <a class="ss-lnk ss-lnk-ext" href="https://www.bmi.com/search/"               target="_blank" rel="noopener noreferrer">BMI Repertoire</a>
            <a class="ss-lnk ss-lnk-ext" href="https://www.sesac.com/#!/repertory/search" target="_blank" rel="noopener noreferrer">SESAC Repertory</a>
            <a class="ss-lnk ss-lnk-ext" href="https://www.globalmusicrights.com/search"  target="_blank" rel="noopener noreferrer">GMR Search</a>
            """, unsafe_allow_html=True)

        with col_legal:
            st.markdown('<div class="ss-col-title">Legal</div>', unsafe_allow_html=True)
            st.button("Copyright & IP",   key="ft_copy",    on_click=_go, args=("legal",), use_container_width=True)
            st.button("Privacy Policy",   key="ft_privacy", on_click=_go, args=("legal",), use_container_width=True)
            st.button("Terms of Service", key="ft_tos",     on_click=_go, args=("legal",), use_container_width=True)

        st.markdown("""
        <div class="ss-bottom">
          <span class="ss-copy">© 2026 Sync-Safe. All rights reserved.</span>
          <div class="ss-trust">
            <span>Stateless</span><span>·</span>
            <span>No Audio Stored</span><span>·</span>
            <span>ZeroGPU</span>
          </div>
        </div>""", unsafe_allow_html=True)
