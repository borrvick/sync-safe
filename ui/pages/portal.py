"""
ui/pages/portal.py
The Portal — the track analysis tool itself.
Separated from the landing/marketing page so the landing page can focus
on communication and the portal page can focus on the task.
"""
from __future__ import annotations

import html

import streamlit as st

from services.metadata_validator import validate_isrc, validate_splits
from ui.components import eq_bars
from ui.nav import render_site_nav, render_site_footer

# Known PRO names shown in the dropdown — ordered by global prevalence
_PRO_OPTIONS: list[str] = [
    "",
    "ASCAP (US)",
    "BMI (US)",
    "SESAC (US)",
    "PRS for Music (UK)",
    "GEMA (Germany)",
    "SACEM (France)",
    "SOCAN (Canada)",
    "APRA AMCOS (Australia)",
    "STIM (Sweden)",
    "TONO (Norway)",
    "KODA (Denmark)",
    "Teosto (Finland)",
    "Buma/Stemra (Netherlands)",
    "SABAM (Belgium)",
    "SIAE (Italy)",
    "SGAE (Spain)",
    "ECAD (Brazil)",
    "JASRAC (Japan)",
    "KOMCA (South Korea)",
    "SACM (Mexico)",
    "Other",
]


def render_portal() -> None:
    """Render the track submission portal."""
    # Initialise metadata session state before access
    if "intake_require_splits" not in st.session_state:
        st.session_state.intake_require_splits = False
    if "intake_metadata" not in st.session_state:
        st.session_state.intake_metadata = {}

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
            Paste a YouTube, Bandcamp, or SoundCloud URL — or upload an audio file. Analysis takes 60–90 seconds.
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
                ["🔗  Stream URL", "📁  Upload File"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if mode == "🔗  Stream URL":
                url = st.text_input(
                    "url",
                    placeholder="YouTube, Bandcamp, SoundCloud, or direct audio URL",
                    label_visibility="collapsed",
                )
                st.markdown("""
                <div style="display:flex;align-items:center;gap:6px;margin:6px 0 10px;flex-wrap:wrap;">
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.5rem;
                               color:var(--dim);letter-spacing:.08em;text-transform:uppercase;
                               margin-right:2px;">Accepts</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.52rem;font-weight:600;
                               padding:2px 7px;border-radius:3px;
                               background:rgba(255,0,0,.08);color:#FF4444;
                               border:1px solid rgba(255,0,0,.18);">YouTube</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.52rem;font-weight:600;
                               padding:2px 7px;border-radius:3px;
                               background:rgba(29,129,96,.1);color:#1DB954;
                               border:1px solid rgba(29,129,96,.25);">Bandcamp</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.52rem;font-weight:600;
                               padding:2px 7px;border-radius:3px;
                               background:rgba(255,85,0,.08);color:#FF5500;
                               border:1px solid rgba(255,85,0,.2);">SoundCloud</span>
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.52rem;font-weight:600;
                               padding:2px 7px;border-radius:3px;
                               background:rgba(245,100,10,.08);color:var(--dim);
                               border:1px solid var(--border-hr);">.mp3 .wav .flac</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption(
                    "⚠️ YouTube audio is compressed (~128 kbps AAC) and may reduce "
                    "accuracy of spectral and AI detection signals. "
                    "For best results, upload a WAV or FLAC file directly."
                )
                if st.button("Initiate Scan →", type="primary",
                             use_container_width=True, key="run_url"):
                    if url:
                        _submit_source(url)
                    else:
                        st.warning("Paste a URL first.")
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

        # ── Rights Metadata Pre-flight ────────────────────────────────────────
        _render_metadata_preflight()

        st.markdown("<br>", unsafe_allow_html=True)

    render_site_footer()


def _render_metadata_preflight() -> None:
    """
    Collapsible rights metadata intake form (optional unless agency mode enabled).

    Stores validated metadata in st.session_state.intake_metadata so that
    loading.py can attach it to the AnalysisResult.
    """
    with st.expander("Rights Metadata  (optional — required for agency placements)"):
        st.markdown(
            "<div style='font-family:\"Figtree\",sans-serif;font-size:.82rem;"
            "color:var(--muted);margin-bottom:12px;'>"
            "Fill in rights details for your split sheet. ISRC and writer splits "
            "are validated before the scan starts."
            "</div>",
            unsafe_allow_html=True,
        )

        require_splits = st.checkbox(
            "Require signed split sheet (agency tier — blocks scan until complete)",
            key="intake_require_splits",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            title = st.text_input("Track title", key="intake_title", placeholder="e.g. Blinding Lights")
            pro = st.selectbox("PRO affiliation", _PRO_OPTIONS, key="intake_pro")
        with col_b:
            artist = st.text_input("Artist name", key="intake_artist", placeholder="e.g. The Weeknd")
            publisher = st.text_input("Publisher name", key="intake_publisher", placeholder="e.g. Warner Chappell")

        isrc_raw = st.text_input(
            "ISRC  (e.g. US-ABC-23-12345)",
            key="intake_isrc",
            placeholder="CC-XXX-YY-NNNNN",
        )

        st.markdown(
            "<div style='font-family:\"Figtree\",sans-serif;font-size:.8rem;"
            "color:var(--muted);margin:12px 0 4px;'>Writer splits (%) — must sum to 100</div>",
            unsafe_allow_html=True,
        )
        num_writers = st.number_input(
            "Number of writers",
            min_value=1,
            max_value=10,
            value=1,
            key="intake_num_writers",
        )
        splits: list[float] = []
        for i in range(int(num_writers)):
            v = st.number_input(
                f"Writer {i + 1} split (%)",
                min_value=0.0,
                max_value=100.0,
                value=round(100.0 / int(num_writers), 2),
                step=0.01,
                format="%.2f",
                key=f"intake_split_{i}",
            )
            splits.append(float(v))

        # ── Inline validation feedback ────────────────────────────────────────
        _show_preflight_feedback(
            title=title,
            artist=artist,
            pro=pro,
            publisher=publisher,
            isrc=isrc_raw,
            splits=splits,
            require_splits=require_splits,
        )

        # Persist metadata for loading.py
        st.session_state.intake_metadata = {
            "title": title.strip(),
            "artist": artist.strip(),
            "pro": pro.strip(),
            "publisher": publisher.strip(),
            "isrc": isrc_raw.strip(),
            "splits": splits,
            "require_splits": require_splits,
        }


def _badge_span(label: str, value: str, ok: bool, suffix: str = "") -> str:
    """Return an HTML <span> badge for a single validation field. Pure — no I/O."""
    color      = "var(--accent)" if ok else "var(--danger)"
    icon_char  = "✓" if ok else "✗"
    icon_label = "pass" if ok else "fail"
    escaped    = html.escape(value.strip() or "—")
    return (
        f'<span style="color:{color};margin-right:12px;">'
        f'<span aria-label="{icon_label}">{icon_char}</span>'
        f' {label}: <strong>{escaped}</strong>{suffix}</span>'
    )


def _build_badge_spans(
    title: str,
    artist: str,
    pro: str,
    publisher: str,
    isrc: str,
    splits: list[float],
) -> tuple[list[str], float]:
    """Build badge span strings and return (spans, split_sum). Pure — no I/O."""
    lines: list[str] = []
    for field_label, value in [
        ("Title", title), ("Artist", artist), ("PRO", pro), ("Publisher", publisher)
    ]:
        lines.append(_badge_span(field_label, value, ok=bool(value.strip())))

    if isrc.strip():
        isrc_ok = validate_isrc(isrc)
        lines.append(_badge_span(
            "ISRC", isrc.strip(), ok=isrc_ok,
            suffix="" if isrc_ok else " — invalid format",
        ))

    splits_ok = validate_splits(splits)
    split_sum = round(sum(splits), 2)
    lines.append(_badge_span(
        "Splits sum", f"{split_sum}%", ok=splits_ok,
        suffix="" if splits_ok else " — must equal 100%",
    ))
    return lines, split_sum


def _show_preflight_feedback(
    title: str,
    artist: str,
    pro: str,
    publisher: str,
    isrc: str,
    splits: list[float],
    require_splits: bool,
) -> None:
    """Render inline field-level validation badges (no I/O — reads session state only)."""
    any_filled = any([title.strip(), artist.strip(), pro.strip(), publisher.strip(), isrc.strip()])
    if not any_filled and not require_splits:
        return  # nothing entered — don't show noise

    lines, split_sum = _build_badge_spans(title, artist, pro, publisher, isrc, splits)

    st.markdown(
        "<div style='font-family:\"JetBrains Mono\",monospace;font-size:.72rem;"
        "line-height:1.9;margin-top:10px;padding:10px 12px;border-radius:6px;"
        "background:var(--surface-raised,rgba(255,255,255,.03));"
        "border:1px solid var(--border-hr);'>"
        + "".join(lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    if require_splits and not validate_splits(splits):
        st.error(
            f"Agency mode: writer splits must sum to 100% (currently {split_sum}%). "
            "Correct the splits before initiating the scan."
        )


def _submit_source(source: object) -> None:
    """Store source in session state and route to the loading page."""
    st.session_state.source   = source
    st.session_state.audio    = None
    st.session_state.analysis = None
    st.session_state.page     = "loading"
    st.rerun()
