"""
ui/pages/landing.py
Landing page: hero, input card, and feature bento grid.
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from ui.components import eq_bars

# ── Bento card data ───────────────────────────────────────────────────────────

_CARDS = [
    {"cls": "bc-c2pa",   "theme": "tb", "icon": "🔬", "cat": "VERIFICATION",      "label": "C2PA Manifest Check",    "desc": "Reads <code>c2pa-python</code> manifests. If a <strong>born-AI</strong> assertion is present, it provides a cryptographically certified AI verdict — the strongest possible signal."},
    {"cls": "bc-groove", "theme": "ta", "icon": "🥁", "cat": "IBI VARIANCE",       "label": "Groove Analysis",         "desc": "Measures inter-beat timing deviations in microseconds. Humans rush and drag <strong>(&gt;8ms²)</strong>. AI generators lock to a perfect grid <strong>(&lt;0.5ms²)</strong>."},
    {"cls": "bc-loop",   "theme": "ta", "icon": "🔁", "cat": "SPECTRAL MATCHING",  "label": "Loop Detection",          "desc": "Splits audio into 4-bar segments and cross-correlates spectral fingerprints. Scores above <strong>0.98</strong> flag stock loops or AI repetition artifacts."},
    {"cls": "bc-struct", "theme": "ta", "icon": "🎼", "cat": "AI ENSEMBLE",        "label": "Structure & Key",         "desc": "Predicts BPM, musical key, and functional section boundaries — intro, verse, chorus, bridge, outro — to help you match scene pacing precisely."},
    {"cls": "bc-sting",  "theme": "ta", "icon": "🔔", "cat": "GALLO-METHOD",       "label": "Sting & Intro Check",     "desc": "Analyses the final 2s RMS energy ratio to detect sting endings that disrupt scene audio. Also flags intro sections exceeding <strong>15s</strong> — the Gallo-Method limit for sync pitches."},
    {"cls": "bc-lyrics", "theme": "tc", "icon": "🎤", "cat": "WHISPER AI",         "label": "Lyric Transcription",     "desc": "Converts vocals to timestamped text via OpenAI Whisper. Essential for flagging explicit content, trademarked phrases, or co-writer obligations before broadcast clearance."},
    {"cls": "bc-auth",   "theme": "tc", "icon": "✍️", "cat": "AUTHORSHIP DETECT", "label": "AI Lyric Detection",      "desc": "Scores lyrics across four linguistic signals — burstiness, vocabulary diversity, rhyme density, repetition — plus a RoBERTa classifier. Returns <strong>Likely Human / Uncertain / Likely AI</strong>."},
    {"cls": "bc-sim",    "theme": "ta", "icon": "🔍", "cat": "DISCOVERY ENGINE",   "label": "Similar Tracks",          "desc": "Uses Last.fm's similarity graph to surface comparable tracks — fully stateless, no database. Each result is resolved to a YouTube preview URL live via yt-dlp."},
    {"cls": "bc-pro",    "theme": "tb", "icon": "⚖️", "cat": "RIGHTS SEARCH",     "label": "PRO Licensing Links",     "desc": "Generates direct search links for ASCAP, BMI, and SESAC repertory databases — so you can identify publishers and rights holders before a single outreach email."},
]

_BENTO = f"""<!DOCTYPE html><html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Figtree:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;padding:2px 0 8px}}
:root{{--bg:#050911;--s1:#0B1320;--s2:#0F1A28;--border:rgba(255,255,255,.06);--accent:#F5640A;--text:#D8E6F2;--muted:#7A95AA;--dim:#364C5C}}
@keyframes eq-a{{0%,100%{{transform:scaleY(.14)}}25%{{transform:scaleY(.88)}}50%{{transform:scaleY(.38)}}75%{{transform:scaleY(.62)}}}}
@keyframes eq-b{{0%,100%{{transform:scaleY(.52)}}25%{{transform:scaleY(.18)}}50%{{transform:scaleY(.92)}}75%{{transform:scaleY(.28)}}}}
@keyframes eq-c{{0%,100%{{transform:scaleY(.72)}}25%{{transform:scaleY(.42)}}50%{{transform:scaleY(.08)}}75%{{transform:scaleY(.82)}}}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:11px}}
.bc-c2pa{{grid-column:1/3}}.bc-groove{{grid-column:3}}.bc-loop{{grid-column:4}}
.bc-struct{{grid-column:1}}.bc-sting{{grid-column:2}}.bc-lyrics{{grid-column:3/5}}
.bc-auth{{grid-column:1/3}}.bc-sim{{grid-column:3}}.bc-pro{{grid-column:4}}
.bc-ph{{display:none}}
.card{{position:relative;border-radius:13px;border:1px solid var(--border);min-height:165px;
       cursor:default;overflow:hidden;
       transition:border-color .25s,box-shadow .25s,transform .3s cubic-bezier(.34,1.56,.64,1)}}
.card:hover{{border-color:rgba(245,100,10,.38);
             box-shadow:0 0 0 1px rgba(245,100,10,.1),0 20px 50px rgba(0,0,0,.6),0 0 28px rgba(245,100,10,.07);
             transform:translateY(-5px)}}
.ta{{background:var(--s1)}}.tb{{background:linear-gradient(140deg,#0D1620 0%,#111E2C 100%)}}
.tc{{background:linear-gradient(140deg,#1E0B00 0%,#2A1200 100%);border-color:rgba(245,100,10,.2)}}
.inner{{position:relative;z-index:1;padding:18px 18px 16px;display:flex;flex-direction:column;justify-content:space-between;height:100%;min-height:165px}}
.cat{{font-family:'Chakra Petch',monospace;font-size:.54rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;
      padding:3px 9px;border-radius:4px;display:inline-block;align-self:flex-start}}
.cat-d{{color:var(--accent);background:rgba(245,100,10,.1);border:1px solid rgba(245,100,10,.2)}}
.cat-a{{color:rgba(245,100,10,.85);background:rgba(245,100,10,.08);border:1px solid rgba(245,100,10,.15)}}
.bot{{display:flex;flex-direction:column;gap:5px}}
.eq{{display:flex;align-items:flex-end;gap:2px;height:20px;margin-bottom:6px;opacity:.45}}
.e{{width:2.5px;border-radius:1px;transform-origin:bottom;background:var(--accent)}}
.ew{{background:rgba(245,100,10,.7)}}
.icon{{font-size:1.5rem;line-height:1;margin-bottom:3px}}
.ttl{{font-family:'Chakra Petch',monospace;font-size:.84rem;font-weight:600;letter-spacing:.01em;color:var(--text)}}
.ttl-a{{color:rgba(245,180,120,.92)}}
.ov{{position:absolute;inset:0;border-radius:inherit;background:rgba(3,7,14,.96);
     backdrop-filter:blur(10px);padding:18px;display:flex;flex-direction:column;gap:7px;
     opacity:0;transform:translateY(7px);transition:opacity .2s,transform .2s;z-index:2;pointer-events:none}}
.card:hover .ov{{opacity:1;transform:translateY(0)}}
.ov-cat{{font-family:'Chakra Petch',monospace;font-size:.54rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--accent)}}
.ov-ttl{{font-family:'Chakra Petch',monospace;font-size:.86rem;font-weight:600;color:var(--text);margin-bottom:1px}}
.ov-desc{{font-family:'Figtree',sans-serif;font-size:.74rem;color:var(--muted);line-height:1.57}}
.ov-desc strong{{color:rgba(245,160,80,.95);font-weight:600}}
.ov-desc code{{font-family:'JetBrains Mono',monospace;font-size:.68rem;background:rgba(245,100,10,.12);color:var(--accent);padding:1px 5px;border-radius:3px}}
.ph{{border-radius:13px;border:1px dashed rgba(245,100,10,.12);min-height:165px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}}
.ph-t{{font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:500;color:var(--dim);letter-spacing:.14em;text-transform:uppercase}}
</style></head><body>
<div class="grid" id="g"></div>
<script>
const C={json.dumps(_CARDS)};
const A=['eq-a','eq-b','eq-c'];
function bars(n,w){{let h='';for(let i=0;i<n;i++){{const d=(1.1+(i%3)*.22).toFixed(2),dl=(i*.07).toFixed(2);h+=`<div class="e${{w?' ew':''}}" style="height:20px;animation:${{A[i%3]}} ${{d}}s ease-in-out ${{dl}}s infinite;"></div>`;}}return h;}}
const g=document.getElementById('g');
C.forEach(c=>{{
  const el=document.createElement('div');
  el.className=`card ${{c.cls}} ${{c.theme}}`;
  const isA=c.theme==='tc';
  el.innerHTML=`
    <div class="inner">
      <span class="cat ${{isA?'cat-a':'cat-d'}}">${{c.cat}}</span>
      <div class="bot">
        <div class="eq">${{bars(9,isA)}}</div>
        <div class="icon">${{c.icon}}</div>
        <div class="ttl ${{isA?'ttl-a':''}}">${{c.label}}</div>
      </div>
    </div>
    <div class="ov">
      <div class="ov-cat">${{c.cat}}</div>
      <div class="ov-ttl">${{c.label}}</div>
      <div class="ov-desc">${{c.desc}}</div>
    </div>`;
  g.appendChild(el);
}});
const ph=document.createElement('div');
ph.className='ph bc-ph';
ph.innerHTML='<div style="font-size:1.1rem;opacity:.2">⊕</div><div class="ph-t">More Features Coming</div>';
g.appendChild(ph);
</script></body></html>"""


def render_landing(logo_b64: str) -> None:
    """Render the landing page and handle source submission."""
    _, col, _ = st.columns([1, 2.2, 1])
    with col:
        # Header bar
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:28px 0 44px;animation:fade-up .5s ease both;">
          <div style="display:flex;align-items:center;gap:12px;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="height:32px;width:auto;display:block;">
            <div>
              <div style="font-family:'Chakra Petch',monospace;font-size:.9rem;
                          font-weight:700;color:#F5640A;letter-spacing:.14em;">SYNC-SAFE</div>
              <div style="font-family:'Chakra Petch',monospace;font-size:.5rem;
                          font-weight:500;color:#364C5C;letter-spacing:.22em;
                          text-transform:uppercase;margin-top:1px;">Forensic Portal</div>
            </div>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                      color:#364C5C;letter-spacing:.05em;">ZeroGPU · Stateless · v2</div>
        </div>
        """, unsafe_allow_html=True)

        # Hero
        st.markdown(f"""
        <div style="text-align:center;padding-bottom:52px;animation:fade-up .65s ease .1s both;">
          <div style="display:flex;align-items:flex-end;justify-content:center;
                      gap:3px;height:44px;margin-bottom:36px;opacity:.55;">
            {eq_bars(28, "#F5640A", 44)}
          </div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.68rem;font-weight:500;
                      color:#364C5C;letter-spacing:.32em;text-transform:uppercase;
                      margin-bottom:18px;">Music Sync Clearance Intelligence</div>
          <div style="font-family:'Chakra Petch',monospace;
                      font-size:clamp(2.8rem,7vw,5rem);font-weight:700;
                      color:#D8E6F2;line-height:.88;letter-spacing:-.03em;
                      margin-bottom:6px;">FORENSIC</div>
          <div style="font-family:'Chakra Petch',monospace;
                      font-size:clamp(2.8rem,7vw,5rem);font-weight:700;
                      color:#F5640A;line-height:.88;letter-spacing:-.03em;
                      margin-bottom:30px;">PORTAL</div>
          <div style="font-family:'Figtree',sans-serif;font-size:.96rem;font-weight:400;
                      color:#7A95AA;line-height:1.65;max-width:480px;margin:0 auto;">
            Verify AI origin · Decode structure · Clear rights<br>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.76rem;
                         color:#364C5C;">→ Built for Music Supervisors who need certainty.</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Input card
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom:22px;animation:fade-up .7s ease .2s both;">
              <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                          letter-spacing:.2em;text-transform:uppercase;color:#364C5C;
                          display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span>▶</span><span>Analyse a Track</span>
                <div style="flex:1;height:1px;background:rgba(255,255,255,.055);"></div>
              </div>
              <div style="font-family:'Chakra Petch',monospace;font-size:1.5rem;font-weight:700;
                          color:#D8E6F2;letter-spacing:-.02em;line-height:1.15;">
                Run your forensic<br><span style="color:#F5640A;">audit.</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            mode = st.radio("mode", ["🔗  YouTube URL", "📁  Upload File"],
                            horizontal=True, label_visibility="collapsed")

            if mode == "🔗  YouTube URL":
                url = st.text_input("url", placeholder="https://youtube.com/watch?v=...",
                                    label_visibility="collapsed")
                if st.button("Initiate Scan →", type="primary",
                             use_container_width=True, key="run_url"):
                    if url:
                        _submit_source(url)
                    else:
                        st.warning("Paste a YouTube URL first.")
            else:
                uploaded = st.file_uploader("file", type=["mp3", "wav", "flac", "m4a", "ogg"],
                                            label_visibility="collapsed")
                if st.button("Initiate Scan →", type="primary",
                             use_container_width=True, key="run_upload",
                             disabled=uploaded is None):
                    if uploaded:
                        _submit_source(uploaded)

        st.markdown("""
        <p style="text-align:center;font-family:'Chakra Petch',monospace;font-size:.56rem;
                  color:#364C5C;font-weight:500;letter-spacing:.14em;text-transform:uppercase;
                  margin-top:10px;">No audio stored · API keys via HF Secrets</p>
        <br>
        """, unsafe_allow_html=True)

        # Feature grid label
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;
                    animation:fade-up .75s ease .3s both;">
          <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                      letter-spacing:.2em;text-transform:uppercase;color:#364C5C;white-space:nowrap;">
            ◈ Feature Matrix
          </div>
          <div style="flex:1;height:1px;background:rgba(255,255,255,.055);"></div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.54rem;color:#364C5C;
                      letter-spacing:.1em;white-space:nowrap;">Hover to inspect</div>
        </div>
        """, unsafe_allow_html=True)

        components.html(_BENTO, height=570, scrolling=False)

        st.markdown("""
        <div style="text-align:center;margin-top:36px;padding-bottom:20px;">
          <span style="font-family:'Chakra Petch',monospace;font-size:.54rem;color:#364C5C;
                       font-weight:500;letter-spacing:.18em;text-transform:uppercase;">
            Hugging Face ZeroGPU · Stateless Architecture · No Audio Stored
          </span>
        </div>
        """, unsafe_allow_html=True)


def _submit_source(source) -> None:
    """Ingest the audio source and transition to the report page."""
    from services.ingestion import Ingestion
    from core.exceptions import SyncSafeError

    try:
        with st.spinner("Fetching audio…"):
            audio = Ingestion().load(source)
        st.session_state.audio          = audio
        st.session_state.analysis       = None
        st.session_state.page           = "report"
        st.rerun()
    except SyncSafeError as exc:
        st.error(f"Could not load audio: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
