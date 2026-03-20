"""
ui/pages/landing.py
Landing page: hero, input card, and feature bento grid.
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from ui.components import eq_bars
from ui.nav import render_site_nav, render_site_footer

# ── Bento card data ───────────────────────────────────────────────────────────

_CARDS = [
    {"cls": "bc-c2pa",   "theme": "tb", "icon": "🔬", "cat": "VERIFICATION",      "label": "C2PA Manifest Check",    "desc": "Reads <code>c2pa-python</code> manifests. If a <strong>born-AI</strong> assertion is present, it provides a cryptographically certified AI verdict — the strongest possible signal."},
    {"cls": "bc-groove", "theme": "ta", "icon": "🥁", "cat": "IBI VARIANCE",       "label": "Groove Analysis",         "desc": "Measures inter-beat timing deviations in microseconds. Humans rush and drag <strong>(&gt;8ms²)</strong>. AI generators lock to a perfect grid <strong>(&lt;0.5ms²)</strong>."},
    {"cls": "bc-loop",   "theme": "ta", "icon": "🔁", "cat": "SPECTRAL MATCHING",  "label": "Loop Detection",          "desc": "Splits audio into 4-bar segments and cross-correlates spectral fingerprints. Scores above <strong>0.98</strong> flag stock loops or AI repetition artifacts."},
    {"cls": "bc-struct", "theme": "ta", "icon": "🎼", "cat": "AI ENSEMBLE",        "label": "Structure & Key",         "desc": "Predicts BPM, musical key, and functional section boundaries — intro, verse, chorus, bridge, outro — to help you match scene pacing precisely."},
    {"cls": "bc-sting",  "theme": "ta", "icon": "🔔", "cat": "SYNC READINESS",     "label": "Sting & Intro Check",     "desc": "Analyses the final 2s RMS energy ratio to detect sting endings that disrupt scene audio. Also flags intro sections exceeding <strong>15s</strong> — the sync readiness limit for sync pitches."},
    {"cls": "bc-lyrics", "theme": "tc", "icon": "🎤", "cat": "WHISPER AI",         "label": "Lyric Transcription",     "desc": "Converts vocals to timestamped text via OpenAI Whisper. Essential for flagging explicit content, trademarked phrases, or co-writer obligations before broadcast clearance."},
    {"cls": "bc-auth",   "theme": "tc", "icon": "✍️", "cat": "AUTHORSHIP DETECT", "label": "AI Lyric Detection",      "desc": "Scores lyrics across four linguistic signals — burstiness, vocabulary diversity, rhyme density, repetition — plus a RoBERTa classifier. Returns <strong>Likely Human / Uncertain / Likely AI</strong>."},
    {"cls": "bc-sim",    "theme": "ta", "icon": "🔍", "cat": "DISCOVERY ENGINE",   "label": "Similar Tracks",          "desc": "Uses Last.fm's similarity graph to surface comparable tracks — fully stateless, no database. Each result is resolved to a YouTube preview URL live via yt-dlp."},
    {"cls": "bc-pro",    "theme": "tb", "icon": "⚖️", "cat": "RIGHTS SEARCH",     "label": "PRO Licensing Links",     "desc": "Generates direct search links for ASCAP, BMI, and SESAC repertory databases — so you can identify publishers and rights holders before a single outreach email."},
]

_BENTO = f"""<!DOCTYPE html><html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=Figtree:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;padding:22px 6px 8px}}
/* ── Dark theme (default) ── */
:root{{--s1:#0C1825;--s2:#111F30;--border:rgba(255,255,255,.07);--accent:#F5640A;--text:#DCE8F4;--muted:#6E8EA8;--dim:#354F64;--ov-bg:rgba(6,12,21,.96)}}
/* ── Light theme overrides ── */
body.light{{--s1:#FAFCFF;--s2:#E4EAF2;--border:rgba(13,27,42,.09);--text:#0D1B2A;--muted:#4B6478;--dim:#8AA4B8;--ov-bg:rgba(255,255,255,.97)}}
body.light .ta{{background:#F4F8FC}}
body.light .tb{{background:linear-gradient(140deg,#EAF0F7 0%,#F4F8FC 100%)}}
body.light .tc{{background:linear-gradient(140deg,#FFF3EE 0%,#FFF8F5 100%)}}
body.light .card:hover,body.light .card:focus{{box-shadow:0 0 0 1px rgba(245,100,10,.2),0 16px 48px rgba(13,27,42,.10),0 0 28px rgba(245,100,10,.08)}}
body.light .ttl-a{{color:rgba(175,65,0,.92)}}
@keyframes eq-a{{0%,100%{{transform:scaleY(.14)}}25%{{transform:scaleY(.88)}}50%{{transform:scaleY(.38)}}75%{{transform:scaleY(.62)}}}}
@keyframes eq-b{{0%,100%{{transform:scaleY(.52)}}25%{{transform:scaleY(.18)}}50%{{transform:scaleY(.92)}}75%{{transform:scaleY(.28)}}}}
@keyframes eq-c{{0%,100%{{transform:scaleY(.72)}}25%{{transform:scaleY(.42)}}50%{{transform:scaleY(.08)}}75%{{transform:scaleY(.82)}}}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;overflow:visible;padding:2px}}
.bc-c2pa{{grid-column:1/3}}.bc-groove{{grid-column:3}}.bc-loop{{grid-column:4}}
.bc-struct{{grid-column:1}}.bc-sting{{grid-column:2}}.bc-lyrics{{grid-column:3/5}}
.bc-auth{{grid-column:1/3}}.bc-sim{{grid-column:3}}.bc-pro{{grid-column:4}}
.bc-ph{{display:none}}
.card{{position:relative;border-radius:13px;border:1px solid var(--border);
       min-height:clamp(180px,22vw,220px);
       cursor:pointer;overflow:visible;
       transition:border-color .25s,box-shadow .3s,transform .32s cubic-bezier(.34,1.56,.64,1)}}
.card:hover,.card:focus{{border-color:rgba(245,100,10,.45);
             box-shadow:0 0 0 1px rgba(245,100,10,.18),0 28px 70px rgba(0,0,0,.75),0 0 48px rgba(245,100,10,.18),0 0 12px rgba(245,100,10,.08);
             transform:translateY(-14px);outline:none;z-index:10}}
.card:focus-visible{{outline:2px solid rgba(245,100,10,.8);outline-offset:3px}}
.ta{{background:var(--s1)}}.tb{{background:linear-gradient(140deg,#0D1620 0%,#111E2C 100%)}}
.tc{{background:linear-gradient(140deg,#1E0B00 0%,#2A1200 100%);border-color:rgba(245,100,10,.2)}}
.inner{{position:relative;z-index:1;padding:18px 18px 16px;display:flex;flex-direction:column;justify-content:space-between;height:100%;min-height:clamp(180px,22vw,220px)}}
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
.ov{{position:absolute;inset:0;border-radius:13px;background:var(--ov-bg);
     backdrop-filter:blur(10px);padding:16px 18px;display:flex;flex-direction:column;gap:6px;
     opacity:0;transform:translateY(6px) scale(.97);
     transition:opacity .2s,transform .2s;z-index:3;pointer-events:none;overflow:auto}}
.card:hover .ov,.card:focus .ov,.card:focus-within .ov{{opacity:1;transform:translateY(0) scale(1)}}
.ov-cat{{font-family:'Chakra Petch',monospace;font-size:.52rem;font-weight:600;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);flex-shrink:0}}
.ov-ttl{{font-family:'Chakra Petch',monospace;font-size:.84rem;font-weight:600;color:var(--text);margin-bottom:2px;flex-shrink:0}}
.ov-desc{{font-family:'Figtree',sans-serif;font-size:.72rem;color:var(--muted);line-height:1.55;
          display:-webkit-box;-webkit-line-clamp:8;-webkit-box-orient:vertical;overflow:hidden}}
.ov-desc strong{{color:rgba(245,160,80,.95);font-weight:600}}
.ov-desc code{{font-family:'JetBrains Mono',monospace;font-size:.66rem;background:rgba(245,100,10,.12);color:var(--accent);padding:1px 5px;border-radius:3px}}
.ph{{border-radius:13px;border:1px dashed rgba(245,100,10,.12);min-height:clamp(180px,22vw,220px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}}
.ph-t{{font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:500;color:var(--dim);letter-spacing:.14em;text-transform:uppercase}}
@media(max-width:500px){{
  .grid{{grid-template-columns:repeat(2,1fr)}}
  .bc-c2pa,.bc-lyrics,.bc-auth{{grid-column:1/3}}
  .bc-groove,.bc-loop,.bc-struct,.bc-sting,.bc-sim,.bc-pro{{grid-column:auto}}
}}
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
  el.setAttribute('role','article');
  el.setAttribute('tabindex','0');
  el.setAttribute('aria-label',c.label+': '+c.cat);
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
/* ── Theme sync ── */
function syncTheme(){{
  try{{
    const isDark=window.parent.document.documentElement.getAttribute('data-theme')==='dark';
    document.body.classList.toggle('light',!isDark);
  }}catch(_){{}}
}}
syncTheme();
try{{
  new MutationObserver(syncTheme).observe(
    window.parent.document.documentElement,
    {{attributes:true,attributeFilter:['data-theme']}}
  );
}}catch(_){{}}
</script></body></html>"""


def render_landing() -> None:
    """Render the landing page and handle source submission."""
    st.markdown(
        '<a href="#main-content" class="skip-link">Skip to main content</a>',
        unsafe_allow_html=True,
    )
    render_site_nav("landing")
    _, col, _ = st.columns([1, 2.2, 1])
    with col:

        # Hero
        st.markdown(f"""
        <div style="text-align:center;padding-bottom:52px;animation:fade-up .65s ease .1s both;">
          <div style="display:flex;align-items:flex-end;justify-content:center;
                      gap:3px;height:44px;margin-bottom:36px;opacity:.55;">
            {eq_bars(28, "#F5640A", 44)}
          </div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.68rem;font-weight:500;
                      color:var(--dim);letter-spacing:.32em;text-transform:uppercase;
                      margin-bottom:18px;">Music Sync Clearance Intelligence</div>
          <div style="font-family:'Chakra Petch',monospace;
                      font-size:clamp(2.8rem,7vw,5rem);font-weight:700;
                      color:var(--text);line-height:.88;letter-spacing:-.03em;
                      margin-bottom:6px;">FORENSIC</div>
          <div style="font-family:'Chakra Petch',monospace;
                      font-size:clamp(2.8rem,7vw,5rem);font-weight:700;
                      color:var(--accent);line-height:.88;letter-spacing:-.03em;
                      margin-bottom:30px;">PORTAL</div>
          <div style="font-family:'Figtree',sans-serif;font-size:.96rem;font-weight:400;
                      color:var(--muted);line-height:1.65;max-width:480px;margin:0 auto;">
            Detect AI authorship · Audit sync compliance · Flag lyric risks<br>
            <span style="font-family:'Figtree',sans-serif;font-size:.88rem;
                         color:var(--muted);">before a track costs you a placement.</span>
          </div>
          <div style="display:flex;align-items:center;justify-content:center;gap:20px;
                      margin-top:22px;flex-wrap:wrap;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;font-weight:500;
                         color:var(--dim);letter-spacing:.12em;text-transform:uppercase;">
              Hugging Face ZeroGPU
            </span>
            <span style="color:var(--border-hr);font-size:.7rem;">|</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;font-weight:500;
                         color:var(--dim);letter-spacing:.12em;text-transform:uppercase;">
              Stateless Architecture
            </span>
            <span style="color:var(--border-hr);font-size:.7rem;">|</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;font-weight:500;
                         color:var(--dim);letter-spacing:.12em;text-transform:uppercase;">
              No Audio Stored
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span id="main-content" tabindex="-1"></span>', unsafe_allow_html=True)

        # CTA
        st.markdown("<div style='margin-bottom:8px;animation:fade-up .7s ease .2s both;'>",
                    unsafe_allow_html=True)
        if st.button("Launch Portal →", type="primary",
                     use_container_width=True, key="cta_portal"):
            st.session_state.page = "portal"
            st.rerun()
        st.markdown("</div><br>", unsafe_allow_html=True)

        # Feature grid label
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;
                    animation:fade-up .75s ease .3s both;">
          <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:600;
                      letter-spacing:.2em;text-transform:uppercase;color:var(--dim);white-space:nowrap;">
            ◈ Feature Matrix
          </div>
          <div style="flex:1;height:1px;background:var(--border-hr);"></div>
          <div style="font-family:'Chakra Petch',monospace;font-size:.54rem;color:var(--dim);
                      letter-spacing:.1em;white-space:nowrap;">Hover or focus to inspect</div>
        </div>
        """, unsafe_allow_html=True)

        components.html(_BENTO, height=740, scrolling=False)

    render_site_footer()
