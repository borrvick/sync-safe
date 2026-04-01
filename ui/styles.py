"""
ui/styles.py
Global CSS for the Sync-Safe Forensic Portal — dual light/dark theme.

Inject once at app startup:
    from ui.styles import STYLES
    st.markdown(STYLES, unsafe_allow_html=True)
"""

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;500;600;700&family=Figtree:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  /* ── Light theme (default) ── */
  --bg:           #EFF3F8;
  --s1:           #FFFFFF;
  --s2:           #E4EAF2;
  --tip-bg:       #0D1B2A;
  --border:       rgba(13,27,42,0.09);
  --border-hr:    rgba(13,27,42,0.06);
  --badge-bg:     rgba(13,27,42,0.05);
  --accent:       #F5640A;
  --ok:           #059669;
  --danger:       #DC2626;
  --text:         #0D1B2A;
  --muted:        #4B6478;
  --dim:          #8AA4B8;
  --shadow:       0 1px 3px rgba(13,27,42,0.07), 0 4px 20px rgba(13,27,42,0.05), inset 0 1px 0 rgba(255,255,255,0.9);
  --shadow-sm:    0 1px 2px rgba(13,27,42,0.06), 0 3px 10px rgba(13,27,42,0.05);
  --pulse-drop:   0 4px 24px rgba(13,27,42,0.12);
  --bg-dot:       rgba(13,27,42,0.04);
  --audio-filter: none;
  /* ── Compliance grade colours ── */
  --grade-b:      #6ECC8A;
  --grade-c:      #F5A623;
  --grade-d:      #F57A35;
  --needs-review: #C8E86A;
  /* ── Issue type colours ── */
  --issue-brand:    #F5A623;
  --issue-explicit: #FF3060;
  --issue-violence: #FF6B35;
  --issue-location: #4FC3F7;
  /* ── Sync status colours ── */
  --sync-pass:      #0DF5A0;
  --sync-fail:      #FF6B35;
  --ui:           'Chakra Petch', monospace;
  --body:         'Figtree', sans-serif;
  --mono:         'JetBrains Mono', monospace;
}

:root[data-theme="dark"] {
  --bg:           #060C15;
  --s1:           #0C1825;
  --s2:           #111F30;
  --tip-bg:       #0C1825;
  --border:       rgba(255,255,255,0.07);
  --border-hr:    rgba(255,255,255,0.055);
  --badge-bg:     rgba(255,255,255,0.06);
  --ok:           #10B981;
  --danger:       #F43F5E;
  --text:         #DCE8F4;
  --muted:        #6E8EA8;
  --dim:          #354F64;
  --shadow:       0 8px 48px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.03);
  --shadow-sm:    0 4px 20px rgba(0,0,0,0.4);
  --pulse-drop:   0 24px 80px rgba(0,0,0,0.65);
  --bg-dot:       rgba(245,100,10,0.05);
  --audio-filter: invert(1) hue-rotate(180deg);
  /* ── Compliance grade colours ── */
  --grade-b:      #6ECC8A;
  --grade-c:      #F5A623;
  --grade-d:      #F57A35;
  --needs-review: #C8E86A;
  /* ── Issue type colours ── */
  --issue-brand:    #F5A623;
  --issue-explicit: #FF3060;
  --issue-violence: #FF6B35;
  --issue-location: #4FC3F7;
  /* ── Sync status colours ── */
  --sync-pass:      #0DF5A0;
  --sync-fail:      #FF6B35;
}

/* NOTE: !important here overrides ALL font families including icon fonts (e.g. Material Symbols).
   Streamlit's expander chevron icon is a font ligature — this rule makes it render as literal text.
   We hide it via [data-testid="stIconMaterial"] and replace it with a CSS-only chevron.
   See the "Expander sections" block below before removing this rule. */
*, *::before, *::after { font-family: var(--body) !important; box-sizing: border-box; }

/* ── Shell ── */
[data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  background-image: radial-gradient(circle, var(--bg-dot) 1px, transparent 0) !important;
  background-size: 30px 30px !important;
  transition: background-color .3s ease !important;
}
[data-testid="stSidebar"]             { display: none !important; }
[data-testid="stMainBlockContainer"]  { padding-top: 0 !important; }
[data-testid="stBottom"]              { display: none !important; }
.block-container                      { padding-left: 2rem !important; padding-right: 2rem !important; }

/* ── Keyframes ── */
@keyframes pulse-border {
  0%,100% { box-shadow: 0 0 0 1px rgba(245,100,10,0.10), var(--pulse-drop); }
  50%     { box-shadow: 0 0 0 1px rgba(245,100,10,0.28), var(--pulse-drop), 0 0 40px rgba(245,100,10,0.08); }
}
@keyframes shimmer {
  from { background-position: -200% center; }
  to   { background-position:  200% center; }
}
@keyframes fade-up {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes eq-a { 0%,100%{transform:scaleY(.14)} 25%{transform:scaleY(.88)} 50%{transform:scaleY(.38)} 75%{transform:scaleY(.62)} }
@keyframes eq-b { 0%,100%{transform:scaleY(.52)} 25%{transform:scaleY(.18)} 50%{transform:scaleY(.92)} 75%{transform:scaleY(.28)} }
@keyframes eq-c { 0%,100%{transform:scaleY(.72)} 25%{transform:scaleY(.42)} 50%{transform:scaleY(.08)} 75%{transform:scaleY(.82)} }

/* ── Input Card (bordered container) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(var(--s1), var(--s1)) padding-box,
              linear-gradient(135deg, rgba(245,100,10,.55) 0%, rgba(245,100,10,.04) 45%,
                              rgba(245,100,10,.04) 55%, rgba(245,100,10,.35) 100%) border-box !important;
  border: 1px solid transparent !important;
  border-radius: 18px !important;
  padding: 36px 36px 28px !important;
  animation: pulse-border 5s ease-in-out infinite !important;
}

/* ── Tabs (radio) ── */
[data-testid="stRadio"] {
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  margin-bottom: 20px !important;
}
[data-testid="stRadio"] [data-testid="stWidgetLabel"] { display: none !important; }
[data-testid="stRadio"] > div:last-child { display: flex !important; gap: 3px !important; }
[data-testid="stRadio"] label {
  flex: 1 !important; display: flex !important; align-items: center !important;
  justify-content: center !important; padding: 10px 14px !important;
  border-radius: 7px !important; cursor: pointer !important; margin: 0 !important;
  font-family: var(--ui) !important; font-size: .78rem !important;
  font-weight: 500 !important; letter-spacing: .09em !important;
  text-transform: uppercase !important; color: var(--muted) !important;
  transition: all .15s !important;
}
[data-testid="stRadio"] input[type="radio"] { display: none !important; }
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] {
  background: var(--s2) !important; color: var(--accent) !important;
  border-radius: 7px !important; box-shadow: 0 0 14px rgba(245,100,10,.1) !important;
}

/* ── Text Input ── */
/* Structural resets — shared across both themes */
[data-testid="stTextInput"] [data-baseweb="input"] {
  border-radius: 10px !important;
  box-shadow: none !important;
  outline: none !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
  border-color: #F5640A !important;
  box-shadow: 0 0 0 3px rgba(245,100,10,.1) !important;
}
[data-testid="stTextInput"] input {
  border: none !important; outline: none !important; box-shadow: none !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .88rem !important;
  caret-color: #F5640A !important;
  padding: 13px 16px !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* ── Light theme input ──
   html:not([data-theme="dark"]) is unambiguous — active when no data-theme
   attribute is present. Explicit hex values sidestep variable cascade issues.
   color-scheme:light resets the browser UA form styles to light mode. */
html:not([data-theme="dark"]) [data-testid="stTextInput"] [data-baseweb="input"],
html:not([data-theme="dark"]) [data-testid="stTextInput"] [data-baseweb="input"] > div {
  background: #FFFFFF !important;
  border: 1px solid rgba(13,27,42,0.12) !important;
  color-scheme: light !important;
}
html:not([data-theme="dark"]) [data-testid="stTextInput"] input {
  background: #FFFFFF !important;
  color: #1A2B3C !important;
  -webkit-text-fill-color: #1A2B3C !important;
}
html:not([data-theme="dark"]) [data-testid="stTextInput"] input::placeholder {
  color: #8AA4B8 !important;
  -webkit-text-fill-color: #8AA4B8 !important;
  opacity: 1 !important;
}
html:not([data-theme="dark"]) [data-testid="stTextInput"] input:-webkit-autofill {
  -webkit-text-fill-color: #1A2B3C !important;
  -webkit-box-shadow: 0 0 0 100px #FFFFFF inset !important;
}

/* ── Dark theme input ──
   html[data-theme="dark"] is set by the theme-toggle JS on <html>.
   color-scheme:dark resets UA form styles to dark mode. */
html[data-theme="dark"] [data-testid="stTextInput"] [data-baseweb="input"],
html[data-theme="dark"] [data-testid="stTextInput"] [data-baseweb="input"] > div {
  background: #0C1825 !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  color-scheme: dark !important;
}
html[data-theme="dark"] [data-testid="stTextInput"] input {
  background: #0C1825 !important;
  color: #DCE8F4 !important;
  -webkit-text-fill-color: #DCE8F4 !important;
}
html[data-theme="dark"] [data-testid="stTextInput"] input::placeholder {
  color: #354F64 !important;
  -webkit-text-fill-color: #354F64 !important;
  opacity: 1 !important;
}
html[data-theme="dark"] [data-testid="stTextInput"] input:-webkit-autofill {
  -webkit-text-fill-color: #DCE8F4 !important;
  -webkit-box-shadow: 0 0 0 100px #0C1825 inset !important;
}

/* ── Primary Button ── */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(105deg,
    var(--accent) 38%, #FF9050 50%, var(--accent) 62%) !important;
  background-size: 200% 100% !important;
  animation: shimmer 3s linear infinite !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-family: var(--ui) !important; font-size: .84rem !important;
  font-weight: 600 !important; letter-spacing: .14em !important;
  text-transform: uppercase !important; min-height: 52px !important;
  box-shadow: 0 4px 24px rgba(245,100,10,.35) !important;
  transition: transform .2s, box-shadow .2s !important;
}
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p {
  color: #fff !important; font-family: var(--ui) !important;
  font-weight: 600 !important; font-size: .84rem !important;
  letter-spacing: .14em !important; text-transform: uppercase !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 36px rgba(245,100,10,.5) !important;
}

/* ── Secondary Button ── */
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  background: transparent !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; font-family: var(--ui) !important;
  font-size: .78rem !important; font-weight: 500 !important;
  letter-spacing: .1em !important; text-transform: uppercase !important;
  color: var(--muted) !important; transition: all .15s !important;
}
button[kind="secondary"] p, [data-testid="stBaseButton-secondary"] p {
  color: var(--muted) !important; font-family: var(--ui) !important;
  letter-spacing: .1em !important; text-transform: uppercase !important;
}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
  background: rgba(245,100,10,.06) !important;
}

/* ── Report card ── */
.sig {
  background: var(--s1); border: 1px solid var(--border); border-radius: 14px;
  padding: 26px 26px 20px; margin-bottom: 14px;
  box-shadow: var(--shadow);
  transition: background .3s, box-shadow .3s;
}
.sig-head {
  font-family: var(--ui); font-size: .58rem; font-weight: 600;
  letter-spacing: .18em; text-transform: uppercase; color: var(--dim);
  display: flex; align-items: center; gap: 10px; margin-bottom: 18px;
}
.sig-head::after { content:''; flex:1; height:1px; background:var(--border); }
.sig-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 0; border-bottom: 1px solid var(--border);
}
.sig-row:last-child { border-bottom: none; }
.sk { font-family:var(--ui); font-size:.64rem; font-weight:500;
      letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
      display:flex; align-items:center; gap:6px; }
.sv { font-family:var(--mono); font-size:.84rem; font-weight:500; color:var(--text); }

/* Metric tooltips */
.tip-wrap { position:relative; display:inline-flex; align-items:center; }
.tip-icon {
  display:inline-flex; align-items:center; justify-content:center;
  width:14px; height:14px; border-radius:50%;
  background:rgba(245,100,10,.12); color:#F5640A;
  font-size:.56rem; font-weight:700; font-family:var(--mono);
  cursor:default; border:1px solid rgba(245,100,10,.28);
  letter-spacing:0; text-transform:none; flex-shrink:0;
  transition:background .15s;
}
.tip-wrap:hover .tip-icon { background:rgba(245,100,10,.28); }
.tip-box {
  display:none; position:absolute; bottom:calc(100% + 8px); left:0;
  background:var(--tip-bg); border:1px solid rgba(245,100,10,.22);
  color:var(--text); font-size:.72rem; font-family:'Figtree',sans-serif;
  font-weight:400; line-height:1.55; letter-spacing:0; text-transform:none;
  padding:9px 13px; border-radius:7px; width:240px; z-index:9999;
  box-shadow:var(--shadow-sm);
  pointer-events:none;
}
.tip-box::after {
  content:''; position:absolute; top:100%; left:14px;
  border:5px solid transparent; border-top-color:var(--tip-bg);
}
.tip-wrap:hover .tip-box { display:block; }

/* Verdicts */
.verd {
  display:inline-flex; align-items:center; gap:7px; padding:5px 14px;
  border-radius:6px; font-family:var(--ui); font-size:.68rem;
  font-weight:600; letter-spacing:.12em; text-transform:uppercase; white-space:nowrap;
}
.verd::before { content:''; width:6px; height:6px; border-radius:50%; background:currentColor; flex-shrink:0; }
.v-h { background:rgba(13,245,160,.08); color:#0DF5A0; border:1px solid rgba(13,245,160,.2); }
.v-a { background:rgba(255,48,96,.08);  color:#FF3060; border:1px solid rgba(255,48,96,.2); }
.v-u { background:rgba(245,100,10,.08); color:var(--accent); border:1px solid rgba(245,100,10,.2); }

/* Pills, track rows */
.s-pill {
  display:inline-flex; font-family:var(--ui); font-size:.6rem; font-weight:500;
  letter-spacing:.07em; text-transform:uppercase; color:var(--muted);
  background:var(--s2); border:1px solid var(--border);
  border-radius:5px; padding:3px 10px; margin:3px;
}
.t-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:11px 14px; border:1px solid var(--border); border-radius:10px;
  margin-bottom:7px; transition:border-color .15s, background .15s;
}
.t-row:hover { border-color:rgba(245,100,10,.4); background:rgba(245,100,10,.03); }
.t-art { font-family:var(--ui); font-size:.58rem; font-weight:600;
         color:var(--accent); letter-spacing:.1em; text-transform:uppercase; }
.t-nm  { font-size:.88rem; font-weight:600; color:var(--text); margin-top:2px; }
.t-btn { font-family:var(--ui); font-size:.58rem; font-weight:600; color:var(--dim);
         letter-spacing:.1em; text-transform:uppercase; padding:5px 11px;
         border-radius:5px; border:1px solid var(--border); text-decoration:none;
         transition:all .15s; }
.t-btn:hover { color:var(--accent); border-color:var(--accent); }

/* ── Expander sections ── */
[data-testid="stExpander"] {
  background: var(--s1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  margin-bottom: 12px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: background .3s, box-shadow .3s !important;
  overflow: visible !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details[open] {
  border-radius: 14px !important;
  background: transparent !important;
}

/* Hide the native browser disclosure triangle */
[data-testid="stExpander"] summary {
  list-style: none !important;
  -webkit-appearance: none !important;
  padding: 16px 20px !important;
  border-radius: 14px !important;
  background: transparent !important;
  cursor: pointer !important;
  transition: background .15s !important;
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
}
[data-testid="stExpander"] summary::marker { content: '' !important; }
[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }

/* Hide Streamlit's icon element entirely.
   Our global * { font-family } overrides the Material Symbols icon font,
   turning "keyboard_arrow_down" into visible literal text.
   Strategy: hide ALL h3 children, then explicitly re-show only the label wrapper. */
[data-testid="stExpander"] summary h3 > * {
  display: none !important;
}
[data-testid="stExpander"] summary h3 > .stExpanderLabelWrapper {
  display: inline-flex !important;
  align-items: center !important;
}
/* Belt-and-suspenders: target by all known testids, aria-hidden, and class patterns */
[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary [aria-hidden="true"],
[data-testid="stExpander"] summary [class*="material"] {
  display: none !important;
}

/* CSS-only chevron — no font, no glyph, always renders correctly */
[data-testid="stExpander"] summary::before {
  content: '' !important;
  width: 7px !important;
  height: 7px !important;
  border-right: 2px solid var(--dim) !important;
  border-bottom: 2px solid var(--dim) !important;
  transform: rotate(-45deg) !important;
  display: inline-block !important;
  flex-shrink: 0 !important;
  transition: transform .2s ease !important;
  margin-top: 1px !important;
}
[data-testid="stExpander"] details[open] > summary::before {
  transform: rotate(45deg) !important;
  margin-top: -3px !important;
}

[data-testid="stExpander"] summary:hover { background: var(--s2) !important; }

/* Open state — theme-aware tint */
[data-testid="stExpander"] details[open] > summary {
  background: var(--s2) !important;
  border-radius: 14px 14px 0 0 !important;
}

/* Label text — only .stExpanderLabelWrapper p, nothing else */
[data-testid="stExpander"] summary .stExpanderLabelWrapper p {
  color: var(--text) !important;
  font-family: var(--ui) !important;
  font-size: .72rem !important;
  font-weight: 600 !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  margin: 0 !important;
}

[data-testid="stExpander"] > div > div { padding: 0 20px 20px !important; }

/* ── Theme FAB ── */
#theme-fab {
  position: fixed;
  bottom: 28px;
  right: 28px;
  z-index: 99999;
}
#theme-toggle {
  width: 46px;
  height: 46px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--s1);
  color: var(--accent);
  font-size: 1.15rem;
  cursor: pointer;
  box-shadow: var(--shadow-sm);
  transition: transform .2s, border-color .2s, box-shadow .2s, background .3s;
  display: flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}
#theme-toggle:hover {
  transform: scale(1.12);
  border-color: var(--accent);
  box-shadow: 0 0 20px rgba(245,100,10,.3);
}

/* ── Skip-to-main-content link ── */
.skip-link {
  position: fixed;
  top: -100%;
  left: 50%;
  transform: translateX(-50%);
  z-index: 999999;
  background: var(--s1);
  color: var(--accent);
  padding: 10px 22px;
  border-radius: 8px;
  border: 2px solid var(--accent);
  font-family: var(--ui);
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-decoration: none;
  text-transform: uppercase;
  white-space: nowrap;
  box-shadow: var(--shadow-sm);
  transition: top .15s;
}
.skip-link:focus { top: 12px; }

/* ── Global focus-visible ring ── */
:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
}

/* ── Per-element focus-visible overrides ── */
button[kind="primary"]:focus-visible,
[data-testid="stBaseButton-primary"]:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 3px !important;
  box-shadow: 0 0 0 4px rgba(245,100,10,.25), 0 4px 24px rgba(245,100,10,.35) !important;
}
button[kind="secondary"]:focus-visible,
[data-testid="stBaseButton-secondary"]:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
  border-color: var(--accent) !important;
}
[data-testid="stTextInput"] input:focus-visible {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(245,100,10,.1), 0 0 20px rgba(245,100,10,.06) !important;
  outline: none !important;
}
[data-testid="stExpander"] summary:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: -2px !important;
  border-radius: 14px !important;
}
#theme-toggle:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 3px !important;
  box-shadow: 0 0 0 4px rgba(245,100,10,.2) !important;
}
audio:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
  border-radius: 10px !important;
}
[data-testid="stLinkButton"] a:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
}

/* ── Tooltip: keyboard access via focus-within ── */
.tip-icon { cursor: pointer !important; }
.tip-wrap:focus-within .tip-box { display: block !important; }
.tip-icon:focus-visible {
  outline: 2px solid var(--accent) !important;
  border-radius: 50% !important;
  outline-offset: 1px !important;
}

/* Misc */
h1,h2,h3 { color:var(--text) !important; }
p,li { color:var(--text); }
.stCaption p { color:var(--muted) !important; }
hr { border-color:var(--border) !important; }
[data-testid="stAlert"] {
  background:rgba(245,100,10,.05) !important; border:1px solid rgba(245,100,10,.18) !important;
  color:var(--accent) !important; border-radius:10px !important;
}
[data-testid="stFileUploader"] section {
  background:var(--bg) !important; border:1px dashed rgba(255,255,255,.07) !important;
  border-radius:10px !important;
}
[data-testid="stFileUploader"] section p { color:var(--muted) !important; }
[data-testid="stTextArea"] textarea {
  background:var(--bg) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; color:var(--text) !important;
  font-family:var(--mono) !important; font-size:.84rem !important;
}
[data-testid="stLinkButton"] a {
  background:var(--s2) !important; border:1px solid var(--border) !important;
  color:var(--muted) !important; border-radius:10px !important;
  font-family:var(--ui) !important; font-size:.76rem !important;
  letter-spacing:.07em !important; transition:all .15s !important;
}
[data-testid="stLinkButton"] a:hover {
  border-color:var(--accent) !important; color:var(--accent) !important;
  background:rgba(245,100,10,.06) !important;
}
audio { border-radius:10px !important; filter:var(--audio-filter) !important; }

/* ── Site nav & footer shared visual styles ── */
.ss-brand { display:flex; align-items:center; gap:12px; padding:14px 0; }
.ss-eq    { display:flex; align-items:flex-end; gap:3px; height:24px; }
.ss-brand-name, .ss-ft-brand-name {
  font-family:'Chakra Petch',monospace !important; font-size:1.05rem; font-weight:700 !important;
  color:var(--accent); letter-spacing:.14em;
}
.ss-brand-sub, .ss-ft-brand-sub {
  font-family:'Chakra Petch',monospace !important; font-size:.5rem; font-weight:500 !important;
  color:var(--dim); letter-spacing:.22em; text-transform:uppercase; margin-top:2px;
}
.ss-ft-brand-name { font-size:.88rem; }
.ss-tagline  { font-family:'Figtree',sans-serif; font-size:.72rem; color:var(--muted); line-height:1.55; margin-top:10px; max-width:200px; }
.ss-badges   { display:flex; flex-direction:column; gap:4px; margin-top:14px; }
.ss-badge    { font-family:'JetBrains Mono',monospace !important; font-size:.52rem; color:var(--dim); letter-spacing:.1em; text-transform:uppercase; }
.ss-col-title { font-family:'Chakra Petch',monospace !important; font-size:.6rem; font-weight:700 !important; letter-spacing:.18em; text-transform:uppercase; color:var(--dim); margin-bottom:14px; }
.ss-lnk      { display:block; font-family:'Figtree',sans-serif !important; font-size:.76rem; font-weight:500 !important; color:var(--muted); margin-bottom:9px; transition:color .15s; text-decoration:none; }
.ss-lnk:hover { color:var(--accent); }
.ss-lnk-ext::after { content:' ↗'; font-size:.65rem; opacity:.6; }
.ss-bottom { display:flex; align-items:center; justify-content:space-between; margin-top:24px; padding-top:20px; border-top:1px solid var(--border); flex-wrap:wrap; gap:8px; }
.ss-copy   { font-family:'JetBrains Mono',monospace !important; font-size:.58rem; color:var(--dim); letter-spacing:.04em; }
.ss-trust  { display:flex; gap:16px; font-family:'JetBrains Mono',monospace !important; font-size:.55rem; color:var(--dim); letter-spacing:.1em; text-transform:uppercase; }

/* ── Nav & footer button-links ──────────────────────────────────────────────
   JS in app.py stamps data-nav-btn="nav" / "footer" onto secondary button
   elements inside the nav/footer containers. data-* attributes survive React
   reconciliation. "html body button" (specificity 0,1,3) beats Streamlit's
   emotion single-class selectors (0,1,0).
*/

/* Brand link — no underline, inherit layout */
a.ss-brand-link {
  text-decoration: none !important;
  display: block !important;
}

/* Nav link buttons — plain text, no button chrome */
html body button[data-nav-btn="nav"] {
  all: unset !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 100% !important;
  padding: 0.5rem 0.6rem !important;
  cursor: pointer !important;
  font-family: 'Figtree', sans-serif !important;
  font-size: 1.25rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  white-space: nowrap !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.15s, border-color 0.15s !important;
}
html body button[data-nav-btn="nav"]:hover {
  color: var(--text) !important;
  background: transparent !important;
  background-color: transparent !important;
}
/* Disabled = current page — accent color + underline indicator */
html body button[data-nav-btn="nav"][disabled] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
  background: transparent !important;
  background-color: transparent !important;
  cursor: default !important;
  opacity: 1 !important;
}
html body button[data-nav-btn="nav"] p,
html body button[data-nav-btn="nav"] [data-testid="stMarkdownContainer"] {
  margin: 0 !important; padding: 0 !important; display: block !important;
  text-align: center !important; width: 100% !important;
}
html body button[data-nav-btn="nav"]:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 3px !important;
  border-radius: 4px !important;
}
/* Kill Streamlit's hover backgrounds on nav button wrappers */
html body div[data-testid="stButton"]:has(button[data-nav-btn="nav"]),
html body div[data-testid="stButton"]:has(button[data-nav-btn="nav"]):hover {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* Nav CTA — "Launch Portal →" primary button, constrained to nav height */
html body button[data-nav-btn="nav-cta"] {
  padding: 0.35rem 0.9rem !important;
  font-size: .82rem !important;
  font-weight: 600 !important;
  font-family: 'Figtree', sans-serif !important;
  white-space: nowrap !important;
  border-radius: 6px !important;
  line-height: 1.4 !important;
  min-height: unset !important;
  height: auto !important;
}
html body button[data-nav-btn="nav-cta"]:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 3px !important;
}

/* Footer link buttons — block-level plain text */
html body button[data-nav-btn="footer"] {
  all: unset !important;
  box-sizing: border-box !important;
  display: block !important;
  cursor: pointer !important;
  margin-bottom: 9px !important;
  font-family: 'Figtree', sans-serif !important;
  font-size: .76rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  transition: color .15s !important;
}
html body button[data-nav-btn="footer"]:hover { color: var(--accent) !important; }
html body button[data-nav-btn="footer"] p,
html body button[data-nav-btn="footer"] [data-testid="stMarkdownContainer"] {
  all: unset !important;
  display: contents !important;
}
html body button[data-nav-btn="footer"]:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
  border-radius: 3px !important;
}

/* ── Suppress gap from 0-height theme-injection iframe ── */
[data-testid="stCustomComponentV1"]:has(iframe[height="0"]) {
  height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}
</style>
"""
