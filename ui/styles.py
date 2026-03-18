"""
ui/styles.py
Global CSS for the Sync-Safe Forensic Portal.

Inject once at app startup:
    from ui.styles import STYLES
    st.markdown(STYLES, unsafe_allow_html=True)
"""

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;500;600;700&family=Figtree:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg:      #050911;
  --s1:      #0B1320;
  --s2:      #0F1A28;
  --border:  rgba(255,255,255,0.06);
  --accent:  #F5640A;
  --ok:      #0DF5A0;
  --danger:  #FF3060;
  --text:    #D8E6F2;
  --muted:   #7A95AA;
  --dim:     #364C5C;
  --ui:      'Chakra Petch', monospace;
  --body:    'Figtree', sans-serif;
  --mono:    'JetBrains Mono', monospace;
}

*, *::before, *::after { font-family: var(--body) !important; box-sizing: border-box; }

/* ── Shell ── */
[data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  background-image: radial-gradient(circle, rgba(245,100,10,0.055) 1px, transparent 0) !important;
  background-size: 30px 30px !important;
}
[data-testid="stSidebar"]             { display: none !important; }
[data-testid="stMainBlockContainer"]  { padding-top: 0 !important; }
[data-testid="stBottom"]              { display: none !important; }
.block-container                      { padding-left: 2rem !important; padding-right: 2rem !important; }

/* ── Keyframes ── */
@keyframes pulse-border {
  0%,100% { box-shadow: 0 0 0 1px rgba(245,100,10,0.10), 0 24px 80px rgba(0,0,0,0.65); }
  50%     { box-shadow: 0 0 0 1px rgba(245,100,10,0.28), 0 24px 80px rgba(0,0,0,0.65), 0 0 40px rgba(245,100,10,0.08); }
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
[data-testid="stTextInput"] input {
  background: var(--bg) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; font-family: var(--mono) !important;
  font-size: .88rem !important; color: var(--text) !important;
  padding: 13px 16px !important; transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(245,100,10,.1), 0 0 20px rgba(245,100,10,.06) !important;
  outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--dim) !important; }
[data-testid="stTextInput"] label { display: none !important; }

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
  box-shadow: 0 8px 48px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.035);
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
  background:#0D1926; border:1px solid rgba(245,100,10,.22);
  color:#B8D0E0; font-size:.72rem; font-family:'Figtree',sans-serif;
  font-weight:400; line-height:1.55; letter-spacing:0; text-transform:none;
  padding:9px 13px; border-radius:7px; width:240px; z-index:9999;
  box-shadow:0 8px 24px rgba(0,0,0,.45);
  pointer-events:none;
}
.tip-box::after {
  content:''; position:absolute; top:100%; left:14px;
  border:5px solid transparent; border-top-color:#0D1926;
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
[data-testid="stExpander"] {
  background:var(--s1) !important; border:1px solid var(--border) !important;
  border-radius:10px !important;
}
[data-testid="stExpander"] summary span { color:var(--muted) !important; }
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
audio { border-radius:10px !important; filter:invert(1) hue-rotate(180deg) !important; }
</style>
"""
