"""
ui/nav.py
Shared site navigation bar and footer rendered on all public-facing pages.

Navigation uses <a href="?p=..." target="_top"> anchor tags rather than
onclick JS. target="_top" navigates the parent browsing context (the
Streamlit page) from within the components.html() iframe — this works
reliably across all browsers without requiring sandbox permissions.

app.py reads st.query_params["p"] on every load and routes accordingly.
"""
from __future__ import annotations

import streamlit.components.v1 as components

_NAV_HEIGHT    = 72
_FOOTER_HEIGHT = 268

_FONTS = (
    "https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700"
    "&family=Figtree:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
)

# EQ bar keyframes — duplicated from styles.py; each iframe has its own style context.
_EQ_KF = (
    "@keyframes eq-a{0%,100%{transform:scaleY(.14)}25%{transform:scaleY(.88)}"
    "50%{transform:scaleY(.38)}75%{transform:scaleY(.62)}}"
    "@keyframes eq-b{0%,100%{transform:scaleY(.52)}25%{transform:scaleY(.18)}"
    "50%{transform:scaleY(.92)}75%{transform:scaleY(.28)}}"
    "@keyframes eq-c{0%,100%{transform:scaleY(.72)}25%{transform:scaleY(.42)}"
    "50%{transform:scaleY(.08)}75%{transform:scaleY(.82)}}"
)

_THEME_JS = """<script>
(function(){
  try{
    function sync(){
      var d=window.parent.document.documentElement.getAttribute('data-theme')||'light';
      document.body.classList.toggle('dark',d==='dark');
    }
    sync();
    new MutationObserver(sync).observe(
      window.parent.document.documentElement,
      {attributes:true,attributeFilter:['data-theme']}
    );
  }catch(e){}
})();
</script>"""

_BASE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --text:#0D1B2A;--muted:#4B6478;--dim:#8AA4B8;
  --accent:#F5640A;--border:rgba(13,27,42,.09);--s1:#FFFFFF;--s2:#E4EAF2;
}
body.dark{
  --text:#DCE8F4;--muted:#6E8EA8;--dim:#354F64;
  --border:rgba(255,255,255,.07);--s1:#0C1825;--s2:#111F30;
}
body{background:transparent;font-family:'Figtree',sans-serif;}
a{text-decoration:none;}
"""


def _eq_html(n: int, color: str, h: int) -> str:
    anims = ["eq-a", "eq-b", "eq-c"]
    out = []
    for i in range(n):
        dur = f"{1.1 + (i % 3) * 0.22:.2f}s"
        dly = f"{i * 0.07:.2f}s"
        out.append(
            f'<div style="width:3px;height:{h}px;background:{color};border-radius:2px;'
            f'transform-origin:bottom;animation:{anims[i%3]} {dur} ease-in-out {dly} infinite;'
            f'flex-shrink:0;"></div>'
        )
    return "".join(out)


def _head(extra_css: str = "") -> str:
    return (
        f'<meta charset="utf-8">'
        f'<link rel="preconnect" href="https://fonts.googleapis.com">'
        f'<link href="{_FONTS}" rel="stylesheet">'
        f"<style>{_BASE_CSS}{_EQ_KF}{extra_css}</style>"
    )


# ── Navigation bar ────────────────────────────────────────────────────────────

_NAV_CSS = """
nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 2px;border-bottom:1px solid var(--border);
}
.brand{
  display:flex;align-items:center;gap:10px;text-decoration:none;
}
.eq{display:flex;align-items:flex-end;gap:2px;height:20px;}
.brand-name{
  font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
  color:var(--accent);letter-spacing:.14em;
}
.brand-sub{
  font-family:'Chakra Petch',monospace;font-size:.44rem;font-weight:500;
  color:var(--dim);letter-spacing:.22em;text-transform:uppercase;margin-top:2px;
}
.links{display:flex;align-items:center;gap:22px;}
.nl{
  font-family:'Figtree',sans-serif;font-size:.72rem;font-weight:500;
  letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  padding:0 0 3px;border-bottom:2px solid transparent;
  transition:color .15s,border-color .15s;display:inline-block;
}
.nl:hover{color:var(--text);}
.nl.active{color:var(--text);border-bottom-color:var(--accent);}
.cta{
  font-family:'Chakra Petch',monospace;font-size:.65rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;
  color:#fff;background:var(--accent);
  padding:8px 18px;border-radius:6px;transition:opacity .15s;display:inline-block;
}
.cta:hover{opacity:.85;}
"""


def render_site_nav(current_page: str) -> None:
    """Render the full-width site navigation bar."""
    eq   = _eq_html(6, "#F5640A", 20)
    hiw  = " active" if current_page == "how_it_works" else ""
    leg  = " active" if current_page == "legal"        else ""

    html = f"""<!doctype html><html><head>{_head(_NAV_CSS)}</head><body>
<nav>
  <a class="brand" href="./" target="_top">
    <div class="eq">{eq}</div>
    <div>
      <div class="brand-name">SYNC-SAFE™</div>
      <div class="brand-sub">Forensic Portal</div>
    </div>
  </a>
  <div class="links">
    <a class="nl{hiw}" href="?p=how_it_works" target="_top">How it Works</a>
    <a class="nl{leg}" href="?p=legal"        target="_top">Legal</a>
    <a class="cta"     href="?p=portal"        target="_top">Launch Portal →</a>
  </div>
</nav>
{_THEME_JS}</body></html>"""

    components.html(html, height=_NAV_HEIGHT, scrolling=False)


# ── Footer ────────────────────────────────────────────────────────────────────

_FOOTER_CSS = """
footer{border-top:1px solid var(--border);padding:36px 4px 24px;}
.grid{display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;gap:32px;}
.brand-name{
  font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
  color:var(--accent);letter-spacing:.14em;
}
.brand-sub{
  font-family:'Chakra Petch',monospace;font-size:.44rem;font-weight:500;
  color:var(--dim);letter-spacing:.22em;text-transform:uppercase;margin-top:3px;
}
.tagline{
  font-family:'Figtree',sans-serif;font-size:.72rem;color:var(--muted);
  line-height:1.55;margin-top:10px;max-width:200px;
}
.badges{display:flex;flex-direction:column;gap:4px;margin-top:14px;}
.badge{
  font-family:'JetBrains Mono',monospace;font-size:.52rem;font-weight:500;
  color:var(--dim);letter-spacing:.1em;text-transform:uppercase;
}
.col-title{
  font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
  letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-bottom:14px;
}
.lnk{
  display:block;font-family:'Figtree',sans-serif;font-size:.76rem;font-weight:500;
  color:var(--muted);margin-bottom:9px;transition:color .15s;
}
.lnk:hover{color:var(--accent);}
.lnk-ext::after{content:' ↗';font-size:.65rem;opacity:.6;}
.bottom{
  display:flex;align-items:center;justify-content:space-between;
  margin-top:32px;padding-top:20px;border-top:1px solid var(--border);
  flex-wrap:wrap;gap:8px;
}
.copy{font-family:'JetBrains Mono',monospace;font-size:.58rem;color:var(--dim);letter-spacing:.04em;}
.trust{
  display:flex;gap:16px;font-family:'JetBrains Mono',monospace;
  font-size:.55rem;font-weight:500;color:var(--dim);letter-spacing:.1em;text-transform:uppercase;
}
"""


def render_site_footer() -> None:
    """Render the full-width site footer."""
    eq = _eq_html(5, "#F5640A", 16)

    html = f"""<!doctype html><html><head>{_head(_FOOTER_CSS)}</head><body>
<footer>
  <div class="grid">
    <div>
      <a href="./" target="_top" style="display:inline-flex;align-items:flex-end;gap:6px;margin-bottom:4px;text-decoration:none;">
        <div style="display:flex;align-items:flex-end;gap:2px;height:16px;">{eq}</div>
        <div class="brand-name">SYNC-SAFE™</div>
      </a>
      <div class="brand-sub">Forensic Portal</div>
      <div class="tagline">
        Detect AI authorship · Audit sync compliance · Flag lyric risks
        before a track costs you a placement.
      </div>
      <div class="badges">
        <span class="badge">⬡ Hugging Face ZeroGPU</span>
        <span class="badge">⬡ Stateless Architecture</span>
        <span class="badge">⬡ No Audio Stored</span>
      </div>
    </div>

    <div>
      <div class="col-title">Product</div>
      <a class="lnk" href="./"              target="_top">Home</a>
      <a class="lnk" href="?p=portal"       target="_top">The Portal</a>
      <a class="lnk" href="?p=how_it_works" target="_top">How it Works</a>
    </div>

    <div>
      <div class="col-title">Rights Resources</div>
      <a class="lnk lnk-ext" href="https://www.ascap.com/repertory"            target="_blank" rel="noopener noreferrer">ASCAP Repertory</a>
      <a class="lnk lnk-ext" href="https://www.bmi.com/search/"                target="_blank" rel="noopener noreferrer">BMI Repertoire</a>
      <a class="lnk lnk-ext" href="https://www.sesac.com/#!/repertory/search"  target="_blank" rel="noopener noreferrer">SESAC Repertory</a>
      <a class="lnk lnk-ext" href="https://www.globalmusicrights.com/search"   target="_blank" rel="noopener noreferrer">GMR Search</a>
    </div>

    <div>
      <div class="col-title">Legal</div>
      <a class="lnk" href="?p=legal" target="_top">Copyright &amp; IP</a>
      <a class="lnk" href="?p=legal" target="_top">Privacy Policy</a>
      <a class="lnk" href="?p=legal" target="_top">Terms of Service</a>
    </div>
  </div>

  <div class="bottom">
    <span class="copy">© 2026 Sync-Safe. All rights reserved.</span>
    <div class="trust">
      <span>Stateless</span><span>·</span>
      <span>No Audio Stored</span><span>·</span>
      <span>ZeroGPU</span>
    </div>
  </div>
</footer>
{_THEME_JS}</body></html>"""

    components.html(html, height=_FOOTER_HEIGHT, scrolling=False)
