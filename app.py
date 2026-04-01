"""
Sync-Safe Forensic Portal
Hugging Face ZeroGPU — Stateless, no database.

app.py is intentionally thin: page config, session state, CSS injection,
and routing. All rendering logic lives in ui/pages/. All business logic
lives in pipeline.py and services/.
"""
import streamlit as st
import streamlit.components.v1 as components

from core.config import get_settings
from core.logging import DEFAULT_LOG_DIR, LogCleaner
from ui.styles import STYLES

# ── Logging setup (runs once per process) ────────────────────────────────────
# Module-level execution ensures LogCleaner runs exactly once per process, not
# once per browser session. session_state would run it once per tab, which
# causes concurrent unlink() races when multiple users open the app.
_log_dir = get_settings().log_dir or DEFAULT_LOG_DIR
LogCleaner(_log_dir).clean()

# ── Theme FAB HTML + JS ───────────────────────────────────────────────────────
# The FAB is injected via st.markdown (position:fixed, always on screen).
# The JS runs via components.html() because st.markdown() does NOT execute
# <script> tags — React's dangerouslySetInnerHTML intentionally skips them.
# The JS accesses window.parent to reach the top-level Streamlit document.

_THEME_FAB = """
<div id="theme-fab">
  <button id="theme-toggle"
          aria-label="Toggle light and dark theme"
          aria-pressed="false">&#9790;</button>
</div>
"""

# NOTE: onclick is intentionally NOT on the button above.
# Putting onclick="string" in st.markdown HTML causes React error #231
# (React sees it as an onClick prop with a string value).
# Instead we attach the listener via addEventListener from the JS below.
#
# localStorage is wrapped in its own try/catch because Safari's ITP blocks
# iframe storage access; the rest of the JS still runs without it.

_THEME_JS = """<script>
(function() {
  try {
    var p = window.parent;

    // No localStorage — accessing window.parent.localStorage in a srcdoc iframe
    // causes a console error in Safari (ITP blocks it below try/catch level).
    // data-theme on <html> already survives Streamlit reruns (React never touches
    // that attribute), so theme persists for the whole browser session.
    var saved = p.document.documentElement.getAttribute('data-theme') || 'light';

    function applyTheme(t) {
      if (t === 'dark') {
        p.document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        p.document.documentElement.removeAttribute('data-theme');
      }
      var b = p.document.getElementById('theme-toggle');
      if (b) {
        b.textContent = t === 'dark' ? '\u2600' : '\u263e';
        b.setAttribute('aria-pressed', t === 'dark' ? 'true' : 'false');
      }
    }

    applyTheme(saved);

    var btn = p.document.getElementById('theme-toggle');
    if (btn && !btn._ssListenerAttached) {
      btn._ssListenerAttached = true;
      btn.addEventListener('click', function() {
        var isDark = p.document.documentElement.getAttribute('data-theme') === 'dark';
        applyTheme(isDark ? 'light' : 'dark');
      });
    }

    /* Tooltip ARIA enhancement — runs on every page */
    function enhanceTips(root) {
      root.querySelectorAll('.tip-icon:not([tabindex])').forEach(function(el) {
        el.setAttribute('tabindex', '0');
        el.setAttribute('role', 'button');
        el.setAttribute('aria-label', 'Show more information');
      });
      root.querySelectorAll('.tip-box:not([role])').forEach(function(el) {
        el.setAttribute('role', 'tooltip');
      });
    }
    enhanceTips(p.document);
    if (!p._ssTooltipObserver) {
      p._ssTooltipObserver = new p.MutationObserver(function() { enhanceTips(p.document); });
      p._ssTooltipObserver.observe(p.document.body, { childList: true, subtree: true });
    }

    // Nav/footer button styling.
    // React re-renders overwrite className, so we stamp data-nav-btn="nav" / "footer"
    // directly onto button elements — data-* attributes survive reconciliation.
    // CSS in styles.py targets these to make buttons look like plain text links.
    function markNavBtns(root) {
      [['ss-nav-marker', 'nav'], ['ss-footer-marker', 'footer']].forEach(function(pair) {
        root.querySelectorAll('.' + pair[0]).forEach(function(marker) {
          var wrapper = marker.closest('[data-testid="stVerticalBlock"]');
          var container = wrapper ? wrapper.parentElement : null;
          while (container && container.dataset.testid !== 'stVerticalBlock') {
            container = container.parentElement;
          }
          if (!container) return;
          container.querySelectorAll('[data-testid="stBaseButton-secondary"]').forEach(function(btn) {
            btn.setAttribute('data-nav-btn', pair[1]);
          });
          // Primary CTA button ("Launch Portal →") — stamp separately so CSS
          // can constrain its size to match the nav bar height.
          if (pair[1] === 'nav') {
            container.querySelectorAll('[data-testid="stBaseButton-primary"]').forEach(function(btn) {
              btn.setAttribute('data-nav-btn', 'nav-cta');
            });
          }
        });
      });
    }
    markNavBtns(p.document);
    if (!p._ssNavObserver) {
      p._ssNavObserver = new p.MutationObserver(function() { markNavBtns(p.document); });
      p._ssNavObserver.observe(p.document.body, { childList: true, subtree: true });
    }
  } catch(e) { console.warn('Theme init:', e); }
})();
</script>"""

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sync-Safe Forensic Portal",
    page_icon="🎚️",
    layout="wide",
)

# ── CSS + Theme FAB ───────────────────────────────────────────────────────────

st.markdown(STYLES, unsafe_allow_html=True)
st.markdown(_THEME_FAB, unsafe_allow_html=True)
components.html(_THEME_JS, height=0, scrolling=False)

# ── Session state defaults ────────────────────────────────────────────────────
# audio:    AudioBuffer | None — ingested on landing page submit
# analysis: AnalysisResult | None — computed on first report page render
# page:     "landing" | "loading" | "report" | "how_it_works" | "legal"
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

# ── URL-based routing (site nav uses ?p= query params) ───────────────────────
# The nav bar runs in a components.html() iframe and navigates via
# window.parent.location.href — a full page reload that Streamlit picks up
# here as a query param. This overrides session state so the correct
# page renders immediately on load without an extra rerun.

_nav_page = st.query_params.get("p", "")
if _nav_page in ("how_it_works", "legal", "portal"):
    st.session_state.page = _nav_page

# ── Routing ───────────────────────────────────────────────────────────────────

if st.session_state.page == "portal":
    from ui.pages.portal import render_portal
    render_portal()
    st.stop()

if st.session_state.page == "how_it_works":
    from ui.pages.how_it_works import render_how_it_works
    render_how_it_works()
    st.stop()

if st.session_state.page == "legal":
    from ui.pages.legal import render_legal
    render_legal()
    st.stop()

if st.session_state.page == "landing":
    from ui.pages.landing import render_landing
    render_landing()
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
render_report(audio, analysis)
