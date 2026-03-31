"""
ui/pages/how_it_works.py
Methodology page — explains the four-stage forensic pipeline in depth.
Audience: music supervisors, clearance agents, and legal teams evaluating
whether to trust the tool's outputs before relying on them in production.
"""
from __future__ import annotations

import streamlit as st

from ui.nav import render_site_nav, render_site_footer


def render_how_it_works() -> None:
    """Render the full How it Works methodology page."""
    render_site_nav("how_it_works")

    _, col, _ = st.columns([1, 3.2, 1])
    with col:
        # ── Page header ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:52px 0 40px;animation:fade-up .5s ease both;">
          <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:600;
                      letter-spacing:.28em;text-transform:uppercase;color:var(--dim);
                      margin-bottom:14px;">Forensic Methodology</div>
          <div style="font-family:'Chakra Petch',monospace;font-size:clamp(2rem,5vw,3.2rem);
                      font-weight:700;color:var(--text);line-height:.95;letter-spacing:-.02em;
                      margin-bottom:8px;">How Sync-Safe<br>
            <span style="color:var(--accent);">Works.</span>
          </div>
          <div style="font-family:'Figtree',sans-serif;font-size:1rem;color:var(--muted);
                      line-height:1.65;max-width:560px;margin-top:20px;">
            Sync-Safe runs four independent analysis stages on every submitted track.
            Each stage answers a distinct question a music supervisor or clearance agent
            needs answered before a placement decision.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Pipeline flow ─────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-bottom:48px;animation:fade-up .6s ease .1s both;">
          <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                      letter-spacing:.2em;text-transform:uppercase;color:var(--dim);
                      display:flex;align-items:center;gap:10px;margin-bottom:20px;">
            <span>▶</span><span>The Pipeline</span>
            <div style="flex:1;height:1px;background:var(--border-hr);"></div>
          </div>
          <div style="display:flex;align-items:stretch;gap:0;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;background:var(--s1);border:1px solid var(--border);
                        border-radius:10px 0 0 10px;padding:16px 14px;text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:6px;">🎵</div>
              <div style="font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
                          color:var(--text);letter-spacing:.06em;">INPUT</div>
              <div style="font-family:'Figtree',sans-serif;font-size:.66rem;color:var(--muted);
                          margin-top:4px;line-height:1.4;">YouTube, Bandcamp, SoundCloud, or audio file</div>
            </div>
            <div style="display:flex;align-items:center;padding:0 4px;color:var(--dim);
                        font-size:.9rem;">→</div>
            <div style="flex:1;min-width:120px;background:var(--s1);border:1px solid var(--border);
                        padding:16px 14px;text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:6px;">🔬</div>
              <div style="font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
                          color:var(--text);letter-spacing:.06em;">AI DETECTION</div>
              <div style="font-family:'Figtree',sans-serif;font-size:.66rem;color:var(--muted);
                          margin-top:4px;line-height:1.4;">C2PA · IBI · Spectral · Loop</div>
            </div>
            <div style="display:flex;align-items:center;padding:0 4px;color:var(--dim);
                        font-size:.9rem;">→</div>
            <div style="flex:1;min-width:120px;background:var(--s1);border:1px solid var(--border);
                        padding:16px 14px;text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:6px;">📋</div>
              <div style="font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
                          color:var(--text);letter-spacing:.06em;">SYNC AUDIT</div>
              <div style="font-family:'Figtree',sans-serif;font-size:.66rem;color:var(--muted);
                          margin-top:4px;line-height:1.4;">Sting · Energy · Intro timer</div>
            </div>
            <div style="display:flex;align-items:center;padding:0 4px;color:var(--dim);
                        font-size:.9rem;">→</div>
            <div style="flex:1;min-width:120px;background:var(--s1);border:1px solid var(--border);
                        padding:16px 14px;text-align:center;">
              <div style="font-size:1.4rem;margin-bottom:6px;">🎤</div>
              <div style="font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
                          color:var(--text);letter-spacing:.06em;">LYRIC AUDIT</div>
              <div style="font-family:'Figtree',sans-serif;font-size:.66rem;color:var(--muted);
                          margin-top:4px;line-height:1.4;">Whisper · NER · Keywords</div>
            </div>
            <div style="display:flex;align-items:center;padding:0 4px;color:var(--dim);
                        font-size:.9rem;">→</div>
            <div style="flex:1;min-width:120px;background:var(--s2);border:1px solid var(--border);
                        border-radius:0 10px 10px 0;padding:16px 14px;text-align:center;
                        border-left:2px solid var(--accent);">
              <div style="font-size:1.4rem;margin-bottom:6px;">📊</div>
              <div style="font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
                          color:var(--accent);letter-spacing:.06em;">REPORT</div>
              <div style="font-family:'Figtree',sans-serif;font-size:.66rem;color:var(--muted);
                          margin-top:4px;line-height:1.4;">Timestamped · Actionable</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Stage 1: AI Authorship Detection ─────────────────────────────────
        with st.expander("Stage 1 — AI Authorship Detection", expanded=True):
            st.markdown("""
            <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                      line-height:1.7;margin:4px 0 20px;">
              The authorship verdict draws on four independent signals. Each signal
              can flag AI origin independently — they are combined for a final
              <strong>Likely AI / Uncertain / Likely Human</strong> verdict.
            </p>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.markdown("""
                <div style="background:var(--s2);border-radius:10px;padding:18px;margin-bottom:12px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🔏 C2PA Manifest</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.6;margin:0;">
                    The C2PA standard embeds cryptographically signed metadata into audio files at
                    creation. AI tools that implement C2PA write a
                    <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">c2pa.ai.generative</code>
                    assertion into the manifest. Sync-Safe parses this using
                    <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">c2pa-python</code> v2.3+.
                    A positive assertion is the strongest possible signal — cryptographically certified.
                  </p>
                </div>""", unsafe_allow_html=True)
                st.markdown("""
                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">📡 Spectral Fingerprint</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.6;margin:0;">
                    Most AI audio synthesis models exhibit characteristic artifacts in the
                    high-frequency spectrum (16 kHz+), shaped by mel-filterbank and vocoder layers.
                    Sync-Safe analyzes spectral contrast across frequency bands and flags anomalous
                    distributions in the 16 kHz+ range that deviate from patterns typical of
                    recorded acoustic instruments and studio-produced audio.
                  </p>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div style="background:var(--s2);border-radius:10px;padding:18px;margin-bottom:12px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🥁 IBI Variance</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.6;margin:0;">
                    Human drummers exhibit natural micro-timing imperfections — inter-beat interval
                    (IBI) deviations typically ranging ±8–30ms. AI-generated drums are locked to a
                    mathematical grid, often below 0.5ms² variance. Sync-Safe uses
                    <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">librosa</code>
                    beat tracking and flags "Perfect Quantization" when variance falls below threshold.
                  </p>
                </div>""", unsafe_allow_html=True)
                st.markdown("""
                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🔁 Loop Detection</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.6;margin:0;">
                    Sync-Safe splits audio into 4-bar and 8-bar windows, extracts mel-spectrogram
                    fingerprints for each, and computes cross-correlation across all segment pairs.
                    A score above <strong>0.98</strong> flags the pair as a likely stock loop or
                    AI repetition artifact — a signal the track may not be original compositional work.
                  </p>
                </div>""", unsafe_allow_html=True)
            st.markdown('<div style="padding-bottom:8px;"></div>', unsafe_allow_html=True)

        # ── Stage 2: Sync Compliance ──────────────────────────────────────────
        with st.expander("Stage 2 — Sync Compliance Auditing"):
            st.markdown("""
            <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                      line-height:1.7;margin:4px 0 20px;">
              Sync licensing has implicit production standards that most AI-generated and
              stock library tracks fail. These three checks identify tracks that will
              cause problems for broadcast editors in post-production.
            </p>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:var(--s2);border-radius:10px;padding:18px;
                        border-left:3px solid var(--accent);margin-bottom:12px;">
              <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                          color:var(--text);letter-spacing:.1em;margin-bottom:10px;">
                🔔 Sting Check — Does the track end cleanly?
              </div>
              <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.65;">
                A <strong>"sting"</strong> is a track that ends on a hard cut — the music stops cleanly
                rather than fading out or ringing. Sync-Safe measures the RMS energy of the final 2 seconds
                as a ratio against the track mean. If the ratio falls below the configured threshold, or if
                the track ends on a root-note hold, the track is flagged.
              </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:var(--s2);border-radius:10px;padding:18px;
                        border-left:3px solid var(--accent);margin-bottom:12px;">
              <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                          color:var(--text);letter-spacing:.1em;margin-bottom:10px;">
                📊 4–8 Bar Energy Evolution — Does the track breathe?
              </div>
              <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.65;">
                Sync-Safe uses the beat grid from
                <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                background:var(--badge-bg);padding:1px 5px;border-radius:3px;">allin1</code>
                combined with librosa spectral contrast to verify that energy evolves by at least the
                minimum configured delta across every 4-bar window. Stagnant segments are flagged with
                their timestamp for supervisor review.
              </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:var(--s2);border-radius:10px;padding:18px;
                        border-left:3px solid var(--accent);margin-bottom:4px;">
              <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                          color:var(--text);letter-spacing:.1em;margin-bottom:10px;">
                ⏱️ Intro Timer — Does the track start within 15 seconds?
              </div>
              <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.65;">
                Sync-Safe reads the section labels produced by the
                <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                background:var(--badge-bg);padding:1px 5px;border-radius:3px;">allin1</code>
                model and flags any segment labelled "intro" that exceeds the 15-second limit —
                a practical constraint of the sync pitch process where supervisors rarely listen
                past the first 15 seconds before making a pass/continue decision.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Stage 3: Lyric Audit ──────────────────────────────────────────────
        with st.expander("Stage 3 — Lyric Risk Audit"):
            st.markdown("""
            <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                      line-height:1.7;margin:4px 0 20px;">
              Lyric clearance is one of the most time-consuming parts of sync licensing.
              Sync-Safe automates the first-pass audit: transcribing the track and running
              four independent checks to surface risks before a human reviewer reads a word.
            </p>
            """, unsafe_allow_html=True)
            for num, title, body in [
                ("①", "Whisper Transcription",
                 "OpenAI Whisper (Base model) converts vocals to timestamped JSON segments. "
                 "Each segment carries a start time, end time, and text — providing the "
                 "timestamp index that all downstream checks attach their flags to."),
                ("②", "Explicit Content Detection",
                 "The <code style=\"font-family:'JetBrains Mono',monospace;font-size:.72rem;"
                 "background:var(--badge-bg);padding:1px 5px;border-radius:3px;\">profanity-check</code> "
                 "library runs against each Whisper segment and flags any segment containing explicit "
                 "language. Flagged segments receive an <strong>EXPLICIT</strong> classification with "
                 "the offending timestamp and a recommendation for editorial action."),
                ("③", "Named Entity Recognition — Brands &amp; Locations",
                 "spaCy's <code style=\"font-family:'JetBrains Mono',monospace;font-size:.72rem;"
                 "background:var(--badge-bg);padding:1px 5px;border-radius:3px;\">en_core_web_sm</code> "
                 "model runs NER on the transcript. <strong>ORG</strong> entities flag potential brand "
                 "mentions that may create trademark liability; <strong>GPE</strong> entities flag "
                 "location references that can restrict territorial clearance."),
                ("④", "Safety Zone Classification",
                 "Curated keyword dictionaries flag violence-adjacent and drug-related terminology "
                 "that would disqualify a track from family-rated content or brand-safe placement. "
                 "Segments matching these terms receive <strong>VIOLENCE</strong> or "
                 "<strong>DRUGS</strong> classifications in the audit table."),
            ]:
                c_num, c_body = st.columns([0.06, 0.94], gap="small")
                with c_num:
                    st.markdown(
                        f'<div style="font-family:\'Chakra Petch\',monospace;font-size:1.2rem;'
                        f'padding-top:18px;text-align:center;color:var(--text);">{num}</div>',
                        unsafe_allow_html=True,
                    )
                with c_body:
                    st.markdown(
                        f'<div style="background:var(--s2);border-radius:10px;padding:16px;margin-bottom:10px;">'
                        f'<div style="font-family:\'Chakra Petch\',monospace;font-size:.62rem;font-weight:700;'
                        f'color:var(--text);letter-spacing:.08em;margin-bottom:6px;">{title}</div>'
                        f'<p style="font-family:\'Figtree\',sans-serif;font-size:.8rem;color:var(--muted);'
                        f'line-height:1.6;margin:0;">{body}</p></div>',
                        unsafe_allow_html=True,
                    )

        # ── Stage 4: Rights Discovery ─────────────────────────────────────────
        with st.expander("Stage 4 — Rights Discovery"):
            st.markdown("""
            <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                      line-height:1.7;margin:4px 0 20px;">
              After analysis, the tool surfaces similar tracks and generates direct
              licensing lookup links — reducing the time between "this track works"
              and "I know who to call."
            </p>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.markdown("""
                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🔍 Audio Similarity</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.6;margin:0;">
                    Last.fm's <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">track.getSimilar</code>
                    API returns a ranked list of comparable tracks based on listener co-occurrence.
                    Each result is resolved to a live YouTube preview via
                    <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">yt-dlp ytsearch1:</code>
                    — fully stateless, no database required.
                  </p>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">⚖️ PRO Licensing Links</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);line-height:1.6;margin:0;">
                    Sync-Safe generates direct deep-link search URLs for ASCAP, BMI, SESAC, and GMR —
                    pre-filled with the artist and track name. No PRO database credentials required;
                    all lookups are performed against public repertory portals.
                  </p>
                </div>""", unsafe_allow_html=True)
            st.markdown('<div style="padding-bottom:8px;"></div>', unsafe_allow_html=True)

        # ── Stateless assurance note ──────────────────────────────────────────
        st.markdown("""
        <div style="margin:40px 0 16px;padding:20px 24px;background:var(--s2);
                    border-radius:12px;border-left:3px solid var(--ok);
                    animation:fade-up .7s ease .2s both;">
          <div style="font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
                      color:var(--ok);letter-spacing:.16em;text-transform:uppercase;
                      margin-bottom:8px;">🔒 Stateless Assurance</div>
          <p style="font-family:'Figtree',sans-serif;font-size:.84rem;color:var(--muted);
                    line-height:1.65;margin:0;">
            Audio submitted for analysis is processed in-memory using <code style="font-family:
            'JetBrains Mono',monospace;font-size:.72rem;background:var(--badge-bg);
            padding:1px 5px;border-radius:3px;">io.BytesIO</code> where possible. Any
            temporary files required by the analysis pipeline are written to isolated temp
            directories and deleted immediately after processing via <code style="font-family:
            'JetBrains Mono',monospace;font-size:.72rem;background:var(--badge-bg);
            padding:1px 5px;border-radius:3px;">try/finally</code> cleanup — they do not
            persist beyond the call that creates them. No audio is retained, logged, or
            transmitted to any storage system after your session ends. All models — Whisper,
            spaCy, allin1, librosa — run locally on the ZeroGPU instance. Your audio never
            leaves the analysis server.
          </p>
        </div>
        """, unsafe_allow_html=True)

    render_site_footer()
