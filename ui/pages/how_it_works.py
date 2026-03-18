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
                          margin-top:4px;line-height:1.4;">YouTube URL or audio file</div>
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
            <div style="padding:4px 0 8px;">
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:20px;">
                The authorship verdict draws on four independent signals. Each signal
                can flag AI origin independently — they are combined for a final
                <strong>Likely AI / Uncertain / Likely Human</strong> verdict.
              </p>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:8px;">

                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🔏 C2PA Manifest</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.6;">
                    The Coalition for Content Provenance and Authenticity (C2PA) standard
                    embeds cryptographically signed metadata directly into audio and image
                    files at the point of creation. AI generation tools that implement C2PA
                    write a <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">
                    c2pa.ai.generative</code> assertion into the manifest.
                    Sync-Safe parses this manifest using <code style="font-family:'JetBrains Mono',
                    monospace;font-size:.72rem;background:var(--badge-bg);padding:1px 5px;
                    border-radius:3px;">c2pa-python</code> v2.3+ and surfaces any AI origin
                    claims. A positive C2PA assertion is the strongest possible signal —
                    cryptographically certified by the creating tool.
                  </p>
                </div>

                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🥁 IBI Variance</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.6;">
                    Human drummers exhibit natural micro-timing imperfections — a phenomenon
                    musicians call "groove." These inter-beat interval (IBI) deviations
                    typically range from ±8–30ms variance. AI-generated drums, by contrast,
                    are locked to a mathematical grid, often with variance below 0.5ms².
                    Sync-Safe uses <code style="font-family:'JetBrains Mono',monospace;
                    font-size:.72rem;background:var(--badge-bg);padding:1px 5px;
                    border-radius:3px;">librosa</code> beat tracking to extract IBI sequences
                    and flags "Perfect Quantization" when variance falls below threshold —
                    a strong statistical signal of AI or MIDI-locked generation.
                  </p>
                </div>

                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">📡 Spectral Fingerprint</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.6;">
                    Most AI audio synthesis models exhibit characteristic artifacts in the
                    high-frequency spectrum (16 kHz+). This region is heavily shaped by
                    the mel-filterbank and vocoder layers used in diffusion-based generation.
                    Sync-Safe analyzes spectral contrast across frequency bands and flags
                    anomalous distributions in the 16 kHz+ range that deviate from patterns
                    typical of recorded acoustic instruments and studio-produced audio.
                  </p>
                </div>

                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🔁 Loop Detection</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.6;">
                    AI composition tools and stock music libraries frequently reuse identical
                    audio segments across a track. Sync-Safe splits the audio into 4-bar and
                    8-bar windows, extracts mel-spectrogram fingerprints for each, and computes
                    cross-correlation across all segment pairs. A correlation score above
                    <strong>0.98</strong> flags the segment pair as a likely stock loop or
                    AI repetition artifact — a signal that the track may not be original
                    compositional work.
                  </p>
                </div>

              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Stage 2: Sync Compliance ──────────────────────────────────────────
        with st.expander("Stage 2 — Sync Compliance Auditing"):
            st.markdown("""
            <div style="padding:4px 0 8px;">
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:20px;">
                Sync licensing has implicit production standards that most AI-generated and
                stock library tracks fail. These three checks identify tracks that will
                cause problems for broadcast editors in post-production.
              </p>

              <div style="display:flex;flex-direction:column;gap:16px;">

                <div style="background:var(--s2);border-radius:10px;padding:18px;
                            border-left:3px solid var(--accent);">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                              color:var(--text);letter-spacing:.1em;margin-bottom:10px;">
                    🔔 Sting Check — Does the track end cleanly?
                  </div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.65;">
                    A <strong>"sting"</strong> is a track that ends on a hard cut — the music
                    stops cleanly rather than fading out or ringing. This is the preferred
                    format for sync because editors can cut cleanly to dialogue or SFX without
                    a tail of reverb or fade bleeding into the scene. Sync-Safe measures the
                    RMS energy of the final 2 seconds as a ratio against the track mean. If the
                    ratio falls below the configured threshold, or if the track ends on a
                    root-note hold, the track is flagged. Conversely, tracks that fade or ring
                    out may require editorial trimming before broadcast use.
                  </p>
                </div>

                <div style="background:var(--s2);border-radius:10px;padding:18px;
                            border-left:3px solid var(--accent);">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                              color:var(--text);letter-spacing:.1em;margin-bottom:10px;">
                    📊 4–8 Bar Energy Evolution — Does the track breathe?
                  </div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.65;">
                    Broadcast editors work with tracks in 4-bar windows — cutting, looping, and
                    layering music against picture. A track that doesn't evolve across those
                    windows (same spectral density, same dynamic level, same energy) becomes
                    aurally fatiguing and fights the edit rather than supporting it. Sync-Safe
                    uses the beat grid from <code style="font-family:'JetBrains Mono',monospace;
                    font-size:.72rem;background:var(--badge-bg);padding:1px 5px;
                    border-radius:3px;">allin1</code> structure analysis combined with librosa
                    spectral contrast to verify that energy evolves by at least the minimum
                    configured delta across every 4-bar window. Stagnant segments are flagged
                    with their timestamp for supervisor review.
                  </p>
                </div>

                <div style="background:var(--s2);border-radius:10px;padding:18px;
                            border-left:3px solid var(--accent);">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                              color:var(--text);letter-spacing:.1em;margin-bottom:10px;">
                    ⏱️ Intro Timer — Does the track start within 15 seconds?
                  </div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.65;">
                    The 15-second intro standard is a practical constraint of the sync pitch
                    process: music supervisors, trailer editors, and ad creative directors
                    audition hundreds of tracks and rarely listen past the first 15 seconds
                    before making a pass/continue decision. A track whose first vocal, melodic
                    hook, or primary instrument doesn't enter until after 15 seconds is
                    structurally unsuited for most sync contexts. Sync-Safe reads the section
                    labels produced by the <code style="font-family:'JetBrains Mono',monospace;
                    font-size:.72rem;background:var(--badge-bg);padding:1px 5px;
                    border-radius:3px;">allin1</code> model and flags any segment labelled
                    "intro" that exceeds the 15-second limit.
                  </p>
                </div>

              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Stage 3: Lyric Audit ──────────────────────────────────────────────
        with st.expander("Stage 3 — Lyric Risk Audit"):
            st.markdown("""
            <div style="padding:4px 0 8px;">
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:20px;">
                Lyric clearance is one of the most time-consuming parts of sync licensing.
                Sync-Safe automates the first-pass audit: transcribing the track and running
                four independent checks to surface risks before a human reviewer reads a word.
              </p>
              <div style="display:flex;flex-direction:column;gap:12px;">

                <div style="display:flex;gap:14px;align-items:flex-start;background:var(--s2);
                            border-radius:10px;padding:16px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:1.2rem;flex-shrink:0;
                              margin-top:2px;">①</div>
                  <div>
                    <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                                color:var(--text);letter-spacing:.08em;margin-bottom:6px;">
                      Whisper Transcription
                    </div>
                    <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                               line-height:1.6;">
                      OpenAI Whisper (Base model) converts vocals to timestamped JSON segments.
                      Each segment carries a start time, end time, and text — providing the
                      timestamp index that all downstream checks attach their flags to.
                    </p>
                  </div>
                </div>

                <div style="display:flex;gap:14px;align-items:flex-start;background:var(--s2);
                            border-radius:10px;padding:16px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:1.2rem;flex-shrink:0;
                              margin-top:2px;">②</div>
                  <div>
                    <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                                color:var(--text);letter-spacing:.08em;margin-bottom:6px;">
                      Explicit Content Detection
                    </div>
                    <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                               line-height:1.6;">
                      The <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                      background:var(--badge-bg);padding:1px 5px;border-radius:3px;">
                      profanity-check</code> library runs against each Whisper segment and flags
                      any segment containing explicit language. Flagged segments receive an
                      <strong>EXPLICIT</strong> classification with the offending timestamp
                      and a recommendation for editorial action.
                    </p>
                  </div>
                </div>

                <div style="display:flex;gap:14px;align-items:flex-start;background:var(--s2);
                            border-radius:10px;padding:16px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:1.2rem;flex-shrink:0;
                              margin-top:2px;">③</div>
                  <div>
                    <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                                color:var(--text);letter-spacing:.08em;margin-bottom:6px;">
                      Named Entity Recognition — Brands &amp; Locations
                    </div>
                    <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                               line-height:1.6;">
                      spaCy's <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                      background:var(--badge-bg);padding:1px 5px;border-radius:3px;">
                      en_core_web_sm</code> model runs NER on the transcript. Organisation
                      entities (<strong>ORG</strong>) flag potential brand mentions that may
                      create trademark liability; geopolitical entities (<strong>GPE</strong>)
                      flag location references that can restrict territorial clearance. Both
                      are surfaced with timestamps and reviewer recommendations.
                    </p>
                  </div>
                </div>

                <div style="display:flex;gap:14px;align-items:flex-start;background:var(--s2);
                            border-radius:10px;padding:16px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:1.2rem;flex-shrink:0;
                              margin-top:2px;">④</div>
                  <div>
                    <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:700;
                                color:var(--text);letter-spacing:.08em;margin-bottom:6px;">
                      Safety Zone Classification
                    </div>
                    <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                               line-height:1.6;">
                      Curated keyword dictionaries flag violence-adjacent and drug-related
                      terminology that would disqualify a track from family-rated content,
                      broadcast standards compliance, or brand-safe placement contexts.
                      Segments matching these terms receive <strong>VIOLENCE</strong> or
                      <strong>DRUGS</strong> classifications in the audit table.
                    </p>
                  </div>
                </div>

              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Stage 4: Rights Discovery ─────────────────────────────────────────
        with st.expander("Stage 4 — Rights Discovery"):
            st.markdown("""
            <div style="padding:4px 0 8px;">
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:20px;">
                After analysis, the tool surfaces similar tracks and generates direct
                licensing lookup links — reducing the time between "this track works"
                and "I know who to call."
              </p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">

                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">🔍 Audio Similarity</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.6;">
                    Last.fm's <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">
                    track.getSimilar</code> API returns a ranked list of comparable tracks
                    based on listener co-occurrence and editorial tagging. Each result is then
                    resolved to a live YouTube preview URL via
                    <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                    background:var(--badge-bg);padding:1px 5px;border-radius:3px;">
                    yt-dlp ytsearch1:</code> — fully stateless, no database required.
                    This gives supervisors an instant reference set for similar placements
                    or safe alternatives if the analyzed track fails audit.
                  </p>
                </div>

                <div style="background:var(--s2);border-radius:10px;padding:18px;">
                  <div style="font-family:'Chakra Petch',monospace;font-size:.58rem;font-weight:700;
                              color:var(--accent);letter-spacing:.14em;text-transform:uppercase;
                              margin-bottom:8px;">⚖️ PRO Licensing Links</div>
                  <p style="font-family:'Figtree',sans-serif;font-size:.8rem;color:var(--muted);
                             line-height:1.6;">
                    Sync-Safe generates direct deep-link search URLs for the four major
                    Performance Rights Organisations — ASCAP, BMI, SESAC, and GMR — pre-filled
                    with the artist and track name. This gives the clearance agent a single
                    click to identify publishers and rights holders, confirm registration
                    status, and initiate licensing outreach. No PRO database credentials
                    are required; all lookups are performed against public repertory portals.
                  </p>
                </div>

              </div>
            </div>
            """, unsafe_allow_html=True)

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
            Every analysis runs in an ephemeral ZeroGPU session scoped to your browser tab.
            Audio data is processed entirely in-memory using <code style="font-family:
            'JetBrains Mono',monospace;font-size:.72rem;background:var(--badge-bg);
            padding:1px 5px;border-radius:3px;">io.BytesIO</code> and is never written
            to disk, cached, logged, or used for model training. When your session ends,
            nothing persists. This is by design — unreleased tracks submitted for clearance
            auditing remain yours alone.
          </p>
        </div>
        """, unsafe_allow_html=True)

    render_site_footer()
