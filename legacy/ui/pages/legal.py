"""
ui/pages/legal.py
Legal, copyright, and privacy information for Sync-Safe Forensic Portal.
Three tabs: Copyright & IP, Privacy Policy, Terms of Service.
"""
from __future__ import annotations

import streamlit as st

from ui.nav import render_site_nav, render_site_footer


def render_legal() -> None:
    """Render the legal information page."""
    render_site_nav("legal")

    _, col, _ = st.columns([1, 3.2, 1])
    with col:
        # ── Page header ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:52px 0 40px;animation:fade-up .5s ease both;">
          <div style="font-family:'Chakra Petch',monospace;font-size:.62rem;font-weight:600;
                      letter-spacing:.28em;text-transform:uppercase;color:var(--dim);
                      margin-bottom:14px;">Legal Information</div>
          <div style="font-family:'Chakra Petch',monospace;font-size:clamp(2rem,5vw,3.2rem);
                      font-weight:700;color:var(--text);line-height:.95;letter-spacing:-.02em;
                      margin-bottom:8px;">Intellectual Property<br>
            <span style="color:var(--accent);">&amp; Legal.</span>
          </div>
          <div style="font-family:'Figtree',sans-serif;font-size:.9rem;color:var(--muted);
                      line-height:1.6;max-width:520px;margin-top:18px;">
            Sync-Safe is a professional forensic tool. The information below governs
            its use, ownership, and data handling commitments.
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Copyright & IP", "Privacy Policy", "Terms of Service"])

        # ── Tab 1: Copyright & IP ─────────────────────────────────────────────
        with tab1:
            st.markdown("""
            <div style="padding:28px 0 16px;">

              <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                          letter-spacing:.2em;text-transform:uppercase;color:var(--dim);
                          margin-bottom:24px;">Last updated: 2026</div>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                1. Ownership of Intellectual Property
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                Copyright © 2026 [OWNER NAME]. All rights reserved.
                The Sync-Safe Forensic Portal (the "Application"), including but not limited
                to its source code, object code, user interface design, logos, spectral
                fingerprinting algorithms, and forensic scoring methodologies, is the exclusive
                intellectual property of [OWNER NAME].
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                2. Proprietary Sync Compliance Logic
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                The sync compliance auditing logic — specifically the automated analysis of
                sting endings, 4–8 bar energy evolution, broadcast-standard intro length
                metrics, broadcast loudness measurement (EBU R128 / ITU-R BS.1770-4), and
                dialogue compatibility scoring — represents a proprietary implementation
                developed by [OWNER NAME]. Unauthorized reverse engineering, replication,
                or redistribution of these analysis frameworks is strictly prohibited.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                3. Data Privacy &amp; Ephemeral Processing
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:12px;">
                Sync-Safe Forensic Portal is a stateless application hosted on Hugging Face
                ZeroGPU. To ensure the highest level of security for music supervisors and
                rights holders:
              </p>
              <ul style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                         line-height:1.7;margin-bottom:24px;padding-left:24px;">
                <li style="margin-bottom:8px;">
                  <strong>No Retention:</strong> User-uploaded audio files are processed in an
                  ephemeral, session-scoped environment.
                </li>
                <li style="margin-bottom:8px;">
                  <strong>Privacy by Design:</strong> No audio data is stored, cached, or
                  persisted after the analysis session is closed.
                </li>
                <li>
                  <strong>No Training:</strong> User data is never used to train AI models
                  or shared with third parties.
                </li>
              </ul>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                4. Forensic Disclaimer (Not Legal Advice)
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:12px;">
                The Application provides automated analytical outputs based on AI authorship
                detection, C2PA manifest analysis, and structural auditing.
              </p>
              <ul style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                         line-height:1.7;margin-bottom:24px;padding-left:24px;">
                <li style="margin-bottom:8px;">
                  <strong>Analytical Only:</strong> These outputs are provided for informational
                  and vetting purposes only.
                </li>
                <li style="margin-bottom:8px;">
                  <strong>No Guarantee:</strong> Sync-Safe Forensic Portal does not provide a
                  guarantee of licensability or a definitive legal clearance of rights.
                </li>
                <li>
                  <strong>Professional Consultation:</strong> Use of this tool does not
                  constitute legal advice. Users are encouraged to consult with legal counsel
                  and official Performance Rights Organisations (e.g., ASCAP, BMI, SESAC)
                  for final rights verification.
                </li>
              </ul>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                5. Trademarks
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:32px;">
                "Sync-Safe," "Sync-Safe Forensic Portal," and all associated logos are
                trademarks of [OWNER NAME]. All other trademarks, including Whisper,
                Hugging Face, ASCAP, BMI, and SESAC, are the property of their respective
                owners and are used herein for identification purposes only.
              </p>

            </div>
            """, unsafe_allow_html=True)

        # ── Tab 2: Privacy Policy ─────────────────────────────────────────────
        with tab2:
            st.markdown("""
            <div style="padding:28px 0 16px;">

              <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                          letter-spacing:.2em;text-transform:uppercase;color:var(--dim);
                          margin-bottom:24px;">Last updated: 2026</div>

              <div style="background:var(--s2);border-radius:12px;padding:20px 24px;
                          border-left:3px solid var(--ok);margin-bottom:28px;">
                <div style="font-family:'Chakra Petch',monospace;font-size:.6rem;font-weight:700;
                            color:var(--ok);letter-spacing:.14em;text-transform:uppercase;
                            margin-bottom:8px;">🔒 Stateless by Design</div>
                <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                           line-height:1.65;margin:0;">
                  Sync-Safe collects no personal data and retains no audio after a session
                  ends. All models run locally on the ZeroGPU server — your audio never
                  leaves the analysis infrastructure.
                </p>
              </div>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                Audio Data
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                Audio submitted for analysis is processed in-memory using <code style="font-family:
                'JetBrains Mono',monospace;font-size:.76rem;background:var(--badge-bg);
                padding:1px 5px;border-radius:3px;">io.BytesIO</code> where possible. Some
                analysis stages (structural analysis, compliance checks) require temporary
                files on disk; these are written to isolated temp directories and deleted
                immediately after processing via <code style="font-family:'JetBrains Mono',
                monospace;font-size:.76rem;background:var(--badge-bg);padding:1px 5px;
                border-radius:3px;">try/finally</code> cleanup. No audio is cached to a
                database, stored in any cloud object store, or retained after your session
                ends. All models — Whisper, spaCy, allin1, librosa — run locally on the
                ZeroGPU instance. Your audio never leaves the analysis server.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                Personal Data
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                Sync-Safe does not require account creation, login, or any personal information
                to use. No names, email addresses, IP addresses, or usage patterns are collected
                or stored. There are no analytics, tracking pixels, or third-party cookies
                on this application.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                Session State
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                Application state (audio buffer, analysis results, current page) is held
                in Streamlit's server-side in-memory session state. This state is never
                serialised to disk or transmitted to any storage system, and is automatically
                discarded when the session ends.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                API Keys &amp; Credentials
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                All API keys (Last.fm) are stored as environment variables in Hugging Face
                Secrets and are never exposed to the client, logged, or included in any
                response payload. All AI models run locally — no external model API
                credentials are required or used.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                AI Model Training
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:32px;">
                Audio submitted to Sync-Safe is never used to train, fine-tune, or evaluate
                any AI or machine learning model. Analysis is inference-only.
              </p>

            </div>
            """, unsafe_allow_html=True)

        # ── Tab 3: Terms of Service ───────────────────────────────────────────
        with tab3:
            st.markdown("""
            <div style="padding:28px 0 16px;">

              <div style="font-family:'Chakra Petch',monospace;font-size:.56rem;font-weight:600;
                          letter-spacing:.2em;text-transform:uppercase;color:var(--dim);
                          margin-bottom:24px;">Last updated: 2026</div>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                1. Acceptance of Terms
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                By using Sync-Safe Forensic Portal you agree to these Terms of Service.
                If you do not agree, do not use the Application.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                2. Permitted Use
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                Sync-Safe is provided for lawful music clearance, forensic auditing, and
                sync licensing research purposes. You must have the legal right to submit
                any audio you analyze — whether through ownership, license, or fair use.
                Do not submit audio to which you have no rights. Automated bulk submissions,
                scraping, or use of the tool to circumvent copyright protections are prohibited.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                3. Analytical Output — Not Legal Advice
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                The reports, scores, flags, and verdicts produced by Sync-Safe are automated
                analytical outputs, not legal opinions. They do not constitute a clearance,
                a license, or a guarantee of any kind. You remain solely responsible for
                all clearance and licensing decisions made in reliance on or separate from
                these outputs. Consult qualified legal counsel before making placement decisions.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                4. Service Availability
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                Sync-Safe runs on Hugging Face ZeroGPU, a free-tier shared GPU infrastructure
                with a 25-minute daily GPU quota per user. The service is provided as-is with
                no uptime guarantee, no SLA, and no warranty of availability. Cold starts
                and queue wait times are expected. We are not liable for delays, failures,
                or inaccuracies caused by infrastructure constraints.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                5. Limitation of Liability
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:24px;">
                To the fullest extent permitted by law, [OWNER NAME] shall not be liable
                for any direct, indirect, incidental, special, consequential, or exemplary
                damages arising from your use of, or inability to use, the Application —
                including but not limited to losses arising from licensing decisions made
                in reliance on the Application's outputs.
              </p>

              <h3 style="font-family:'Chakra Petch',monospace;font-size:.88rem;font-weight:700;
                         color:var(--text);letter-spacing:.06em;margin-bottom:12px;">
                6. Changes to Terms
              </h3>
              <p style="font-family:'Figtree',sans-serif;font-size:.88rem;color:var(--muted);
                        line-height:1.7;margin-bottom:32px;">
                These terms may be updated at any time. Continued use of the Application
                following any update constitutes acceptance of the revised terms.
              </p>

            </div>
            """, unsafe_allow_html=True)

    render_site_footer()
