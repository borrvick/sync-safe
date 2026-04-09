"""
services/compliance/_orchestrator.py
Sync readiness compliance checks — implements the ComplianceChecker protocol.

Five checks, one class:
  1. Sting / ending type  — librosa RMS + onset strength
  2. 4-8 bar energy rule  — librosa spectral contrast per window
  3. Intro timer          — allin1 sections + Whisper fallback
  4. Lyric audit          — Detoxify transformer + spaCy NER
  5. Brand keywords       — curated regex list (data/brand_keywords.py)
"""
from __future__ import annotations

import re

import numpy as np

from core.config import CONSTANTS, ModelParams
from core.exceptions import ModelInferenceError
from core.models import (
    AudioBuffer,
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    IntroResult,
    Section,
    StingResult,
    TranscriptSegment,
)
from data.brand_keywords import BRAND_KEYWORDS
from data.profanity_supplement import PROFANITY_SUPPLEMENT

from ._pure import (
    _EXPLICIT_CONFIRMED,
    _EXPLICIT_POTENTIAL,
    _HARD_OBSCENE_THRESHOLD,
    _MIN_SEGMENT_CHARS,
    _NER_ISSUE_MAP,
    _NER_RECOMMENDATIONS,
    _WINDOW_SIZE,
    _build_windows,
    _check_brand_keywords,
    _classify_cut_type,
    _compute_fade_severity,
    _compute_grade,
    _deduplicate_flags,
    _detect_onset_intro_end,
    _score_detoxify,
    _section_energy_note,
)

try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn


class Compliance:
    """
    Applies sync readiness compliance rules to audio and lyrics.

    Implements: ComplianceChecker protocol (core/protocols.py)

    Constructor injection:
        params — ModelParams controlling NLI model name and batch size.

    Usage:
        service = Compliance()
        report  = service.check(audio, transcript, sections, beats)
    """

    def __init__(self, params: ModelParams | None = None) -> None:
        self._params = params or ModelParams()
        # Lazy model handles — loaded on first call, not at import time
        self._detoxify_model   = None
        self._profanity_filter = None
        self._spacy_nlp        = None
        # Brand patterns compiled once here, not at module import
        self._brand_patterns: list[tuple[str, re.Pattern]] = [
            (
                name,
                re.compile(
                    r"(?<!\w)(?:" + "|".join(re.escape(p) for p in patterns) + r")(?!\w)",
                    re.IGNORECASE,
                ),
            )
            for name, patterns in BRAND_KEYWORDS
        ]

    # ------------------------------------------------------------------
    # Public interface (ComplianceChecker protocol)
    # ------------------------------------------------------------------

    def check(
        self,
        audio: AudioBuffer,
        transcript: list[TranscriptSegment],
        sections: list[Section],
        beats: list[float],
    ) -> ComplianceReport:
        """
        Run all sync readiness checks and return a typed ComplianceReport.

        Args:
            audio:      In-memory audio buffer.
            transcript: Whisper segments for the lyric audit.
            sections:   allin1 structural sections for the intro check.
            beats:      allin1 beat timestamps for the energy evolution check.

        Returns:
            ComplianceReport with flags, sub-results, and an A–F grade.

        Raises:
            ModelInferenceError: if the NLI model fails to load.
        """
        y, sr = audio.to_array(CONSTANTS.SAMPLE_RATE)

        sting     = self._check_sting(y, sr, beats)
        evolution = self._check_energy_evolution(y, sr, beats, sections)
        intro     = self._check_intro(y, sr, beats, sections, transcript)
        flags     = self._audit_lyrics(transcript)

        grade = _compute_grade(flags=flags)

        return ComplianceReport(
            flags=flags,
            sting=sting,
            evolution=evolution,
            intro=intro,
            grade=grade,
        )

    # ------------------------------------------------------------------
    # Private: Sting / ending-type check  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _check_sting(
        self, audio: np.ndarray, sr: int, beats: list[float]
    ) -> StingResult:
        """
        Classify the track ending as sting, fade, or cut.

        Args:
            audio: Pre-decoded mono array at CONSTANTS.SAMPLE_RATE.
            sr:    Sample rate of the decoded array.
            beats: Beat timestamps (seconds) for cut-type classification (#104).
        """
        import librosa

        if len(audio) < sr * 4:
            # Too short — report as cut (neutral), no flag
            return StingResult(
                ending_type="cut",
                sync_ready=True,
                final_energy_ratio=1.0,
                flag=False,
                cut_type=_classify_cut_type(len(audio) / sr, beats, CONSTANTS.CUT_BEAT_TOLERANCE_S),
            )

        hop = CONSTANTS.STING_HOP_LENGTH
        rms          = librosa.feature.rms(y=audio, frame_length=CONSTANTS.STING_FRAME_LENGTH, hop_length=hop)[0]
        overall_mean = float(np.mean(rms)) + 1e-9

        # Fade: declining RMS slope over last 10s AND low tail energy
        fade_frames = int((sr * CONSTANTS.FADE_WINDOW_SECONDS) / hop)
        tail_rms    = rms[-fade_frames:]
        tail_mean   = float(np.mean(tail_rms))
        final_ratio = tail_mean / overall_mean
        x           = np.arange(len(tail_rms), dtype=float)
        norm_slope  = float(np.polyfit(x, tail_rms, 1)[0]) / overall_mean
        is_fade     = (norm_slope < CONSTANTS.FADE_SLOPE_THRESHOLD) and (final_ratio < CONSTANTS.FADE_RATIO_MAX)

        # Sting: onset spike ≥ STING_SPIKE_FACTOR × local mean AND
        #        ≥ STING_RMS_DROP_RATIO energy collapse within 1s after peak
        sting_window = audio[-(sr * 3):]
        onset_env    = librosa.onset.onset_strength(y=sting_window, sr=sr)
        onset_mean   = float(np.mean(onset_env)) + 1e-9
        peak_idx     = int(np.argmax(onset_env))
        onset_spike  = float(onset_env[peak_idx]) / onset_mean
        peak_sample  = peak_idx * hop
        pre_rms      = float(
            np.mean(np.abs(sting_window[max(0, peak_sample - sr // 4):peak_sample]))
        ) + 1e-9
        post_end     = min(peak_sample + sr, len(sting_window))
        post_rms     = (
            float(np.mean(np.abs(sting_window[peak_sample:post_end])))
            if post_end > peak_sample else pre_rms
        )
        energy_drop  = 1.0 - (post_rms / pre_rms)
        is_sting     = (
            onset_spike >= CONSTANTS.STING_SPIKE_FACTOR
            and energy_drop >= CONSTANTS.STING_RMS_DROP_RATIO
            and not is_fade
        )

        if is_sting:
            return StingResult(
                ending_type="sting",
                sync_ready=True,
                final_energy_ratio=round(final_ratio, 3),
                flag=False,
                norm_slope=round(norm_slope, 4),
                onset_spike_factor=round(onset_spike, 3),
            )
        if is_fade:
            fade_sev, fade_tail = _compute_fade_severity(
                rms=rms,
                overall_mean=overall_mean,
                sr=sr,
                hop=hop,
                threshold_ratio=CONSTANTS.FADE_TAIL_THRESHOLD_RATIO,
                max_seconds=CONSTANTS.FADE_SEVERITY_MAX_SECONDS,
            )
            return StingResult(
                ending_type="fade",
                sync_ready=False,
                final_energy_ratio=round(final_ratio, 3),
                flag=True,
                fade_severity=fade_sev,
                fade_tail_seconds=fade_tail,
                norm_slope=round(norm_slope, 4),
                onset_spike_factor=round(onset_spike, 3),
            )
        return StingResult(
            ending_type="cut",
            sync_ready=True,
            final_energy_ratio=round(final_ratio, 3),
            flag=False,
            cut_type=_classify_cut_type(len(audio) / sr, beats, CONSTANTS.CUT_BEAT_TOLERANCE_S),
            norm_slope=round(norm_slope, 4),
            onset_spike_factor=round(onset_spike, 3),
        )

    # ------------------------------------------------------------------
    # Private: 4-8 bar energy evolution check  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _check_energy_evolution(
        self,
        audio: np.ndarray,
        sr: int,
        beats: list[float],
        sections: list[Section],
    ) -> EnergyEvolutionResult:
        """
        Verify spectral contrast evolves across every 4-bar window.

        Args:
            audio:    Pre-decoded mono array at CONSTANTS.SAMPLE_RATE.
            sr:       Sample rate of the decoded array.
            beats:    allin1 beat timestamps; falls back to librosa if sparse.
            sections: allin1 sections for per-section breakdown (#106).
        """
        import librosa

        # Fall back to librosa beat tracking when beats are absent or sparse
        if not beats or len(beats) < 8:
            _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        if len(beats) < 8:
            return EnergyEvolutionResult(
                stagnant_windows=0,
                total_windows=0,
                flag=False,
                detail="Not enough beats detected for 4-bar analysis",
            )

        beats_arr = np.asarray(beats, dtype=float)

        def _scan_windows(
            seg_audio: np.ndarray, seg_beats: np.ndarray
        ) -> tuple[int, int]:
            """Return (stagnant, total) window counts for a beat-sliced segment."""
            step = CONSTANTS.BEATS_PER_WINDOW
            wins: list[np.ndarray] = []
            for j in range(0, len(seg_beats) - step, step):
                t0 = seg_beats[j]
                t1 = seg_beats[min(j + step, len(seg_beats) - 1)]
                s0, s1 = int(t0 * sr), int(t1 * sr)
                chunk = seg_audio[max(0, s0):min(s1, len(seg_audio))]
                if len(chunk) > 0:
                    wins.append(chunk)
            if len(wins) < 2:
                return 0, 0
            cs = [float(np.mean(librosa.feature.spectral_contrast(y=w, sr=sr))) for w in wins]
            cmin, cmax = min(cs), max(cs)
            crange = cmax - cmin + 1e-9
            norm = [(c - cmin) / crange for c in cs]
            stag = sum(
                1 for k in range(1, len(wins))
                if abs(norm[k] - norm[k - 1]) < CONSTANTS.ENERGY_DELTA_MIN
            )
            return stag, len(wins) - 1

        # Global scan (existing behaviour — uses full audio with absolute beat times)
        step = CONSTANTS.BEATS_PER_WINDOW
        global_window_starts: list[float] = []
        global_windows: list[np.ndarray]  = []
        for i in range(0, len(beats_arr) - step, step):
            t_start = float(beats_arr[i])
            t_end   = float(beats_arr[min(i + step, len(beats_arr) - 1)])
            s_start, s_end = int(t_start * sr), int(t_end * sr)
            if s_end > s_start:
                global_window_starts.append(t_start)
                global_windows.append(audio[s_start:s_end])

        if len(global_windows) < 2:
            return EnergyEvolutionResult(
                stagnant_windows=0,
                total_windows=0,
                flag=False,
                detail="Track too short for 4-bar analysis",
            )

        contrasts = [
            float(np.mean(librosa.feature.spectral_contrast(y=seg, sr=sr)))
            for seg in global_windows
        ]
        c_min, c_max = min(contrasts), max(contrasts)
        c_range = c_max - c_min + 1e-9
        norm    = [(c - c_min) / c_range for c in contrasts]

        # Collect per-window delta and stagnant window start timestamps
        per_window_deltas: list[float] = [
            round(abs(norm[i] - norm[i - 1]), 4)
            for i in range(1, len(global_windows))
        ]
        stagnant_ts: list[float] = [
            round(global_window_starts[i], 3)
            for i in range(1, len(global_windows))
            if per_window_deltas[i - 1] < CONSTANTS.ENERGY_DELTA_MIN
        ]

        stagnant = len(stagnant_ts)
        total = len(global_windows) - 1
        flag  = stagnant > 0
        detail = (
            f"{stagnant} of {total} window{'s' if total != 1 else ''} below "
            f"{CONSTANTS.ENERGY_DELTA_MIN:.0%} spectral contrast delta"
            if flag else ""
        )

        # Per-section breakdown (#106): slice beats by section boundaries
        section_details: list[dict[str, str | int | bool]] = []
        for sec in sections:
            sec_beats = beats_arr[(beats_arr >= sec.start) & (beats_arr <= sec.end)]
            if len(sec_beats) < 4:
                continue
            # Beats are absolute; slice the audio the same way
            sec_audio = audio[int(sec.start * sr):min(int(sec.end * sr), len(audio))]
            # Normalise beat times to section-local zero origin
            sec_stag, sec_total = _scan_windows(sec_audio, sec_beats - sec.start)
            section_details.append({
                "label":            sec.label,
                "stagnant_windows": sec_stag,
                "total_windows":    sec_total,
                "flag":             sec_stag > 0,
                "note":             _section_energy_note(sec_stag, sec_total),
            })

        # Which section contains the track end? (#106)
        track_end_s = len(audio) / sr
        ending_section: str | None = next(
            (sec.label for sec in sections if sec.start <= track_end_s <= sec.end),
            None,
        )

        return EnergyEvolutionResult(
            stagnant_windows=stagnant,
            total_windows=total,
            flag=flag,
            detail=detail,
            section_details=section_details,
            ending_section=ending_section,
            stagnant_timestamps=stagnant_ts,
            per_window_contrasts=per_window_deltas,
        )

    # ------------------------------------------------------------------
    # Private: Intro length check  (CPU)
    # ------------------------------------------------------------------

    def _check_intro(
        self,
        audio: np.ndarray,
        sr: int,
        beats: list[float],
        sections: list[Section],
        transcript: list[TranscriptSegment],
    ) -> IntroResult:
        """
        Flag intros longer than CONSTANTS.INTRO_MAX_SECONDS.

        Three signals (#105):
          1. allin1 section label "intro" — most reliable structural signal
          2. Onset RMS energy jump — first significant energy increase after
             INTRO_ONSET_MIN_BEATS beats (hardware-detectable, label-independent)
          3. Whisper first lyric timestamp — pre-vocal proxy / last resort fallback

        Confidence:
          "High"   — ≥2 signals agree within INTRO_CONFIDENCE_AGREEMENT_S seconds
          "Medium" — only one non-whisper signal present (allin1 or onset)
          "Low"    — whisper-only estimate

        Never raises — missing data degrades to IntroResult with flag=False.
        """
        # Signal 1: allin1 section label
        allin1_intro: float | None = None
        if sections:
            total = sum(
                s.end - s.start
                for s in sections
                if s.label.lower().startswith("intro")
            )
            if total > 0:
                allin1_intro = total

        # Signal 2: onset RMS energy jump
        onset_intro: float | None = _detect_onset_intro_end(audio, sr, beats)

        # Signal 3: Whisper first lyric timestamp
        whisper_intro: float | None = None
        if transcript:
            vocal_starts = [seg.start for seg in transcript if seg.text]
            if vocal_starts:
                whisper_intro = float(min(vocal_starts))

        # --- Resolve estimate and confidence ---
        strong_signals: list[float] = [
            v for v in (allin1_intro, onset_intro) if v is not None
        ]

        if len(strong_signals) >= 2:
            # Both allin1 and onset fired — check agreement
            agreement = abs(strong_signals[0] - strong_signals[1]) <= CONSTANTS.INTRO_CONFIDENCE_AGREEMENT_S
            intro_secs = strong_signals[0]  # allin1 takes priority
            confidence = "High" if agreement else "Medium"
            source = "allin1"
        elif allin1_intro is not None:
            # Only allin1 — onset is None here (otherwise strong_signals would have len ≥ 2)
            intro_secs = allin1_intro
            confidence = "Medium"
            source = "allin1"
        elif onset_intro is not None:
            # Only onset (no allin1 labels)
            intro_secs = onset_intro
            # Upgrade to High if whisper agrees within tolerance
            if whisper_intro is not None and abs(onset_intro - whisper_intro) <= CONSTANTS.INTRO_CONFIDENCE_AGREEMENT_S:
                confidence = "High"
            else:
                confidence = "Medium"
            source = "onset"
        elif whisper_intro is not None:
            # Only whisper fallback
            intro_secs = whisper_intro
            confidence = "Low"
            source = "whisper_fallback"
        else:
            return IntroResult(
                intro_seconds=0.0,
                flag=False,
                source="none",
                confidence="",
            )

        flagged = intro_secs > CONSTANTS.INTRO_MAX_SECONDS
        return IntroResult(
            intro_seconds=round(intro_secs, 1),
            flag=flagged,
            source=source,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Private: Lyric audit — NLI + NER + brand keywords  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _audit_lyrics(
        self, transcript: list[TranscriptSegment]
    ) -> list[ComplianceFlag]:
        """
        Run Detoxify toxicity scoring, spaCy NER, and brand keyword scan on
        each segment.

        Raises:
            ModelInferenceError: if Detoxify fails to load.
        """
        if not transcript:
            return []

        detector  = self._load_detoxify()        # raises ModelInferenceError on failure
        pf        = self._load_profanity()       # None on failure — supplementary
        nlp       = self._load_spacy()           # None on failure — NER is optional


        classifiable = [s for s in transcript if len(s.text) >= _MIN_SEGMENT_CHARS]
        windows      = _build_windows(classifiable, _WINDOW_SIZE)

        raw_flags: list[ComplianceFlag] = []

        # Pass 1: profanity word-list on ALL segments — no minimum length,
        # so short segments like "(Holy shit)" are not skipped.
        # Each word-list hit is cross-validated against Detoxify's obscenity
        # score to suppress idiomatic false positives (e.g. "tied up right now").
        if pf is not None:
            for seg in transcript:
                if pf.contains_profanity(seg.text):
                    try:
                        obs = float(detector.predict(seg.text).get("obscene", 0.0))
                    except (RuntimeError, ValueError, TypeError, KeyError):
                        # Detoxify predict is best-effort; treat as potential on any
                        # numeric/runtime failure rather than dropping the profanity hit.
                        obs = _EXPLICIT_POTENTIAL

                    if obs >= _EXPLICIT_CONFIRMED:
                        confidence: str = "confirmed"
                    elif obs >= _EXPLICIT_POTENTIAL:
                        confidence = "potential"
                    else:
                        continue  # Detoxify disagrees — drop as false positive

                    severity: str = "hard" if obs >= _HARD_OBSCENE_THRESHOLD else "soft"
                    raw_flags.append(ComplianceFlag(
                        timestamp_s=int(seg.start),
                        issue_type="EXPLICIT",
                        text=seg.text,
                        recommendation=(
                            "Clean edit required — hard explicit content."
                            if severity == "hard"
                            else "Mild language — acceptable for some placements. Supervisor discretion."
                        ),
                        confidence=confidence,
                        severity=severity,
                    ))

        # Pass 2: Detoxify + NER + brand keywords on segments ≥ MIN chars
        for i, seg in enumerate(classifiable):
            ts = int(seg.start)

            # Detoxify → EXPLICIT / VIOLENCE / DRUGS
            for flag in _score_detoxify(seg.text, windows[i], detector):
                raw_flags.append(ComplianceFlag(
                    timestamp_s=ts,
                    issue_type=flag[0],
                    text=seg.text,
                    recommendation=flag[1],
                    confidence=flag[2],
                    severity=flag[3],
                ))

            # Curated brand keywords → BRAND (potential)
            raw_flags.extend(_check_brand_keywords(seg.text, ts, self._brand_patterns))

            # spaCy NER → BRAND (ORG) / LOCATION (GPE) — both downgraded to
            # potential because NER in song lyrics produces many false positives
            # (terms of endearment mis-classified as ORG, etc.).
            if nlp is not None:
                try:
                    doc = nlp(seg.text)
                    for ent in doc.ents:
                        issue_type = _NER_ISSUE_MAP.get(ent.label_)
                        if issue_type:
                            raw_flags.append(ComplianceFlag(
                                timestamp_s=ts,
                                issue_type=issue_type,
                                text=ent.text,
                                recommendation=_NER_RECOMMENDATIONS[issue_type],
                                confidence="potential",
                            ))
                except (OSError, RuntimeError, ValueError):
                    # spaCy model load errors or tokenizer failures are non-fatal;
                    # NER is supplementary — pipeline continues without it.
                    pass

        return _deduplicate_flags(raw_flags)

    # ------------------------------------------------------------------
    # Private: lazy model loaders
    # ------------------------------------------------------------------

    def _load_detoxify(self):
        """
        Load and cache the Detoxify 'original' toxicity classifier.

        Raises:
            ModelInferenceError: if Detoxify cannot be imported or loaded.
        """
        if self._detoxify_model is not None:
            return self._detoxify_model
        try:
            from detoxify import Detoxify
            self._detoxify_model = Detoxify("original")
            return self._detoxify_model
        except Exception as exc:
            raise ModelInferenceError(
                "Detoxify model failed to load.",
                context={"original_error": str(exc)},
            ) from exc

    def _load_profanity(self):
        """
        Load and cache the better-profanity word-list filter.

        Word sources (additive):
          1. better-profanity built-in list (handles leetspeak variants)
          2. LDNOOBW English list fetched from GitHub — the canonical open-source
             profanity list originally created by Shutterstock.

        Fetch failure is silently swallowed — the built-in list still works
        if GitHub is unreachable.

        Returns None on failure — profanity detection is supplementary to Detoxify.
        """
        if self._profanity_filter is not None:
            return self._profanity_filter
        try:
            from better_profanity import profanity as bp
            bp.load_censor_words()

            # Extend with LDNOOBW English word list
            try:
                import urllib.request
                url = (
                    "https://raw.githubusercontent.com/"
                    "LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words"
                    "/master/en"
                )
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "sync-safe-forensic-portal/1.0"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    ldnoobw_words = [
                        line.strip()
                        for line in resp.read().decode("utf-8").splitlines()
                        if line.strip()
                    ]
                bp.add_censor_words(ldnoobw_words)
            except Exception:  # noqa: BLE001 — LDNOOBW fetch is best-effort
                pass

            # Project-specific supplement (data/profanity_supplement.py)
            if PROFANITY_SUPPLEMENT:
                bp.add_censor_words(list(PROFANITY_SUPPLEMENT))

            self._profanity_filter = bp
            return self._profanity_filter
        except Exception:  # noqa: BLE001
            return None

    def _load_spacy(self):
        """
        Load and cache the spaCy NER model.

        Returns None on failure — NER is supplementary to NLI.
        """
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_sm")
            return self._spacy_nlp
        except Exception:  # noqa: BLE001
            return None
