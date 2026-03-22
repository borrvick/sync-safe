"""
services/compliance.py
Sync readiness compliance checks — implements the ComplianceChecker protocol.

Five checks, one class:
  1. Sting / ending type  — librosa RMS + onset strength
  2. 4-8 bar energy rule  — librosa spectral contrast per window
  3. Intro timer          — allin1 sections + Whisper fallback
  4. Lyric audit          — Detoxify transformer + spaCy NER
  5. Brand keywords       — curated regex list (data/brand_keywords.py)

Design notes:
- Detoxify and spaCy models are lazy-initialized on first use and cached as
  instance attributes — no module-level globals.
- Brand patterns are compiled once in __init__, not at module import.
- Detoxify load failure → ModelInferenceError (hard — audit cannot run).
- spaCy load failure → graceful degradation (NER is supplementary).
- All thresholds are module-level constants; no inline magic numbers.
- _compute_grade and all small helpers are pure module-level functions for
  independent testability.
"""
from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np

from core.config import CONSTANTS, ModelParams
from core.exceptions import ModelInferenceError
from core.models import (
    AudioBuffer,
    ComplianceFlag,
    ComplianceReport,
    EnergyEvolutionResult,
    IntroResult,
    IssueType,
    Section,
    Severity,
    StingResult,
    TranscriptSegment,
)
from data.brand_keywords import BRAND_KEYWORDS
from data.drug_keywords import DRUG_KEYWORDS
from data.profanity_supplement import PROFANITY_SUPPLEMENT

try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn


# ---------------------------------------------------------------------------
# Detoxify thresholds
# ---------------------------------------------------------------------------

# Detoxify score thresholds — sourced from CONSTANTS (core/config.py).
# All numeric values live there; references here are read-only aliases for brevity.
_EXPLICIT_CONFIRMED: float = CONSTANTS.EXPLICIT_CONFIRMED
_EXPLICIT_POTENTIAL: float = CONSTANTS.EXPLICIT_POTENTIAL
_HARD_OBSCENE_THRESHOLD: float = CONSTANTS.EXPLICIT_HARD
_VIOLENCE_CONFIRMED: float = CONSTANTS.VIOLENCE_CONFIRMED
_VIOLENCE_POTENTIAL: float = CONSTANTS.VIOLENCE_POTENTIAL
_HARD_THREAT_THRESHOLD: float = CONSTANTS.VIOLENCE_HARD
_DRUGS_TOXIC_MIN: float = CONSTANTS.DRUGS_TOXIC_MIN

# Number of segments in a sliding window used to confirm borderline scores
_WINDOW_SIZE: int = 3

# Segments shorter than this carry no classifiable signal
_MIN_SEGMENT_CHARS: int = 20

# Only these issue types count toward the A–F content grade
_GRADE_ISSUE_TYPES: frozenset[str] = frozenset({"EXPLICIT", "VIOLENCE", "DRUGS"})

# NER entity labels mapped to our issue types
_NER_ISSUE_MAP: dict[str, IssueType] = {
    "ORG": "BRAND",
}

_NER_RECOMMENDATIONS: dict[IssueType, str] = {
    "BRAND": "Confirm explicit brand clearance with rights holder.",
}


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
        raw = audio.raw

        sting     = self._check_sting(raw)
        evolution = self._check_energy_evolution(raw, beats)
        intro     = self._check_intro(sections, transcript)
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
    def _check_sting(self, raw: bytes) -> StingResult:
        """
        Classify the track ending as sting, fade, or cut.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Sting check: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        if len(audio) < sr * 4:
            # Too short — report as cut (neutral), no flag
            return StingResult(
                ending_type="cut",
                sync_ready=True,
                final_energy_ratio=1.0,
                flag=False,
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
            )
        if is_fade:
            return StingResult(
                ending_type="fade",
                sync_ready=False,
                final_energy_ratio=round(final_ratio, 3),
                flag=True,
            )
        return StingResult(
            ending_type="cut",
            sync_ready=True,
            final_energy_ratio=round(final_ratio, 3),
            flag=False,
        )

    # ------------------------------------------------------------------
    # Private: 4-8 bar energy evolution check  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _check_energy_evolution(
        self, raw: bytes, beats: list[float]
    ) -> EnergyEvolutionResult:
        """
        Verify spectral contrast evolves across every 4-bar window.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Energy evolution check: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

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

        windows: list[tuple[float, float, np.ndarray]] = []
        step = CONSTANTS.BEATS_PER_WINDOW
        for i in range(0, len(beats) - step, step):
            t_start = beats[i]
            t_end   = beats[min(i + step, len(beats) - 1)]
            s_start, s_end = int(t_start * sr), int(t_end * sr)
            if s_end > s_start:
                windows.append((t_start, t_end, audio[s_start:s_end]))

        if len(windows) < 2:
            return EnergyEvolutionResult(
                stagnant_windows=0,
                total_windows=0,
                flag=False,
                detail="Track too short for 4-bar analysis",
            )

        contrasts = [
            float(np.mean(librosa.feature.spectral_contrast(y=seg, sr=sr)))
            for _, _, seg in windows
        ]
        c_min, c_max = min(contrasts), max(contrasts)
        c_range = c_max - c_min + 1e-9
        norm    = [(c - c_min) / c_range for c in contrasts]

        stagnant = sum(
            1 for i in range(1, len(windows))
            if abs(norm[i] - norm[i - 1]) < CONSTANTS.ENERGY_DELTA_MIN
        )
        total = len(windows) - 1
        flag  = stagnant > 0
        detail = (
            f"{stagnant} of {total} window{'s' if total != 1 else ''} below "
            f"{CONSTANTS.ENERGY_DELTA_MIN:.0%} spectral contrast delta"
            if flag else ""
        )

        return EnergyEvolutionResult(
            stagnant_windows=stagnant,
            total_windows=total,
            flag=flag,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Private: Intro length check  (CPU)
    # ------------------------------------------------------------------

    def _check_intro(
        self,
        sections: list[Section],
        transcript: list[TranscriptSegment],
    ) -> IntroResult:
        """
        Flag intros longer than CONSTANTS.INTRO_MAX_SECONDS.

        Primary source: allin1 sections labelled 'intro'.
        Fallback: first Whisper lyric timestamp as pre-vocal proxy.
        Never raises — missing data degrades to IntroResult with flag=False.
        """
        intro_secs = sum(
            s.end - s.start
            for s in sections
            if s.label.lower().startswith("intro")
        )
        source = "allin1" if sections else "none"

        if intro_secs == 0 and transcript:
            vocal_starts = [seg.start for seg in transcript if seg.text]
            if vocal_starts:
                intro_secs = float(min(vocal_starts))
                source     = "whisper_fallback"

        flagged = intro_secs > CONSTANTS.INTRO_MAX_SECONDS
        return IntroResult(
            intro_seconds=round(intro_secs, 1),
            flag=flagged,
            source=source,
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
                    except Exception:  # noqa: BLE001 — Detoxify predict is best-effort; fall back to potential
                        obs = _EXPLICIT_POTENTIAL

                    if obs >= _EXPLICIT_CONFIRMED:
                        confidence: str = "confirmed"
                    elif obs >= _EXPLICIT_POTENTIAL:
                        confidence = "potential"
                    else:
                        continue  # Detoxify disagrees — drop as false positive

                    severity: Severity = "hard" if obs >= _HARD_OBSCENE_THRESHOLD else "soft"
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

            # spaCy NER → BRAND (ORG, potential) — downgraded to potential
            # because NER in song lyrics produces too many false positives
            # (e.g. terms of endearment and common nouns mis-classified as ORG).
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
                except Exception:  # noqa: BLE001 — NER is supplementary
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


# ---------------------------------------------------------------------------
# Module-level pure functions — independently testable
# ---------------------------------------------------------------------------

def _build_windows(
    segments: list[TranscriptSegment], window_size: int
) -> dict[int, str]:
    """
    Build a sliding text window for each segment index.

    Returns a dict mapping segment index → concatenated text of the next
    `window_size` segments (inclusive). Used for borderline NLI confirmation.
    """
    return {
        i: " ".join(
            segments[j].text
            for j in range(i, min(i + window_size, len(segments)))
        )
        for i in range(len(segments))
    }


def _score_detoxify(
    seg_text: str,
    window_text: str,
    detector,
) -> list[tuple[IssueType, str, str, str]]:
    """
    Score a segment with Detoxify and return any flags.

    Uses the segment score first; falls back to the sliding window to confirm
    borderline scores before promoting them to "confirmed".

    Returns:
        List of (issue_type, recommendation, confidence, severity) tuples.
        severity is "hard" for absolute deal-breakers, "soft" for
        placement-dependent issues that are the sync director's call.
        Returns empty list if clean.

    Pure function aside from the detector call — no side effects.
    """
    def _predict(text: str) -> dict[str, float]:
        try:
            return {k: float(v) for k, v in detector.predict(text).items()}
        except Exception:  # noqa: BLE001
            return {}

    solo = _predict(seg_text)
    if not solo:
        return []

    flags: list[tuple[IssueType, str, str, str]] = []

    # --- EXPLICIT ---
    explicit_score = solo.get("obscene", 0.0)
    if explicit_score >= _EXPLICIT_CONFIRMED:
        sev = "hard" if explicit_score >= _HARD_OBSCENE_THRESHOLD else "soft"
        rec = (
            "Clean edit required — hard explicit content."
            if sev == "hard"
            else "Mild language — acceptable for some placements. Supervisor discretion."
        )
        flags.append(("EXPLICIT", rec, "confirmed", sev))
    elif explicit_score >= _EXPLICIT_POTENTIAL:
        win = _predict(window_text)
        win_score = win.get("obscene", 0.0)
        if win_score >= _EXPLICIT_CONFIRMED:
            sev = "hard" if win_score >= _HARD_OBSCENE_THRESHOLD else "soft"
            rec = (
                "Clean edit required — hard explicit content."
                if sev == "hard"
                else "Mild language — acceptable for some placements. Supervisor discretion."
            )
            flags.append(("EXPLICIT", rec, "confirmed", sev))
        else:
            flags.append((
                "EXPLICIT",
                "Possible explicit content — supervisor should review before submission.",
                "potential",
                "soft",
            ))

    # --- VIOLENCE ---
    threat_score = solo.get("threat", 0.0)
    if threat_score >= _VIOLENCE_CONFIRMED:
        sev = "hard" if threat_score >= _HARD_THREAT_THRESHOLD else "soft"
        rec = (
            "Threatening language — disqualifies family and broadcast placements."
            if sev == "hard"
            else "Flag for brand-safety review; may disqualify family placements."
        )
        flags.append(("VIOLENCE", rec, "confirmed", sev))
    elif threat_score >= _VIOLENCE_POTENTIAL:
        win = _predict(window_text)
        win_score = win.get("threat", 0.0)
        if win_score >= _VIOLENCE_CONFIRMED:
            sev = "hard" if win_score >= _HARD_THREAT_THRESHOLD else "soft"
            rec = (
                "Threatening language — disqualifies family and broadcast placements."
                if sev == "hard"
                else "Flag for brand-safety review; may disqualify family placements."
            )
            flags.append(("VIOLENCE", rec, "confirmed", sev))
        else:
            flags.append((
                "VIOLENCE",
                "Possible violent language — review in context before family placement.",
                "potential",
                "soft",
            ))

    # --- DRUGS (toxicity + word-list gate) — always hard ---
    toxic_score = solo.get("toxicity", 0.0)
    text_lower  = seg_text.lower()
    has_drug_word = any(w in text_lower.split() for w in DRUG_KEYWORDS)
    if toxic_score >= _DRUGS_TOXIC_MIN and has_drug_word:
        flags.append((
            "DRUGS",
            "Drug reference — disqualifies broadcast and most brand placements.",
            "confirmed",
            "hard",
        ))

    return flags


def _check_brand_keywords(
    seg_text: str,
    timestamp_s: int,
    brand_patterns: list[tuple[str, re.Pattern]],
) -> list[ComplianceFlag]:
    """
    Scan a lyric segment against the compiled brand keyword patterns.

    Returns a list of potential BRAND ComplianceFlags (one per matched brand).
    Pure function — the patterns are passed in rather than read from a global.
    """
    flags: list[ComplianceFlag] = []
    seen: set[str] = set()
    for display_name, pattern in brand_patterns:
        if display_name not in seen and pattern.search(seg_text):
            seen.add(display_name)
            flags.append(ComplianceFlag(
                timestamp_s=timestamp_s,
                issue_type="BRAND",
                text=display_name,
                recommendation=(
                    "Possible brand/trademark reference — supervisor should verify sync clearance."
                ),
                confidence="potential",
            ))
    return flags


def _deduplicate_flags(flags: list[ComplianceFlag]) -> list[ComplianceFlag]:
    """
    Remove duplicate flags — keep the first occurrence of each (text, issue_type) pair.

    Deduplication is keyed on normalized text + issue type, NOT timestamp, so
    repeated chorus lines collapse to a single flag at the earliest timestamp.
    Confirmed flags take priority over potential ones for the same key.
    """
    # Sort: confirmed before potential, then by timestamp (earliest first)
    ordered = sorted(flags, key=lambda f: (0 if f.confidence == "confirmed" else 1, f.timestamp_s))
    seen:   set[tuple[str, str]] = set()
    unique: list[ComplianceFlag] = []
    for f in ordered:
        key = (f.text.strip().lower(), f.issue_type)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _compute_grade(flags: list[ComplianceFlag]) -> str:
    """
    Compute an A–F lyric-content grade.

    Only hard-severity EXPLICIT, VIOLENCE, and DRUGS flags count toward the grade.
    BRAND flags and structural issues (fade, energy, intro) do not affect this
    grade — structural fitness is handled by the Sync Snapshot verdict.
    Soft confirmed flags (mild language, brand mentions, borderline metaphors) are
    shown in the audit table but do not lower the grade below B — they are the
    sync director's placement call, not absolute blockers.

    Grading scale:
        A — 0 hard confirmed issues, 0 soft/potential flags
        B — 0 hard confirmed issues, but soft confirmed or potential flags present
        C — 1 hard confirmed issue
        D — 2–3 hard confirmed issues
        F — 4+ hard confirmed issues (or any DRUGS hard confirmed)
    """
    hard_confirmed = sum(
        1 for f in flags
        if f.confidence == "confirmed"
        and f.severity == "hard"
        and f.issue_type in _GRADE_ISSUE_TYPES
    )
    has_drugs_hard = any(
        f.issue_type == "DRUGS" and f.confidence == "confirmed" and f.severity == "hard"
        for f in flags
    )
    if has_drugs_hard or hard_confirmed >= 4:
        return "F"
    if hard_confirmed == 3:
        return "D"
    if hard_confirmed == 2:
        return "D"
    if hard_confirmed == 1:
        return "C"
    # 0 hard confirmed — grade on whether any soft/potential flags exist
    has_advisory = any(
        f.confidence in ("confirmed", "potential") for f in flags
    )
    return "B" if has_advisory else "A"
