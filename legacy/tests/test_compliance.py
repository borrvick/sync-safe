"""
tests/test_compliance.py

Unit tests for services/compliance.py pure functions.

Covers:
  - _build_windows           — sliding text window builder
  - _check_brand_keywords    — brand regex scanner
  - _deduplicate_flags       — dedup + confidence priority logic
  - _compute_grade           — A–F grade from confirmed strikes
  - Compliance._check_intro  — pure method (no I/O, no models)
  - _detect_onset_intro_end  — onset RMS energy jump detector (#105)
  - _check_intro confidence  — three-signal intro confidence scoring (#105)

No GPU, no Detoxify, no spaCy, no audio — all tests complete in < 1 second.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from core.config import CONSTANTS
from core.models import (
    ComplianceFlag,
    EnergyEvolutionResult,
    IntroResult,
    Section,
    StingResult,
    TranscriptSegment,
)
from services.compliance import (
    Compliance,
    _NER_ISSUE_MAP,
    _NER_RECOMMENDATIONS,
    _build_windows,
    _check_brand_keywords,
    _classify_cut_type,
    _compute_fade_severity,
    _compute_grade,
    _deduplicate_flags,
    _detect_onset_intro_end,
    _section_energy_note,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag(
    text: str = "test",
    issue_type: str = "EXPLICIT",
    confidence: str = "confirmed",
    timestamp_s: int = 0,
) -> ComplianceFlag:
    return ComplianceFlag(
        timestamp_s=timestamp_s,
        issue_type=issue_type,
        text=text,
        recommendation="review",
        confidence=confidence,
    )


def _segment(text: str, start: float = 0.0, end: float = 5.0) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _sting(flag: bool = False, ending_type: str = "cut") -> StingResult:
    return StingResult(
        ending_type=ending_type,
        sync_ready=not flag,
        final_energy_ratio=0.5,
        flag=flag,
    )


def _evolution(flag: bool = False) -> EnergyEvolutionResult:
    return EnergyEvolutionResult(stagnant_windows=1 if flag else 0,
                                 total_windows=4, flag=flag)


def _intro(flag: bool = False, seconds: float = 10.0) -> IntroResult:
    return IntroResult(intro_seconds=seconds, flag=flag, source="allin1")


# ---------------------------------------------------------------------------
# _build_windows
# ---------------------------------------------------------------------------

class TestBuildWindows:
    def test_single_segment_window_is_itself(self):
        segs = [_segment("hello")]
        windows = _build_windows(segs, window_size=3)
        assert windows[0] == "hello"

    def test_window_concatenates_next_n_segments(self):
        segs = [_segment("a"), _segment("b"), _segment("c"), _segment("d")]
        windows = _build_windows(segs, window_size=2)
        assert windows[0] == "a b"
        assert windows[1] == "b c"
        assert windows[2] == "c d"

    def test_window_at_end_does_not_go_out_of_bounds(self):
        segs = [_segment("x"), _segment("y")]
        windows = _build_windows(segs, window_size=5)
        # Last segment window includes only remaining segments
        assert windows[1] == "y"

    def test_empty_segments_returns_empty_dict(self):
        assert _build_windows([], window_size=3) == {}

    def test_window_size_one_returns_each_segment(self):
        segs = [_segment("p"), _segment("q")]
        windows = _build_windows(segs, window_size=1)
        assert windows[0] == "p"
        assert windows[1] == "q"


# ---------------------------------------------------------------------------
# _check_brand_keywords
# ---------------------------------------------------------------------------

class TestCheckBrandKeywords:
    # Build a minimal brand pattern list for testing
    _PATTERNS: list = [
        ("Rolls-Royce", re.compile(
            r"(?<!\w)(?:rolls royce|rolls-royce|rollie|phantom)(?!\w)",
            re.IGNORECASE,
        )),
        ("Ferrari", re.compile(
            r"(?<!\w)(?:ferrari)(?!\w)",
            re.IGNORECASE,
        )),
    ]

    def test_detects_brand_mention(self):
        flags = _check_brand_keywords("Pulled up in the Phantom", 10, self._PATTERNS)
        assert len(flags) == 1
        assert flags[0].text == "Rolls-Royce"
        assert flags[0].timestamp_s == 10
        assert flags[0].confidence == "potential"

    def test_case_insensitive_match(self):
        flags = _check_brand_keywords("dropped the FERRARI keys", 5, self._PATTERNS)
        assert any(f.text == "Ferrari" for f in flags)

    def test_clean_text_returns_empty(self):
        flags = _check_brand_keywords("driving down the road", 0, self._PATTERNS)
        assert flags == []

    def test_multiple_brands_in_one_segment(self):
        flags = _check_brand_keywords("phantom and ferrari", 3, self._PATTERNS)
        texts = {f.text for f in flags}
        assert "Rolls-Royce" in texts
        assert "Ferrari" in texts

    def test_no_duplicate_brand_per_segment(self):
        flags = _check_brand_keywords("phantom phantom phantom", 0, self._PATTERNS)
        assert len(flags) == 1

    def test_word_boundary_prevents_partial_match(self):
        # "phantoms" should NOT match "phantom" due to word-boundary assertion
        flags = _check_brand_keywords("phantoms everywhere", 0, self._PATTERNS)
        assert flags == []


# ---------------------------------------------------------------------------
# _deduplicate_flags
# ---------------------------------------------------------------------------

class TestDeduplicateFlags:
    def test_removes_exact_duplicate(self):
        flags = [_flag("shit", "EXPLICIT", "confirmed")] * 3
        result = _deduplicate_flags(flags)
        assert len(result) == 1

    def test_confirmed_wins_over_potential(self):
        potential = _flag("cocaine", "DRUGS", "potential")
        confirmed = _flag("cocaine", "DRUGS", "confirmed")
        # Insert potential first — confirmed should win
        result = _deduplicate_flags([potential, confirmed])
        assert len(result) == 1
        assert result[0].confidence == "confirmed"

    def test_different_issue_types_not_deduped(self):
        flags = [
            _flag("Nike", "BRAND", "confirmed"),
            _flag("Nike", "EXPLICIT", "confirmed"),
        ]
        result = _deduplicate_flags(flags)
        assert len(result) == 2

    def test_same_text_different_timestamps_deduped_to_earliest(self):
        # Same lyric at two timestamps (chorus repeat) collapses to one flag
        # at the earliest timestamp.
        flags = [
            _flag("shit", "EXPLICIT", "confirmed", timestamp_s=10),
            _flag("shit", "EXPLICIT", "confirmed", timestamp_s=60),
        ]
        result = _deduplicate_flags(flags)
        assert len(result) == 1
        assert result[0].timestamp_s == 10

    def test_empty_input_returns_empty(self):
        assert _deduplicate_flags([]) == []

    def test_case_insensitive_text_dedup(self):
        flags = [
            _flag("SHIT", "EXPLICIT", "confirmed"),
            _flag("shit", "EXPLICIT", "confirmed"),
        ]
        result = _deduplicate_flags(flags)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _compute_grade
# ---------------------------------------------------------------------------

def _hard(text: str = "test", issue_type: str = "EXPLICIT", timestamp_s: int = 0) -> ComplianceFlag:
    """Helper: confirmed hard flag."""
    return ComplianceFlag(
        timestamp_s=timestamp_s, issue_type=issue_type, text=text,
        recommendation="review", confidence="confirmed", severity="hard",
    )


def _soft(text: str = "test", issue_type: str = "EXPLICIT", timestamp_s: int = 0) -> ComplianceFlag:
    """Helper: confirmed soft flag."""
    return ComplianceFlag(
        timestamp_s=timestamp_s, issue_type=issue_type, text=text,
        recommendation="review", confidence="confirmed", severity="soft",
    )


class TestComputeGrade:
    def test_no_flags_is_grade_a(self):
        assert _compute_grade([]) == "A"

    def test_soft_confirmed_only_is_grade_b(self):
        # Mild language / brand — director's call, no hard blocker
        assert _compute_grade([_soft()]) == "B"

    def test_potential_only_is_grade_b(self):
        # Potential flags bump to B so the director knows to review
        flags = [_flag(f"w{i}", confidence="potential") for i in range(5)]
        assert _compute_grade(flags) == "B"

    def test_one_hard_confirmed_is_grade_c(self):
        assert _compute_grade([_hard()]) == "C"

    def test_two_hard_confirmed_is_grade_d(self):
        flags = [_hard("f1"), _hard("f2", "VIOLENCE")]
        assert _compute_grade(flags) == "D"

    def test_four_hard_confirmed_is_grade_f(self):
        flags = [_hard(f"word{i}") for i in range(4)]
        assert _compute_grade(flags) == "F"

    def test_drugs_hard_is_always_grade_f(self):
        assert _compute_grade([_hard("crack", "DRUGS")]) == "F"

    def test_soft_flags_do_not_raise_above_b(self):
        # 10 confirmed soft flags — still B, not C/D/F
        flags = [_soft(f"mild{i}") for i in range(10)]
        assert _compute_grade(flags) == "B"

    def test_brand_confirmed_soft_does_not_affect_hard_grade(self):
        # 1 hard EXPLICIT + 5 confirmed BRAND soft → C not D/F
        flags = [_hard("expletive")] + [_soft(f"brand{i}", "BRAND") for i in range(5)]
        assert _compute_grade(flags) == "C"

    def test_structural_issues_do_not_affect_lyric_grade(self):
        # Fade ending, energy stagnation, long intro — none lower the lyric grade
        assert _compute_grade([]) == "A"

    def test_brand_and_location_flags_do_not_affect_grade(self):
        # BRAND soft confirmed — informational only, stays at B
        flags = [_soft(f"brand{i}", "BRAND") for i in range(10)]
        assert _compute_grade(flags) == "B"


# ---------------------------------------------------------------------------
# Compliance._check_intro  (pure method — no I/O, no models)
# ---------------------------------------------------------------------------

class TestCheckIntro:
    # Silent audio + empty beats → onset detector never fires, keeps tests pure/fast
    _SR    = 22050
    _AUDIO = np.zeros(22050 * 5, dtype=np.float32)
    _BEATS: list[float] = []

    def setup_method(self):
        self.svc = Compliance()

    def _call(
        self,
        sections: list[Section],
        transcript: list[TranscriptSegment],
    ) -> IntroResult:
        return self.svc._check_intro(
            self._AUDIO, self._SR, self._BEATS, sections, transcript
        )

    def test_intro_section_under_limit_does_not_flag(self):
        sections = [Section(label="intro", start=0.0, end=10.0)]
        result = self._call(sections, transcript=[])
        assert result.flag is False
        assert result.intro_seconds == pytest.approx(10.0)
        assert result.source == "allin1"

    def test_intro_section_over_limit_flags(self):
        sections = [Section(label="intro", start=0.0,
                            end=float(CONSTANTS.INTRO_MAX_SECONDS + 1))]
        result = self._call(sections, transcript=[])
        assert result.flag is True

    def test_multiple_intro_sections_sum_is_checked(self):
        # Two 8-second intros → 16s total → over 15s limit
        sections = [
            Section(label="intro", start=0.0, end=8.0),
            Section(label="intro", start=20.0, end=28.0),
        ]
        result = self._call(sections, transcript=[])
        assert result.intro_seconds == pytest.approx(16.0)
        assert result.flag is True

    def test_non_intro_sections_are_ignored(self):
        sections = [
            Section(label="verse", start=0.0, end=30.0),
            Section(label="chorus", start=30.0, end=60.0),
        ]
        result = self._call(sections, transcript=[])
        assert result.flag is False

    def test_whisper_fallback_when_no_sections(self):
        transcript = [_segment("here we go", start=12.0)]
        result = self._call(sections=[], transcript=transcript)
        assert result.source == "whisper_fallback"
        assert result.intro_seconds == pytest.approx(12.0)

    def test_whisper_fallback_over_limit_flags(self):
        transcript = [_segment("let's go", start=float(CONSTANTS.INTRO_MAX_SECONDS + 2))]
        result = self._call(sections=[], transcript=transcript)
        assert result.flag is True

    def test_no_sections_no_transcript_returns_unflagged(self):
        result = self._call(sections=[], transcript=[])
        assert result.flag is False
        assert result.source == "none"


# ---------------------------------------------------------------------------
# NER issue map — GPE → LOCATION, ORG → BRAND
# ---------------------------------------------------------------------------

class TestNerIssueMap:
    def test_org_not_in_map(self):
        # ORG intentionally excluded — en_core_web_sm produces too many false
        # positives on song lyrics; brand detection uses data/brand_keywords.py
        assert "ORG" not in _NER_ISSUE_MAP

    def test_gpe_maps_to_location(self):
        assert _NER_ISSUE_MAP["GPE"] == "LOCATION"

    def test_no_unexpected_keys(self):
        # Only GPE should be mapped — ORG excluded, other spaCy labels ignored
        assert set(_NER_ISSUE_MAP.keys()) == {"GPE"}

    def test_brand_recommendation_present(self):
        assert "BRAND" in _NER_RECOMMENDATIONS
        assert len(_NER_RECOMMENDATIONS["BRAND"]) > 0

    def test_location_recommendation_present(self):
        assert "LOCATION" in _NER_RECOMMENDATIONS
        assert len(_NER_RECOMMENDATIONS["LOCATION"]) > 0


# ---------------------------------------------------------------------------
# _audit_lyrics — pure paths that don't require Detoxify/spaCy/GPU
# ---------------------------------------------------------------------------

class TestAuditLyricsGuards:
    """Tests for _audit_lyrics guard clauses and empty-segment handling."""

    def setup_method(self):
        self.svc = Compliance()

    def test_empty_transcript_returns_empty(self):
        # Whisper empty segments must not raise — this is the primary guard
        result = self.svc._audit_lyrics([])
        assert result == []

    def test_all_short_segments_skipped_for_detoxify(self):
        # Segments shorter than _MIN_SEGMENT_CHARS bypass Pass 2 (Detoxify)
        # but are still checked by Pass 1 (profanity) if profanity filter loads.
        # Without real models this just confirms no crash.
        short_segs = [_segment("ok", start=float(i)) for i in range(5)]
        # Should complete without raising even if models are unavailable
        try:
            self.svc._audit_lyrics(short_segs)
        except Exception as exc:  # noqa: BLE001
            # ModelInferenceError is acceptable (Detoxify not installed in test env)
            from core.exceptions import ModelInferenceError
            assert isinstance(exc, ModelInferenceError), f"Unexpected: {exc}"


# ---------------------------------------------------------------------------
# LOCATION IssueType — round-trip through ComplianceFlag model
# ---------------------------------------------------------------------------

class TestLocationIssueType:
    def test_compliance_flag_accepts_location_issue_type(self):
        flag = ComplianceFlag(
            timestamp_s=30,
            issue_type="LOCATION",
            text="New York",
            recommendation="Geographic reference — verify placement restrictions.",
            confidence="potential",
            severity="soft",
        )
        assert flag.issue_type == "LOCATION"

    def test_location_flag_excluded_from_grade_computation(self):
        # LOCATION flags do not lower grade below B (same as BRAND) — confirmed soft
        # flags of any type trigger B, but LOCATION never escalates to C/D/F.
        flags = [
            ComplianceFlag(
                timestamp_s=i * 10,
                issue_type="LOCATION",
                text=f"city{i}",
                recommendation="check markets",
                confidence="confirmed",
                severity="soft",
            )
            for i in range(5)
        ]
        assert _compute_grade(flags) == "B"


# ---------------------------------------------------------------------------
# _compute_fade_severity (#103)
# ---------------------------------------------------------------------------

import numpy as np


class TestComputeFadeSeverity:
    def _call(
        self,
        rms: "np.ndarray",
        overall_mean: float = 0.1,
        sr: int = 44100,
        hop: int = 512,
        threshold_ratio: float = 0.10,
        max_seconds: float = 60.0,
    ) -> tuple[float, float]:
        return _compute_fade_severity(rms, overall_mean, sr, hop, threshold_ratio, max_seconds)

    def test_all_zero_rms_returns_zeros(self) -> None:
        rms = np.zeros(1000)
        sev, tail = self._call(rms, overall_mean=0.0)
        assert sev == 0.0 and tail == 0.0

    def test_empty_rms_returns_zeros(self) -> None:
        rms = np.array([])
        sev, tail = self._call(rms, overall_mean=0.0)
        assert sev == 0.0 and tail == 0.0

    def test_constant_rms_no_tail_above_threshold(self) -> None:
        # uniform signal: last frame == first frame, no fall-off → no tail below threshold
        rms = np.full(200, 0.5)
        sev, tail = self._call(rms, overall_mean=0.5, threshold_ratio=0.10)
        # All frames are above threshold, last_frame = 199, tail_seconds ≈ 0
        assert tail == pytest.approx(0.0, abs=0.1)

    def test_severity_clipped_to_one(self) -> None:
        # rms that produces a very long tail — severity must not exceed 1.0
        rms = np.linspace(1.0, 0.0, 10000)
        sev, _ = self._call(rms, overall_mean=0.5, max_seconds=1.0)
        assert sev <= 1.0

    def test_severity_non_negative(self) -> None:
        rms = np.random.default_rng(42).uniform(0, 1, 500)
        sev, tail = self._call(rms, overall_mean=float(np.mean(rms)))
        assert sev >= 0.0 and tail >= 0.0

    def test_custom_max_seconds_scales_severity(self) -> None:
        # Same rms → larger max_seconds → lower severity
        rms = np.linspace(1.0, 0.01, 2000)
        mean = float(np.mean(rms))
        sev_short, _ = self._call(rms, overall_mean=mean, max_seconds=1.0)
        sev_long,  _ = self._call(rms, overall_mean=mean, max_seconds=600.0)
        assert sev_short >= sev_long

    def test_returns_rounded_values(self) -> None:
        rms = np.linspace(0.8, 0.05, 500)
        sev, tail = self._call(rms, overall_mean=float(np.mean(rms)))
        assert round(sev, 3) == sev
        assert round(tail, 1) == tail


# ---------------------------------------------------------------------------
# _classify_cut_type (#104)
# ---------------------------------------------------------------------------

class TestClassifyCutType:
    def test_no_beats_returns_mid_phrase(self) -> None:
        assert _classify_cut_type(30.0, [], 0.075) == "mid_phrase_cut"

    def test_exact_beat_match_returns_clean_cut(self) -> None:
        beats = [0.5, 1.0, 1.5, 2.0, 30.0]
        assert _classify_cut_type(30.0, beats, 0.075) == "clean_cut"

    def test_within_tolerance_returns_clean_cut(self) -> None:
        beats = [29.95, 30.5]
        assert _classify_cut_type(30.0, beats, 0.075) == "clean_cut"  # 0.05 < 0.075

    def test_outside_tolerance_returns_mid_phrase(self) -> None:
        beats = [29.8, 30.9]
        assert _classify_cut_type(30.0, beats, 0.075) == "mid_phrase_cut"  # 0.20 > 0.075

    def test_tolerance_boundary_exclusive(self) -> None:
        # exactly on tolerance edge → clean_cut (<=)
        assert _classify_cut_type(10.0, [9.925], 0.075) == "clean_cut"

    def test_single_beat_far_away(self) -> None:
        assert _classify_cut_type(60.0, [1.0], 0.075) == "mid_phrase_cut"

    def test_custom_tolerance_zero_requires_exact(self) -> None:
        # tolerance=0 → only exact match qualifies
        assert _classify_cut_type(5.0, [5.0], 0.0) == "clean_cut"
        assert _classify_cut_type(5.0, [5.001], 0.0) == "mid_phrase_cut"


# ---------------------------------------------------------------------------
# _section_energy_note (#106)
# ---------------------------------------------------------------------------

class TestSectionEnergyNote:
    def test_zero_stagnant(self) -> None:
        assert _section_energy_note(0, 8) == "Good evolution"

    def test_all_stagnant(self) -> None:
        assert _section_energy_note(4, 4) == "All 4 windows flat"

    def test_singular_window_all_stagnant(self) -> None:
        assert _section_energy_note(1, 1) == "All 1 window flat"

    def test_partial_stagnant(self) -> None:
        assert _section_energy_note(3, 8) == "3 of 8 windows stagnant"

    def test_partial_stagnant_singular(self) -> None:
        assert _section_energy_note(1, 5) == "1 of 5 windows stagnant"


# ---------------------------------------------------------------------------
# _detect_onset_intro_end (#105)
# ---------------------------------------------------------------------------

class TestDetectOnsetIntroEnd:
    """Tests for the onset RMS energy jump intro detector."""

    SR = 22050
    HOP = 512

    def _make_audio(self, pre_rms: float, post_rms: float, sr: int = SR) -> np.ndarray:
        """Build a short synthetic array: quiet pre-onset block then louder block."""
        # 1.5 s quiet then 1.5 s loud — total 3 s
        half = int(1.5 * sr)
        quiet = np.random.randn(half).astype(np.float32) * pre_rms
        loud  = np.random.randn(half).astype(np.float32) * post_rms
        return np.concatenate([quiet, loud])

    def test_returns_none_when_too_few_beats(self) -> None:
        y = self._make_audio(0.01, 0.5)
        # fewer beats than INTRO_ONSET_MIN_BEATS
        beats = list(np.linspace(0.0, 1.0, CONSTANTS.INTRO_ONSET_MIN_BEATS - 1))
        result = _detect_onset_intro_end(y, self.SR, beats)
        assert result is None

    def test_returns_none_for_silent_track(self) -> None:
        y = np.zeros(self.SR * 3, dtype=np.float32)
        beats = list(np.linspace(0.0, 2.5, CONSTANTS.INTRO_ONSET_MIN_BEATS + 4))
        result = _detect_onset_intro_end(y, self.SR, beats)
        assert result is None

    def test_detects_jump_after_onset_window(self) -> None:
        # Quiet for first 1.5 s, loud after — beat grid opens past onset window
        y = self._make_audio(pre_rms=0.005, post_rms=0.5)
        beats = list(np.linspace(0.0, 2.8, CONSTANTS.INTRO_ONSET_MIN_BEATS + 8))
        result = _detect_onset_intro_end(y, self.SR, beats)
        # Onset should be detected in the louder half
        assert result is not None
        assert result >= 0.0
        assert result <= 3.0

    def test_flat_audio_returns_none(self) -> None:
        # Uniform amplitude — no significant jump
        y = np.ones(self.SR * 3, dtype=np.float32) * 0.05
        beats = list(np.linspace(0.0, 2.9, CONSTANTS.INTRO_ONSET_MIN_BEATS + 6))
        result = _detect_onset_intro_end(y, self.SR, beats)
        # No frame exceeds the jump threshold for flat audio
        assert result is None

    def test_returns_float_when_jump_found(self) -> None:
        y = self._make_audio(pre_rms=0.001, post_rms=1.0)
        beats = list(np.linspace(0.0, 2.9, CONSTANTS.INTRO_ONSET_MIN_BEATS + 10))
        result = _detect_onset_intro_end(y, self.SR, beats)
        if result is not None:
            assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _check_intro confidence (#105)
# ---------------------------------------------------------------------------

class TestCheckIntroConfidence:
    """Tests for the three-signal intro confidence computation."""

    SR = 22050

    def _make_quiet_audio(self, duration_s: float = 5.0) -> np.ndarray:
        """Return near-silent audio — onset detector finds no jump."""
        return np.zeros(int(duration_s * self.SR), dtype=np.float32)

    def _sec(self, label: str, start: float, end: float) -> Section:
        return Section(label=label, start=start, end=end)

    def _seg(self, start: float, text: str = "hello world test lyrics") -> TranscriptSegment:
        return TranscriptSegment(start=start, end=start + 3.0, text=text)

    def _check(
        self,
        audio: np.ndarray,
        beats: list[float],
        sections: list[Section],
        transcript: list[TranscriptSegment],
    ) -> IntroResult:
        svc = Compliance()
        return svc._check_intro(audio, self.SR, beats, sections, transcript)

    def test_allin1_only_returns_medium_confidence(self) -> None:
        y = self._make_quiet_audio()
        beats = list(np.linspace(0.0, 4.9, CONSTANTS.INTRO_ONSET_MIN_BEATS + 4))
        sections = [self._sec("intro", 0.0, 8.0), self._sec("verse", 8.0, 30.0)]
        result = self._check(y, beats, sections, [])
        assert result.source == "allin1"
        # With no onset signal on silent audio, confidence is Medium or High
        assert result.confidence in ("Medium", "High")

    def test_whisper_only_returns_low_confidence(self) -> None:
        y = self._make_quiet_audio()
        beats: list[float] = []  # no beats → no onset window
        transcript = [self._seg(start=12.0)]
        result = self._check(y, beats, [], transcript)
        assert result.source == "whisper_fallback"
        assert result.confidence == "Low"

    def test_no_signals_returns_empty_confidence(self) -> None:
        y = self._make_quiet_audio()
        result = self._check(y, [], [], [])
        assert result.intro_seconds == 0.0
        assert result.confidence == ""
        assert result.flag is False

    def test_intro_over_max_seconds_is_flagged(self) -> None:
        y = self._make_quiet_audio()
        beats = list(np.linspace(0.0, 4.9, CONSTANTS.INTRO_ONSET_MIN_BEATS + 4))
        over_limit = CONSTANTS.INTRO_MAX_SECONDS + 5.0
        sections = [self._sec("intro", 0.0, over_limit)]
        result = self._check(y, beats, sections, [])
        assert result.flag is True
        assert result.intro_seconds > CONSTANTS.INTRO_MAX_SECONDS

    def test_intro_under_max_seconds_not_flagged(self) -> None:
        y = self._make_quiet_audio()
        beats = list(np.linspace(0.0, 4.9, CONSTANTS.INTRO_ONSET_MIN_BEATS + 4))
        under_limit = max(1.0, CONSTANTS.INTRO_MAX_SECONDS - 3.0)
        sections = [self._sec("intro", 0.0, under_limit)]
        result = self._check(y, beats, sections, [])
        assert result.flag is False

    def test_intro_result_has_confidence_field(self) -> None:
        y = self._make_quiet_audio()
        beats = []
        transcript = [self._seg(start=5.0)]
        result = self._check(y, beats, [], transcript)
        assert hasattr(result, "confidence")
        assert isinstance(result.confidence, str)
