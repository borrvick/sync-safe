"""
tests/test_compliance.py

Unit tests for services/compliance.py pure functions.

Covers:
  - _build_windows       — sliding text window builder
  - _check_brand_keywords — brand regex scanner
  - _deduplicate_flags   — dedup + confidence priority logic
  - _compute_grade       — A–F grade from confirmed strikes
  - Compliance._check_intro — pure method (no I/O, no models)

No GPU, no Detoxify, no spaCy, no audio — all tests complete in < 1 second.
"""
from __future__ import annotations

import re

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
    _build_windows,
    _check_brand_keywords,
    _compute_grade,
    _deduplicate_flags,
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

    def test_different_timestamps_not_deduped(self):
        flags = [
            _flag("shit", "EXPLICIT", "confirmed", timestamp_s=10),
            _flag("shit", "EXPLICIT", "confirmed", timestamp_s=60),
        ]
        result = _deduplicate_flags(flags)
        assert len(result) == 2

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

class TestComputeGrade:
    def test_zero_confirmed_issues_is_grade_a(self):
        assert _compute_grade([], _sting(), _evolution(), _intro()) == "A"

    def test_one_confirmed_flag_is_grade_b(self):
        assert _compute_grade([_flag()], _sting(), _evolution(), _intro()) == "B"

    def test_two_confirmed_flags_is_grade_c(self):
        flags = [_flag(), _flag("violence", "VIOLENCE")]
        assert _compute_grade(flags, _sting(), _evolution(), _intro()) == "C"

    def test_four_confirmed_flags_is_grade_d(self):
        flags = [_flag(f"word{i}") for i in range(4)]
        assert _compute_grade(flags, _sting(), _evolution(), _intro()) == "D"

    def test_six_confirmed_flags_is_grade_f(self):
        flags = [_flag(f"word{i}") for i in range(6)]
        assert _compute_grade(flags, _sting(), _evolution(), _intro()) == "F"

    def test_fade_ending_is_always_grade_f(self):
        # Even with zero other issues, a fade ending → F
        assert _compute_grade([], _sting(flag=True, ending_type="fade"),
                               _evolution(), _intro()) == "F"

    def test_sting_flag_adds_structural_strike(self):
        # Sting flag (non-fade) adds 1 confirmed strike → B
        assert _compute_grade([], _sting(flag=True, ending_type="sting"),
                               _evolution(), _intro()) == "B"

    def test_energy_evolution_flag_adds_strike(self):
        assert _compute_grade([], _sting(), _evolution(flag=True), _intro()) == "B"

    def test_intro_flag_adds_strike(self):
        assert _compute_grade([], _sting(), _evolution(), _intro(flag=True)) == "B"

    def test_potential_flags_do_not_lower_grade(self):
        # 5 potential flags should leave grade at A (only confirmed count)
        flags = [_flag(f"w{i}", confidence="potential") for i in range(5)]
        assert _compute_grade(flags, _sting(), _evolution(), _intro()) == "A"

    def test_none_inputs_handled_gracefully(self):
        # All structural results can be None (e.g. audio too short for analysis)
        grade = _compute_grade([], None, None, None)
        assert grade == "A"

    def test_combined_structural_and_lyric_strikes(self):
        # 1 lyric flag + intro flag + evolution flag = 3 confirmed → C
        flags = [_flag()]
        grade = _compute_grade(flags, _sting(), _evolution(flag=True), _intro(flag=True))
        assert grade == "C"


# ---------------------------------------------------------------------------
# Compliance._check_intro  (pure method — no I/O, no models)
# ---------------------------------------------------------------------------

class TestCheckIntro:
    def setup_method(self):
        self.svc = Compliance()

    def test_intro_section_under_limit_does_not_flag(self):
        sections = [Section(label="intro", start=0.0, end=10.0)]
        result = self.svc._check_intro(sections, transcript=[])
        assert result.flag is False
        assert result.intro_seconds == pytest.approx(10.0)
        assert result.source == "allin1"

    def test_intro_section_over_limit_flags(self):
        sections = [Section(label="intro", start=0.0,
                            end=float(CONSTANTS.INTRO_MAX_SECONDS + 1))]
        result = self.svc._check_intro(sections, transcript=[])
        assert result.flag is True

    def test_multiple_intro_sections_sum_is_checked(self):
        # Two 8-second intros → 16s total → over 15s limit
        sections = [
            Section(label="intro", start=0.0, end=8.0),
            Section(label="intro", start=20.0, end=28.0),
        ]
        result = self.svc._check_intro(sections, transcript=[])
        assert result.intro_seconds == pytest.approx(16.0)
        assert result.flag is True

    def test_non_intro_sections_are_ignored(self):
        sections = [
            Section(label="verse", start=0.0, end=30.0),
            Section(label="chorus", start=30.0, end=60.0),
        ]
        result = self.svc._check_intro(sections, transcript=[])
        assert result.flag is False

    def test_whisper_fallback_when_no_sections(self):
        transcript = [_segment("here we go", start=12.0)]
        result = self.svc._check_intro(sections=[], transcript=transcript)
        assert result.source == "whisper_fallback"
        assert result.intro_seconds == pytest.approx(12.0)

    def test_whisper_fallback_over_limit_flags(self):
        transcript = [_segment("let's go", start=float(CONSTANTS.INTRO_MAX_SECONDS + 2))]
        result = self.svc._check_intro(sections=[], transcript=transcript)
        assert result.flag is True

    def test_no_sections_no_transcript_returns_unflagged(self):
        result = self.svc._check_intro(sections=[], transcript=[])
        assert result.flag is False
        assert result.source == "none"
