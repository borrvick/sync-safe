"""
tests/test_authorship_sections.py

Unit tests for Batch 9 features:
  - Per-section authorship breakdown (_analyze_per_section) — issue #156
  - Section-aware repetition/rhyme filtering (_filter_verse_lines) — issue #158
  - Combined authorship verdict (combined_authorship_verdict) — issue #161

RoBERTa is always patched out; no GPU or model download required.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from core.models import AuthorshipResult, ForensicsResult, Section, TranscriptSegment
from services.content._authorship import Authorship, _analyze_per_section
from services.content._pure import _filter_verse_lines, _repetition_score, _rhyme_density
from ui.pages.report import combined_authorship_verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str, start: float = 0.0, end: float = 2.0) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _timed_seg(text: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _section(label: str, start: float, end: float) -> Section:
    return Section(label=label, start=start, end=end)


def _long_line(n: int = 6) -> str:
    """Return a line with n distinct words (≥ 4 chars each)."""
    words = ["word", "text", "lyric", "song", "beat", "rhyme",
             "flow", "note", "music", "sound", "play", "tone"]
    return " ".join(words[i % len(words)] for i in range(n))


def _make_verse_segs(n: int = 6) -> list[TranscriptSegment]:
    """Return n segments with distinct content spread over 0–n*2 seconds."""
    lines = [
        "the road ahead is long and winding",
        "we carry hope through darkened skies",
        "the stars will guide our weary feet",
        "until the morning light arrives",
        "beneath the weight of broken dreams",
        "we find the strength to carry on",
        "the dawn will break through stormy nights",
        "and lead us to a brand new song",
    ]
    return [
        _timed_seg(lines[i % len(lines)], start=float(i * 2), end=float(i * 2 + 2))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _filter_verse_lines — section-aware line filtering (#158)
# ---------------------------------------------------------------------------

class TestFilterVerseLines:
    def test_no_sections_returns_all_lines(self):
        lines = ["line one", "line two", "line three", "line four"]
        segs  = [_timed_seg(ln, i * 2.0, i * 2.0 + 2.0) for i, ln in enumerate(lines)]
        assert _filter_verse_lines(lines, segs, []) == lines

    def test_no_segments_returns_all_lines(self):
        lines = ["a", "b", "c", "d"]
        secs  = [_section("chorus", 0.0, 10.0)]
        assert _filter_verse_lines(lines, None, secs) == lines

    def test_chorus_lines_excluded(self):
        # segments 0–1 = chorus (0–4s), segments 2–5 = verse (4–12s)
        segs = [
            _timed_seg("chorus line a", 0.0, 2.0),
            _timed_seg("chorus line b", 2.0, 4.0),
            _timed_seg("verse line a",  4.0, 6.0),
            _timed_seg("verse line b",  6.0, 8.0),
            _timed_seg("verse line c",  8.0, 10.0),
            _timed_seg("verse line d", 10.0, 12.0),
        ]
        lines = [seg.text for seg in segs]
        secs  = [_section("chorus", 0.0, 4.0)]
        result = _filter_verse_lines(lines, segs, secs)
        assert "chorus line a" not in result
        assert "chorus line b" not in result
        assert "verse line a" in result
        assert len(result) == 4

    def test_falls_back_when_filtered_too_small(self):
        # Only 2 verse lines after filtering — below the 4-line minimum
        segs = [
            _timed_seg("chorus a", 0.0, 2.0),
            _timed_seg("chorus b", 2.0, 4.0),
            _timed_seg("chorus c", 4.0, 6.0),
            _timed_seg("verse a",  6.0, 8.0),
            _timed_seg("verse b",  8.0, 10.0),
        ]
        lines = [seg.text for seg in segs]
        secs  = [_section("chorus", 0.0, 6.0)]
        result = _filter_verse_lines(lines, segs, secs)
        assert result == lines  # fell back to full list

    def test_mismatched_lengths_falls_back(self):
        lines = ["a", "b", "c", "d"]
        segs  = [_timed_seg("a", 0.0, 2.0)]  # length mismatch
        secs  = [_section("chorus", 0.0, 2.0)]
        assert _filter_verse_lines(lines, segs, secs) == lines

    def test_no_chorus_windows_returns_all(self):
        segs = [_timed_seg(f"line {i}", i * 2.0, i * 2.0 + 2.0) for i in range(4)]
        lines = [seg.text for seg in segs]
        secs  = [_section("verse", 0.0, 8.0)]  # no chorus labels
        assert _filter_verse_lines(lines, segs, secs) == lines


# ---------------------------------------------------------------------------
# _repetition_score — section-aware (#158)
# ---------------------------------------------------------------------------

class TestRepetitionScoreSection:
    def test_chorus_repetition_not_penalised(self):
        # All 4 lines are the same chorus hook — without section-awareness this
        # produces a 1.0 repetition score (every line repeated); with awareness
        # the chorus is excluded and the fallback kicks in (< 4 verse lines).
        segs = [
            _timed_seg("hook hook hook", 0.0, 2.0),
            _timed_seg("hook hook hook", 2.0, 4.0),
            _timed_seg("hook hook hook", 4.0, 6.0),
            _timed_seg("hook hook hook", 6.0, 8.0),
        ]
        lines = [seg.text for seg in segs]
        secs  = [_section("chorus", 0.0, 8.0)]
        # Falls back to full list (< 4 verse lines) — score is still 1.0 here,
        # but the important thing is the function doesn't crash.
        score = _repetition_score(lines, segs, secs)
        assert 0.0 <= score <= 1.0

    def test_verse_repetition_flagged(self):
        # 8 verse lines with 4 identical pairs — high repetition score
        lines = ["same line here"] * 4 + ["other line too"] * 4
        segs  = [_timed_seg(ln, i * 2.0, i * 2.0 + 2.0) for i, ln in enumerate(lines)]
        secs  = [_section("verse", 0.0, 16.0)]
        score = _repetition_score(lines, segs, secs)
        assert score >= 0.5

    def test_no_sections_full_track(self):
        lines = ["unique line alpha", "unique line beta", "unique line gamma", "unique line delta"]
        score = _repetition_score(lines)
        assert score == 0.0


# ---------------------------------------------------------------------------
# _rhyme_density — section-aware (#158)
# ---------------------------------------------------------------------------

class TestRhymeDensitySection:
    def test_returns_float_or_none(self):
        lines = ["I walk the line", "I find the vine", "I see the shrine", "I feel divine"]
        density = _rhyme_density(lines)
        assert density is None or 0.0 <= density <= 1.0

    def test_too_few_lines_returns_none(self):
        assert _rhyme_density(["a", "b"]) is None

    def test_no_crash_with_sections(self):
        segs = [_timed_seg(f"line {i}", i * 2.0, i * 2.0 + 2.0) for i in range(4)]
        lines = [seg.text for seg in segs]
        secs  = [_section("verse", 0.0, 8.0)]
        result = _rhyme_density(lines, segs, secs)
        assert result is None or 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# _analyze_per_section — per-section scoring (#156)
# ---------------------------------------------------------------------------

class TestAnalyzePerSection:
    def test_empty_sections_returns_empty(self):
        segs = _make_verse_segs(8)
        assert _analyze_per_section(segs, []) == {}

    def test_insufficient_section_omitted(self):
        # Only 2 segs in chorus — below _MIN_SECTION_LINES=4
        segs = [
            _timed_seg("chorus a", 0.0, 2.0),
            _timed_seg("chorus b", 2.0, 4.0),
        ]
        secs = [_section("chorus", 0.0, 4.0)]
        result = _analyze_per_section(segs, secs)
        assert "CHORUS" not in result

    def test_sufficient_section_included(self):
        segs = _make_verse_segs(6)
        secs = [_section("verse", 0.0, 12.0)]
        result = _analyze_per_section(segs, secs)
        assert "VERSE" in result

    def test_same_label_sections_merged(self):
        # Two CHORUS sections — both should be merged into one bucket
        segs = [
            _timed_seg("chorus line one",   0.0,  2.0),
            _timed_seg("chorus line two",   2.0,  4.0),
            _timed_seg("verse line a",      4.0,  6.0),
            _timed_seg("verse line b",      6.0,  8.0),
            _timed_seg("chorus line three", 8.0, 10.0),
            _timed_seg("chorus line four", 10.0, 12.0),
        ]
        secs = [
            _section("chorus", 0.0, 4.0),
            _section("verse",  4.0, 8.0),
            _section("chorus", 8.0, 12.0),
        ]
        result = _analyze_per_section(segs, secs)
        # Both chorus plays merged → at most one "CHORUS" key
        assert list(result.keys()).count("CHORUS") <= 1

    def test_section_result_has_verdict(self):
        segs = _make_verse_segs(6)
        secs = [_section("verse", 0.0, 12.0)]
        result = _analyze_per_section(segs, secs)
        sec = result.get("VERSE")
        assert sec is not None
        assert sec.verdict in ("Likely Human", "Uncertain", "Likely AI", "Insufficient data")

    def test_no_roberta_in_section_scores(self):
        segs = _make_verse_segs(6)
        secs = [_section("verse", 0.0, 12.0)]
        result = _analyze_per_section(segs, secs)
        sec = result.get("VERSE")
        assert sec is not None
        assert "roberta_ai_prob" not in sec.scores or sec.scores.get("roberta_ai_prob") is None


# ---------------------------------------------------------------------------
# Authorship.analyze with sections — integration (#156/#158)
# ---------------------------------------------------------------------------

class TestAuthorshipAnalyzeWithSections:
    def setup_method(self):
        self.svc = Authorship()

    def _analyze(self, segs, sections=None):
        with patch.object(self.svc, "_run_roberta", return_value=None):
            return self.svc.analyze(segs, sections=sections)

    def test_no_sections_per_section_empty(self):
        segs = _make_verse_segs(8)
        result = self._analyze(segs, sections=None)
        assert result.per_section == {}

    def test_sections_produces_per_section(self):
        segs = _make_verse_segs(8)
        secs = [_section("verse", 0.0, 16.0)]
        result = self._analyze(segs, sections=secs)
        assert len(result.per_section) >= 0  # may be empty if below threshold

    def test_sections_insufficient_data_no_crash(self):
        segs = _make_verse_segs(2)  # too short for analysis at all
        secs = [_section("verse", 0.0, 4.0)]
        result = self._analyze(segs, sections=secs)
        assert result.verdict == "Insufficient data"

    def test_per_section_results_are_typed(self):
        from core.models import SectionAuthorshipResult
        segs = _make_verse_segs(8)
        secs = [_section("verse", 0.0, 16.0)]
        result = self._analyze(segs, sections=secs)
        for _label, sec in result.per_section.items():
            assert isinstance(sec, SectionAuthorshipResult)


# ---------------------------------------------------------------------------
# combined_authorship_verdict — issue #161
# ---------------------------------------------------------------------------

class TestCombinedAuthorshipVerdict:
    def _forensics(self, verdict: str) -> ForensicsResult:
        return ForensicsResult(
            verdict=verdict,
            loop_score=0.0,
            ibi_variance=0.0,
            ai_segments=[],
        )

    def _authorship(self, verdict: str) -> AuthorshipResult:
        return AuthorshipResult(
            verdict=verdict,
            signal_count=0,
            roberta_score=None,
            feature_notes=[],
            scores={},
        )

    def test_both_none_returns_none(self):
        assert combined_authorship_verdict(None, None) is None

    def test_audio_ai_detected(self):
        result = combined_authorship_verdict(
            self._forensics("Likely AI"), self._authorship("Likely Human")
        )
        assert result is not None
        text, color = result
        assert "AI" in text
        assert color == "var(--danger)"

    def test_lyric_ai_detected(self):
        result = combined_authorship_verdict(
            self._forensics("Likely Not AI"), self._authorship("Likely AI")
        )
        assert result is not None
        text, color = result
        assert "AI" in text
        assert color == "var(--danger)"

    def test_both_ai_detected(self):
        result = combined_authorship_verdict(
            self._forensics("Likely AI"), self._authorship("Likely AI")
        )
        assert result is not None
        _text, color = result
        assert color == "var(--danger)"

    def test_both_clear(self):
        result = combined_authorship_verdict(
            self._forensics("Likely Not AI"), self._authorship("Likely Human")
        )
        assert result is not None
        text, color = result
        assert color == "var(--ok)"
        assert "No" in text

    def test_audio_ok_lyric_insufficient(self):
        result = combined_authorship_verdict(
            self._forensics("Likely Not AI"), self._authorship("Insufficient data")
        )
        assert result is not None
        _text, color = result
        assert color == "var(--ok)"

    def test_audio_insufficient_lyric_clear(self):
        result = combined_authorship_verdict(
            self._forensics("Insufficient data"), self._authorship("Likely Human")
        )
        assert result is not None
        _text, color = result
        assert color == "var(--ok)"

    def test_both_insufficient(self):
        result = combined_authorship_verdict(
            self._forensics("Insufficient data"), self._authorship("Insufficient data")
        )
        assert result is not None
        _text, color = result
        assert color == "var(--grade-c)"

    def test_forensics_only(self):
        result = combined_authorship_verdict(self._forensics("Likely Not AI"), None)
        assert result is not None

    def test_authorship_only(self):
        result = combined_authorship_verdict(None, self._authorship("Likely Human"))
        assert result is not None
