"""
tests/test_authorship_skip_reason.py

Unit tests for the four skip_reason tiers in Authorship.analyze().

Tier logic:
  "instrumental" — empty transcript OR full_text < 80 chars
  "too_short"    — < 4 non-empty lines
  "short"        — 4–7 lines (analysis runs; result is advisory)
  None           — 8+ lines (normal path)

RoBERTa is patched out for the "short" and None paths so no GPU or
model download is required.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from core.models import TranscriptSegment
from services.content._authorship import Authorship


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str, start: float = 0.0, end: float = 5.0) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def _lines(n: int, length: int = 20) -> list[TranscriptSegment]:
    """Return n segments each with `length` characters of text."""
    return [_seg("a" * length, start=float(i)) for i in range(n)]


# ---------------------------------------------------------------------------
# "instrumental" — empty or whitespace-only transcript
# ---------------------------------------------------------------------------

class TestSkipReasonInstrumental:
    def setup_method(self):
        self.svc = Authorship()

    def test_empty_transcript(self):
        result = self.svc.analyze([])
        assert result.skip_reason == "instrumental"
        assert result.verdict == "Insufficient data"
        assert result.signal_count == 0

    def test_whitespace_only_segments(self):
        result = self.svc.analyze([_seg("   "), _seg("\t"), _seg("")])
        assert result.skip_reason == "instrumental"

    def test_full_text_under_80_chars(self):
        # 3 lines, each 10 chars = 30 chars total — under the 80-char threshold
        result = self.svc.analyze([_seg("a" * 10), _seg("b" * 10), _seg("c" * 10)])
        assert result.skip_reason == "instrumental"

    def test_single_long_line_under_80(self):
        result = self.svc.analyze([_seg("word " * 10)])  # 50 chars
        assert result.skip_reason == "instrumental"


# ---------------------------------------------------------------------------
# "too_short" — < 4 non-empty lines, but text >= 80 chars
# ---------------------------------------------------------------------------

class TestSkipReasonTooShort:
    def setup_method(self):
        self.svc = Authorship()

    def test_three_lines_over_80_chars(self):
        # 3 lines × 30 chars each = 90 chars — passes the char guard, fails the line guard
        result = self.svc.analyze([_seg("a" * 30), _seg("b" * 30), _seg("c" * 30)])
        assert result.skip_reason == "too_short"
        assert result.verdict == "Insufficient data"

    def test_one_long_line(self):
        result = self.svc.analyze([_seg("word " * 20)])  # 100 chars, 1 line
        assert result.skip_reason == "too_short"

    def test_two_lines_over_80_chars(self):
        result = self.svc.analyze([_seg("a" * 50), _seg("b" * 50)])
        assert result.skip_reason == "too_short"


# ---------------------------------------------------------------------------
# "short" — 4–7 lines (analysis runs; skip_reason advisory only)
# ---------------------------------------------------------------------------

class TestSkipReasonShort:
    def setup_method(self):
        self.svc = Authorship()

    def _analyze_with_mocked_roberta(self, segs: list[TranscriptSegment]):
        with patch.object(self.svc, "_run_roberta", return_value=None):
            return self.svc.analyze(segs)

    def test_four_lines(self):
        result = self._analyze_with_mocked_roberta(_lines(4))
        assert result.skip_reason == "short"
        assert result.verdict != "Insufficient data"  # analysis ran

    def test_five_lines(self):
        result = self._analyze_with_mocked_roberta(_lines(5))
        assert result.skip_reason == "short"

    def test_seven_lines(self):
        result = self._analyze_with_mocked_roberta(_lines(7))
        assert result.skip_reason == "short"

    def test_short_still_returns_verdict(self):
        result = self._analyze_with_mocked_roberta(_lines(5))
        assert result.verdict in ("Likely Human", "Uncertain", "Likely AI")


# ---------------------------------------------------------------------------
# None — 8+ lines (normal path, no advisory message)
# ---------------------------------------------------------------------------

class TestSkipReasonNormal:
    def setup_method(self):
        self.svc = Authorship()

    def _analyze_with_mocked_roberta(self, segs: list[TranscriptSegment]):
        with patch.object(self.svc, "_run_roberta", return_value=None):
            return self.svc.analyze(segs)

    def test_eight_lines(self):
        result = self._analyze_with_mocked_roberta(_lines(8))
        assert result.skip_reason is None

    def test_twenty_lines(self):
        result = self._analyze_with_mocked_roberta(_lines(20))
        assert result.skip_reason is None

    def test_normal_path_returns_verdict(self):
        result = self._analyze_with_mocked_roberta(_lines(10))
        assert result.verdict in ("Likely Human", "Uncertain", "Likely AI")
        assert result.skip_reason is None
