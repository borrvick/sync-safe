"""
tests/test_theme_mood.py
Unit tests for ThemeMoodAnalyzer (keyword-only path — no Groq dependency).
"""
from __future__ import annotations

import pytest

from core.models import ThemeMoodResult, TranscriptSegment
from services.content import ThemeMoodAnalyzer
from services.content._theme import _keyword_analyze


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=start + 2.0, text=text)


# ---------------------------------------------------------------------------
# _keyword_analyze (pure function)
# ---------------------------------------------------------------------------

class TestKeywordAnalyze:
    def test_returns_theme_mood_result(self):
        result = _keyword_analyze("I love you forever in my heart kiss")
        assert isinstance(result, ThemeMoodResult)

    def test_romantic_lyrics_detect_romantic_mood(self):
        result = _keyword_analyze(
            "love kiss heart forever darling embrace passion adore yours"
        )
        assert result.mood == "Romantic"

    def test_uplifting_lyrics_detect_uplifting_mood(self):
        result = _keyword_analyze(
            "rise shine hope alive bright free joy believe dream glory"
        )
        assert result.mood == "Uplifting"

    def test_empty_text_returns_neutral(self):
        result = _keyword_analyze("")
        assert result.mood == "Neutral"
        assert result.confidence == 0.0
        assert result.themes == []

    def test_themes_list_capped_at_three(self):
        # Carpet-bomb every taxonomy keyword — should still return ≤3 themes
        text = " ".join(
            "love heart kiss broken goodbye tears party dance night"
            " rise strong survive fight power".split()
            * 5
        )
        result = _keyword_analyze(text)
        assert len(result.themes) <= 3

    def test_confidence_between_zero_and_one(self):
        result = _keyword_analyze("run rush go fast move pump jump race drive push")
        assert 0.0 <= result.confidence <= 1.0

    def test_groq_enriched_always_false(self):
        result = _keyword_analyze("some random lyrics here")
        assert result.groq_enriched is False

    def test_raw_keywords_is_list_of_strings(self):
        result = _keyword_analyze("love heart fire gone")
        assert all(isinstance(k, str) for k in result.raw_keywords)

    def test_raw_keywords_capped_at_twenty(self):
        # Long text hitting many taxonomy words
        text = " ".join(["love", "kiss", "broken", "fire", "party", "rise",
                          "shine", "hope", "dance", "fight", "power", "dream",
                          "tears", "gone", "alone", "pain", "run", "race",
                          "fast", "grind", "hustle", "win", "shadow", "dark"])
        result = _keyword_analyze(text)
        assert len(result.raw_keywords) <= 20


# ---------------------------------------------------------------------------
# ThemeMoodAnalyzer (keyword-only, groq_client=None)
# ---------------------------------------------------------------------------

class TestThemeMoodAnalyzer:
    def test_analyze_returns_theme_mood_result(self):
        segs = [_seg("I love you forever in my heart")]
        result = ThemeMoodAnalyzer().analyze(segs)
        assert isinstance(result, ThemeMoodResult)

    def test_empty_transcript_returns_neutral(self):
        result = ThemeMoodAnalyzer().analyze([])
        assert result.mood == "Neutral"
        assert result.themes == []

    def test_whitespace_only_segments_treated_as_empty(self):
        segs = [_seg("   "), _seg("\t"), _seg("")]
        result = ThemeMoodAnalyzer().analyze(segs)
        assert result.mood == "Neutral"

    def test_groq_enriched_false_without_client(self):
        segs = [_seg("love kiss heart forever darling embrace")]
        result = ThemeMoodAnalyzer(groq_client=None).analyze(segs)
        assert result.groq_enriched is False

    def test_romantic_multi_segment(self):
        segs = [
            _seg("love kiss heart forever darling", start=0.0),
            _seg("embrace passion adore yours tender", start=2.0),
            _seg("devotion warmth longing intimate", start=4.0),
        ]
        result = ThemeMoodAnalyzer().analyze(segs)
        assert result.mood == "Romantic"
        assert "Love & Romance" in result.themes

    def test_dark_lyrics_detect_dark_mood(self):
        segs = [
            _seg("dark shadow night death fear demon void hollow", start=0.0),
            _seg("ghost grave haunt nightmare dread abyss wicked", start=2.0),
        ]
        result = ThemeMoodAnalyzer().analyze(segs)
        assert result.mood == "Dark"

    def test_melancholic_lyrics_detect_melancholic_mood(self):
        segs = [
            _seg("cry tears broken miss gone alone pain hurt goodbye", start=0.0),
        ]
        result = ThemeMoodAnalyzer().analyze(segs)
        assert result.mood == "Melancholic"
