"""
tests/test_content.py
Unit tests for services/content/_theme.py — keyword scoring, negation handling,
Groq enrichment path, and ThemeMoodAnalyzer.analyze() / enrich() (#167/#168/#169).
"""
from __future__ import annotations

import pytest

from core.models import ThemeMoodResult, TranscriptSegment
from services.content import ThemeMoodAnalyzer
from services.content._theme import _keyword_analyze


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(text: str) -> TranscriptSegment:
    return TranscriptSegment(start=0.0, end=1.0, text=text)


# ---------------------------------------------------------------------------
# _keyword_analyze — empty / blank input
# ---------------------------------------------------------------------------

class TestKeywordAnalyzeEmpty:
    def test_empty_string_returns_neutral(self) -> None:
        r = _keyword_analyze("")
        assert r.mood == "Neutral"
        assert r.confidence == 0.0
        assert r.themes == []

    def test_blank_string_returns_neutral(self) -> None:
        r = _keyword_analyze("   ")
        assert r.mood == "Neutral"
        assert r.themes == []

    def test_empty_has_no_theme_scores(self) -> None:
        r = _keyword_analyze("")
        assert r.theme_scores == {}

    def test_empty_mood_summary_is_none(self) -> None:
        r = _keyword_analyze("")
        assert r.mood_summary is None


# ---------------------------------------------------------------------------
# _keyword_analyze — mood detection
# ---------------------------------------------------------------------------

class TestKeywordAnalyzeMood:
    def test_uplifting_lyrics_detected(self) -> None:
        r = _keyword_analyze("rise shine hope alive bright free joy win")
        assert r.mood == "Uplifting"
        assert r.confidence > 0.0

    def test_dark_lyrics_detected(self) -> None:
        r = _keyword_analyze("dark shadow night death fear demon void hollow ghost")
        assert r.mood == "Dark"

    def test_confidence_capped_at_one(self) -> None:
        # Repeat keywords many times — confidence must never exceed 1.0
        text = ("love heart kiss " * 50)
        r = _keyword_analyze(text)
        assert r.confidence <= 1.0

    def test_neutral_when_no_keywords_match(self) -> None:
        r = _keyword_analyze("the cat sat on the mat")
        assert r.mood == "Neutral"


# ---------------------------------------------------------------------------
# _keyword_analyze — theme scoring
# ---------------------------------------------------------------------------

class TestKeywordAnalyzeThemes:
    def test_party_theme_detected(self) -> None:
        r = _keyword_analyze("party dance night celebrate drinks club move")
        assert "Party & Celebration" in r.themes

    def test_theme_scores_are_zero_to_one(self) -> None:
        r = _keyword_analyze("love heart kiss forever darling hold")
        for score in r.theme_scores.values():
            assert 0.0 <= score <= 1.0

    def test_top_theme_has_highest_score(self) -> None:
        r = _keyword_analyze("love heart kiss together forever darling hold embrace")
        if r.themes:
            top = r.themes[0]
            assert r.theme_scores[top] == max(r.theme_scores.values())

    def test_at_most_three_themes_returned(self) -> None:
        # Dense multi-theme lyric — still capped at 3
        text = (
            "love heart kiss party dance celebrate "
            "rise strong survive fight hustle grind"
        )
        r = _keyword_analyze(text)
        assert len(r.themes) <= 3

    def test_low_scoring_themes_excluded(self) -> None:
        # Only one theme reinforced — others should be below THEME_MIN_CONFIDENCE
        r = _keyword_analyze("god faith pray heaven soul blessed spirit grace holy lord")
        for t in r.themes:
            assert r.theme_scores[t] >= 0.25

    def test_top_category_populated(self) -> None:
        r = _keyword_analyze("party dance celebrate night drinks club move lit crowd")
        if r.themes:
            assert r.top_category in ("energy", "emotional", "seasonal")


# ---------------------------------------------------------------------------
# _keyword_analyze — negation handling
# ---------------------------------------------------------------------------

class TestNegationHandling:
    def test_negated_keyword_reduces_theme_score(self) -> None:
        positive = _keyword_analyze("love heart kiss together forever darling hold embrace")
        negated  = _keyword_analyze("not love no heart never kiss without together")
        pos_score = positive.theme_scores.get("Love & Romance", 0.0)
        neg_score = negated.theme_scores.get("Love & Romance", 0.0)
        assert neg_score < pos_score

    def test_negation_does_not_go_below_zero(self) -> None:
        # Flood with negations — raw_pts floor is 0
        r = _keyword_analyze("not love no heart never kiss without darling")
        for score in r.theme_scores.values():
            assert score >= 0.0


# ---------------------------------------------------------------------------
# _keyword_analyze — raw_keywords transparency
# ---------------------------------------------------------------------------

class TestRawKeywords:
    def test_raw_keywords_not_empty_when_matches_exist(self) -> None:
        r = _keyword_analyze("love heart kiss")
        assert len(r.raw_keywords) > 0

    def test_raw_keywords_capped_at_twenty(self) -> None:
        # Dense lyrics covering many themes
        text = (
            "love heart kiss party dance celebrate rise strong survive fight "
            "summer beach sun vacation pool waves god faith pray heaven soul "
            "hustle grind remember childhood memory yesterday"
        )
        r = _keyword_analyze(text)
        assert len(r.raw_keywords) <= 20


# ---------------------------------------------------------------------------
# ThemeMoodAnalyzer.analyze() — integration
# ---------------------------------------------------------------------------

class TestThemeMoodAnalyzerAnalyze:
    def setup_method(self) -> None:
        self.svc = ThemeMoodAnalyzer()

    def test_returns_theme_mood_result(self) -> None:
        segments = [_seg("love heart kiss together forever")]
        r = self.svc.analyze(segments)
        assert isinstance(r, ThemeMoodResult)

    def test_groq_not_enriched_by_default(self) -> None:
        segments = [_seg("rise shine hope alive")]
        r = self.svc.analyze(segments)
        assert r.groq_enriched is False
        assert r.mood_summary is None

    def test_empty_transcript_handled(self) -> None:
        r = self.svc.analyze([])
        assert r.mood == "Neutral"

    def test_blank_segments_handled(self) -> None:
        r = self.svc.analyze([_seg("   "), _seg("")])
        assert r.mood == "Neutral"


# ---------------------------------------------------------------------------
# ThemeMoodAnalyzer.enrich() — no Groq key (keyword-only fallback)
# ---------------------------------------------------------------------------

class TestThemeMoodAnalyzerEnrichNoKey:
    def test_enrich_returns_original_when_no_groq(self) -> None:
        svc    = ThemeMoodAnalyzer(groq_client=None)
        result = _keyword_analyze("love heart kiss together")
        # Patch _try_init_groq to return None (no key in test env)
        import services.content._theme as _theme_mod
        orig = _theme_mod._try_init_groq
        _theme_mod._try_init_groq = lambda: None
        try:
            enriched = svc.enrich(result, "love heart kiss together")
        finally:
            _theme_mod._try_init_groq = orig
        assert enriched is result  # same object — no enrichment
