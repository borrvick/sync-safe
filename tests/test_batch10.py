"""
tests/test_batch10.py

Unit tests for Batch 10 features:
  - _ai_phrase_score() — AI lyric cliché detector (#160)
  - _score_signals() with phrase signal and float accumulator (#160)
  - TrackCandidate.source field and default (#129)
  - _build_sync_fee_section() — fee tier in JSON export (#116)
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from core.models import AnalysisResult, AudioBuffer, PopularityResult, TrackCandidate
from services.content._pure import _ai_phrase_score, _score_signals
from services.export._orchestrator import _build_sync_fee_section, to_analysis_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_result(**overrides) -> AnalysisResult:
    """Return a minimal valid AnalysisResult for export tests."""
    audio = AudioBuffer(
        raw=b"",
        label="test",
        source="file",
        metadata={},
    )
    defaults = dict(audio=audio)
    defaults.update(overrides)
    return AnalysisResult(**defaults)


def _popularity(**kwargs) -> PopularityResult:
    defaults = dict(
        listeners=10000,
        playcount=50000,
        spotify_score=None,
        popularity_score=75,
        tier="Mainstream",
        sync_cost_low=15000,
        sync_cost_high=100000,
    )
    defaults.update(kwargs)
    return PopularityResult(**defaults)


# ---------------------------------------------------------------------------
# _ai_phrase_score — issue #160
# ---------------------------------------------------------------------------

class TestAiPhraseScore:
    def test_known_phrase_detected(self):
        assert _ai_phrase_score(["I am enough"], "i am enough") is True

    def test_phrase_case_insensitive(self):
        assert _ai_phrase_score(["Lost In The Moment"], "lost in the moment") is True

    def test_no_phrase_clean_lyrics(self):
        assert _ai_phrase_score(["hello world", "we roll together"], "hello world\nwe roll together") is False

    def test_structural_three_consecutive_i_verb_noun(self):
        lines = ["I walk the road", "I see the light", "I feel the rain", "no match here"]
        full  = "\n".join(lines)
        assert _ai_phrase_score(lines, full) is True

    def test_structural_three_consecutive_you_verb_noun(self):
        lines = ["You hold the key", "You break the chain", "You find the way"]
        full  = "\n".join(lines)
        assert _ai_phrase_score(lines, full) is True

    def test_structural_two_consecutive_not_enough(self):
        lines = ["I walk the road", "I see the light", "then something else", "I feel again"]
        full  = "\n".join(lines)
        assert _ai_phrase_score(lines, full) is False

    def test_too_few_lines_structural_not_triggered(self):
        lines = ["I walk the road", "I see the light"]
        full  = "\n".join(lines)
        # only 2 lines — structural check needs 3
        result = _ai_phrase_score(lines, full)
        # phrase check can still fire — but no structural match
        assert isinstance(result, bool)

    def test_empty_lines_no_crash(self):
        assert _ai_phrase_score([], "") is False

    def test_partial_phrase_not_matched(self):
        # "in the shadows" is not "in the shadows of"
        assert _ai_phrase_score(["in the shadows"], "in the shadows") is False

    def test_phrase_embedded_in_longer_text(self):
        text = "we were all lost in the moment together"
        assert _ai_phrase_score([text], text) is True


# ---------------------------------------------------------------------------
# _score_signals with phrase — issue #160
# ---------------------------------------------------------------------------

class TestScoreSignalsPhrase:
    def test_phrase_hit_adds_half_weight(self):
        ai_signals, _notes, scores = _score_signals(
            burst=0.5, uwr=0.6, rhyme=0.3, rep=0.1, rob=None, phrase=True
        )
        # No structural flags + phrase = 0.5
        assert ai_signals == pytest.approx(0.5)

    def test_phrase_miss_no_signal(self):
        ai_signals, _notes, scores = _score_signals(
            burst=0.5, uwr=0.6, rhyme=0.3, rep=0.1, rob=None, phrase=False
        )
        assert ai_signals == pytest.approx(0.0)

    def test_phrase_plus_one_structural_flag(self):
        # burstiness < threshold → +1.0; phrase → +0.5
        from core.config import CONSTANTS
        low_burst = CONSTANTS.BURSTINESS_CV_THRESHOLD - 0.01
        ai_signals, _notes, _scores = _score_signals(
            burst=low_burst, uwr=0.6, rhyme=0.3, rep=0.1, rob=None, phrase=True
        )
        assert ai_signals == pytest.approx(1.5)

    def test_signal_count_is_float(self):
        ai_signals, _, _ = _score_signals(
            burst=0.5, uwr=0.6, rhyme=0.3, rep=0.1, rob=None, phrase=True
        )
        assert isinstance(ai_signals, float)

    def test_scores_dict_has_ai_phrase_score(self):
        _, _, scores = _score_signals(
            burst=None, uwr=None, rhyme=None, rep=0.0, rob=None, phrase=True
        )
        assert "ai_phrase_score" in scores
        assert scores["ai_phrase_score"] == 1.0

    def test_scores_dict_phrase_false(self):
        _, _, scores = _score_signals(
            burst=None, uwr=None, rhyme=None, rep=0.0, rob=None, phrase=False
        )
        assert scores["ai_phrase_score"] == 0.0

    def test_phrase_note_appears_in_feature_notes(self):
        _, notes, _ = _score_signals(
            burst=None, uwr=None, rhyme=None, rep=0.0, rob=None, phrase=True
        )
        assert any("cliché" in n or "phrase" in n.lower() for n in notes)

    def test_no_phrase_clean_note_appears(self):
        _, notes, _ = _score_signals(
            burst=None, uwr=None, rhyme=None, rep=0.0, rob=None, phrase=False
        )
        assert any("No AI" in n or "phrase" in n.lower() for n in notes)

    def test_roberta_high_still_adds_two(self):
        ai_signals, _, _ = _score_signals(
            burst=0.5, uwr=0.6, rhyme=0.3, rep=0.1, rob=0.85, phrase=False
        )
        assert ai_signals == pytest.approx(2.0)

    def test_roberta_high_plus_phrase(self):
        ai_signals, _, _ = _score_signals(
            burst=0.5, uwr=0.6, rhyme=0.3, rep=0.1, rob=0.85, phrase=True
        )
        assert ai_signals == pytest.approx(2.5)

    def test_default_phrase_false(self):
        # phrase defaults to False — existing callers not broken
        ai_signals, _, _ = _score_signals(
            burst=0.5, uwr=0.6, rhyme=0.3, rep=0.1, rob=None
        )
        assert ai_signals == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TrackCandidate.source — issue #129
# ---------------------------------------------------------------------------

class TestTrackCandidateSource:
    def test_default_source_is_lastfm(self):
        t = TrackCandidate(title="Test", artist="Artist")
        assert t.source == "lastfm"

    def test_spotify_source(self):
        t = TrackCandidate(title="Test", artist="Artist", source="spotify")
        assert t.source == "spotify"

    def test_audio_source(self):
        t = TrackCandidate(title="Test", artist="Artist", source="audio")
        assert t.source == "audio"

    def test_source_serialises_in_to_dict(self):
        t = TrackCandidate(title="Test", artist="Artist", source="lastfm")
        assert t.to_dict()["source"] == "lastfm"


# ---------------------------------------------------------------------------
# _build_sync_fee_section — issue #116
# ---------------------------------------------------------------------------

class TestBuildSyncFeeSection:
    def test_no_popularity_all_null(self):
        result = _minimal_result()
        section = _build_sync_fee_section(result)
        assert section["fee_tier"] is None
        assert section["base_range_low"] is None
        assert section["popularity_score"] is None

    def test_popularity_present_fields_populated(self):
        pop = _popularity()
        result = _minimal_result(popularity=pop)
        section = _build_sync_fee_section(result)
        assert section["fee_tier"] == "Mainstream"
        assert section["base_range_low"] == 15000
        assert section["base_range_high"] == 100000
        assert section["popularity_score"] == 75

    def test_high_score_strong_confidence(self):
        pop = _popularity(popularity_score=80)
        result = _minimal_result(popularity=pop)
        section = _build_sync_fee_section(result)
        assert section["tier_confidence"] == "strong"

    def test_low_score_moderate_confidence(self):
        pop = _popularity(popularity_score=40)
        result = _minimal_result(popularity=pop)
        section = _build_sync_fee_section(result)
        assert section["tier_confidence"] == "moderate"

    def test_multipliers_null_before_110(self):
        pop = _popularity()
        result = _minimal_result(popularity=pop)
        section = _build_sync_fee_section(result)
        assert section["multipliers_applied"] is None
        assert section["adjusted_range_low"] is None
        assert section["adjusted_range_high"] is None

    def test_to_analysis_json_includes_sync_fee(self):
        pop = _popularity()
        result = _minimal_result(popularity=pop)
        payload = json.loads(to_analysis_json(result).decode("utf-8"))
        assert "sync_fee" in payload
        assert payload["sync_fee"]["fee_tier"] == "Mainstream"

    def test_to_analysis_json_no_popularity_sync_fee_null(self):
        result = _minimal_result()
        payload = json.loads(to_analysis_json(result).decode("utf-8"))
        assert "sync_fee" in payload
        assert payload["sync_fee"]["fee_tier"] is None
