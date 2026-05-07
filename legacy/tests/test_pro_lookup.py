"""
tests/test_pro_lookup.py
Unit tests for services/pro_lookup.py — pure helper functions only.
ProLookup.lookup() makes a network call and is tested manually.
"""
from __future__ import annotations

import pytest

from services.legal import _infer_pro, _parse_first_recording


# ---------------------------------------------------------------------------
# _infer_pro
# ---------------------------------------------------------------------------

class TestInferPro:
    def test_us_returns_combined_label(self) -> None:
        assert _infer_pro("US") == "ASCAP / BMI / SESAC (US)"

    def test_gb_returns_prs(self) -> None:
        assert _infer_pro("GB") == "PRS for Music (UK)"

    def test_de_returns_gema(self) -> None:
        assert _infer_pro("DE") == "GEMA (Germany)"

    def test_unknown_country_returns_none(self) -> None:
        assert _infer_pro("ZZ") is None

    def test_lowercase_input_normalised(self) -> None:
        assert _infer_pro("us") == "ASCAP / BMI / SESAC (US)"

    def test_empty_string_returns_none(self) -> None:
        assert _infer_pro("") is None


# ---------------------------------------------------------------------------
# _parse_first_recording
# ---------------------------------------------------------------------------

class TestParseFirstRecording:
    def test_empty_recordings_returns_none_tuple(self) -> None:
        assert _parse_first_recording({}) == (None, None, None, None)
        assert _parse_first_recording({"recordings": []}) == (None, None, None, None)

    def test_isrc_extracted_from_first_recording(self) -> None:
        data = {"recordings": [{"isrcs": ["US-ABC-23-12345"]}]}
        isrc, *_ = _parse_first_recording(data)
        assert isrc == "US-ABC-23-12345"

    def test_pro_inferred_from_isrc_country(self) -> None:
        data = {"recordings": [{"isrcs": ["GB-XYZ-23-99999"]}]}
        _, pro, *_ = _parse_first_recording(data)
        assert pro == "PRS for Music (UK)"

    def test_no_isrcs_returns_none_for_both(self) -> None:
        data = {"recordings": [{"isrcs": []}]}
        isrc, pro, *_ = _parse_first_recording(data)
        assert isrc is None
        assert pro is None

    def test_unknown_country_isrc_returns_none_pro(self) -> None:
        data = {"recordings": [{"isrcs": ["ZZ-ABC-23-12345"]}]}
        isrc, pro, *_ = _parse_first_recording(data)
        assert isrc == "ZZ-ABC-23-12345"
        assert pro is None

    def test_second_recording_ignored(self) -> None:
        """Only the first result should be used."""
        data = {
            "recordings": [
                {"isrcs": ["US-AAA-23-00001"]},
                {"isrcs": ["GB-BBB-23-00002"]},
            ]
        }
        isrc, pro, *_ = _parse_first_recording(data)
        assert isrc == "US-AAA-23-00001"
        assert pro == "ASCAP / BMI / SESAC (US)"

    def test_mb_score_and_artist_extracted(self) -> None:
        data = {
            "recordings": [{
                "isrcs": ["US-ABC-23-12345"],
                "score": "95",
                "artist-credit": [{"artist": {"name": "The Weeknd"}}],
            }]
        }
        _, _, mb_score, mb_artist = _parse_first_recording(data)
        assert mb_score == 95
        assert mb_artist == "The Weeknd"

    def test_missing_score_and_artist_returns_none(self) -> None:
        data = {"recordings": [{"isrcs": ["US-ABC-23-12345"]}]}
        _, _, mb_score, mb_artist = _parse_first_recording(data)
        assert mb_score == 0 or mb_score is None  # parsed as int(0) from absent field
        assert mb_artist is None
