"""
tests/test_discovery.py
Unit tests for services/discovery.py — network calls and subprocess are mocked.
"""
from unittest.mock import MagicMock, patch

import pytest

from services.discovery import (
    Discovery,
    _classify_popularity,
    _find_binary,
    _parse_track_list,
    _resolve_youtube_url,
)
from core.config import CONSTANTS
from core.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# _classify_popularity (pure helper — no mocks needed)
# ---------------------------------------------------------------------------

class TestClassifyPopularity:
    """Tests for _classify_popularity — uses CONSTANTS directly so tier boundaries
    come from config, not magic numbers in tests."""

    def test_emerging_tier(self) -> None:
        result = _classify_popularity(5_000, 50_000, CONSTANTS)
        assert result.tier == "Emerging"
        assert result.listeners == 5_000
        assert result.playcount == 50_000
        assert result.sync_cost_low  == CONSTANTS.SYNC_COST_EMERGING[0]
        assert result.sync_cost_high == CONSTANTS.SYNC_COST_EMERGING[1]

    def test_regional_tier_at_boundary(self) -> None:
        result = _classify_popularity(CONSTANTS.POPULARITY_REGIONAL_MIN, 0, CONSTANTS)
        assert result.tier == "Regional"

    def test_mainstream_tier_at_boundary(self) -> None:
        result = _classify_popularity(CONSTANTS.POPULARITY_MAINSTREAM_MIN, 0, CONSTANTS)
        assert result.tier == "Mainstream"

    def test_global_tier_at_boundary(self) -> None:
        result = _classify_popularity(CONSTANTS.POPULARITY_GLOBAL_MIN, 0, CONSTANTS)
        assert result.tier == "Global"
        assert result.sync_cost_low  == CONSTANTS.SYNC_COST_GLOBAL[0]
        assert result.sync_cost_high == CONSTANTS.SYNC_COST_GLOBAL[1]

    def test_zero_listeners(self) -> None:
        result = _classify_popularity(0, 0, CONSTANTS)
        assert result.tier == "Emerging"
        assert result.listeners == 0

    def test_just_below_regional(self) -> None:
        result = _classify_popularity(CONSTANTS.POPULARITY_REGIONAL_MIN - 1, 0, CONSTANTS)
        assert result.tier == "Emerging"


# ---------------------------------------------------------------------------
# _parse_track_list (pure helper — no mocks needed)
# ---------------------------------------------------------------------------

class TestParseTrackList:
    def test_parses_standard_payload(self):
        payload = [
            {"name": "Creep", "artist": {"name": "Radiohead"}},
            {"name": "High and Dry", "artist": {"name": "Radiohead"}},
        ]
        assert _parse_track_list(payload) == [
            ("Radiohead", "Creep"),
            ("Radiohead", "High and Dry"),
        ]

    def test_respects_max_similar_limit(self):
        payload = [{"name": f"Track {i}", "artist": {"name": "Artist"}} for i in range(10)]
        assert len(_parse_track_list(payload)) <= 5

    def test_skips_entries_missing_name(self):
        payload = [
            {"name": "", "artist": {"name": "Artist"}},
            {"name": "Valid Track", "artist": {"name": "Artist"}},
        ]
        result = _parse_track_list(payload)
        assert len(result) == 1
        assert result[0][1] == "Valid Track"

    def test_handles_string_artist(self):
        payload = [{"name": "Song", "artist": "Solo Artist"}]
        assert _parse_track_list(payload) == [("Solo Artist", "Song")]

    def test_empty_list(self):
        assert _parse_track_list([]) == []


# ---------------------------------------------------------------------------
# _resolve_youtube_url (subprocess mocked)
# ---------------------------------------------------------------------------

class TestResolveYoutubeUrl:
    def test_returns_url_on_success(self):
        mock_result = MagicMock()
        mock_result.stdout = "https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"

        with patch("services.discovery.subprocess.run", return_value=mock_result), \
             patch("services.discovery._find_binary", return_value="/usr/bin/yt-dlp"):
            url = _resolve_youtube_url("Rick Astley", "Never Gonna Give You Up")

        assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_returns_none_on_empty_output(self):
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("services.discovery.subprocess.run", return_value=mock_result), \
             patch("services.discovery._find_binary", return_value="/usr/bin/yt-dlp"):
            assert _resolve_youtube_url("Unknown", "Track") is None

    def test_returns_none_on_non_https_output(self):
        mock_result = MagicMock()
        mock_result.stdout = "not-a-url"

        with patch("services.discovery.subprocess.run", return_value=mock_result), \
             patch("services.discovery._find_binary", return_value="/usr/bin/yt-dlp"):
            assert _resolve_youtube_url("Artist", "Track") is None

    def test_returns_none_on_subprocess_exception(self):
        with patch("services.discovery.subprocess.run", side_effect=FileNotFoundError), \
             patch("services.discovery._find_binary", return_value="/usr/bin/yt-dlp"):
            assert _resolve_youtube_url("Artist", "Track") is None

    def test_returns_none_when_binary_not_found(self):
        with patch("services.discovery._find_binary", return_value=None):
            assert _resolve_youtube_url("Artist", "Track") is None


# ---------------------------------------------------------------------------
# Discovery.find_similar (requests + subprocess mocked)
# ---------------------------------------------------------------------------

class TestDiscovery:
    _LASTFM_RESPONSE = {
        "similartracks": {
            "track": [
                {"name": "Creep", "artist": {"name": "Radiohead"}},
                {"name": "Karma Police", "artist": {"name": "Radiohead"}},
            ]
        }
    }

    def _mock_yt_run(self, *args, **kwargs):
        result = MagicMock()
        result.stdout = "https://www.youtube.com/watch?v=fake123\n"
        return result

    def test_returns_similar_tracks(self):
        from core.config import get_settings
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._LASTFM_RESPONSE
        mock_resp.raise_for_status.return_value = None

        with patch.object(get_settings(), "lastfm_api_key", "fakekey"), \
             patch("services.discovery.get_settings") as mock_settings, \
             patch("services.discovery.requests.get", return_value=mock_resp), \
             patch("services.discovery.subprocess.run", side_effect=self._mock_yt_run), \
             patch("services.discovery._find_binary", return_value="/usr/bin/yt-dlp"):

            mock_settings.return_value.lastfm_api_key = "fakekey"
            svc     = Discovery()
            results = svc.find_similar("Fake Plastic Trees", "Radiohead")

        assert len(results) == 2
        assert results[0].title  == "Creep"
        assert results[0].artist == "Radiohead"
        assert results[0].youtube_url == "https://www.youtube.com/watch?v=fake123"

    def test_raises_configuration_error_without_api_key(self):
        with patch("services.discovery.get_settings") as mock_settings:
            mock_settings.return_value.lastfm_api_key = ""
            svc = Discovery()
            with pytest.raises(ConfigurationError):
                svc.find_similar("Any Title", "Any Artist")

    def test_returns_empty_for_empty_inputs(self):
        with patch("services.discovery.get_settings") as mock_settings:
            mock_settings.return_value.lastfm_api_key = "fakekey"
            svc = Discovery()
            assert svc.find_similar("", "") == []

    def test_youtube_url_is_none_when_resolve_fails(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._LASTFM_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_yt = MagicMock()
        mock_yt.stdout = ""

        with patch("services.discovery.get_settings") as mock_settings, \
             patch("services.discovery.requests.get", return_value=mock_resp), \
             patch("services.discovery.subprocess.run", return_value=mock_yt), \
             patch("services.discovery._find_binary", return_value="/usr/bin/yt-dlp"):

            mock_settings.return_value.lastfm_api_key = "fakekey"
            svc     = Discovery()
            results = svc.find_similar("Title", "Artist")

        assert all(r.youtube_url is None for r in results)
