"""
tests/test_discovery.py
Unit tests for services/discovery.py — network calls and subprocess are mocked.
"""
from unittest.mock import MagicMock, patch

import pytest

from services.discovery import (
    Discovery,
    _classify_popularity,
    _clean_title_for_lastfm,
    _find_binary,
    _normalise_lastfm,
    _normalise_views,
    _parse_track_list,
    _resolve_youtube_url,
)
from core.config import CONSTANTS
from core.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# _normalise_lastfm / _normalise_views (pure helpers — no mocks needed)
# ---------------------------------------------------------------------------

class TestNormaliseLastfm:
    def test_zero_listeners_returns_zero(self) -> None:
        assert _normalise_lastfm(0, CONSTANTS) == 0

    def test_global_ceiling_returns_75(self) -> None:
        # The global boundary maps to exactly 75 — above it extends toward 100
        assert _normalise_lastfm(CONSTANTS.LASTFM_LISTENERS_GLOBAL, CONSTANTS) == 75

    def test_above_ceiling_clamped_to_100(self) -> None:
        assert _normalise_lastfm(CONSTANTS.LASTFM_LISTENERS_GLOBAL * 10, CONSTANTS) == 100

    def test_small_count_returns_low_score(self) -> None:
        assert _normalise_lastfm(100, CONSTANTS) < 30


class TestNormaliseViews:
    def test_zero_views_returns_zero(self) -> None:
        assert _normalise_views(0, CONSTANTS) == 0

    def test_global_ceiling_returns_75(self) -> None:
        # The global boundary maps to exactly 75 — above it extends toward 100
        assert _normalise_views(CONSTANTS.PLATFORM_VIEWS_GLOBAL, CONSTANTS) == 75

    def test_above_ceiling_clamped_to_100(self) -> None:
        assert _normalise_views(CONSTANTS.PLATFORM_VIEWS_GLOBAL * 2, CONSTANTS) == 100

    def test_1m_views_returns_low_score(self) -> None:
        score = _normalise_views(1_000_000, CONSTANTS)
        assert 0 < score < 50


# ---------------------------------------------------------------------------
# _classify_popularity (pure helper — no mocks needed)
# ---------------------------------------------------------------------------

class TestClassifyPopularity:
    """Tests for _classify_popularity — uses CONSTANTS directly so tier boundaries
    come from config, not magic numbers in tests."""

    def test_emerging_tier_lastfm_only(self) -> None:
        result = _classify_popularity(5_000, 50_000, constants=CONSTANTS)
        assert result.tier == "Emerging"
        assert result.listeners == 5_000
        assert result.playcount == 50_000
        assert result.sync_cost_low  == CONSTANTS.SYNC_COST_EMERGING[0]
        assert result.sync_cost_high == CONSTANTS.SYNC_COST_EMERGING[1]

    def test_zero_listeners(self) -> None:
        result = _classify_popularity(0, 0, constants=CONSTANTS)
        assert result.tier == "Emerging"
        assert result.popularity_score == 0

    def test_spotify_score_elevates_tier(self) -> None:
        # Spotify score of 80 alone scores high, but the signal gate requires
        # POPULARITY_MIN_SIGNALS_GLOBAL signals for Global — so with only 1
        # signal the tier is capped below Global.
        result = _classify_popularity(10, 0, spotify_score=80, constants=CONSTANTS)
        assert result.tier != "Emerging"          # score is elevated
        assert result.tier != "Global"            # gate fires — not enough signals
        assert result.spotify_score == 80

    def test_high_view_count_elevates_tier(self) -> None:
        # 300M YouTube views is a strong signal, but with only 1 signal the
        # Mainstream gate fires (requires POPULARITY_MIN_SIGNALS_MAINSTREAM).
        # Score is elevated well above Emerging though.
        result = _classify_popularity(
            1, 0,
            platform_metrics={"view_count": 300_000_000},
            constants=CONSTANTS,
        )
        assert result.tier not in ("Emerging",)   # score is elevated
        assert result.popularity_score > 25        # clearly above Emerging range

    def test_bad_lastfm_overridden_by_youtube_views(self) -> None:
        # Simulates the "24K Magic shows as Emerging" bug:
        # 13 Last.fm listeners + 500M YouTube views → should NOT be Emerging
        result = _classify_popularity(
            13, 0,
            platform_metrics={"view_count": 500_000_000},
            constants=CONSTANTS,
        )
        assert result.tier != "Emerging"

    def test_max_of_signals_taken(self) -> None:
        # Spotify score 60 > last.fm score → popularity_score should equal 60
        result = _classify_popularity(
            100, 0, spotify_score=60, constants=CONSTANTS
        )
        assert result.popularity_score == 60

    def test_platform_metrics_stored(self) -> None:
        metrics = {"view_count": 1_000_000, "like_count": 50_000}
        result = _classify_popularity(0, 0, platform_metrics=metrics, constants=CONSTANTS)
        assert result.platform_metrics == metrics

    def test_no_signals_returns_emerging(self) -> None:
        result = _classify_popularity(0, 0, constants=CONSTANTS)
        assert result.tier == "Emerging"
        assert result.popularity_score == 0


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

        with patch("services.discovery._orchestrator.subprocess.run", return_value=mock_result), \
             patch("services.discovery._orchestrator._find_binary", return_value="/usr/bin/yt-dlp"):
            url = _resolve_youtube_url("Rick Astley", "Never Gonna Give You Up")

        assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_returns_none_on_empty_output(self):
        mock_result = MagicMock()
        mock_result.stdout = ""

        with patch("services.discovery._orchestrator.subprocess.run", return_value=mock_result), \
             patch("services.discovery._orchestrator._find_binary", return_value="/usr/bin/yt-dlp"):
            assert _resolve_youtube_url("Unknown", "Track") is None

    def test_returns_none_on_non_https_output(self):
        mock_result = MagicMock()
        mock_result.stdout = "not-a-url"

        with patch("services.discovery._orchestrator.subprocess.run", return_value=mock_result), \
             patch("services.discovery._orchestrator._find_binary", return_value="/usr/bin/yt-dlp"):
            assert _resolve_youtube_url("Artist", "Track") is None

    def test_returns_none_on_subprocess_exception(self):
        with patch("services.discovery._orchestrator.subprocess.run", side_effect=FileNotFoundError), \
             patch("services.discovery._orchestrator._find_binary", return_value="/usr/bin/yt-dlp"):
            assert _resolve_youtube_url("Artist", "Track") is None

    def test_returns_none_when_binary_not_found(self):
        with patch("services.discovery._orchestrator._find_binary", return_value=None):
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
             patch("services.discovery._orchestrator.get_settings") as mock_settings, \
             patch("services.discovery._orchestrator.requests.get", return_value=mock_resp), \
             patch("services.discovery._orchestrator.subprocess.run", side_effect=self._mock_yt_run), \
             patch("services.discovery._orchestrator._find_binary", return_value="/usr/bin/yt-dlp"):

            mock_settings.return_value.lastfm_api_key = "fakekey"
            svc     = Discovery()
            results = svc.find_similar("Fake Plastic Trees", "Radiohead")

        assert len(results) == 2
        assert results[0].title  == "Creep"
        assert results[0].artist == "Radiohead"
        assert results[0].youtube_url == "https://www.youtube.com/watch?v=fake123"

    def test_raises_configuration_error_without_api_key(self):
        with patch("services.discovery._orchestrator.get_settings") as mock_settings:
            mock_settings.return_value.lastfm_api_key = ""
            svc = Discovery()
            with pytest.raises(ConfigurationError):
                svc.find_similar("Any Title", "Any Artist")

    def test_returns_empty_for_empty_inputs(self):
        with patch("services.discovery._orchestrator.get_settings") as mock_settings:
            mock_settings.return_value.lastfm_api_key = "fakekey"
            svc = Discovery()
            assert svc.find_similar("", "") == []

    def test_youtube_url_is_none_when_resolve_fails(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._LASTFM_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_yt = MagicMock()
        mock_yt.stdout = ""

        with patch("services.discovery._orchestrator.get_settings") as mock_settings, \
             patch("services.discovery._orchestrator.requests.get", return_value=mock_resp), \
             patch("services.discovery._orchestrator.subprocess.run", return_value=mock_yt), \
             patch("services.discovery._orchestrator._find_binary", return_value="/usr/bin/yt-dlp"):

            mock_settings.return_value.lastfm_api_key = "fakekey"
            svc     = Discovery()
            results = svc.find_similar("Title", "Artist")

        assert all(r.youtube_url is None for r in results)


# ---------------------------------------------------------------------------
# _classify_popularity — signal gate (#88)
# ---------------------------------------------------------------------------

class TestSignalGate:
    """Tests for the minimum-signal gate that guards Mainstream/Global tiers."""

    def test_single_high_view_count_capped_at_regional_not_mainstream(self) -> None:
        # 50M views → score 50 (Mainstream boundary), but only 1 signal — gate fires
        result = _classify_popularity(
            0, 0,
            platform_metrics={"view_count": CONSTANTS.PLATFORM_VIEWS_MAINSTREAM},
            constants=CONSTANTS,
        )
        # With only 1 signal and POPULARITY_MIN_SIGNALS_MAINSTREAM == 2, tier is capped
        assert result.tier in ("Regional", "Emerging")

    def test_two_signals_unlocks_mainstream(self) -> None:
        # Last.fm + views together satisfy the 2-signal gate
        result = _classify_popularity(
            CONSTANTS.LASTFM_LISTENERS_MAINSTREAM,  # score 50
            0,
            platform_metrics={"view_count": CONSTANTS.PLATFORM_VIEWS_MAINSTREAM},  # score 50
            constants=CONSTANTS,
        )
        assert result.tier == "Mainstream"

    def test_single_spotify_score_capped_below_global(self) -> None:
        # Spotify 90 alone → score 90, but 1 signal → Global gate fires
        result = _classify_popularity(0, 0, spotify_score=90, constants=CONSTANTS)
        assert result.tier != "Global"

    def test_two_signals_unlocks_global(self) -> None:
        # Spotify 80 + Last.fm at global boundary → both score ≥ 75, 2 signals
        result = _classify_popularity(
            CONSTANTS.LASTFM_LISTENERS_GLOBAL,
            0,
            spotify_score=80,
            constants=CONSTANTS,
        )
        assert result.tier == "Global"

    def test_signal_count_stored_in_popularity_score(self) -> None:
        # When two signals are present the blended score is max of both
        result = _classify_popularity(
            100, 0, spotify_score=60, constants=CONSTANTS
        )
        assert result.popularity_score == 60


# ---------------------------------------------------------------------------
# _clean_title_for_lastfm — fuzzy Last.fm fallback (#89)
# ---------------------------------------------------------------------------

class TestCleanTitleForLastfm:
    def test_strips_feat_dot(self) -> None:
        assert _clean_title_for_lastfm("24K Magic (feat. Bruno Mars)") == "24K Magic"

    def test_strips_ft_dot(self) -> None:
        assert _clean_title_for_lastfm("Song (ft. Artist)") == "Song"

    def test_strips_featuring(self) -> None:
        assert _clean_title_for_lastfm("Track featuring Someone") == "Track"

    def test_strips_bracket_suffix(self) -> None:
        assert _clean_title_for_lastfm("Mr. Brightside [Official Video]") == "Mr. Brightside"

    def test_strips_bare_parenthetical(self) -> None:
        assert _clean_title_for_lastfm("Blue (Extended Version)") == "Blue"

    def test_clean_title_unchanged(self) -> None:
        assert _clean_title_for_lastfm("Blinding Lights") == "Blinding Lights"

    def test_empty_string_returns_empty(self) -> None:
        assert _clean_title_for_lastfm("") == ""

    def test_feat_inline_without_parens(self) -> None:
        result = _clean_title_for_lastfm("Sunflower feat. Swae Lee")
        assert "feat" not in result.lower()
        assert "Sunflower" in result


# ---------------------------------------------------------------------------
# get_track_popularity — fuzzy Last.fm retry integration test (#89)
# ---------------------------------------------------------------------------

class TestFuzzyLastfmRetry:
    """
    Verifies that get_track_popularity() retries with a cleaned title when
    the first Last.fm lookup returns suspiciously few listeners.
    """

    def _make_settings(self):
        s = MagicMock()
        s.lastfm_api_key = "fakekey"
        s.spotify_client_id = ""
        s.spotify_client_secret = ""
        return s

    def test_retry_fires_and_upgrades_listeners(self) -> None:
        """First call returns 13 listeners; retry returns 1_500_000."""
        low_resp = MagicMock()
        low_resp.raise_for_status.return_value = None
        low_resp.json.return_value = {
            "track": {"listeners": "13", "playcount": "50"}
        }

        high_resp = MagicMock()
        high_resp.raise_for_status.return_value = None
        high_resp.json.return_value = {
            "track": {"listeners": "1500000", "playcount": "10000000"}
        }

        with patch("services.discovery._orchestrator.get_settings", return_value=self._make_settings()), \
             patch("services.discovery._orchestrator.requests.get", side_effect=[low_resp, high_resp]):
            svc = Discovery()
            result = svc.get_track_popularity(
                "24K Magic (feat. Bruno Mars)", "Bruno Mars"
            )

        assert result is not None
        assert result.listeners == 1_500_000

    def test_retry_does_not_downgrade(self) -> None:
        """If the retry returns fewer listeners, keep the original."""
        first_resp = MagicMock()
        first_resp.raise_for_status.return_value = None
        first_resp.json.return_value = {
            "track": {"listeners": "500", "playcount": "2000"}
        }

        lower_resp = MagicMock()
        lower_resp.raise_for_status.return_value = None
        lower_resp.json.return_value = {
            "track": {"listeners": "10", "playcount": "100"}
        }

        with patch("services.discovery._orchestrator.get_settings", return_value=self._make_settings()), \
             patch("services.discovery._orchestrator.requests.get", side_effect=[first_resp, lower_resp]):
            svc = Discovery()
            result = svc.get_track_popularity("Song (feat. X)", "Artist")

        assert result is not None
        assert result.listeners == 500

    def test_no_retry_when_clean_title_unchanged(self) -> None:
        """If the title has no decorations, the retry should not fire (only 1 HTTP call)."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "track": {"listeners": "200", "playcount": "1000"}
        }

        with patch("services.discovery._orchestrator.get_settings", return_value=self._make_settings()), \
             patch("services.discovery._orchestrator.requests.get", return_value=resp) as mock_get:
            svc = Discovery()
            svc.get_track_popularity("Blinding Lights", "The Weeknd")

        # Only 1 call — no retry needed because title has no decorations
        assert mock_get.call_count == 1
