"""
tests/test_ingestion.py
Unit tests for services/ingestion.py — subprocess.Popen is mocked throughout.
"""
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import AudioSourceError, ConfigurationError, ValidationError
from core.models import AudioBuffer
from services.ingestion import Ingestion, _check_size, _label_from_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_upload(content: bytes, name: str = "track.mp3") -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = content
    mock.name = name
    return mock


def _make_popen_mock(
    ytdlp_returncode: int = 0,
    ffmpeg_returncode: int = 0,
    wav_data: bytes = b"RIFF....",
) -> tuple[MagicMock, MagicMock]:
    """Return a (ytdlp_mock, ffmpeg_mock) pair for subprocess.Popen patching."""
    ytdlp_mock = MagicMock()
    ytdlp_mock.stdout = MagicMock()
    ytdlp_mock.returncode = ytdlp_returncode
    ytdlp_mock.wait.return_value = ytdlp_returncode
    ytdlp_mock.communicate.return_value = (b"", b"yt-dlp error")
    ytdlp_mock.poll.return_value = ytdlp_returncode

    ffmpeg_mock = MagicMock()
    ffmpeg_mock.returncode = ffmpeg_returncode
    ffmpeg_mock.communicate.return_value = (wav_data, b"ffmpeg error")
    ffmpeg_mock.poll.return_value = ffmpeg_returncode

    return ytdlp_mock, ffmpeg_mock


# ---------------------------------------------------------------------------
# _label_from_url — pure function
# ---------------------------------------------------------------------------

class TestLabelFromUrl:
    def test_extracts_video_id_from_watch_url(self):
        assert _label_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube:dQw4w9WgXcQ"

    def test_extracts_video_id_from_short_url(self):
        assert _label_from_url("https://youtu.be/dQw4w9WgXcQ") == "youtube:dQw4w9WgXcQ"

    def test_falls_back_on_malformed_url(self):
        assert _label_from_url("not-a-url") == "not-a-url"

    def test_truncates_long_fallback_to_60_chars(self):
        assert len(_label_from_url("x" * 100)) == 60


# ---------------------------------------------------------------------------
# _check_size — pure function
# ---------------------------------------------------------------------------

class TestCheckSize:
    def test_passes_within_limit(self):
        _check_size(10 * 1024 * 1024, max_mb=50)   # 10 MB < 50 MB — no raise

    def test_raises_at_exact_limit(self):
        with pytest.raises(AudioSourceError) as exc_info:
            _check_size(51 * 1024 * 1024, max_mb=50)
        assert exc_info.value.context["limit_mb"] == 50
        assert exc_info.value.context["size_mb"] == 51.0

    def test_context_contains_actual_size(self):
        with pytest.raises(AudioSourceError) as exc_info:
            _check_size(75 * 1024 * 1024, max_mb=50)
        assert exc_info.value.context["size_mb"] == 75.0


# ---------------------------------------------------------------------------
# Ingestion.load — file upload path
# ---------------------------------------------------------------------------

class TestLoadUpload:
    def setup_method(self):
        self.svc = Ingestion()

    def test_returns_audio_buffer_with_correct_fields(self):
        upload = _make_upload(b"fake audio bytes", "song.mp3")
        buf = self.svc.load(upload)
        assert isinstance(buf, AudioBuffer)
        assert buf.raw == b"fake audio bytes"
        assert buf.label == "song.mp3"
        assert buf.sample_rate == 22_050

    def test_to_bytesio_returns_fresh_cursor(self):
        upload = _make_upload(b"data", "track.wav")
        buf = self.svc.load(upload)
        b1 = buf.to_bytesio()
        b2 = buf.to_bytesio()
        assert b1.read() == b"data"
        assert b2.read() == b"data"   # independent cursors

    def test_preserves_filename_with_spaces(self):
        upload = _make_upload(b"data", "my track (final).flac")
        assert self.svc.load(upload).label == "my track (final).flac"

    def test_raises_audio_source_error_on_empty_file(self):
        with pytest.raises(AudioSourceError, match="empty"):
            self.svc.load(_make_upload(b""))

    def test_raises_audio_source_error_when_exceeds_limit(self):
        large = b"x" * (51 * 1024 * 1024)
        with pytest.raises(AudioSourceError, match="size limit"):
            self.svc.load(_make_upload(large))


# ---------------------------------------------------------------------------
# Ingestion.load — YouTube path
# ---------------------------------------------------------------------------

class TestFetchYoutubeAudio:
    VALID_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def setup_method(self):
        self.svc = Ingestion()

    def test_returns_audio_buffer_on_success(self):
        wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
        ytdlp_mock, ffmpeg_mock = _make_popen_mock(wav_data=wav)

        with patch("services.ingestion.subprocess.Popen",
                   side_effect=[ytdlp_mock, ffmpeg_mock]):
            buf = self.svc.load(self.VALID_URL)

        assert isinstance(buf, AudioBuffer)
        assert buf.raw == wav
        assert "dQw4w9WgXcQ" in buf.label

    def test_raises_audio_source_error_on_ffmpeg_failure(self):
        ytdlp_mock, ffmpeg_mock = _make_popen_mock(ffmpeg_returncode=1)

        with patch("services.ingestion.subprocess.Popen",
                   side_effect=[ytdlp_mock, ffmpeg_mock]):
            with pytest.raises(AudioSourceError, match="ffmpeg"):
                self.svc.load(self.VALID_URL)

    def test_raises_audio_source_error_on_ytdlp_failure(self):
        ytdlp_mock, ffmpeg_mock = _make_popen_mock(ytdlp_returncode=1)

        with patch("services.ingestion.subprocess.Popen",
                   side_effect=[ytdlp_mock, ffmpeg_mock]):
            with pytest.raises(AudioSourceError, match="yt-dlp"):
                self.svc.load(self.VALID_URL)

    def test_raises_audio_source_error_on_empty_output(self):
        ytdlp_mock, ffmpeg_mock = _make_popen_mock(wav_data=b"")

        with patch("services.ingestion.subprocess.Popen",
                   side_effect=[ytdlp_mock, ffmpeg_mock]):
            with pytest.raises(AudioSourceError, match="empty"):
                self.svc.load(self.VALID_URL)

    def test_raises_validation_error_for_unsupported_url(self):
        # Spotify is not in the supported source list; "unknown" classification → ValidationError
        with pytest.raises(ValidationError) as exc_info:
            self.svc.load("https://open.spotify.com/track/abc123")
        assert "url" in exc_info.value.context

    def test_raises_validation_error_for_http_url(self):
        with pytest.raises(ValidationError):
            self.svc.load("http://www.youtube.com/watch?v=abc")

    def test_processes_are_killed_on_exception(self):
        ytdlp_mock, ffmpeg_mock = _make_popen_mock(ffmpeg_returncode=1)
        ytdlp_mock.poll.return_value = None
        ffmpeg_mock.poll.return_value = None

        with patch("services.ingestion.subprocess.Popen",
                   side_effect=[ytdlp_mock, ffmpeg_mock]):
            with pytest.raises(AudioSourceError):
                self.svc.load(self.VALID_URL)

        ytdlp_mock.kill.assert_called_once()
        ffmpeg_mock.kill.assert_called_once()

    def test_satisfies_audio_provider_protocol(self):
        from core.protocols import AudioProvider
        assert isinstance(self.svc, AudioProvider)


# ---------------------------------------------------------------------------
# _classify_url
# ---------------------------------------------------------------------------

class TestClassifyUrl:
    """Tests for the pure URL classification function."""

    def test_youtube_dot_com(self):
        from services.ingestion import _classify_url
        assert _classify_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "youtube"

    def test_youtu_be_short(self):
        from services.ingestion import _classify_url
        assert _classify_url("https://youtu.be/dQw4w9WgXcQ") == "youtube"

    def test_bandcamp_subdomain(self):
        from services.ingestion import _classify_url
        assert _classify_url("https://artist.bandcamp.com/track/song-name") == "bandcamp"

    def test_soundcloud(self):
        from services.ingestion import _classify_url
        assert _classify_url("https://soundcloud.com/artist/track") == "soundcloud"

    def test_direct_mp3(self):
        from services.ingestion import _classify_url
        assert _classify_url("https://example.com/audio/track.mp3") == "direct"

    def test_direct_wav(self):
        from services.ingestion import _classify_url
        assert _classify_url("https://cdn.example.com/files/track.wav") == "direct"

    def test_unknown_url(self):
        from services.ingestion import _classify_url
        assert _classify_url("https://spotify.com/track/abc") == "unknown"

    def test_malformed_url_returns_unknown(self):
        from services.ingestion import _classify_url
        assert _classify_url("not-a-url") == "unknown"


# ---------------------------------------------------------------------------
# _clean_title / _split_artist_title
# ---------------------------------------------------------------------------

class TestCleanTitle:
    def _clean(self, title: str) -> str:
        from services.ingestion import _clean_title
        return _clean_title(title)

    def test_lyrics_suffix_stripped(self):
        assert self._clean("24K Magic (Lyrics)") == "24K Magic"

    def test_lyric_video_stripped(self):
        assert self._clean("Blinding Lights (Lyric Video)") == "Blinding Lights"

    def test_official_video_stripped(self):
        assert self._clean("Shape of You (Official Video)") == "Shape of You"

    def test_official_music_video_stripped(self):
        assert self._clean("Stay (Official Music Video)") == "Stay"

    def test_official_audio_stripped(self):
        assert self._clean("Levitating (Official Audio)") == "Levitating"

    def test_hq_stripped(self):
        assert self._clean("Bohemian Rhapsody (HQ)") == "Bohemian Rhapsody"

    def test_hd_stripped(self):
        assert self._clean("Purple Rain (HD)") == "Purple Rain"

    def test_clean_version_stripped(self):
        assert self._clean("WAP (Clean)") == "WAP"

    def test_explicit_stripped(self):
        assert self._clean("WAP (Explicit)") == "WAP"

    def test_extended_version_stripped(self):
        assert self._clean("Blue (Extended Version)") == "Blue"

    def test_slowed_reverb_stripped(self):
        assert self._clean("drivers license (Slowed + Reverb)") == "drivers license"

    def test_sped_up_stripped(self):
        assert self._clean("As It Was (Sped Up)") == "As It Was"

    def test_nightcore_stripped(self):
        assert self._clean("Fireflies (Nightcore)") == "Fireflies"

    def test_remastered_stripped(self):
        assert self._clean("Hotel California (Remastered)") == "Hotel California"

    def test_clean_title_unchanged(self):
        assert self._clean("24K Magic") == "24K Magic"

    def test_bracket_variant_stripped(self):
        assert self._clean("Mr. Brightside [Official Video]") == "Mr. Brightside"


class TestSplitArtistTitle:
    def _split(self, title: str) -> tuple:
        from services.ingestion import _split_artist_title
        return _split_artist_title(title)

    def test_standard_dash_split(self):
        artist, title = self._split("Bruno Mars - 24K Magic (Lyrics)")
        assert artist == "Bruno Mars"
        assert title == "24K Magic"

    def test_lyrics_stripped_before_split(self):
        _, title = self._split("The Weeknd - Blinding Lights (Lyrics)")
        assert title == "Blinding Lights"

    def test_no_dash_returns_empty_artist(self):
        artist, title = self._split("24K Magic (Lyrics)")
        assert artist == ""

    def test_official_video_stripped(self):
        _, title = self._split("Adele - Hello (Official Video)")
        assert title == "Hello"
