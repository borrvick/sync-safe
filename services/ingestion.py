"""
services/ingestion.py
Audio ingestion — implements the AudioProvider protocol.

All audio is loaded fully into memory as an AudioBuffer (raw WAV bytes).
No temporary files are written to disk.

Design notes:
- Ingestion.load() is the single public entry point. It dispatches
  on the source type so callers never need an if/else.
- Subprocess processes are always killed in a finally block — no zombies
  if the caller catches an exception and continues.
- Magic numbers (sample rate, codec) come from CONSTANTS, not inline literals.
- _find_binary and _label_from_url are module-level pure functions so they
  can be unit-tested independently.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path
from typing import Union
from urllib.parse import parse_qs, quote, urlparse

from core.config import CONSTANTS, Settings, get_settings
from core.exceptions import AudioSourceError, ConfigurationError, ValidationError
from core.models import AudioBuffer
from utils.security import validate_url

# ---------------------------------------------------------------------------
# URL classification constants (module-level for testability)
# ---------------------------------------------------------------------------

_YOUTUBE_HOSTS: frozenset[str] = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
})
_BANDCAMP_HOSTS: frozenset[str] = frozenset({"bandcamp.com"})
_SOUNDCLOUD_HOSTS: frozenset[str] = frozenset({
    "soundcloud.com", "www.soundcloud.com", "on.soundcloud.com",
})
_DIRECT_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
})


class Ingestion:
    """
    Loads audio from a YouTube URL or a Streamlit UploadedFile into memory.

    Implements: AudioProvider protocol (core/protocols.py)

    Constructor injection: pass a Settings instance to override defaults,
    e.g. in tests or when a paid tier raises the upload size ceiling.

    Usage:
        service = Ingestion()
        buffer  = service.load("https://www.youtube.com/watch?v=...")
        buffer  = service.load(uploaded_file)   # st.file_uploader result
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public interface (AudioProvider protocol)
    # ------------------------------------------------------------------

    def load(
        self,
        source: Union[str, object],   # str | st.runtime.UploadedFile
    ) -> AudioBuffer:
        """
        Load audio from any supported source.

        Args:
            source: A YouTube URL string or a Streamlit UploadedFile object.

        Returns:
            AudioBuffer with raw WAV bytes at 22 050 Hz mono 16-bit.

        Raises:
            ValidationError:     URL is malformed or not a permitted domain.
            AudioSourceError:    Download failed, file is empty/corrupt,
                                 or exceeds the configured size limit.
            ConfigurationError:  yt-dlp or ffmpeg binary not found.
        """
        if isinstance(source, str):
            kind = _classify_url(source)
            if kind == "unknown":
                raise ValidationError(
                    "Unsupported URL — only YouTube, Bandcamp, SoundCloud, "
                    "and direct audio links (.mp3/.wav/.flac/.ogg/.m4a/.aac) are supported.",
                    context={"url": source},
                )
            if kind == "direct":
                return self._load_direct(source)
            return self._load_ytdlp(source, source_label=kind)
        return self._load_upload(source)

    # ------------------------------------------------------------------
    # Private: yt-dlp path (YouTube, Bandcamp, SoundCloud)
    # ------------------------------------------------------------------

    def _load_ytdlp(self, url: str, source_label: str = "youtube") -> AudioBuffer:
        try:
            validate_url(url)
        except ValueError as exc:
            raise ValidationError(
                str(exc),
                context={"url": url},
            ) from exc

        ytdlp_bin  = _find_binary("yt-dlp")
        ffmpeg_bin = _find_binary("ffmpeg")

        # Fetch metadata before the download so YouTube doesn't rate-limit
        # two consecutive requests for the same video. --dump-json is a
        # lightweight info-only call (~2s); the download follows separately.
        track_metadata = _fetch_youtube_metadata(url, ytdlp_bin)

        ytdlp_cmd = [
            ytdlp_bin,
            "--quiet",
            "--no-warnings",
            "--format", "bestaudio/best",
            "--output", "-",
            "--no-playlist",
            "--extractor-args", "youtube:player_client=android",
            url,
        ]

        # Transcode to CONSTANTS.SAMPLE_RATE Hz mono 16-bit WAV on stdout.
        ffmpeg_cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "wav",
            "-ar", str(CONSTANTS.SAMPLE_RATE),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]

        ytdlp_proc: subprocess.Popen | None = None
        ffmpeg_proc: subprocess.Popen | None = None
        try:
            ytdlp_proc = subprocess.Popen(
                ytdlp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=ytdlp_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Allow ytdlp_proc to receive SIGPIPE if ffmpeg exits early.
            if ytdlp_proc.stdout:
                ytdlp_proc.stdout.close()

            wav_bytes, ffmpeg_err = ffmpeg_proc.communicate()
            ytdlp_proc.wait()

            if ytdlp_proc.returncode not in (0, None):
                _, ytdlp_err = ytdlp_proc.communicate()
                raise AudioSourceError(
                    "yt-dlp failed to download audio.",
                    context={
                        "url": url,
                        "exit_code": ytdlp_proc.returncode,
                        "stderr": ytdlp_err.decode(errors="replace").strip(),
                    },
                )
            if ffmpeg_proc.returncode != 0:
                raise AudioSourceError(
                    "ffmpeg failed to transcode audio.",
                    context={
                        "url": url,
                        "exit_code": ffmpeg_proc.returncode,
                        "stderr": ffmpeg_err.decode(errors="replace").strip(),
                    },
                )
            if not wav_bytes:
                raise AudioSourceError(
                    "Audio download produced empty output.",
                    context={"url": url},
                )

            _check_size(len(wav_bytes), self._settings.max_upload_mb)

            track_metadata.update(_wav_info(wav_bytes))

            return AudioBuffer(
                raw=wav_bytes,
                sample_rate=CONSTANTS.SAMPLE_RATE,
                label=_label_from_url(url),
                metadata=track_metadata,
                source=source_label,  # type: ignore[arg-type]
            )

        finally:
            for proc in (ytdlp_proc, ffmpeg_proc):
                if proc and proc.poll() is None:
                    proc.kill()

    # ------------------------------------------------------------------
    # Private: direct HTTP audio download
    # ------------------------------------------------------------------

    def _load_direct(self, url: str) -> AudioBuffer:
        """Download a direct audio URL into an AudioBuffer."""
        import urllib.error
        max_bytes = CONSTANTS.MAX_UPLOAD_BYTES
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sync-safe/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise AudioSourceError(
                            f"Direct download exceeds {max_bytes} byte limit.",
                            context={"url": url},
                        )
                    chunks.append(chunk)
                raw = b"".join(chunks)
        except AudioSourceError:
            raise
        except urllib.error.URLError as exc:
            raise AudioSourceError(
                "Direct audio download failed.",
                context={"url": url, "error": str(exc)},
            ) from exc

        if not raw:
            raise AudioSourceError("Direct download produced empty bytes.", context={"url": url})

        _check_size(len(raw), self._settings.max_upload_mb)

        from pathlib import PurePosixPath
        from urllib.parse import urlparse as _urlparse
        label = PurePosixPath(_urlparse(url).path).name or url
        return AudioBuffer(
            raw=raw,
            sample_rate=CONSTANTS.SAMPLE_RATE,
            label=label,
            source="direct",
        )

    # ------------------------------------------------------------------
    # Private: file-upload path
    # ------------------------------------------------------------------

    def _load_upload(self, file: object) -> AudioBuffer:
        """
        Wrap a Streamlit UploadedFile in an AudioBuffer.

        The file bytes are stored as-is (no resampling here). Downstream
        services call librosa.load(buffer.to_bytesio(), sr=...) as needed.
        """
        try:
            raw: bytes = file.read()        # type: ignore[union-attr]
            name: str  = getattr(file, "name", "upload")
        except Exception as exc:
            raise AudioSourceError(
                "Could not read uploaded file.",
                context={"original_error": str(exc)},
            ) from exc

        if not raw:
            raise AudioSourceError(
                "Uploaded file is empty.",
                context={"filename": getattr(file, "name", "unknown")},
            )

        _check_size(len(raw), self._settings.max_upload_mb)

        return AudioBuffer(
            raw=raw,
            sample_rate=CONSTANTS.SAMPLE_RATE,
            label=name,
            source="file",
        )


# ---------------------------------------------------------------------------
# Module-level pure functions (testable without instantiating the service)
# ---------------------------------------------------------------------------

def _classify_url(url: str) -> str:
    """
    Classify a URL as one of: youtube / bandcamp / soundcloud / direct / unknown.

    Pure function — no I/O, no side effects, deterministic.
    Returns "unknown" on any parse error rather than raising.
    """
    try:
        parsed = urlparse(url.strip())
        host   = parsed.netloc.lower().removeprefix("www.")
        if host in _YOUTUBE_HOSTS or host == "youtu.be":
            return "youtube"
        if host in _BANDCAMP_HOSTS or host.endswith(".bandcamp.com"):
            return "bandcamp"
        if host in _SOUNDCLOUD_HOSTS:
            return "soundcloud"
        from pathlib import PurePosixPath
        ext = PurePosixPath(parsed.path).suffix.lower()
        if ext in _DIRECT_AUDIO_EXTENSIONS:
            return "direct"
        return "unknown"
    except Exception:   # noqa: BLE001 — pure parse helper; never fatal
        return "unknown"


def _find_binary(name: str) -> str:
    """
    Resolve a system binary by name.

    Search order:
      1. The active venv's bin/ directory (sys.executable parent) — preferred
         so venv-installed binaries take precedence over stale system installs.
      2. PATH via shutil.which.
      3. The pip --user bin directory (macOS / Linux fallback).

    Raises:
        ConfigurationError: if the binary cannot be found anywhere.
    """
    import sys

    # 1. Venv bin — most reliable when running inside a virtualenv
    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)

    # 2. PATH
    path_bin = shutil.which(name)
    if path_bin:
        return path_bin

    # 3. pip --user bin
    import site
    user_bin = Path(site.getuserbase()) / "bin" / name
    if user_bin.exists():
        return str(user_bin)

    raise ConfigurationError(
        f"Required binary '{name}' not found.",
        context={
            "binary": name,
            "suggestion": f"pip install {name}",
            "searched": [str(venv_bin), path_bin, str(user_bin)],
        },
    )


def _label_from_url(url: str) -> str:
    """
    Derive a short human-readable label from a YouTube URL.

    Returns the video ID if parseable, otherwise the first 60 characters.
    This is a pure function — no network calls.
    """
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if "v" in params:
            return f"youtube:{params['v'][0]}"
        if parsed.netloc == "youtu.be" and parsed.path.lstrip("/"):
            return f"youtube:{parsed.path.lstrip('/')}"
    except Exception:
        pass
    return url[:60]


def _fetch_youtube_metadata(url: str, ytdlp_bin: str) -> dict[str, str]:
    """
    Fetch title and artist via YouTube's public oEmbed API.

    oEmbed is a lightweight JSON endpoint that requires no auth and no yt-dlp.
    It returns in ~200ms and is not subject to the bot-detection blocking that
    affects yt-dlp metadata calls made outside the main download subprocess.

    Response fields used:
      title       — e.g. "Sabrina Carpenter - Espresso"
      author_name — channel name; may have " - Topic" suffix for auto-channels

    Artist resolution:
      1. Strip " - Topic" from author_name if present
      2. Use author_name directly (official artist channels)
      3. Parse "Artist - Title" from the video title as a last resort

    Never raises — metadata is supplementary. Any failure returns empty strings.
    """
    try:
        oembed_url = (
            "https://www.youtube.com/oembed"
            f"?url={quote(url, safe='')}&format=json"
        )
        req = urllib.request.Request(
            oembed_url,
            headers={"User-Agent": "sync-safe-forensic-portal/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict = json.loads(resp.read().decode("utf-8"))

        title_raw   = str(data.get("title") or "")
        author_name = str(data.get("author_name") or "")

        # " - Topic" channels → strip suffix; otherwise use as-is
        artist = _artist_from_uploader(author_name) or author_name

        # If title follows "Artist - Title" convention, prefer the parsed
        # artist (more precise than the channel name).
        if title_raw and " - " in title_raw:
            parsed_artist, parsed_title = _split_artist_title(title_raw)
            if parsed_artist:
                return {"title": _clean_title(parsed_title), "artist": parsed_artist}

        return {"title": _clean_title(title_raw), "artist": artist}
    except Exception:  # noqa: BLE001 — metadata is always best-effort
        return {"title": "", "artist": ""}


def _artist_from_uploader(uploader: str) -> str:
    """
    Extract an artist name from YouTube's 'Artist - Topic' channel convention.

    Pure function — no I/O.
    """
    suffix = " - Topic"
    if uploader.endswith(suffix):
        return uploader[: -len(suffix)]
    return ""


def _split_artist_title(video_title: str) -> tuple[str, str]:
    """
    Parse artist and track title from YouTube's common 'Artist - Title' format.

    Also strips trailing qualifiers like '(Official Video)', '(Lyric Video)',
    '(Official Music Video)', '(Audio)' that appear in YouTube titles but
    would confuse a lyrics API search.

    Returns (artist, clean_title). If the ' - ' separator is absent, returns
    ("", original_title) so the caller can decide what to do.

    Pure function — no I/O.
    """
    import re
    # Strip parenthetical/bracketed suffixes common in music video titles
    clean = re.sub(
        r"\s*[\(\[]"
        r"(Official\s*(Music\s*|Lyric\s*|Audio\s*)?Video"
        r"|Lyric\s*Video|Audio|Live( Version)?|Visualizer)"
        r"[^\)\]]*[\)\]]\s*$",
        "",
        video_title,
        flags=re.IGNORECASE,
    ).strip()

    if " - " in clean:
        artist_part, title_part = clean.split(" - ", 1)
        return artist_part.strip(), title_part.strip()

    return "", video_title


def _clean_title(title: str) -> str:
    """
    Strip common YouTube parenthetical suffixes from a track title.

    Removes qualifiers like '(Official Audio)', '(Official Video)',
    '(Official Music Video)', '(Lyric Video)', '(Visualizer)', '(Live)',
    '(Radio Edit)', and bracketed equivalents.

    Pure function — no I/O.
    """
    clean = re.sub(
        r"\s*[\(\[]"
        r"(Official\s*(Music\s*|Lyric\s*|Audio\s*)?Video"
        r"|Official\s*Audio"
        r"|Lyric\s*Video|Audio|Live( Version)?|Visualizer"
        r"|Radio\s*Edit|Remaster(ed)?|feat\.[^\)\]]*)"
        r"[^\)\]]*[\)\]]\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    return clean or title


def _wav_info(wav_bytes: bytes) -> dict[str, str]:
    """
    Extract technical audio metadata from raw WAV bytes.

    Returns a dict with keys: duration, sample_rate, bit_depth, channels.
    All values are strings. Returns an empty dict on any parse error.

    ffmpeg piped output writes 0xFFFFFFFF as the RIFF data-chunk size
    placeholder because it cannot seek back to update the header on a
    stream. wave.getnframes() would read that placeholder (~27 hours).
    Instead, duration is derived from the actual byte length of wav_bytes
    using the header fields ffmpeg does write correctly.

    WAV header layout (standard 44-byte PCM header):
      offset 22 — num channels  (uint16 LE)
      offset 24 — sample rate   (uint32 LE)
      offset 34 — bits/sample   (uint16 LE)
      offset 44+ — audio data

    Pure function — no I/O beyond reading the provided bytes.
    """
    import struct
    try:
        if len(wav_bytes) < 44:
            return {}
        num_channels    = struct.unpack_from("<H", wav_bytes, 22)[0]
        sample_rate     = struct.unpack_from("<I", wav_bytes, 24)[0]
        bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
        if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
            return {}
        bytes_per_frame = (bits_per_sample // 8) * num_channels
        data_bytes      = len(wav_bytes) - 44
        total_seconds   = int(data_bytes / bytes_per_frame / sample_rate)
        minutes, seconds = divmod(total_seconds, 60)
        channel_label   = "Mono" if num_channels == 1 else ("Stereo" if num_channels == 2 else str(num_channels))
        return {
            "duration":    f"{minutes}:{seconds:02d}",
            "sample_rate": f"{sample_rate:,} Hz",
            "bit_depth":   f"{bits_per_sample}-bit",
            "channels":    channel_label,
        }
    except Exception:
        return {}


def _check_size(byte_count: int, max_mb: int) -> None:
    """
    Raise AudioSourceError if the byte count exceeds the configured limit.

    Pure function — extracted so it can be tested independently of the
    subprocess pipeline.
    """
    max_bytes = max_mb * 1024 * 1024
    if byte_count > max_bytes:
        raise AudioSourceError(
            f"Audio exceeds the {max_mb} MB size limit.",
            context={
                "size_mb": round(byte_count / 1024 / 1024, 1),
                "limit_mb": max_mb,
            },
        )
