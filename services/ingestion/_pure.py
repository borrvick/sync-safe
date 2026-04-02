"""
services/ingestion/_pure.py
Pure ingestion helpers — no yt-dlp, no network, no I/O.
"""
from __future__ import annotations

import re
import struct
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlparse

from core.exceptions import AudioSourceError

# ---------------------------------------------------------------------------
# URL classification constants
# ---------------------------------------------------------------------------

_YOUTUBE_HOSTS: frozenset[str] = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
})
_BANDCAMP_HOSTS: frozenset[str] = frozenset({"bandcamp.com"})
_SOUNDCLOUD_HOSTS: frozenset[str] = frozenset({
    "soundcloud.com", "www.soundcloud.com", "on.soundcloud.com",
})
_TIKTOK_HOSTS: frozenset[str] = frozenset({
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "m.tiktok.com",
})
_INSTAGRAM_HOSTS: frozenset[str] = frozenset({
    "instagram.com", "www.instagram.com",
})
_FACEBOOK_HOSTS: frozenset[str] = frozenset({
    "facebook.com", "www.facebook.com", "fb.com", "fb.watch",
})
_DIRECT_AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
})


def _classify_url(url: str) -> str:
    """
    Classify a URL as one of:
      youtube / bandcamp / soundcloud / tiktok / instagram / facebook /
      direct / ytdlp / unknown

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return "unknown"
        host = parsed.netloc.lower().removeprefix("www.")
        if host in _YOUTUBE_HOSTS or host == "youtu.be":
            return "youtube"
        if host in _BANDCAMP_HOSTS or host.endswith(".bandcamp.com"):
            return "bandcamp"
        if host in _SOUNDCLOUD_HOSTS:
            return "soundcloud"
        if host in _TIKTOK_HOSTS:
            return "tiktok"
        if host in _INSTAGRAM_HOSTS:
            return "instagram"
        if host in _FACEBOOK_HOSTS:
            return "facebook"
        ext = PurePosixPath(parsed.path).suffix.lower()
        if ext in _DIRECT_AUDIO_EXTENSIONS:
            return "direct"
        return "ytdlp"
    except Exception:   # noqa: BLE001
        return "unknown"


def _label_from_url(url: str) -> str:
    """
    Derive a short human-readable label from a YouTube URL.

    Pure function — no network calls.
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


def _check_size(byte_count: int, max_mb: int) -> None:
    """
    Raise AudioSourceError if the byte count exceeds the configured limit.

    Pure function — no I/O.
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


def _artist_from_uploader(uploader: str) -> str:
    """Extract an artist name from YouTube's 'Artist - Topic' channel convention."""
    suffix = " - Topic"
    if uploader.endswith(suffix):
        return uploader[: -len(suffix)]
    return ""


def _split_artist_title(video_title: str) -> tuple[str, str]:
    """
    Parse artist and track title from YouTube's common 'Artist - Title' format.

    Returns (artist, clean_title). If the ' - ' separator is absent, returns
    ("", original_title).

    Pure function — no I/O.
    """
    clean = re.sub(
        r"\s*[\(\[]"
        r"(Official\s*(Music\s*|Lyric\s*|Audio\s*)?Video"
        r"|Official\s*Audio"
        r"|Lyric\s*Video|Lyrics?|Audio|Live( Version)?"
        r"|Visualizer|HQ|HD|Full\s*Song|Clean|Explicit"
        r"|Extended(\s*Version)?|Slowed(\s*\+?\s*Reverb)?"
        r"|Sped\s*Up|Nightcore|Remaster(ed)?)"
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

    Pure function — no I/O.
    """
    clean = re.sub(
        r"\s*[\(\[]"
        r"(Official\s*(Music\s*|Lyric\s*|Audio\s*)?Video"
        r"|Official\s*Audio"
        r"|Lyric\s*Video|Lyrics?|Audio|Live( Version)?"
        r"|Visualizer|HQ|HD|Full\s*Song|Clean|Explicit"
        r"|Extended(\s*Version)?|Slowed(\s*\+?\s*Reverb)?"
        r"|Sped\s*Up|Nightcore|Remaster(ed)?"
        r"|Radio\s*Edit|feat\.[^\)\]]*)"
        r"[^\)\]]*[\)\]]\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()
    return clean or title


def _wav_info(wav_bytes: bytes) -> dict[str, str]:
    """
    Extract technical audio metadata from raw WAV bytes.

    Pure function — no I/O beyond reading the provided bytes.
    """
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
