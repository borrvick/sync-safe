"""
services/ingestion/_orchestrator.py
Audio ingestion — implements the AudioProvider protocol.

All audio is loaded fully into memory as an AudioBuffer (raw WAV bytes).
No temporary files are written to disk.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Union
from urllib.parse import urlparse as _urlparse

from core.config import CONSTANTS, Settings, get_settings
from core.exceptions import AudioSourceError, ConfigurationError, ValidationError
from core.models import AudioBuffer
from utils.security import validate_url

from ._metadata import _fetch_platform_engagement, _fetch_youtube_metadata
from ._pure import (
    _check_size,
    _classify_url,
    _label_from_url,
    _wav_info,
)


def _find_binary(name: str) -> str:
    """
    Resolve a system binary by name.

    Search order:
      1. The active venv's bin/ directory (sys.executable parent)
      2. PATH via shutil.which.
      3. The pip --user bin directory (macOS / Linux fallback).

    Raises:
        ConfigurationError: if the binary cannot be found anywhere.
    """
    import site

    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)

    path_bin = shutil.which(name)
    if path_bin:
        return path_bin

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


class Ingestion:
    """
    Loads audio from a YouTube URL or a Streamlit UploadedFile into memory.

    Implements: AudioProvider protocol (core/protocols.py)

    Constructor injection: pass a Settings instance to override defaults,
    e.g. in tests or when a paid tier raises the upload size ceiling.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public interface (AudioProvider protocol)
    # ------------------------------------------------------------------

    def load(
        self,
        source: Union[str, object],
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
                    "Unsupported URL — paste a YouTube, TikTok, Instagram, Facebook, "
                    "Bandcamp, or SoundCloud link, a direct audio file URL "
                    "(.mp3/.wav/.flac/.ogg/.m4a/.aac), or any other URL supported by yt-dlp.",
                    context={"url": source},
                )
            if kind == "direct":
                return self._load_direct(source)
            return self._load_ytdlp(source, source_label=kind)
        return self._load_upload(source)

    # ------------------------------------------------------------------
    # Private: yt-dlp path
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

        track_metadata = _fetch_youtube_metadata(url, ytdlp_bin)
        track_metadata.update(_fetch_platform_engagement(url, ytdlp_bin))

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
        """Wrap a Streamlit UploadedFile in an AudioBuffer."""
        try:
            raw: bytes = file.read()        # type: ignore[union-attr]
            name: str  = getattr(file, "name", "upload")
        except (AttributeError, OSError, ValueError) as exc:
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
