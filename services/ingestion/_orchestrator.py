"""
services/ingestion/_orchestrator.py
Audio ingestion — implements the AudioProvider protocol.

All audio is loaded fully into memory as an AudioBuffer (raw WAV bytes).
No temporary files are written to disk.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import PurePosixPath
from typing import Union
from urllib.parse import urlparse as _urlparse

from core.config import CONSTANTS, Settings, get_settings
from core.exceptions import AudioSourceError, ValidationError
from core.models import AudioBuffer
from core.protocols import YtDlpProvider
from utils.security import validate_url

from ._metadata import _fetch_youtube_metadata
from ._pure import (
    _check_size,
    _classify_url,
    _label_from_url,
    _wav_info,
)
from ._ytdlp import YtDlpClient


class Ingestion:
    """
    Loads audio from a YouTube URL or a Streamlit UploadedFile into memory.

    Implements: AudioProvider protocol (core/protocols.py)

    Constructor injection: pass a Settings instance to override defaults,
    e.g. in tests or when a paid tier raises the upload size ceiling.
    Pass a YtDlpProvider to swap the download backend — e.g. for a paid
    service or in integration tests that stub the subprocess calls.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        ytdlp_client: YtDlpProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ytdlp    = ytdlp_client or YtDlpClient()

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

        track_metadata = _fetch_youtube_metadata(url)
        track_metadata.update(self._ytdlp.fetch_engagement(url))

        wav_bytes = self._ytdlp.download_audio(url, CONSTANTS.SAMPLE_RATE)

        _check_size(len(wav_bytes), self._settings.max_upload_mb)
        track_metadata.update(_wav_info(wav_bytes))

        return AudioBuffer(
            raw=wav_bytes,
            sample_rate=CONSTANTS.SAMPLE_RATE,
            label=_label_from_url(url),
            metadata=track_metadata,
            source=source_label,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Private: direct HTTP audio download
    # ------------------------------------------------------------------

    def _load_direct(self, url: str) -> AudioBuffer:
        """Download a direct audio URL into an AudioBuffer."""
        # Direct audio URLs are not host-allowlisted (any CDN is valid) but must
        # use HTTPS to prevent cleartext transport and HTTP-based SSRF.
        if _urlparse(url).scheme != "https":
            raise ValidationError(
                "Direct audio URLs must use HTTPS.",
                context={"url": url},
            )
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
