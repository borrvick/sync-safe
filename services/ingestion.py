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
import shutil
import subprocess
from pathlib import Path
from typing import Union
from urllib.parse import parse_qs, urlparse

from core.config import CONSTANTS, Settings, get_settings
from core.exceptions import AudioSourceError, ConfigurationError, ValidationError
from core.models import AudioBuffer
from utils.security import validate_url


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
            return self._load_youtube(source)
        return self._load_upload(source)

    # ------------------------------------------------------------------
    # Private: YouTube path
    # ------------------------------------------------------------------

    def _load_youtube(self, url: str) -> AudioBuffer:
        try:
            validate_url(url)
        except ValueError as exc:
            raise ValidationError(
                str(exc),
                context={"url": url},
            ) from exc

        ytdlp_bin  = _find_binary("yt-dlp")
        ffmpeg_bin = _find_binary("ffmpeg")

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

            return AudioBuffer(
                raw=wav_bytes,
                sample_rate=CONSTANTS.SAMPLE_RATE,
                label=_label_from_url(url),
            )

        finally:
            for proc in (ytdlp_proc, ffmpeg_proc):
                if proc and proc.poll() is None:
                    proc.kill()

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
        )


# ---------------------------------------------------------------------------
# Module-level pure functions (testable without instantiating the service)
# ---------------------------------------------------------------------------

def _find_binary(name: str) -> str:
    """
    Resolve a system binary by name.

    Checks PATH first, then the pip --user bin directory (macOS / Linux).

    Raises:
        ConfigurationError: if the binary cannot be found anywhere.
    """
    found = shutil.which(name)
    if found:
        return found

    import site
    user_bin = Path(site.getuserbase()) / "bin" / name
    if user_bin.exists():
        return str(user_bin)

    raise ConfigurationError(
        f"Required binary '{name}' not found.",
        context={
            "binary": name,
            "suggestion": f"pip install {name}",
            "searched": [shutil.which(name), str(user_bin)],
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
