"""
services/ingestion/_ytdlp.py
YtDlpClient — isolates all yt-dlp and ffmpeg subprocess calls.

Implements: YtDlpProvider protocol (core/protocols.py)

All three subprocess operations (download, engagement, search) are owned
here so that swapping yt-dlp for a paid service requires changes only in
this file, not in Ingestion or Discovery.

Binary resolution runs at construction time so the cost is paid once
rather than on every download call.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from core.exceptions import AudioSourceError, ConfigurationError


# Engagement fields requested from yt-dlp --print for popularity signals.
_ENGAGEMENT_FIELDS: list[str] = [
    "view_count",
    "like_count",
    "share_count",
    "repost_count",
    "channel_follower_count",
]


def _find_binary(name: str) -> Optional[str]:
    """
    Locate a CLI binary — venv bin, PATH, then user site-packages bin.
    Returns None rather than raising so callers can decide how to handle absence.
    """
    import shutil
    import site

    venv_bin = Path(sys.executable).parent / name
    if venv_bin.exists():
        return str(venv_bin)

    path_bin = shutil.which(name)
    if path_bin:
        return path_bin

    try:
        user_bin = Path(site.getuserbase()) / "bin" / name
        if user_bin.exists():
            return str(user_bin)
    except Exception:
        pass

    return None


class YtDlpClient:
    """
    Wraps yt-dlp and ffmpeg subprocess calls for audio download, engagement
    metrics, and YouTube URL search.

    Implements: YtDlpProvider protocol (core/protocols.py)

    Constructor injection: pass ytdlp_bin / ffmpeg_bin paths explicitly in
    tests or paid-service configurations.  When omitted, binaries are
    auto-discovered from the active venv and PATH.

    Usage:
        client    = YtDlpClient()
        wav_bytes = client.download_audio("https://youtu.be/...", 22050)
        metrics   = client.fetch_engagement("https://youtu.be/...")
        yt_url    = client.search_url("The Weeknd", "Blinding Lights")
    """

    def __init__(
        self,
        ytdlp_bin: str | None = None,
        ffmpeg_bin: str | None = None,
    ) -> None:
        self._ytdlp  = ytdlp_bin  or self._require_binary("yt-dlp")
        self._ffmpeg = ffmpeg_bin or self._require_binary("ffmpeg")

    # ------------------------------------------------------------------
    # Public interface (YtDlpProvider protocol)
    # ------------------------------------------------------------------

    def download_audio(self, url: str, sample_rate: int) -> bytes:
        """
        Download audio from a URL and transcode to WAV bytes.

        Args:
            url:         Source URL (YouTube, SoundCloud, etc.)
            sample_rate: Target WAV sample rate in Hz.

        Returns:
            Raw WAV bytes (16-bit PCM mono).

        Raises:
            AudioSourceError: if yt-dlp or ffmpeg fail, or output is empty.
        """
        ytdlp_cmd = [
            self._ytdlp,
            "--quiet",
            "--no-warnings",
            "--format", "bestaudio/best",
            "--output", "-",
            "--no-playlist",
            "--extractor-args", "youtube:player_client=android",
            url,
        ]
        ffmpeg_cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "wav",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]

        ytdlp_proc: subprocess.Popen | None = None
        ffmpeg_proc: subprocess.Popen | None = None
        try:
            ytdlp_proc = subprocess.Popen(  # nosec B603 — cmd is a list (no shell=True); URL is validated upstream (HTTPS-only) before reaching this call
                ytdlp_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            ffmpeg_proc = subprocess.Popen(  # nosec B603 — ffmpeg_cmd is a hardcoded list of flags; no user input interpolated
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

            return wav_bytes

        finally:
            for proc in (ytdlp_proc, ffmpeg_proc):
                if proc and proc.poll() is None:
                    proc.kill()

    def fetch_engagement(self, url: str) -> dict[str, int]:
        """
        Fetch per-platform engagement metrics for a URL via yt-dlp --print.

        Never raises — engagement data is always supplementary.

        Returns:
            Dict of available engagement field names to integer values.
            Empty dict on any failure.
        """
        print_template = "\n".join(f"%({f})s" for f in _ENGAGEMENT_FIELDS)
        try:
            result = subprocess.run(  # nosec B603 — list form, no shell=True; url validated as HTTPS upstream before fetch_engagement is called
                [
                    self._ytdlp,
                    "--quiet",
                    "--no-warnings",
                    "--no-playlist",
                    "--print", print_template,
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return {}

            lines   = result.stdout.splitlines()
            metrics: dict[str, int] = {}
            for field, raw in zip(_ENGAGEMENT_FIELDS, lines):
                raw = raw.strip()
                if raw and raw not in ("NA", "None", "none"):
                    try:
                        metrics[field] = int(float(raw.replace(",", "")))
                    except ValueError:
                        pass
            return metrics
        except Exception:  # noqa: BLE001 — engagement is always best-effort
            return {}

    def search_url(self, artist: str, title: str) -> Optional[str]:
        """
        Use yt-dlp ytsearch1: to resolve a YouTube watch URL.

        No audio is downloaded — only metadata is fetched (--no-download).
        Returns None on any failure so a missing URL never blocks the caller.

        Args:
            artist: Artist name.
            title:  Track title.

        Returns:
            YouTube watch URL string, or None on failure.
        """
        query = f"ytsearch1:{artist} - {title}"
        cmd   = [
            self._ytdlp,
            "--quiet",
            "--no-warnings",
            "--no-download",
            "--print", "webpage_url",
            query,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)  # nosec B603 — list form, no shell=True; query string passed as single argv element, not interpolated into a shell
            url    = result.stdout.strip()
            if url.startswith("https://"):
                return url
        except Exception:  # noqa: BLE001
            pass
        return None

    # ------------------------------------------------------------------
    # Private: binary resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _require_binary(name: str) -> str:
        """
        Find a required binary or raise ConfigurationError.

        Raises:
            ConfigurationError: if the binary cannot be found anywhere.
        """
        found = _find_binary(name)
        if found:
            return found
        raise ConfigurationError(
            f"Required binary '{name}' not found.",
            context={
                "binary": name,
                "suggestion": f"pip install {name}",
            },
        )
