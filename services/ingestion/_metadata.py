"""
services/ingestion/_metadata.py
YouTube oEmbed metadata and platform engagement fetching via yt-dlp.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import urllib.request
from urllib.parse import quote

from ._pure import _artist_from_uploader, _clean_title, _split_artist_title

# Engagement fields requested from yt-dlp --print for popularity signals.
_ENGAGEMENT_FIELDS: list[str] = [
    "view_count",
    "like_count",
    "share_count",
    "repost_count",
    "channel_follower_count",
]


def _fetch_youtube_metadata(url: str, ytdlp_bin: str) -> dict[str, str]:
    """
    Fetch title and artist via YouTube's public oEmbed API.

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

        artist = _artist_from_uploader(author_name) or author_name

        if title_raw and " - " in title_raw:
            parsed_artist, parsed_title = _split_artist_title(title_raw)
            if parsed_artist:
                return {"title": _clean_title(parsed_title), "artist": parsed_artist}

        return {"title": _clean_title(title_raw), "artist": artist}
    except Exception:  # noqa: BLE001 — metadata is always best-effort
        return {"title": "", "artist": ""}


def _fetch_platform_engagement(url: str, ytdlp_bin: str) -> dict[str, int]:
    """
    Fetch per-platform engagement metrics for a URL via yt-dlp --print.

    Never raises — engagement data is always supplementary.
    """
    print_template = "\n".join(f"%({f})s" for f in _ENGAGEMENT_FIELDS)
    try:
        result = subprocess.run(
            [
                ytdlp_bin,
                "--quiet",
                "--no-warnings",
                "--no-playlist",
                "--print", print_template,
                shlex.quote(url),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {}

        lines = result.stdout.splitlines()
        metrics: dict[str, int] = {}
        for field, raw in zip(_ENGAGEMENT_FIELDS, lines):
            raw = raw.strip()
            if raw and raw not in ("NA", "None", "none"):
                try:
                    metrics[field] = int(float(raw.replace(",", "")))
                except ValueError:
                    pass
        return metrics
    except Exception:  # noqa: BLE001
        return {}
