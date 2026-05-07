"""
services/ingestion/_metadata.py
YouTube oEmbed metadata fetching (HTTP only — no yt-dlp dependency).

Engagement metrics have moved to YtDlpClient.fetch_engagement()
(services/ingestion/_ytdlp.py) where all subprocess calls are co-located.
"""
from __future__ import annotations

import json
import urllib.request
from urllib.parse import quote

from ._pure import _artist_from_uploader, _clean_title, _split_artist_title


def _fetch_youtube_metadata(url: str) -> dict[str, str]:
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
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310 — base URL is hardcoded (youtube.com/oembed); only title/artist are interpolated as query params
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
