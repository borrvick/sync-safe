"""
services/tag_injector.py
Write Sync-Safe audit results into audio file tags.

Embeds compliance metadata as custom tags so a file carries its audit trail
when shared with supervisors, music supervisors, or licensing portals.

Tag schema version: 1.0

Supported formats and tag frames:
- MP3 / ID3:  TXXX custom frames + COMM comment
- FLAC / OGG: Vorbis comment tags (same key names, lowercase)
- MP4 / M4A:  freeform atoms under "----:SYNC_SAFE:..."
- Unknown:    bytes returned unchanged; warning logged

Design:
- All I/O is in a tempfile.TemporaryDirectory with try/finally (no permanent
  disk writes beyond the temp directory lifetime).
- The injector never modifies existing title/artist tags — it only appends
  SYNC_SAFE_* custom fields.
- Returns the original bytes on any mutagen failure so the download
  still works even if tagging is unsupported.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.models import AnalysisResult

_log = logging.getLogger(__name__)

# Tag schema version — bump when the set of injected keys changes.
_TAG_SCHEMA_VERSION: str = "1.0"

# Verdicts that indicate a failing audit for the SYNC_SAFE_VERDICT tag.
_FAIL_GRADES:    frozenset[str] = frozenset({"D", "F"})
_AI_VERDICTS:    frozenset[str] = frozenset({"AI", "Likely AI"})


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _build_tag_values(result: AnalysisResult) -> dict[str, str]:
    """
    Derive the SYNC_SAFE_* tag dict from an AnalysisResult.

    Pure function — no I/O.

    Returns:
        Mapping of tag key (uppercase) → value string.
    """
    grade   = (result.compliance.grade or "") if result.compliance else ""
    verdict = result.forensics.verdict if result.forensics else ""
    ai_prob = str(round(result.forensics.ai_probability * 100.0, 1)) if result.forensics else ""

    flag_types: list[str] = sorted(
        {f.issue_type for f in result.compliance.flags}
    ) if result.compliance else []

    # Three-level sync verdict: PASS / CAUTION / FAIL
    if grade in _FAIL_GRADES or (result.compliance and result.compliance.hard_flags) or verdict in _AI_VERDICTS:
        sync_verdict = "FAIL"
    elif grade in ("A", "B") and verdict not in _AI_VERDICTS:
        sync_verdict = "PASS"
    else:
        sync_verdict = "CAUTION"

    summary_parts = [f"Grade: {grade}"] if grade else []
    if ai_prob and verdict:
        summary_parts.append(f"AI: {ai_prob}% ({verdict})")
    if flag_types:
        summary_parts.append(f"Flags: {', '.join(flag_types)}")
    summary_parts.append(f"Scanned: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}")

    return {
        "SYNC_SAFE_VERSION":        _TAG_SCHEMA_VERSION,
        "SYNC_SAFE_VERDICT":        sync_verdict,
        "SYNC_SAFE_AI_PROBABILITY": ai_prob,
        "SYNC_SAFE_FLAGS":          ", ".join(flag_types) if flag_types else "none",
        "SYNC_SAFE_SCAN_DATE":      datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "SYNC_SAFE_SUMMARY":        " | ".join(summary_parts),
    }


# ---------------------------------------------------------------------------
# Format-specific writers
# ---------------------------------------------------------------------------

def _inject_id3(path: Path, tags: dict[str, str]) -> None:
    """Inject TXXX custom frames and a COMM comment into an ID3/MP3 file."""
    from mutagen.id3 import COMM, ID3, TXXX, error as ID3Error  # type: ignore[import]

    try:
        id3 = ID3(str(path))
    except ID3Error:
        id3 = ID3()

    for key, value in tags.items():
        id3.add(TXXX(encoding=3, desc=key, text=[value]))

    summary = tags.get("SYNC_SAFE_SUMMARY", "")
    if summary:
        id3.add(COMM(encoding=3, lang="eng", desc="Sync-Safe Report", text=[summary]))

    id3.save(str(path), v2_version=3)


def _inject_vorbis(path: Path, tags: dict[str, str]) -> None:
    """Inject Vorbis comment tags into a FLAC or OGG file."""
    from mutagen import File as MutagenFile  # type: ignore[import]

    audio_file = MutagenFile(str(path))
    if audio_file is None:
        raise ValueError(f"mutagen could not open {path.name} for tag writing")
    for key, value in tags.items():
        audio_file[key.lower()] = [value]  # Vorbis keys are lowercase
    audio_file.save()


def _inject_mp4(path: Path, tags: dict[str, str]) -> None:
    """Inject freeform atoms into an MP4/M4A file."""
    from mutagen.mp4 import MP4, MP4FreeForm  # type: ignore[import]

    audio_file = MP4(str(path))
    for key, value in tags.items():
        atom_key = f"----:SYNC_SAFE:{key}"
        audio_file[atom_key] = [MP4FreeForm(value.encode("utf-8"))]
    audio_file.save()


def _detect_format(raw: bytes) -> Optional[str]:
    """
    Detect audio container format from magic bytes.

    Returns:
        "mp3" | "flac" | "ogg" | "mp4" | None (unknown)
    """
    if raw[:3] == b"ID3" or (len(raw) >= 2 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0):
        return "mp3"
    if raw[:4] == b"fLaC":
        return "flac"
    if raw[:4] == b"OggS":
        return "ogg"
    if raw[4:8] in (b"ftyp", b"moov") or raw[:4] == b"\x00\x00\x00\x20":
        return "mp4"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TagInjector:
    """
    Writes Sync-Safe audit results into audio file tags.

    Uses a TemporaryDirectory so no file persists beyond the method call.
    Returns the original bytes on any tagging failure (non-fatal).

    Implements: TagInjectorProvider protocol (core/protocols.py)
    """

    def inject(self, audio_bytes: bytes, result: AnalysisResult) -> bytes:
        """
        Embed audit results into audio file tags and return the modified bytes.

        Args:
            audio_bytes: Raw audio bytes from AudioBuffer.raw.
            result:      Complete AnalysisResult from the pipeline.

        Returns:
            Audio bytes with SYNC_SAFE_* tags injected.
            Returns *audio_bytes* unchanged if the format is unsupported or if
            mutagen raises — tag injection is always non-fatal.
        """
        if not audio_bytes:
            return audio_bytes

        fmt = _detect_format(audio_bytes)
        if fmt is None:
            _log.warning("TagInjector: unrecognised audio format — tags not written")
            return audio_bytes

        ext_map = {"mp3": ".mp3", "flac": ".flac", "ogg": ".ogg", "mp4": ".m4a"}
        suffix  = ext_map[fmt]
        tags    = _build_tag_values(result)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"track{suffix}"
            path.write_bytes(audio_bytes)
            try:
                if fmt == "mp3":
                    _inject_id3(path, tags)
                elif fmt in ("flac", "ogg"):
                    _inject_vorbis(path, tags)
                elif fmt == "mp4":
                    _inject_mp4(path, tags)
                return path.read_bytes()
            except Exception as exc:  # noqa: BLE001 — tag injection is non-fatal
                _log.warning("TagInjector: failed to write tags (%s) — returning original", exc)
                return audio_bytes
