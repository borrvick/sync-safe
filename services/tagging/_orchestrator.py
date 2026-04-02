"""
services/tagging/_orchestrator.py
TagInjector class — detects format and dispatches to _formats.py injectors.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.models import AnalysisResult

from ._formats import _inject_id3, _inject_mp4, _inject_vorbis

_log = logging.getLogger(__name__)

_TAG_SCHEMA_VERSION: str = "1.0"
_FAIL_GRADES:    frozenset[str] = frozenset({"D", "F"})
_AI_VERDICTS:    frozenset[str] = frozenset({"AI", "Likely AI"})


def _build_tag_values(result: AnalysisResult) -> dict[str, str]:
    """
    Derive the SYNC_SAFE_* tag dict from an AnalysisResult.

    Pure function — no I/O.
    """
    grade   = (result.compliance.grade or "") if result.compliance else ""
    verdict = result.forensics.verdict if result.forensics else ""
    ai_prob = str(round(result.forensics.ai_probability * 100.0, 1)) if result.forensics else ""

    flag_types: list[str] = sorted(
        {f.issue_type for f in result.compliance.flags}
    ) if result.compliance else []

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
