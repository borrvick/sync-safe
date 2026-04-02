"""
services/tagging/_formats.py
Per-format tag injection functions — independently importable, no class dependencies.
"""
from __future__ import annotations

from pathlib import Path


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
