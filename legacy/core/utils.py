"""
core/utils.py
Shared pure utility functions used by both services and UI layers.

Rule: no Streamlit imports, no GPU, no I/O. Functions here must be
importable by any layer without side effects.
"""
from __future__ import annotations

from core.models import Section, TranscriptSegment


def assign_sections(
    segments: list[TranscriptSegment],
    sections: list[Section],
) -> list[tuple[str, list[TranscriptSegment]]]:
    """
    Group Whisper transcript segments into allin1 structural sections.

    Each segment is assigned to the section whose time window contains the
    segment midpoint ((start + end) / 2). Using the midpoint rather than
    the segment start prevents segments that begin just before a section
    boundary from being misassigned to the preceding section.

    Sections with the same label are kept as separate entries — the caller
    decides whether to merge them (e.g. Authorship merges all CHORUS instances;
    the lyric column renders them separately).

    Args:
        segments: Whisper/LRCLib segments in chronological order.
        sections: allin1 structural sections; may be empty.

    Returns:
        List of (label, segment_group) pairs in section order.
        Falls back to [("TRACK", segments)] when *sections* is empty.
        Segments not covered by any section are appended as ("OTHER", [...]).

    Pure function — no I/O, no side effects.
    """
    if not sections:
        return [("TRACK", segments)]

    ordered   = sorted(sections, key=lambda s: s.start)
    result: list[tuple[str, list[TranscriptSegment]]] = []
    captured: set[int] = set()

    for sec in ordered:
        grp = [
            seg for seg in segments
            if sec.start <= (seg.start + seg.end) / 2 < sec.end
        ]
        if grp:
            result.append((sec.label.upper(), grp))
            captured.update(id(seg) for seg in grp)

    leftovers = [seg for seg in segments if id(seg) not in captured]
    if leftovers:
        result.append(("OTHER", leftovers))

    return result
