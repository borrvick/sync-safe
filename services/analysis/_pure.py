"""
services/analysis/_pure.py
Pure section parsing and merging functions — no allin1, no librosa, no I/O.
"""
from __future__ import annotations

from core.models import Section


def _merge_consecutive_sections(sections: list[Section]) -> list[Section]:
    """
    Collapse adjacent Section objects with the same label into one.

    Example:
        [chorus 0:30-0:38, chorus 0:38-0:46, verse 0:46-1:02]
        → [chorus 0:30-0:46, verse 0:46-1:02]

    Pure function — no side effects.
    """
    if not sections:
        return []
    merged: list[Section] = [sections[0]]
    for sec in sections[1:]:
        if sec.label.lower() == merged[-1].label.lower():
            merged[-1] = Section(
                label=merged[-1].label,
                start=merged[-1].start,
                end=sec.end,
            )
        else:
            merged.append(sec)
    return merged


def _parse_sections(result: object) -> list[Section]:
    """
    Convert an allin1 AnalysisResult's segments into typed Section objects.

    Consecutive segments sharing the same label are merged into a single
    Section.  Pure function: takes the allin1 result object, returns a list
    of Section.
    """
    if not hasattr(result, "segments"):
        return []
    raw: list[Section] = [
        Section(
            label=str(getattr(seg, "label", "unknown")),
            start=float(getattr(seg, "start", 0.0)),
            end=float(getattr(seg, "end", 0.0)),
        )
        for seg in result.segments
    ]
    return _merge_consecutive_sections(raw)
