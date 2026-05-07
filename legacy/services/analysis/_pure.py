"""
services/analysis/_pure.py
Pure section parsing and merging functions — no allin1, no librosa, no I/O.
"""
from __future__ import annotations

import statistics
from typing import Optional

from core.models import Section

# Canonical section label vocabulary (Harmonix dataset).
# Any label produced by allin1 that is not in this set gets normalised below.
HARMONIX_LABELS: frozenset[str] = frozenset({
    "chorus", "verse", "bridge", "intro", "outro",
    "pre-chorus", "post-chorus", "instrumental",
    "solo", "break", "interlude", "hook", "refrain",
})

# Map common allin1 abbreviations / raw outputs → canonical Harmonix label.
_LABEL_ALIASES: dict[str, str] = {
    # single-letter allin1 shorthand
    "c":    "chorus",
    "v":    "verse",
    "b":    "bridge",
    "i":    "intro",
    "o":    "outro",
    "p":    "pre-chorus",
    "s":    "solo",
    # common variant spellings
    "inst": "instrumental",
    "instrumental break": "instrumental",
    "prechorus": "pre-chorus",
    "pre chorus": "pre-chorus",
    "post chorus": "post-chorus",
    "postchorus": "post-chorus",
    # "refrain" and "hook" are in HARMONIX_LABELS — no alias needed; they return early.
    "drop": "chorus",
    "buildup": "pre-chorus",
    "build up": "pre-chorus",
    "build": "pre-chorus",
    "fade": "outro",
    "coda": "outro",
    "ending": "outro",
    "end": "outro",
    "breakdown": "break",
    "interlude": "break",
    "ad lib": "outro",
    "ad-lib": "outro",
}


def _normalize_section_label(label: str, index: int, total: int) -> str:
    """
    Normalise an allin1 section label to the canonical Harmonix vocabulary.

    Steps:
    1. Lowercase and strip whitespace.
    2. Return unchanged if already a canonical Harmonix label.
    3. Apply the alias table.
    4. Fall back to positional heuristics: index 0 → "intro",
       last index → "outro", otherwise → "verse".

    Pure function — no I/O.
    """
    lo = label.strip().lower()
    if lo in HARMONIX_LABELS:
        return lo
    if lo in _LABEL_ALIASES:
        return _LABEL_ALIASES[lo]
    # positional fallback for unrecognised labels
    if index == 0:
        return "intro"
    if index == total - 1:
        return "outro"
    return "verse"


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

    Each raw label is normalised to the canonical Harmonix vocabulary via
    `_normalize_section_label` before consecutive same-label sections are
    merged.  Pure function: takes the allin1 result object, returns a list
    of Section.
    """
    if not hasattr(result, "segments"):
        return []
    segs = list(result.segments)
    total = len(segs)
    raw: list[Section] = [
        Section(
            label=_normalize_section_label(
                str(getattr(seg, "label", "unknown")), i, total
            ),
            start=float(getattr(seg, "start", 0.0)),
            end=float(getattr(seg, "end", 0.0)),
        )
        for i, seg in enumerate(segs)
    ]
    return _merge_consecutive_sections(raw)


def section_ibi_tightness(
    section_start: float,
    section_end: float,
    all_beats: list[float],
    locked_threshold_ms: float,
    loose_threshold_ms: float,
    min_beats: int = 4,
) -> Optional[str]:
    """
    Classify the beat-grid tightness within a single section.

    Filters `all_beats` to only those within [section_start, section_end] before
    computing inter-beat intervals — this prevents cross-section IBI bleed where
    the first beat's interval would measure distance from the last beat of the
    prior section, skewing the std dev.

    Args:
        section_start:       Section start time in seconds.
        section_end:         Section end time in seconds.
        all_beats:           Full track beat grid in seconds.
        locked_threshold_ms: Std dev ≤ this → "Locked" (quantized).
        loose_threshold_ms:  Std dev ≥ this → "Loose" (rubato/live).
        min_beats:           Minimum in-section beats required for a valid estimate.

    Returns:
        "Locked", "Moderate", "Loose", or None (insufficient beats).

    Pure function — no I/O.
    """
    section_beats = [b for b in all_beats if section_start <= b <= section_end]
    if len(section_beats) < min_beats:
        return None  # too few beats for a reliable std dev

    ibis_ms = [
        (section_beats[i + 1] - section_beats[i]) * 1000
        for i in range(len(section_beats) - 1)
    ]
    std_ms = statistics.stdev(ibis_ms)

    if std_ms <= locked_threshold_ms:
        return "Locked"
    if std_ms >= loose_threshold_ms:
        return "Loose"
    return "Moderate"
