"""
services/content/_pure.py
Pure linguistic signal functions for lyric authorship detection — no GPU, no I/O.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np

from core.config import CONSTANTS
from core.models import Section, TranscriptSegment


def _burstiness(lines: list[str]) -> float | None:
    """
    Coefficient of variation (std / mean) of word counts per line.

    Human writers vary line length naturally; AI tends toward uniform output.
    Returns None when there are fewer than 4 non-empty lines.
    """
    counts = [len(ln.split()) for ln in lines if ln.strip()]
    if len(counts) < 4:
        return None
    mean = float(np.mean(counts))
    std  = float(np.std(counts))
    return std / (mean + 1e-9)


def _unique_word_ratio(text: str) -> float | None:
    """
    Type-token ratio: unique_words / total_words.

    Returns None when the text contains fewer than 20 words.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 20:
        return None
    return len(set(words)) / len(words)


def _rhyme_density(
    lines: list[str],
    segments: Optional[list[TranscriptSegment]] = None,
    sections: Optional[list[Section]] = None,
) -> float | None:
    """
    Fraction of consecutive non-empty line pairs that share a rhyme ending.

    When *segments* and *sections* are both provided, verse-only rhyme density
    is computed (chorus/outro lines excluded) because dense rhyming in those
    sections is structurally normal (#158). Falls back to full-track scoring
    when the filtered verse set has fewer than 4 lines.

    Returns None when there are fewer than 4 non-empty lines with words.
    """
    active_lines = _filter_verse_lines(lines, segments, sections)

    endings = []
    for ln in active_lines:
        words = re.findall(r"\b\w+\b", ln.lower())
        if words:
            endings.append(words[-1])

    if len(endings) < 4:
        return None

    pairs = rhymes = 0
    for i in range(len(endings) - 1):
        a, b = endings[i], endings[i + 1]
        if len(a) >= 2 and len(b) >= 2 and a != b and a[-2:] == b[-2:]:
            rhymes += 1
        pairs += 1

    return rhymes / pairs if pairs else 0.0


def _repetition_score(
    lines: list[str],
    segments: Optional[list[TranscriptSegment]] = None,
    sections: Optional[list[Section]] = None,
) -> float:
    """
    Fraction of lines whose normalised text appears more than once.

    When *segments* and *sections* are both provided, only lines from
    non-chorus/outro sections are scored — chorus repetition is intentional
    and not an AI signal (#158). Falls back to full-track scoring when the
    filtered verse set has fewer than 4 lines.

    Returns 0.0 when there are no non-empty lines.
    """
    active_lines = _filter_verse_lines(lines, segments, sections)
    clean = [re.sub(r"\s+", " ", ln.strip().lower()) for ln in active_lines if ln.strip()]
    if not clean:
        return 0.0
    counts   = Counter(clean)
    repeated = sum(1 for ln in clean if counts[ln] > 1)
    return repeated / len(clean)


def _filter_verse_lines(
    lines: list[str],
    segments: Optional[list[TranscriptSegment]],
    sections: Optional[list[Section]],
) -> list[str]:
    """
    Return only the lines that belong to non-chorus/outro sections (#158).

    When sections are unavailable or the filtered set is too small (< 4 lines),
    returns the original full list so the signal degrades gracefully rather than
    silently hiding a real verse-level problem.

    Pure helper — no I/O.
    """
    if not segments or not sections:
        return lines

    # Build a set of chorus/outro section time windows
    chorus_windows: list[tuple[float, float]] = [
        (sec.start, sec.end)
        for sec in sections
        if sec.label.lower() in CONSTANTS.CHORUS_OUTRO_LABELS
    ]
    if not chorus_windows:
        return lines

    # Map each segment to a line index (segments and lines are parallel)
    if len(segments) != len(lines):
        return lines  # mismatched lengths — fall back to full set

    verse_lines: list[str] = []
    for seg, ln in zip(segments, lines):
        mid = (seg.start + seg.end) / 2
        in_chorus = any(start <= mid < end for start, end in chorus_windows)
        if not in_chorus:
            verse_lines.append(ln)

    return verse_lines if len(verse_lines) >= 4 else lines


def _score_signals(
    burst: float | None,
    uwr:   float | None,
    rhyme: float | None,
    rep:   float,
    rob:   float | None,
) -> tuple[int, list[str], dict[str, float | None]]:
    """
    Map feature values to AI signal counts and human-readable notes.

    Returns:
        (ai_signals, feature_notes, scores_dict)

    Pure function — all thresholds come from CONSTANTS.
    """
    ai_signals:    int       = 0
    feature_notes: list[str] = []
    scores: dict[str, float | None] = {
        "burstiness":        round(burst, 3) if burst is not None else None,
        "unique_word_ratio": round(uwr, 3)   if uwr   is not None else None,
        "rhyme_density":     round(rhyme, 3) if rhyme is not None else None,
        "repetition_score":  round(rep, 3),
        "roberta_ai_prob":   round(rob, 3)   if rob   is not None else None,
    }

    if burst is not None:
        if burst < CONSTANTS.BURSTINESS_CV_THRESHOLD:
            ai_signals += 1
            feature_notes.append(
                f"Uniform line lengths — burstiness CV {burst:.2f} "
                f"(AI threshold <{CONSTANTS.BURSTINESS_CV_THRESHOLD})"
            )
        else:
            feature_notes.append(f"Variable line lengths — burstiness CV {burst:.2f} ✓")

    if uwr is not None:
        if uwr < CONSTANTS.UNIQUE_WORD_RATIO_THRESHOLD:
            ai_signals += 1
            feature_notes.append(
                f"Low vocabulary diversity — {uwr:.0%} unique words "
                f"(AI threshold <{CONSTANTS.UNIQUE_WORD_RATIO_THRESHOLD:.0%})"
            )
        else:
            feature_notes.append(f"Healthy vocabulary diversity — {uwr:.0%} unique words ✓")

    if rhyme is not None:
        if rhyme > CONSTANTS.RHYME_DENSITY_THRESHOLD:
            ai_signals += 1
            feature_notes.append(
                f"Over-rhymed — {rhyme:.0%} consecutive pairs rhyme "
                f"(AI threshold >{CONSTANTS.RHYME_DENSITY_THRESHOLD:.0%})"
            )
        else:
            feature_notes.append(f"Natural rhyme density — {rhyme:.0%} ✓")

    if rep > CONSTANTS.REPETITION_SCORE_THRESHOLD:
        ai_signals += 1
        feature_notes.append(
            f"High repetition — {rep:.0%} of lines repeated "
            f"(AI threshold >{CONSTANTS.REPETITION_SCORE_THRESHOLD:.0%})"
        )
    else:
        feature_notes.append(f"Normal repetition — {rep:.0%} of lines repeated ✓")

    if rob is not None:
        if rob >= 0.70:
            ai_signals += 2
            feature_notes.append(f"Classifier: AI-generated ({rob:.0%} confidence)")
        elif rob >= 0.50:
            ai_signals += 1
            feature_notes.append(f"Classifier: borderline ({rob:.0%} AI probability)")
        else:
            feature_notes.append(f"Classifier: likely human ({rob:.0%} AI probability) ✓")

    return ai_signals, feature_notes, scores
