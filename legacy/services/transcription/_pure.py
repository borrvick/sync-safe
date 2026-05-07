"""
services/transcription/_pure.py
Pure transcription helpers — no GPU, no I/O, no model calls.
"""
from __future__ import annotations

import re

from core.models import TranscriptSegment


def _parse_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    """
    Convert Whisper's raw segment dicts into typed TranscriptSegment objects,
    then strip hallucinated repetition runs.

    Pure function — no I/O, no model calls. Handles missing or malformed
    keys gracefully so a partial Whisper result still produces valid output.
    """
    segments: list[TranscriptSegment] = []
    for seg in raw_segments:
        text = _collapse_intra_repetitions(seg.get("text", "").strip())
        if not text:
            continue
        segments.append(TranscriptSegment(
            start=round(float(seg.get("start", 0.0)), 2),
            end=round(float(seg.get("end", 0.0)), 2),
            text=text,
        ))
    return _strip_repetition_runs(segments)


def _collapse_intra_repetitions(text: str, max_keep: int = 2) -> str:
    """
    Collapse repeated phrases within a single Whisper segment.

    Whisper sometimes loops *within* a single decoding window, e.g.:
      "we are not, we are not, we are not, ..." × 55  (comma-separated)
      "we are not we are not we are not ..."           (space-separated)

    Uses two passes:
      1. Comma-split: detects runs in comma-delimited clauses.
      2. Regex: detects n-gram repetitions (1–6 words) with any separator.

    Natural commas ("See you can't sleep, baby, I know") are unaffected
    because no clause there appears more than max_keep times in a row.

    Pure function — no I/O.
    """
    # Pass 1 — comma-separated runs
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= max_keep + 2:
            result: list[str] = []
            i = 0
            while i < len(parts):
                phrase = parts[i].lower()
                run_end = i + 1
                while run_end < len(parts) and parts[run_end].lower() == phrase:
                    run_end += 1
                keep = min(run_end - i, max_keep)
                result.extend(parts[i : i + keep])
                i = run_end
            collapsed = ", ".join(result)
            if len(collapsed) < len(text):
                text = collapsed

    # Pass 2 — regex n-gram repetition (catches space-only separators)
    # Matches a phrase of 1–6 words repeated 4+ times (with any separator).
    pattern = re.compile(
        r'\b((?:\w[\w\']*(?:\s+|,\s*))){1,6}(?=(?:\1){3,})',
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        phrase = match.group(0).strip(", ")
        # Replace the entire repetitive block with max_keep copies
        repetition_block = re.compile(
            r'(?:' + re.escape(phrase) + r'(?:[,\s]+|$)){4,}',
            re.IGNORECASE,
        )
        replacement = (", ".join([phrase] * max_keep))
        text = repetition_block.sub(replacement, text)

    return text.strip(", ")


def _strip_repetition_runs(
    segments: list[TranscriptSegment],
    max_run: int = 3,
    uniform_gap_tolerance: float = 0.15,
) -> list[TranscriptSegment]:
    """
    Remove hallucinated repetition runs from Whisper output.

    When Whisper transcribes near-silence (e.g. an instrumental intro after
    vocal isolation), it hallucinates the same phrase repeatedly at fixed
    intervals — e.g. "I can't relate" × 14, each exactly 2.0s apart.

    Real repeated lyrics (a chorus, a hook) also repeat, but their inter-segment
    gaps vary naturally. This function only drops a run when BOTH conditions hold:
      1. The run length exceeds max_run (> 3 identical consecutive segments)
      2. The gaps between segments in the run are suspiciously uniform
         (all within uniform_gap_tolerance seconds of each other)

    Pure function — no I/O.
    """
    if not segments:
        return segments

    result: list[TranscriptSegment] = []
    i = 0
    while i < len(segments):
        run_text = segments[i].text.lower()
        run_end  = i + 1
        while run_end < len(segments) and segments[run_end].text.lower() == run_text:
            run_end += 1
        run_length = run_end - i

        # Drop if EITHER: very long run (≥5 = always hallucination, real
        # choruses don't repeat the same phrase identically 5+ times in a row)
        # OR shorter run with machine-regular gaps (uniform-interval
        # hallucination on silence).
        is_hallucination = (
            run_length >= 5
            or (run_length > max_run and _gaps_are_uniform(segments[i:run_end], uniform_gap_tolerance))
        )
        if is_hallucination:
            pass  # drop
        else:
            result.extend(segments[i:run_end])

        i = run_end

    return result


def _gaps_are_uniform(
    run: list[TranscriptSegment],
    tolerance: float,
) -> bool:
    """
    Return True if the gaps between consecutive segments in a run are all
    within `tolerance` seconds of the median gap.

    Whisper hallucination runs land at perfectly even intervals (e.g. every
    2.000s). Real repeated lyrics have natural variation in spacing.

    Pure function — no I/O.
    """
    if len(run) < 2:
        return False
    gaps = [run[k + 1].start - run[k].start for k in range(len(run) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2]
    return all(abs(g - median_gap) <= tolerance for g in gaps)
