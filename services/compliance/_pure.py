"""
services/compliance/_pure.py
Pure compliance helpers and module-level constants — no GPU, no I/O.
"""
from __future__ import annotations

import re

from core.config import CONSTANTS
from core.models import ComplianceFlag, IssueType, Severity, TranscriptSegment
from data.drug_keywords import DRUG_KEYWORDS

# ---------------------------------------------------------------------------
# Detoxify thresholds — read-only aliases for brevity
# ---------------------------------------------------------------------------

_EXPLICIT_CONFIRMED: float = CONSTANTS.EXPLICIT_CONFIRMED
_EXPLICIT_POTENTIAL: float = CONSTANTS.EXPLICIT_POTENTIAL
_HARD_OBSCENE_THRESHOLD: float = CONSTANTS.EXPLICIT_HARD
_VIOLENCE_CONFIRMED: float = CONSTANTS.VIOLENCE_CONFIRMED
_VIOLENCE_POTENTIAL: float = CONSTANTS.VIOLENCE_POTENTIAL
_HARD_THREAT_THRESHOLD: float = CONSTANTS.VIOLENCE_HARD
_DRUGS_TOXIC_MIN: float = CONSTANTS.DRUGS_TOXIC_MIN

# Number of segments in a sliding window used to confirm borderline scores
_WINDOW_SIZE: int = 3

# Segments shorter than this carry no classifiable signal
_MIN_SEGMENT_CHARS: int = 20

# Only these issue types count toward the A–F content grade
_GRADE_ISSUE_TYPES: frozenset[str] = frozenset({"EXPLICIT", "VIOLENCE", "DRUGS"})

# NER entity labels mapped to our issue types.
# ORG → BRAND: trademarked company/product names (Nike, Spotify, etc.)
# GPE → LOCATION: geo-political entities (countries, cities, regions) that may
#   create placement restrictions in international licensing contexts.
_NER_ISSUE_MAP: dict[str, IssueType] = {
    "ORG":  "BRAND",
    "GPE":  "LOCATION",
}

_NER_RECOMMENDATIONS: dict[IssueType, str] = {
    "BRAND":    "Confirm explicit brand clearance with rights holder.",
    "LOCATION": "Geographic reference — verify placement restrictions for international markets.",
}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _build_windows(
    segments: list[TranscriptSegment], window_size: int
) -> dict[int, str]:
    """
    Build a sliding text window for each segment index.

    Returns a dict mapping segment index → concatenated text of the next
    `window_size` segments (inclusive). Used for borderline NLI confirmation.
    """
    return {
        i: " ".join(
            segments[j].text
            for j in range(i, min(i + window_size, len(segments)))
        )
        for i in range(len(segments))
    }


def _score_detoxify(
    seg_text: str,
    window_text: str,
    detector,
) -> list[tuple[IssueType, str, str, str]]:
    """
    Score a segment with Detoxify and return any flags.

    Uses the segment score first; falls back to the sliding window to confirm
    borderline scores before promoting them to "confirmed".

    Returns:
        List of (issue_type, recommendation, confidence, severity) tuples.
        severity is "hard" for absolute deal-breakers, "soft" for
        placement-dependent issues that are the sync director's call.
        Returns empty list if clean.

    Pure function aside from the detector call — no side effects.
    """
    def _predict(text: str) -> dict[str, float]:
        try:
            return {k: float(v) for k, v in detector.predict(text).items()}
        except Exception:  # noqa: BLE001
            return {}

    solo = _predict(seg_text)
    if not solo:
        return []

    flags: list[tuple[IssueType, str, str, str]] = []

    # --- EXPLICIT ---
    explicit_score = solo.get("obscene", 0.0)
    if explicit_score >= _EXPLICIT_CONFIRMED:
        sev = "hard" if explicit_score >= _HARD_OBSCENE_THRESHOLD else "soft"
        rec = (
            "Clean edit required — hard explicit content."
            if sev == "hard"
            else "Mild language — acceptable for some placements. Supervisor discretion."
        )
        flags.append(("EXPLICIT", rec, "confirmed", sev))
    elif explicit_score >= _EXPLICIT_POTENTIAL:
        win = _predict(window_text)
        win_score = win.get("obscene", 0.0)
        if win_score >= _EXPLICIT_CONFIRMED:
            sev = "hard" if win_score >= _HARD_OBSCENE_THRESHOLD else "soft"
            rec = (
                "Clean edit required — hard explicit content."
                if sev == "hard"
                else "Mild language — acceptable for some placements. Supervisor discretion."
            )
            flags.append(("EXPLICIT", rec, "confirmed", sev))
        else:
            flags.append((
                "EXPLICIT",
                "Possible explicit content — supervisor should review before submission.",
                "potential",
                "soft",
            ))

    # --- VIOLENCE ---
    threat_score = solo.get("threat", 0.0)
    if threat_score >= _VIOLENCE_CONFIRMED:
        sev = "hard" if threat_score >= _HARD_THREAT_THRESHOLD else "soft"
        rec = (
            "Threatening language — disqualifies family and broadcast placements."
            if sev == "hard"
            else "Flag for brand-safety review; may disqualify family placements."
        )
        flags.append(("VIOLENCE", rec, "confirmed", sev))
    elif threat_score >= _VIOLENCE_POTENTIAL:
        win = _predict(window_text)
        win_score = win.get("threat", 0.0)
        if win_score >= _VIOLENCE_CONFIRMED:
            sev = "hard" if win_score >= _HARD_THREAT_THRESHOLD else "soft"
            rec = (
                "Threatening language — disqualifies family and broadcast placements."
                if sev == "hard"
                else "Flag for brand-safety review; may disqualify family placements."
            )
            flags.append(("VIOLENCE", rec, "confirmed", sev))
        else:
            flags.append((
                "VIOLENCE",
                "Possible violent language — review in context before family placement.",
                "potential",
                "soft",
            ))

    # --- DRUGS (toxicity + word-list gate) — always hard ---
    toxic_score = solo.get("toxicity", 0.0)
    text_lower  = seg_text.lower()
    has_drug_word = any(w in text_lower.split() for w in DRUG_KEYWORDS)
    if toxic_score >= _DRUGS_TOXIC_MIN and has_drug_word:
        flags.append((
            "DRUGS",
            "Drug reference — disqualifies broadcast and most brand placements.",
            "confirmed",
            "hard",
        ))

    return flags


def _check_brand_keywords(
    seg_text: str,
    timestamp_s: int,
    brand_patterns: list[tuple[str, re.Pattern]],
) -> list[ComplianceFlag]:
    """
    Scan a lyric segment against the compiled brand keyword patterns.

    Returns a list of potential BRAND ComplianceFlags (one per matched brand).
    Pure function — the patterns are passed in rather than read from a global.
    """
    flags: list[ComplianceFlag] = []
    seen: set[str] = set()
    for display_name, pattern in brand_patterns:
        if display_name not in seen and pattern.search(seg_text):
            seen.add(display_name)
            flags.append(ComplianceFlag(
                timestamp_s=timestamp_s,
                issue_type="BRAND",
                text=display_name,
                recommendation=(
                    "Possible brand/trademark reference — supervisor should verify sync clearance."
                ),
                confidence="potential",
            ))
    return flags


def _deduplicate_flags(flags: list[ComplianceFlag]) -> list[ComplianceFlag]:
    """
    Remove duplicate flags — keep the first occurrence of each (text, issue_type) pair.

    Deduplication is keyed on normalized text + issue type, NOT timestamp, so
    repeated chorus lines collapse to a single flag at the earliest timestamp.
    Confirmed flags take priority over potential ones for the same key.
    """
    # Sort: confirmed before potential, then by timestamp (earliest first)
    ordered = sorted(flags, key=lambda f: (0 if f.confidence == "confirmed" else 1, f.timestamp_s))
    seen:   set[tuple[str, str]] = set()
    unique: list[ComplianceFlag] = []
    for f in ordered:
        key = (f.text.strip().lower(), f.issue_type)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _compute_grade(flags: list[ComplianceFlag]) -> str:
    """
    Compute an A–F lyric-content grade.

    Only hard-severity EXPLICIT, VIOLENCE, and DRUGS flags count toward the grade.
    BRAND flags and structural issues (fade, energy, intro) do not affect this
    grade — structural fitness is handled by the Sync Snapshot verdict.
    Soft confirmed flags (mild language, brand mentions, borderline metaphors) are
    shown in the audit table but do not lower the grade below B — they are the
    sync director's placement call, not absolute blockers.

    Grading scale:
        A — 0 hard confirmed issues, 0 soft/potential flags
        B — 0 hard confirmed issues, but soft confirmed or potential flags present
        C — 1 hard confirmed issue
        D — 2–3 hard confirmed issues
        F — 4+ hard confirmed issues (or any DRUGS hard confirmed)
    """
    hard_confirmed = sum(
        1 for f in flags
        if f.confidence == "confirmed"
        and f.severity == "hard"
        and f.issue_type in _GRADE_ISSUE_TYPES
    )
    has_drugs_hard = any(
        f.issue_type == "DRUGS" and f.confidence == "confirmed" and f.severity == "hard"
        for f in flags
    )
    if has_drugs_hard or hard_confirmed >= 4:
        return "F"
    if hard_confirmed == 3:
        return "D"
    if hard_confirmed == 2:
        return "D"
    if hard_confirmed == 1:
        return "C"
    # 0 hard confirmed — grade on whether any soft/potential flags exist
    has_advisory = any(
        f.confidence in ("confirmed", "potential") for f in flags
    )
    return "B" if has_advisory else "A"
