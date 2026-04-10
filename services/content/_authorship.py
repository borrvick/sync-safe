"""
services/content/_authorship.py
Authorship class — orchestrates linguistic signals and RoBERTa classifier.
"""
from __future__ import annotations

import numpy as np

from core.config import ModelParams
from core.exceptions import ModelInferenceError
from core.models import AuthorshipResult, Section, SectionAuthorshipResult, TranscriptSegment
from core.utils import assign_sections

from ._pure import (
    _ai_phrase_score,
    _burstiness,
    _repetition_score,
    _rhyme_density,
    _score_signals,
    _unique_word_ratio,
)
from ._scoring import _compute_verdict

try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn


class Authorship:
    """
    Estimates whether lyrics were written by a human or AI.

    Implements: AuthorshipAnalyzer protocol (core/protocols.py)

    Constructor injection: pass a ModelParams instance to control the
    RoBERTa model ID and chunk size without touching environment variables.

    Usage:
        service = Authorship()
        result  = service.analyze(transcript)
        print(result.verdict, result.signal_count)
    """

    def __init__(self, params: ModelParams | None = None) -> None:
        self._params = params or ModelParams()
        self._roberta = None   # lazy — loaded on first call

    # ------------------------------------------------------------------
    # Public interface (AuthorshipAnalyzer protocol)
    # ------------------------------------------------------------------

    def analyze(
        self,
        transcript: list[TranscriptSegment],
        sections: list[Section] | None = None,
    ) -> AuthorshipResult:
        """
        Run all authorship signals and return a typed AuthorshipResult.

        Args:
            transcript: Whisper segments (text fields are the model input).
            sections:   allin1 structural sections, used for per-section scoring
                        (#156) and section-aware signal filtering (#158).
                        Defaults to None (full-track analysis only).

        Returns:
            AuthorshipResult with verdict, signal count, per-signal data, and
            optional per_section breakdown when sections are provided.

        Raises:
            ModelInferenceError: if the RoBERTa model fails to load.
        """
        lines     = [seg.text.strip() for seg in transcript if seg.text.strip()]
        full_text = "\n".join(lines)

        if not lines or len(full_text) < 80:
            return AuthorshipResult(
                verdict="Insufficient data",
                signal_count=0,
                roberta_score=None,
                feature_notes=[],
                scores={},
                skip_reason="instrumental",
            )

        if len(lines) < 4:
            return AuthorshipResult(
                verdict="Insufficient data",
                signal_count=0,
                roberta_score=None,
                feature_notes=[],
                scores={},
                skip_reason="too_short",
            )

        active_sections = sections or []
        burst  = _burstiness(lines)
        uwr    = _unique_word_ratio(full_text)
        rhyme  = _rhyme_density(lines, transcript, active_sections)
        rep    = _repetition_score(lines, transcript, active_sections)
        rob    = self._run_roberta(full_text)
        phrase = _ai_phrase_score(lines, full_text)

        ai_signals, feature_notes, scores = _score_signals(
            burst=burst, uwr=uwr, rhyme=rhyme, rep=rep, rob=rob, phrase=phrase,
        )
        verdict     = _compute_verdict(ai_signals)
        skip_reason = "short" if len(lines) < 8 else None
        per_section = _analyze_per_section(transcript, active_sections)

        return AuthorshipResult(
            verdict=verdict,
            signal_count=ai_signals,
            roberta_score=round(rob, 3) if rob is not None else None,
            feature_notes=feature_notes,
            scores=scores,
            skip_reason=skip_reason,
            per_section=per_section,
        )

    # ------------------------------------------------------------------
    # Private: RoBERTa inference  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _run_roberta(self, text: str) -> float | None:
        """
        Run the RoBERTa AI-text classifier on full lyric text.

        Chunks text into blocks of ModelParams.roberta_chunk_words words
        and averages the AI-probability scores.

        Returns:
            Mean AI probability (0.0–1.0), or None if all chunks fail.

        Raises:
            ModelInferenceError: if the model fails to load.
        """
        clf = self._load_roberta()

        words  = text.split()
        size   = self._params.roberta_chunk_words
        chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

        ai_probs: list[float] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                result = clf(chunk)[0]
                label  = result["label"].upper()
                score  = float(result["score"])
                # LABEL_1 = AI, LABEL_0 = Human (Hello-SimpleAI label convention)
                ai_prob = (
                    score
                    if ("1" in label or "fake" in label or "ai" in label)
                    else 1.0 - score
                )
                ai_probs.append(ai_prob)
            except Exception:  # noqa: BLE001 — partial result preferred
                continue

        return float(np.mean(ai_probs)) if ai_probs else None

    def _load_roberta(self):
        """
        Lazy-load and cache the RoBERTa text-classification pipeline.

        Raises:
            ModelInferenceError: if transformers cannot load the model.
        """
        if self._roberta is not None:
            return self._roberta
        try:
            from transformers import pipeline as hf_pipeline
            self._roberta = hf_pipeline(
                "text-classification",
                model=self._params.roberta_model,
                truncation=True,
                max_length=512,
            )
            return self._roberta
        except Exception as exc:
            raise ModelInferenceError(
                "RoBERTa authorship model failed to load.",
                context={
                    "model": self._params.roberta_model,
                    "original_error": str(exc),
                },
            ) from exc


# ---------------------------------------------------------------------------
# Per-section authorship — pure helper (#156)
# ---------------------------------------------------------------------------

_MIN_SECTION_LINES: int = 4
_MIN_SECTION_CHARS: int = 80


def _analyze_per_section(
    transcript: list[TranscriptSegment],
    sections: list[Section],
) -> dict[str, SectionAuthorshipResult]:
    """
    Run linguistic signals independently per allin1 section (#156).

    All instances of the same label (e.g. every CHORUS) are merged into a
    single bucket before scoring — avoids penalising intentional structural
    repetition across multiple chorus plays.

    Sections below the minimum data threshold (_MIN_SECTION_LINES lines or
    _MIN_SECTION_CHARS characters) are omitted from the result entirely.
    RoBERTa is NOT run per-section (GPU cost is prohibitive at this granularity).

    Pure function — no GPU, no I/O.
    """
    if not sections:
        return {}

    # Merge same-label sections into buckets
    buckets: dict[str, list[TranscriptSegment]] = {}
    for label, segs in assign_sections(transcript, sections):
        buckets.setdefault(label, []).extend(segs)

    result: dict[str, SectionAuthorshipResult] = {}
    for label, segs in buckets.items():
        lines     = [seg.text.strip() for seg in segs if seg.text.strip()]
        full_text = "\n".join(lines)
        if len(lines) < _MIN_SECTION_LINES or len(full_text) < _MIN_SECTION_CHARS:
            continue  # insufficient data — omit rather than produce noise

        burst = _burstiness(lines)
        uwr   = _unique_word_ratio(full_text)
        rhyme = _rhyme_density(lines)   # no section filtering within a single section
        rep   = _repetition_score(lines)

        ai_signals, feature_notes, scores = _score_signals(
            burst=burst, uwr=uwr, rhyme=rhyme, rep=rep, rob=None,
        )
        result[label] = SectionAuthorshipResult(
            verdict=_compute_verdict(ai_signals),
            signal_count=ai_signals,
            feature_notes=feature_notes,
            scores=scores,
        )

    return result
