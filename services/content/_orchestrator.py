"""
services/lyric_authorship.py
AI lyric authorship detection — implements the AuthorshipAnalyzer protocol.

Four linguistic signals (CPU, instant):
  1. Burstiness        — coefficient of variation of line word counts
  2. Unique word ratio — type-token ratio across full lyrics
  3. Rhyme density     — fraction of consecutive line pairs that end-rhyme
  4. Repetition score  — fraction of lines repeated verbatim elsewhere

One classifier signal (GPU):
  5. Hello-SimpleAI/chatgpt-detector-roberta → AI probability

Design notes:
- Authorship.analyze() is the single public entry point.
- RoBERTa is lazy-loaded on first call and cached as an instance attribute —
  no module-level globals.
- All thresholds come from CONSTANTS — no inline magic numbers.
- The four signal functions are pure module-level functions for independent
  unit testing.
- RoBERTa load failure → ModelInferenceError.
  Inference failures on individual chunks → silently skipped (partial result
  is better than none).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from core.config import CONSTANTS, ModelParams
from core.exceptions import ModelInferenceError
from core.models import AuthorshipResult, TranscriptSegment

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

    def analyze(self, transcript: list[TranscriptSegment]) -> AuthorshipResult:
        """
        Run all authorship signals and return a typed AuthorshipResult.

        Args:
            transcript: Whisper segments (text fields are the model input).

        Returns:
            AuthorshipResult with verdict, signal count, and per-signal data.

        Raises:
            ModelInferenceError: if the RoBERTa model fails to load.
        """
        lines     = [seg.text.strip() for seg in transcript if seg.text.strip()]
        full_text = "\n".join(lines)

        if len(lines) < 4 or len(full_text) < 80:
            return AuthorshipResult(
                verdict="Insufficient data",
                signal_count=0,
                roberta_score=None,
                feature_notes=["Not enough lyric content to analyse."],
                scores={},
            )

        burst = _burstiness(lines)
        uwr   = _unique_word_ratio(full_text)
        rhyme = _rhyme_density(lines)
        rep   = _repetition_score(lines)
        rob   = self._run_roberta(full_text)   # may raise ModelInferenceError

        ai_signals, feature_notes, scores = _score_signals(
            burst=burst,
            uwr=uwr,
            rhyme=rhyme,
            rep=rep,
            rob=rob,
        )

        verdict = _compute_verdict(ai_signals)

        return AuthorshipResult(
            verdict=verdict,
            signal_count=ai_signals,
            roberta_score=round(rob, 3) if rob is not None else None,
            feature_notes=feature_notes,
            scores=scores,
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
# Module-level pure functions — independently testable
# ---------------------------------------------------------------------------

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

    AI lyrics reuse vocabulary and phrases more than humans.
    Returns None when the text contains fewer than 20 words.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 20:
        return None
    return len(set(words)) / len(words)


def _rhyme_density(lines: list[str]) -> float | None:
    """
    Fraction of consecutive non-empty line pairs that share a rhyme ending
    (last 2+ chars of terminal word, case-insensitive, excluding identical words).

    AI over-rhymes; human songs mix rhymed and unrhymed lines.
    Returns None when there are fewer than 4 non-empty lines with words.
    """
    endings = []
    for ln in lines:
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


def _repetition_score(lines: list[str]) -> float:
    """
    Fraction of lines whose normalised text appears more than once.

    AI generators repeat hooks and phrases beyond normal chorus repetition.
    Returns 0.0 when there are no non-empty lines.
    """
    clean = [re.sub(r"\s+", " ", ln.strip().lower()) for ln in lines if ln.strip()]
    if not clean:
        return 0.0
    counts   = Counter(clean)
    repeated = sum(1 for ln in clean if counts[ln] > 1)
    return repeated / len(clean)


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


def _compute_verdict(ai_signals: int) -> str:
    """
    Map a signal count to a verdict string.

    Thresholds from CONSTANTS:
        AI_SIGNAL_COUNT_CERTAIN   → "Likely AI"
        AI_SIGNAL_COUNT_UNCERTAIN → "Uncertain"
        else                      → "Likely Human"
    """
    if ai_signals >= CONSTANTS.AI_SIGNAL_COUNT_CERTAIN:
        return "Likely AI"
    if ai_signals >= CONSTANTS.AI_SIGNAL_COUNT_UNCERTAIN:
        return "Uncertain"
    return "Likely Human"
