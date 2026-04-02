"""
services/content/_theme.py
Theme and mood analysis — implements the ThemeMoodAnalyzer protocol.

Two-tier strategy:
  1. Keyword taxonomy fast-path (always runs, no API required)
     Scores each mood and theme category against lyric word-frequency counts.
  2. Optional Groq enrichment (gated by Settings.groq_api_key)
     Sends a compact lyric summary to Groq's LLM for richer classification.
     Falls back to keyword-only if the API call fails.

Design notes:
  - Constructor injection: pass groq_client=None to force keyword-only mode.
  - All string constants live in _labels.py — none inline here.
  - ModelInferenceError raised on Groq failure; keyword result returned as
    the fallback so the pipeline never drops Theme & Mood entirely.
"""
from __future__ import annotations

from collections import Counter

from core.config import get_settings
from core.exceptions import ModelInferenceError
from core.models import ThemeMoodResult, TranscriptSegment

from ._labels import MOOD_LABELS, THEME_TAXONOMY

# Maximum lyrics characters sent to Groq (keeps token cost predictable)
_GROQ_MAX_CHARS: int = 2000
# Groq model to use for enrichment
_GROQ_MODEL: str = "llama3-8b-8192"


class ThemeMoodAnalyzer:
    """
    Classifies lyric themes and overall mood from a transcript.

    Implements: ThemeMoodAnalyzer protocol (core/protocols.py)

    Constructor injection:
        groq_client — pass an initialised Groq client to enable enrichment,
                      or None (default) for keyword-only mode.

    Usage:
        service = ThemeMoodAnalyzer()
        result  = service.analyze(transcript)
        print(result.mood, result.themes)
    """

    def __init__(self, groq_client=None) -> None:
        self._groq = groq_client

    # ------------------------------------------------------------------
    # Public interface (ThemeMoodAnalyzer protocol)
    # ------------------------------------------------------------------

    def analyze(self, transcript: list[TranscriptSegment]) -> ThemeMoodResult:
        """
        Return a ThemeMoodResult from keyword analysis and optional Groq enrichment.

        Args:
            transcript: Whisper/LRCLib segments.

        Returns:
            ThemeMoodResult with themes, mood, confidence, and raw keywords.

        Raises:
            ModelInferenceError: if Groq is configured but its API call fails
                                 and the caller needs to know (keyword result
                                 is returned as fallback — this is non-fatal).
        """
        text = " ".join(seg.text for seg in transcript if seg.text.strip())
        keyword_result = _keyword_analyze(text)

        if self._groq is None:
            # Attempt lazy init if API key is configured
            self._groq = _try_init_groq()

        if self._groq is None:
            return keyword_result

        try:
            return _groq_enrich(self._groq, text, keyword_result)
        except ModelInferenceError:
            return keyword_result


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _keyword_analyze(text: str) -> ThemeMoodResult:
    """
    Score mood and themes from word-frequency matching against the taxonomy.

    Pure function — no I/O, no API calls.
    """
    if not text.strip():
        return ThemeMoodResult(
            themes=[],
            mood="Neutral",
            confidence=0.0,
            groq_enriched=False,
            raw_keywords=[],
        )

    words = text.lower().split()
    word_set = set(words)
    word_freq = Counter(words)
    total_words = max(len(words), 1)

    # Score each mood
    mood_scores: dict[str, float] = {}
    for mood, keywords in MOOD_LABELS.items():
        hits = sum(word_freq.get(kw, 0) for kw in keywords if " " not in kw)
        # Multi-word phrases checked against full text
        phrase_hits = sum(1 for kw in keywords if " " in kw and kw in text.lower())
        mood_scores[mood] = (hits + phrase_hits * 2) / total_words

    # Score each theme
    theme_scores: dict[str, float] = {}
    for theme, keywords in THEME_TAXONOMY.items():
        hits = sum(word_freq.get(kw, 0) for kw in keywords if " " not in kw)
        phrase_hits = sum(1 for kw in keywords if " " in kw and kw in text.lower())
        theme_scores[theme] = (hits + phrase_hits * 2) / total_words

    # Pick top mood
    best_mood = max(mood_scores, key=lambda m: mood_scores[m])
    best_mood_score = mood_scores[best_mood]
    mood = best_mood if best_mood_score > 0.0 else "Neutral"

    # Pick themes above a minimum signal threshold
    threshold = max(0.002, best_mood_score * 0.3)
    themes = [
        t for t, s in sorted(theme_scores.items(), key=lambda x: -x[1])
        if s >= threshold
    ][:3]  # cap at 3 themes

    # Collect matched keywords for transparency
    all_kw = [kw for kws in MOOD_LABELS.values() for kw in kws]
    all_kw += [kw for kws in THEME_TAXONOMY.values() for kw in kws]
    raw_keywords = sorted({kw for kw in all_kw if kw in word_set or kw in text.lower()})[:20]

    # Confidence: normalise top mood score to a 0–1 range (empirical ceiling ~0.05)
    confidence = min(1.0, best_mood_score / 0.05)

    return ThemeMoodResult(
        themes=themes,
        mood=mood,
        confidence=round(confidence, 3),
        groq_enriched=False,
        raw_keywords=raw_keywords,
    )


def _try_init_groq():
    """
    Attempt to lazily initialise a Groq client from Settings.groq_api_key.

    Returns None if the key is absent or the groq package is not installed.
    Never raises.
    """
    try:
        settings = get_settings()
        key = getattr(settings, "groq_api_key", None)
        if not key:
            return None
        from groq import Groq
        return Groq(api_key=key)
    except Exception:  # noqa: BLE001 — Groq init is always best-effort
        return None


def _groq_enrich(
    groq_client,
    text: str,
    keyword_result: ThemeMoodResult,
) -> ThemeMoodResult:
    """
    Use Groq's LLM to enrich the keyword-derived result.

    Sends a compact lyric extract and asks for JSON-structured themes + mood.
    Falls through to keyword_result on any parse failure.

    Raises:
        ModelInferenceError: if the Groq API call itself errors.
    """
    import json

    excerpt = text[:_GROQ_MAX_CHARS]
    prompt = (
        "Analyse these song lyrics and respond with ONLY valid JSON — no prose, "
        "no markdown fences. Schema: "
        '{\"themes\": [<up to 3 short theme labels>], \"mood\": \"<single mood label>\", '
        '\"confidence\": <float 0-1>}\n\nLyrics:\n' + excerpt
    )

    try:
        response = groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        themes = [str(t) for t in data.get("themes", keyword_result.themes)][:3]
        mood = str(data.get("mood", keyword_result.mood))
        confidence = float(data.get("confidence", keyword_result.confidence))
        return ThemeMoodResult(
            themes=themes or keyword_result.themes,
            mood=mood or keyword_result.mood,
            confidence=round(min(1.0, max(0.0, confidence)), 3),
            groq_enriched=True,
            raw_keywords=keyword_result.raw_keywords,
        )
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        # LLM returned unparseable output — fall back silently
        return keyword_result
    except Exception as exc:
        raise ModelInferenceError(
            "Groq theme enrichment failed.",
            context={"original_error": str(exc)},
        ) from exc
