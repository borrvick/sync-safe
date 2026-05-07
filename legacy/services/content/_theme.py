"""
services/content/_theme.py
Theme and mood analysis — implements the ThemeMoodAnalyzer protocol.

Two-tier strategy:
  1. Keyword taxonomy fast-path (always runs, no API required)
     Scores each theme using keyword (3 pts), synonym (1 pt), and intensity
     modifier (+2 bonus) matches with negation handling.
  2. On-demand Groq enrichment (gated by Settings.groq_api_key, user toggle)
     Sends a compact lyric summary to Groq for a richer mood_summary.
     Exposed as ThemeMoodAnalyzer.enrich() — never called automatically.

Design notes:
  - Constructor injection: pass groq_client=None to force keyword-only mode.
  - All string constants live in _labels.py — none inline here.
  - ModelInferenceError raised on Groq API failure; caller receives keyword
    result as the fallback so the pipeline never drops Theme & Mood entirely.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from core.config import CONSTANTS, get_settings
from core.exceptions import ModelInferenceError
from core.models import ThemeMoodResult, TranscriptSegment

from ._labels import MOOD_LABELS, NEGATION_TOKENS, THEME_TAXONOMY

# Empirical word-frequency score ceiling for mood confidence normalisation (0–1)
_CONFIDENCE_CEILING: float = 0.05

# Points per match type
_PTS_KEYWORD:  int = 3
_PTS_SYNONYM:  int = 1
_PTS_MODIFIER: int = 2   # bonus added when modifier found near a keyword hit


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
        # On-demand Groq enrichment (user triggered):
        enriched = service.enrich(result, full_lyrics_text)
    """

    def __init__(self, groq_client=None) -> None:
        self._groq = groq_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, transcript: list[TranscriptSegment]) -> ThemeMoodResult:
        """
        Return a ThemeMoodResult from keyword analysis only.

        Groq enrichment is intentionally NOT called here — it is on-demand
        via enrich() to avoid incurring API cost on every analysis (#169).

        Args:
            transcript: Whisper/LRCLib segments.

        Returns:
            ThemeMoodResult with themes, mood, confidence, theme_scores,
            top_category, and raw_keywords. mood_summary is always None
            until enrich() is called.
        """
        text = " ".join(seg.text for seg in transcript if seg.text.strip())
        return _keyword_analyze(text)

    def enrich(
        self,
        result: ThemeMoodResult,
        transcript_text: str,
    ) -> ThemeMoodResult:
        """
        Call Groq to generate mood_summary and optionally refine themes (#169).

        Only called when the user explicitly requests it (UI toggle). Returns
        the original result unchanged if Groq is not configured or fails.

        Args:
            result:          Existing keyword-only ThemeMoodResult.
            transcript_text: Full joined transcript for Groq prompt.

        Returns:
            New ThemeMoodResult with mood_summary populated (and groq_enriched=True).
            Falls back to original result on any API or parse error.

        Raises:
            ModelInferenceError: if the Groq API call errors (non-parse failure).
        """
        if self._groq is None:
            self._groq = _try_init_groq()
        if self._groq is None:
            return result

        return _groq_enrich(self._groq, transcript_text, result)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _keyword_analyze(text: str) -> ThemeMoodResult:
    """
    Score mood and themes using weighted keyword matching with negation handling.

    Scoring per theme:
      Exact keyword match (whole word)   → _PTS_KEYWORD pts each
      Synonym match                      → _PTS_SYNONYM pt each
      Intensity modifier (near keyword)  → +_PTS_MODIFIER bonus (per theme, not per word)

    Negation: if any NEGATION_TOKENS appear within THEME_NEGATION_WINDOW tokens
    before a keyword or synonym match, the match is penalised by _PTS_KEYWORD pts
    (floor 0 — never goes negative).

    Score normalised by max(word_count, 50) to prevent repetitive short tracks
    from inflating confidence.

    Pure function — no I/O, no API calls.
    """
    if not text.strip():
        return ThemeMoodResult(
            themes=[],
            mood="Neutral",
            confidence=0.0,
            groq_enriched=False,
            raw_keywords=[],
            theme_scores={},
            top_category="",
            mood_summary=None,
        )

    words       = text.lower().split()
    word_freq   = Counter(words)
    # Normalise floor prevents inflation on short/repetitive tracks
    total_words = max(len(words), 50)

    # --- Mood scoring (flat keyword list, no negation — mood is holistic) ---
    mood_scores: dict[str, float] = {}
    for mood, keywords in MOOD_LABELS.items():
        hits = sum(word_freq.get(kw, 0) for kw in keywords if " " not in kw)
        phrase_hits = sum(1 for kw in keywords if " " in kw and kw in text.lower())
        mood_scores[mood] = (hits + phrase_hits * 2) / total_words

    best_mood       = max(mood_scores, key=lambda m: mood_scores[m])
    best_mood_score = mood_scores[best_mood]
    mood            = best_mood if best_mood_score > 0.0 else "Neutral"
    confidence      = round(min(1.0, best_mood_score / _CONFIDENCE_CEILING), 3)

    # --- Theme scoring (rich taxonomy with negation handling) ---
    raw_theme_scores: dict[str, float] = {}

    for theme, data in THEME_TAXONOMY.items():
        raw_pts = 0

        for kw in data["keywords"]:
            if " " in kw:
                for match in re.finditer(re.escape(kw), text.lower()):
                    word_pos = len(text.lower()[:match.start()].split())
                    window   = words[max(0, word_pos - CONSTANTS.THEME_NEGATION_WINDOW):word_pos]
                    if any(t in NEGATION_TOKENS for t in window):
                        raw_pts -= _PTS_KEYWORD
                    else:
                        raw_pts += _PTS_KEYWORD
            else:
                for idx, w in enumerate(words):
                    if w == kw:
                        window = words[max(0, idx - CONSTANTS.THEME_NEGATION_WINDOW):idx]
                        if any(t in NEGATION_TOKENS for t in window):
                            raw_pts -= _PTS_KEYWORD
                        else:
                            raw_pts += _PTS_KEYWORD

        for syn in data["synonyms"]:
            if " " in syn:
                if syn in text.lower():
                    raw_pts += _PTS_SYNONYM
            else:
                raw_pts += word_freq.get(syn, 0) * _PTS_SYNONYM

        # Modifier bonus: applied once per theme when any modifier is present
        if any(
            (mod in text.lower() if " " in mod else mod in words)
            for mod in data["intensity_modifiers"]
        ):
            raw_pts += _PTS_MODIFIER

        raw_theme_scores[theme] = max(0, raw_pts) / total_words

    # Normalise theme scores to 0–1 against the ceiling of the highest raw score
    max_raw = max(raw_theme_scores.values()) if raw_theme_scores else 0.0
    ceiling = max(max_raw, 1e-9)
    theme_scores: dict[str, float] = {
        t: round(s / ceiling, 3)
        for t, s in raw_theme_scores.items()
    }

    # Pick themes above THEME_MIN_CONFIDENCE (post-normalisation)
    ranked = sorted(theme_scores.items(), key=lambda x: -x[1])
    themes = [t for t, s in ranked if s >= CONSTANTS.THEME_MIN_CONFIDENCE][:3]

    # Top category from the highest-scoring theme
    top_category = ""
    if themes:
        top_category = THEME_TAXONOMY[themes[0]].get("category", "")

    # Collect matched raw keywords for transparency
    all_kw: list[str] = []
    for d in THEME_TAXONOMY.values():
        all_kw.extend(d["keywords"])
        all_kw.extend(d["synonyms"])
    for kws in MOOD_LABELS.values():
        all_kw.extend(kws)
    word_set     = set(words)
    raw_keywords = sorted(
        {kw for kw in all_kw if kw in word_set or (" " in kw and kw in text.lower())}
    )[:20]

    return ThemeMoodResult(
        themes=themes,
        mood=mood,
        confidence=confidence,
        groq_enriched=False,
        raw_keywords=raw_keywords,
        theme_scores=theme_scores,
        top_category=top_category,
        mood_summary=None,
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
    Use Groq LLM to generate a mood_summary for the keyword result (#169).

    Sends a compact lyric extract and asks for a JSON-structured summary.
    Falls through to keyword_result on any parse failure.

    Raises:
        ModelInferenceError: if the Groq API call itself errors (not a parse failure).
    """
    excerpt  = text[:CONSTANTS.THEME_GROQ_LYRICS_CAP]
    existing = ", ".join(keyword_result.themes) or "none detected"
    prompt   = (
        "You are a music sync licensing assistant. Given these song lyrics, write:\n"
        "1. A 1–2 sentence mood summary a music supervisor would use in a brief.\n"
        "2. Up to 3 refined theme tags (short phrases).\n\n"
        f"Existing keyword-detected themes: {existing}\n\n"
        f"Lyrics:\n{excerpt}\n\n"
        'Respond as JSON only — no markdown fences:\n'
        '{"themes": ["tag1", "tag2"], "summary": "..."}'
    )

    try:
        response = groq_client.chat.completions.create(
            model=CONSTANTS.THEME_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        raw     = response.choices[0].message.content.strip()
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data    = json.loads(cleaned)

        groq_themes: list[str] = [str(t) for t in data.get("themes", [])][:3]
        summary: str           = str(data.get("summary", ""))[:300].strip()

        # Groq themes take precedence only when keyword themes are absent
        merged_themes = groq_themes if groq_themes else keyword_result.themes

        return ThemeMoodResult(
            themes=merged_themes,
            mood=keyword_result.mood,
            confidence=keyword_result.confidence,
            groq_enriched=True,
            raw_keywords=keyword_result.raw_keywords,
            theme_scores=keyword_result.theme_scores,
            top_category=keyword_result.top_category,
            mood_summary=summary or None,
        )
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        # Unparseable LLM output — fall back silently to keyword result
        return keyword_result
    except Exception as exc:
        raise ModelInferenceError(
            "Groq theme enrichment failed.",
            context={"original_error": str(exc)},
        ) from exc
