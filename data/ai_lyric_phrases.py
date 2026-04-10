"""
data/ai_lyric_phrases.py
Curated list of AI lyric clichés used by the phrase-detector signal (#160).

Phrases are stored lowercase; callers compare against full_text.lower().
This list is intentionally conservative — false positives on legitimate
human lyrics are worse than missed detections for this soft signal.
"""
from __future__ import annotations

AI_LYRIC_PHRASES: frozenset[str] = frozenset({
    "in the shadows of",
    "echoes of yesterday",
    "dancing in the rain",
    "lost in the moment",
    "i am enough",
    "my journey",
    "but deep inside i know",
    "on this journey",
    "you are my everything",
    "in the tapestry of",
    "rise above the storm",
    "where the light finds me",
    "carry on through the night",
    "the world is ours to",
    "i will find my way",
    "face the unknown",
    "we are stronger together",
    "chasing after dreams",
    "brighter days ahead",
})
