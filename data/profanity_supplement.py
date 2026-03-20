"""
data/profanity_supplement.py
Custom profanity words to supplement better-profanity's built-in list and the
LDNOOBW word list fetched at runtime.

Add words here that are specific to music/broadcasting contexts and are not
already covered by the two primary sources. Keep this list minimal — the
runtime-fetched LDNOOBW list is comprehensive and should handle most cases.
"""

# Words added beyond better-profanity built-in + LDNOOBW English list.
# Use lowercase. better-profanity handles common leetspeak variants automatically.
PROFANITY_SUPPLEMENT: list[str] = []
