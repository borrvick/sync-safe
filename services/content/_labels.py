"""
services/content/_labels.py
Mood and theme taxonomy constants for ThemeMoodAnalyzer.
All string constants live here — never inline in logic code.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Mood labels — flat keyword lists (unchanged scoring model)
# ---------------------------------------------------------------------------

MOOD_LABELS: dict[str, list[str]] = {
    "Uplifting": [
        "rise", "shine", "hope", "alive", "bright", "free", "joy", "win",
        "celebrate", "glory", "soar", "fly", "believe", "dream", "glow",
        "triumph", "smile", "light", "together", "strong",
    ],
    "Melancholic": [
        "cry", "tears", "broken", "miss", "gone", "alone", "pain", "hurt",
        "goodbye", "lost", "leave", "fading", "empty", "sorrow", "grieve",
        "apart", "fade", "cold", "silence", "regret",
    ],
    "Intense": [
        "fire", "rage", "fight", "blood", "war", "power", "crush", "burn",
        "storm", "force", "attack", "destroy", "survive", "battle", "explode",
        "fury", "blast", "danger", "wild", "unstoppable",
    ],
    "Romantic": [
        "love", "heart", "kiss", "together", "forever", "darling", "hold",
        "embrace", "tender", "yours", "mine", "desire", "passion", "adore",
        "cherish", "sweetheart", "devoted", "longing", "warmth", "intimate",
    ],
    "Dark": [
        "dark", "shadow", "night", "death", "fear", "demon", "void", "hollow",
        "ghost", "grave", "haunt", "nightmare", "dread", "abyss", "wicked",
        "curse", "fallen", "bleed", "suffer", "decay",
    ],
    "Chill": [
        "easy", "drift", "slow", "flow", "gentle", "calm", "peace", "rest",
        "mellow", "breeze", "smooth", "float", "unwind", "sway", "lazy",
        "hazy", "quiet", "serene", "lull", "still",
    ],
    "Energetic": [
        "run", "rush", "go", "fast", "move", "pump", "jump", "race", "drive",
        "push", "hustle", "grind", "sprint", "charge", "bounce", "hype",
        "loud", "turn up", "lit", "ignite",
    ],
    "Nostalgic": [
        "remember", "back", "used to", "childhood", "memory", "yesterday",
        "old", "when we", "time", "years", "ago", "once", "before", "past",
        "familiar", "return", "miss", "then", "those days", "grew up",
    ],
}

# ---------------------------------------------------------------------------
# Theme taxonomy — rich structure for scoring (#167)
#
# Each entry has:
#   keywords           — primary matches, 3 pts each
#   synonyms           — secondary matches, 1 pt each
#   intensity_modifiers — bonus +2 pts when co-occurring near a keyword
#   sub_themes         — display labels for sub-categorisation (not scored)
#   category           — one of "energy" | "emotional" | "seasonal"
#                        used for pill color and briefing copy
# ---------------------------------------------------------------------------

THEME_TAXONOMY: dict[str, dict] = {
    "Party & Celebration": {
        "keywords":            ["party", "dance", "night", "celebrate", "drinks", "club",
                                "move", "turn up", "lit", "crowd", "floor", "vibe", "jump"],
        "synonyms":            ["banger", "rave", "groove", "anthem", "hands up", "shots"],
        "intensity_modifiers": ["wild", "crazy", "all night", "on fire", "nonstop"],
        "sub_themes":          ["club_banger", "summer_anthem", "weekend_vibe"],
        "category":            "energy",
    },
    "Love & Romance": {
        "keywords":            ["love", "heart", "kiss", "together", "forever", "darling",
                                "hold", "embrace", "desire", "passion", "adore", "cherish"],
        "synonyms":            ["devotion", "sweetheart", "yours", "mine", "intimate", "warmth"],
        "intensity_modifiers": ["deeply", "completely", "endlessly", "unconditionally"],
        "sub_themes":          ["new_love", "slow_burn", "commitment"],
        "category":            "emotional",
    },
    "Heartbreak & Loss": {
        "keywords":            ["broken", "goodbye", "tears", "apart", "miss", "lost", "gone",
                                "leave", "cry", "empty", "alone", "pain", "hurt", "ending"],
        "synonyms":            ["shattered", "devastated", "ruins", "over", "done", "cold"],
        "intensity_modifiers": ["completely", "utterly", "forever"],
        "sub_themes":          ["breakup", "longing", "moving_on"],
        "category":            "emotional",
    },
    "Empowerment & Resilience": {
        "keywords":            ["rise", "strong", "survive", "fight", "overcome", "stand",
                                "believe", "warrior", "refuse", "power", "champion"],
        "synonyms":            ["fearless", "unstoppable", "relentless", "never give up"],
        "intensity_modifiers": ["absolutely", "always", "never stop"],
        "sub_themes":          ["workout", "comeback", "empowerment"],
        "category":            "energy",
    },
    "Nostalgia & Memory": {
        "keywords":            ["remember", "back then", "childhood", "memory", "yesterday",
                                "used to", "before", "past", "years ago", "grew up"],
        "synonyms":            ["reminisce", "flashback", "those days", "old days", "once"],
        "intensity_modifiers": ["vividly", "clearly", "still", "always will"],
        "sub_themes":          ["coming_of_age", "hometown", "simpler_times"],
        "category":            "emotional",
    },
    "Summery & Carefree": {
        "keywords":            ["summer", "beach", "sun", "vacation", "pool", "waves",
                                "heat", "warm", "holiday", "sand", "ocean"],
        "synonyms":            ["island", "tropical", "paradise", "sunshine", "carefree"],
        "intensity_modifiers": ["blazing", "golden", "endless"],
        "sub_themes":          ["beach_day", "road_trip", "festival"],
        "category":            "seasonal",
    },
    "Ambition & Success": {
        "keywords":            ["hustle", "grind", "top", "money", "goal", "win", "achieve",
                                "rise", "success", "build", "level up", "throne", "rich"],
        "synonyms":            ["empire", "boss", "king", "queen", "crown", "legacy"],
        "intensity_modifiers": ["relentless", "unstoppable", "always"],
        "sub_themes":          ["grind_culture", "wealth", "self_made"],
        "category":            "energy",
    },
    "Nature & Journey": {
        "keywords":            ["road", "river", "mountain", "sky", "stars", "wind", "sun",
                                "travel", "journey", "horizon", "earth", "flow", "wandering"],
        "synonyms":            ["ocean", "forest", "breeze", "open road", "horizon"],
        "intensity_modifiers": ["vast", "endless", "boundless"],
        "sub_themes":          ["road_trip", "adventure", "reflection"],
        "category":            "seasonal",
    },
    "Spirituality & Faith": {
        "keywords":            ["god", "faith", "pray", "heaven", "soul", "blessed", "spirit",
                                "grace", "holy", "lord", "worship"],
        "synonyms":            ["divine", "amen", "hallelujah", "sacred", "higher power"],
        "intensity_modifiers": ["deeply", "truly", "fully"],
        "sub_themes":          ["gospel", "devotion", "transcendence"],
        "category":            "emotional",
    },
    "Identity & Self-Expression": {
        "keywords":            ["myself", "who i am", "my way", "real me", "own", "true",
                                "voice", "authentic", "unique", "freedom", "express"],
        "synonyms":            ["individuality", "unapologetic", "original", "genuine"],
        "intensity_modifiers": ["completely", "proudly", "fearlessly"],
        "sub_themes":          ["self_discovery", "authenticity", "independence"],
        "category":            "emotional",
    },
}

# ---------------------------------------------------------------------------
# Negation tokens — presence within THEME_NEGATION_WINDOW tokens before a
# keyword match reduces its score contribution (#167).
# ---------------------------------------------------------------------------

NEGATION_TOKENS: frozenset[str] = frozenset({"not", "never", "no", "without", "ain't", "neither"})

# ---------------------------------------------------------------------------
# Category → CSS variable color mapping (#168)
# Imported by ui/pages/report.py for pill and bar coloring.
# ---------------------------------------------------------------------------

THEME_CATEGORY_COLORS: dict[str, str] = {
    "energy":   "var(--accent)",
    "emotional": "var(--info)",
    "seasonal": "var(--sync-pass)",
}
