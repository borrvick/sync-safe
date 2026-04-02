"""
services/content/_labels.py
Mood and theme taxonomy constants for ThemeMoodAnalyzer.
All string constants live here — never inline in logic code.
"""
from __future__ import annotations

# Mood labels with representative trigger keywords
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

# Theme taxonomy: category → keywords
THEME_TAXONOMY: dict[str, list[str]] = {
    "Love & Romance": [
        "love", "heart", "kiss", "together", "forever", "darling", "hold",
        "embrace", "desire", "yours", "mine", "romantic", "passion", "adore",
    ],
    "Heartbreak & Loss": [
        "broken", "goodbye", "tears", "apart", "miss", "lost", "gone",
        "leave", "cry", "empty", "alone", "pain", "hurt", "ending",
    ],
    "Empowerment & Resilience": [
        "rise", "strong", "survive", "fight", "overcome", "power", "stand",
        "never give up", "believe", "warrior", "unbreakable", "refuse",
    ],
    "Party & Celebration": [
        "party", "dance", "night", "celebrate", "drinks", "club", "move",
        "turn up", "lit", "crowd", "hands up", "jump", "floor", "vibe",
    ],
    "Identity & Self-Expression": [
        "myself", "who i am", "my way", "real me", "own", "true", "voice",
        "authentic", "unique", "freedom", "express", "individuality",
    ],
    "Nostalgia & Memory": [
        "remember", "back then", "childhood", "memory", "yesterday", "old",
        "used to", "before", "past", "those days", "years ago", "grew up",
    ],
    "Spirituality & Faith": [
        "god", "faith", "pray", "heaven", "soul", "blessed", "spirit",
        "divine", "grace", "holy", "believe", "lord", "worship", "amen",
    ],
    "Social Commentary": [
        "society", "system", "war", "justice", "rights", "truth", "change",
        "revolution", "power", "oppression", "freedom", "voice", "stand up",
    ],
    "Nature & Journey": [
        "road", "river", "mountain", "sky", "ocean", "stars", "wind", "sun",
        "travel", "journey", "horizon", "earth", "flow", "wandering",
    ],
    "Ambition & Success": [
        "hustle", "grind", "top", "money", "goal", "win", "achieve", "rise",
        "success", "empire", "rich", "build", "level up", "throne",
    ],
}
