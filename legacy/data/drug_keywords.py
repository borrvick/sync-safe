"""
data/drug_keywords.py
Drug and substance reference keywords for lyric compliance scanning.

Used alongside Detoxify toxicity scoring — a segment is flagged DRUGS only
when BOTH a drug keyword matches AND Detoxify toxicity score exceeds the
configured threshold. This two-gate approach prevents false positives from
medical or metaphorical usage.

Categories:
    - Cannabis slang
    - Stimulant and amphetamine slang
    - Opioid and prescription drug slang
    - Cocaine and crack slang
    - Party / MDMA slang
    - General intoxication slang
    - Cough syrup / lean slang
    - Methamphetamine slang
"""

DRUG_KEYWORDS: frozenset[str] = frozenset({
    # Cannabis
    "weed",
    "blunt",
    "joint",
    "stoned",

    # Stimulants / amphetamines
    "adderall",
    "roll",
    "rolling",
    "rolled",

    # Opioids / prescription
    "percocet",
    "oxy",
    "xan",
    "xanax",
    "lean",
    "codeine",
    "syrup",
    "pill",
    "pills",

    # Cocaine / crack
    "coke",
    "cocaine",
    "crack",
    "smack",
    "dope",

    # MDMA / party drugs
    "molly",
    "mdma",

    # Methamphetamine
    "meth",

    # General intoxication (context-dependent — gated by Detoxify score)
    "high",
    "trip",
    "tripping",
})
