"""
core/models/_types.py
Shared Literal type aliases used across all model submodules.
"""
from __future__ import annotations

from typing import Literal

IssueType       = Literal["EXPLICIT", "BRAND", "LOCATION", "VIOLENCE", "DRUGS"]
Confidence      = Literal["confirmed", "potential"]
Severity        = Literal["hard", "soft"]
EndingType      = Literal["sting", "fade", "cut"]
AIVerdict       = Literal["Likely Human", "Uncertain", "Likely AI", "Insufficient data"]
ForensicVerdict = Literal["AI", "Likely AI", "Likely Not AI", "Not AI", "Insufficient data"]
AudioSource     = Literal["youtube", "file", "bandcamp", "soundcloud", "direct"]
