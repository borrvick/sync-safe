"""
services/content/_orchestrator.py
Backward-compatibility shim — all logic lives in sub-modules.
"""
from ._authorship import Authorship
from ._pure import (
    _burstiness,
    _repetition_score,
    _rhyme_density,
    _score_signals,
    _unique_word_ratio,
)
from ._scoring import _compute_verdict

__all__ = [
    "Authorship",
    "_burstiness",
    "_unique_word_ratio",
    "_rhyme_density",
    "_repetition_score",
    "_score_signals",
    "_compute_verdict",
]
