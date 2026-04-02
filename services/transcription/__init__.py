from ._orchestrator import LyricsOrchestrator, Transcription
from ._pure import (
    _collapse_intra_repetitions,
    _gaps_are_uniform,
    _parse_segments,
    _strip_repetition_runs,
)

__all__ = [
    "LyricsOrchestrator",
    "Transcription",
    "_parse_segments",
    "_collapse_intra_repetitions",
    "_strip_repetition_runs",
    "_gaps_are_uniform",
]
