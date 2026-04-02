"""
services/analysis/_allin1.py
allin1-specific helpers: monkey-patch and result extraction.
"""
from __future__ import annotations

from core.models import Section

from ._pure import _parse_sections


def _patch_allin1_metrical() -> None:
    """
    Monkey-patch allin1.helpers.postprocess_metrical_structure so that a
    madmom DBN failure (e.g. "inhomogeneous shape" on some tracks) returns
    empty beats instead of crashing — letting allin1 still return segments.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    try:
        import allin1.helpers as _h

        if getattr(_h, "_metrical_patched", False):
            return

        _original = _h.postprocess_metrical_structure

        def _safe_postprocess(logits: object, cfg: object) -> dict:
            try:
                return _original(logits, cfg)
            except Exception:
                return {"beats": [], "downbeats": [], "beat_positions": []}

        _h.postprocess_metrical_structure = _safe_postprocess
        _h._metrical_patched = True
    except Exception:
        pass  # If patching fails, let allin1 run unpatched


def _extract_allin1_results(
    result: object, raw: bytes
) -> tuple[float | str, list[Section], list[float]]:
    """
    Extract BPM, sections, and beats from an allin1 AnalysisResult.

    When allin1 returns empty beats (madmom DBN produced nothing), falls back
    to librosa beat_track so BPM and beats are always populated.

    Args:
        result: allin1.typings.AnalysisResult
        raw:    original audio bytes (used for librosa fallback)

    Returns:
        (bpm, sections, beats)
    """
    sections = _parse_sections(result)
    beats = (
        [float(b) for b in result.beats]
        if hasattr(result, "beats") and result.beats
        else []
    )
    bpm: float | str = (
        float(result.bpm)
        if hasattr(result, "bpm") and result.bpm is not None
        else "N/A"
    )

    # Fallback: if allin1's madmom beat tracker returned nothing, use librosa.
    if not beats:
        try:
            import io as _io
            import librosa

            audio_np, sr = librosa.load(_io.BytesIO(raw), sr=None, mono=True)
            tempo, beat_frames = librosa.beat.beat_track(y=audio_np, sr=sr)
            bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
            beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        except Exception:
            pass  # librosa fallback failed; leave beats=[] and bpm="N/A"

    return bpm, sections, beats
