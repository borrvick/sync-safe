"""
services/loudness/_orchestrator.py
Broadcast loudness and dialogue-readiness analysis.
"""
from __future__ import annotations

import io
import logging

import librosa

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import AudioBuffer, AudioQualityResult, Section

from ._pure import (
    _classify_dialogue,
    _classify_loudness,
    _compute_vo_headroom,
    _dialogue_score,
    _genre_lra_context,
    _measure_loudness,
    _measure_section_loudness,
)

_log = logging.getLogger(__name__)


class AudioQualityAnalyzer:
    """
    Measures broadcast loudness and dialogue-readiness of an audio track.

    Usage:
        result = AudioQualityAnalyzer().analyze(buffer)
        print(result.integrated_lufs, result.dialogue_label)
    """

    def analyze(
        self,
        buffer: AudioBuffer,
        sections: list[Section] | None = None,
        genre: str | None = None,
    ) -> AudioQualityResult:
        """
        Run loudness + dialogue analysis on an AudioBuffer.

        Args:
            buffer:   AudioBuffer with raw audio bytes.
            sections: Optional list of structural sections from allin1. When
                      provided, per-section LUFS and LRA are computed (#96).
            genre:    Optional genre string (e.g. "hip-hop", "cinematic"). When
                      provided, a soft LRA context note is computed (#99).

        Returns:
            AudioQualityResult with LUFS, true peak, LRA, dialogue score,
            per-section loudness breakdown, and genre-aware LRA context.

        Raises:
            ModelInferenceError: if audio is too short or malformed.
        """
        try:
            y, sr = librosa.load(io.BytesIO(buffer.raw), sr=None, mono=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "AudioQualityAnalyzer: failed to decode audio.",
                context={"original_error": str(exc)},
            ) from exc

        duration = len(y) / sr
        if duration < CONSTANTS.LUFS_MIN_DURATION_S:
            raise ModelInferenceError(
                f"AudioQualityAnalyzer: audio too short ({duration:.1f}s < "
                f"{CONSTANTS.LUFS_MIN_DURATION_S}s minimum for LUFS measurement).",
                context={"duration_s": duration},
            )

        integrated_lufs, true_peak, lra = _measure_loudness(y, sr)
        diag_score       = _dialogue_score(y, sr)
        diag_label       = _classify_dialogue(diag_score)
        section_loudness = _measure_section_loudness(y, sr, sections) if sections else []
        lra_context      = _genre_lra_context(genre, lra)

        delta_spotify     = round(integrated_lufs - CONSTANTS.LUFS_SPOTIFY,     1)
        delta_apple_music = round(integrated_lufs - CONSTANTS.LUFS_APPLE_MUSIC, 1)
        delta_youtube     = round(integrated_lufs - CONSTANTS.LUFS_YOUTUBE,     1)
        delta_broadcast   = round(integrated_lufs - CONSTANTS.LUFS_BROADCAST,   1)
        tp_warning        = true_peak > CONSTANTS.TRUE_PEAK_WARN_DBFS

        return AudioQualityResult(
            integrated_lufs=integrated_lufs,
            true_peak_dbfs=true_peak,
            loudness_range_lu=lra,
            delta_spotify=delta_spotify,
            delta_apple_music=delta_apple_music,
            delta_youtube=delta_youtube,
            delta_broadcast=delta_broadcast,
            true_peak_warning=tp_warning,
            gain_spotify_db=round(-delta_spotify, 1),
            gain_apple_music_db=round(-delta_apple_music, 1),
            gain_youtube_db=round(-delta_youtube, 1),
            gain_broadcast_db=round(-delta_broadcast, 1),
            loudness_verdict=_classify_loudness(integrated_lufs, tp_warning, delta_broadcast),
            dialogue_score=diag_score,
            dialogue_label=diag_label,
            vo_headroom_db=_compute_vo_headroom(diag_score, CONSTANTS.VO_HEADROOM_MAX_DB),
            section_loudness=section_loudness,
            genre_lra_context=lra_context,
        )
