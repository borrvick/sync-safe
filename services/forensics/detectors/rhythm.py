"""
services/forensics/detectors/rhythm.py
RhythmAnalyzer — IBI groove, loop cross-correlation, loop autocorrelation,
and sub-beat grid deviation.

All signals are computed from the pre-decoded standard-SR array in _AudioData.
Beat tracking is shared between IBI and spectral slop to avoid redundant work.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import SectionRepetition

from .._audio import _AudioData
from .._bundle import _SignalBundle
from .._pure import (
    _check_spectral_slop,
    _cross_correlate,
    _spectral_fingerprint,
    compute_loop_autocorrelation_score,
    compute_subbeat_grid_deviation,
)

if TYPE_CHECKING:
    from core.models import Section


class RhythmAnalyzer:
    """
    Computes rhythm and loop structure signals.

    Signals populated:
        ibi_variance, spectral_slop, loop_score,
        loop_autocorr_score, subbeat_grid_deviation
    """

    def process(
        self,
        data: _AudioData,
        bundle: _SignalBundle,
        sections: list[Section] | None = None,
    ) -> None:
        import librosa
        # Beat tracking is expensive — run once and share between _groove and _loops.
        tempo, beat_frames = librosa.beat.beat_track(y=data.y, sr=data.sr)
        bundle.ibi_variance, bundle.spectral_slop = self._groove(data.y, data.sr, beat_frames)
        bundle.loop_score                          = self._loops(data.y, data.sr, tempo)
        bundle.loop_window_scores                  = self._loops_windowed(data.y, data.sr, tempo)
        bundle.loop_autocorr_score                 = compute_loop_autocorrelation_score(
            data.y, data.sr,
            CONSTANTS.LOOP_PEAK_COUNT_THRESHOLD,
            CONSTANTS.LOOP_PEAK_SPACING_MAX,
        )
        bundle.subbeat_grid_deviation = compute_subbeat_grid_deviation(data.y, data.sr)
        if sections:
            bundle.section_similarities, bundle.section_internal_repetition = (
                self._loops_by_section(data.y, data.sr, tempo, sections)
            )

    # ------------------------------------------------------------------
    # IBI + spectral slop (shares beat-tracker result)
    # ------------------------------------------------------------------

    def _groove(self, y: np.ndarray, sr: int, beat_frames: np.ndarray) -> tuple[float, float]:
        """
        Compute inter-beat interval variance and spectral slop ratio.

        Args:
            beat_frames: Pre-computed beat frames from process() — avoids a
                         second beat_track call.

        Returns:
            (ibi_variance, spectral_slop)
            ibi_variance = -1.0 when the track is too short to analyse.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        try:
            import librosa

            beat_times_ms   = librosa.frames_to_time(beat_frames, sr=sr) * 1000.0

            if len(beat_times_ms) < 2:
                return -1.0, 0.0

            ibi          = np.diff(beat_times_ms)
            ibi_variance = float(np.var(ibi))
            spectral_slop = _check_spectral_slop(
                y, sr,
                CONSTANTS.SPECTRAL_SLOP_HZ,
                CONSTANTS.SPECTRAL_SLOP_RATIO,
            )
            return ibi_variance, spectral_slop

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Groove analysis failed.",
                context={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Loop cross-correlation
    # ------------------------------------------------------------------

    def _loops(self, y: np.ndarray, sr: int, tempo: np.ndarray) -> float:
        """
        Cross-correlate 4-bar spectral fingerprints across the track.

        Args:
            tempo: Pre-computed tempo from process() — avoids a second
                   beat_track call.

        Returns:
            Maximum cosine similarity (0.0–1.0). 0.0 when track is too short.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        try:
            bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

            if not (CONSTANTS.LOOP_BPM_MIN <= bpm <= CONSTANTS.LOOP_BPM_MAX):
                return 0.0

            segment_len = int((60.0 / bpm) * CONSTANTS.BEATS_PER_WINDOW * sr)
            if segment_len > len(y):
                return 0.0

            segments     = [y[i : i + segment_len] for i in range(0, len(y) - segment_len, segment_len)]
            if len(segments) < 2:
                return 0.0

            fingerprints = [_spectral_fingerprint(s) for s in segments]

            max_score = 0.0
            for i in range(len(fingerprints)):
                for j in range(i + 1, len(fingerprints)):
                    score = _cross_correlate(fingerprints[i], fingerprints[j])
                    if score > max_score:
                        max_score = score

            return max_score

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Loop detection failed.",
                context={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Per-window loop scores (#142)
    # ------------------------------------------------------------------

    def _loops_windowed(
        self,
        y: np.ndarray,
        sr: int,
        tempo: np.ndarray,
    ) -> list[tuple[float, float]]:
        """
        Return per-4-bar-window loop scores for heatmap rendering (#142).

        Returns:
            List of (start_s, max_pairwise_similarity) per non-overlapping
            4-bar window.  Empty when the track is too short or BPM is out
            of range.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        try:
            bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
            if not (CONSTANTS.LOOP_BPM_MIN <= bpm <= CONSTANTS.LOOP_BPM_MAX):
                return []

            segment_len = int((60.0 / bpm) * CONSTANTS.BEATS_PER_WINDOW * sr)
            if segment_len > len(y):
                return []

            windows: list[tuple[float, float]] = []
            for i in range(0, len(y) - segment_len, segment_len):
                seg = y[i : i + segment_len]
                fp  = _spectral_fingerprint(seg)
                # Compare this window against all subsequent windows
                max_score = 0.0
                for j in range(i + segment_len, len(y) - segment_len, segment_len):
                    other_score = _cross_correlate(fp, _spectral_fingerprint(y[j : j + segment_len]))
                    if other_score > max_score:
                        max_score = other_score
                windows.append((round(i / sr, 3), round(max_score, 4)))

            return windows

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Windowed loop detection failed.",
                context={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Section-aware repetition: inter + intra (#143, #145)
    # ------------------------------------------------------------------

    def _loops_by_section(
        self,
        y: np.ndarray,
        sr: int,
        tempo: np.ndarray,
        sections: list[Section],
    ) -> tuple[dict[str, SectionRepetition], dict[str, SectionRepetition]]:
        """
        Compute inter-section and intra-section repetition scores (#143, #145).

        Inter-section: compare same-label sections against each other.
        Intra-section: compare 2-bar sub-windows within each section instance.

        Labels with only one valid section are omitted from inter results.
        Sections shorter than SECTION_MIN_DURATION_S are skipped entirely.

        Returns:
            (section_similarities, section_internal_repetition)
            Both are dict[label, SectionRepetition].  Keys are lowercase labels.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        if not sections:
            return {}, {}

        try:
            bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
            if not (CONSTANTS.LOOP_BPM_MIN <= bpm <= CONSTANTS.LOOP_BPM_MAX):
                return {}, {}

            min_samples   = int(CONSTANTS.SECTION_MIN_DURATION_S * sr)
            trim_samples  = int(CONSTANTS.SECTION_BOUNDARY_TRIM_S * sr)

            # Build slices grouped by normalised label
            slices_by_label: dict[str, list[np.ndarray]] = {}
            for sec in sections:
                label = sec.label.lower()
                start = int(sec.start * sr)
                end   = int(sec.end   * sr)
                # Trim 50 ms from each boundary
                start = min(start + trim_samples, end)
                end   = max(end   - trim_samples, start)
                seg   = y[start:end]
                if len(seg) < min_samples:
                    continue
                slices_by_label.setdefault(label, []).append(seg)

            # --- Inter-section similarity (#143) ---
            inter: dict[str, SectionRepetition] = {}
            for label, slices in slices_by_label.items():
                if len(slices) < 2:
                    continue  # can't compare a section to itself — omit key
                fps = [_spectral_fingerprint(s) for s in slices]
                scores: list[float] = []
                for i in range(len(fps)):
                    for j in range(i + 1, len(fps)):
                        scores.append(_cross_correlate(fps[i], fps[j]))
                inter[label] = SectionRepetition(
                    max_similarity  = round(max(scores), 4),
                    mean_similarity = round(float(np.mean(scores)), 4),
                    pair_count      = len(scores),
                )

            # --- Intra-section internal repetition (#145) ---
            sub_win_len = int((60.0 / bpm) * CONSTANTS.INTERNAL_LOOP_BEATS_PER_WINDOW * sr)
            intra: dict[str, SectionRepetition] = {}
            # Collect best intra score per label (max across all instances of that label)
            intra_best: dict[str, list[float]] = {}
            for sec in sections:
                label = sec.label.lower()
                start = int(sec.start * sr)
                end   = int(sec.end   * sr)
                start = min(start + trim_samples, end)
                end   = max(end   - trim_samples, start)
                seg   = y[start:end]
                if len(seg) < min_samples or sub_win_len > len(seg):
                    continue
                sub_windows = [
                    seg[k : k + sub_win_len]
                    for k in range(0, len(seg) - sub_win_len, sub_win_len)
                ]
                if len(sub_windows) < 2:
                    continue
                fps = [_spectral_fingerprint(sw) for sw in sub_windows]
                scores = []
                for i in range(len(fps)):
                    for j in range(i + 1, len(fps)):
                        scores.append(_cross_correlate(fps[i], fps[j]))
                intra_best.setdefault(label, []).extend(scores)

            for label, scores in intra_best.items():
                intra[label] = SectionRepetition(
                    max_similarity  = round(max(scores), 4),
                    mean_similarity = round(float(np.mean(scores)), 4),
                    pair_count      = len(scores),
                )

            return inter, intra

        except (OSError, ValueError, RuntimeError) as exc:
            raise ModelInferenceError(
                "Section-aware loop analysis failed.",
                context={"original_error": str(exc)},
            ) from exc
