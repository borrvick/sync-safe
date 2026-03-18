"""
services/forensics.py
Forensic AI-origin detection — implements the ForensicsAnalyzer protocol.

Four detection signals, each independent:
  1. C2PA manifest check   — cryptographic provenance (no librosa needed)
  2. IBI groove analysis   — inter-beat interval variance via librosa beat tracking
  3. Loop detection        — 4-bar spectral fingerprint cross-correlation
  4. SynthID-style scan    — phase coherence in 18–22 kHz band (44.1 kHz load)

Design rules:
- Forensics.analyze() is the single public entry point.
- Each check populates specific numeric fields on ForensicsResult; _compute_verdict
  compares those numbers against CONSTANTS — no string matching on human labels.
- Hard failures (librosa OOM, unexpected shapes) raise ModelInferenceError.
- C2PA degrades gracefully: "no manifest" is the normal case for most tracks.
- Pure helper functions live at module level for independent unit testing.

Implementation constants (STFT params, phase thresholds) are named module-level
variables prefixed with _ . Business-level thresholds live in core/config.CONSTANTS.
"""
from __future__ import annotations

import io
import json

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import AudioBuffer, ForensicVerdict, ForensicsResult

try:
    import spaces
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn):
            return fn


# ---------------------------------------------------------------------------
# Implementation constants — technical parameters, not business thresholds
# ---------------------------------------------------------------------------

_SYNTHID_LOAD_SR: int   = 44_100   # must be 44.1 kHz — Nyquist = 22.05 kHz
_SYNTHID_N_FFT: int     = 8_192    # ~5.4 Hz/bin at 44.1 kHz
_SYNTHID_HOP: int       = 2_048
_SYNTHID_PHASE_STD_MAX: float  = 0.10   # radians; below this = phase-locked bin
_SYNTHID_MAG_SPIKE_DB: float   = 12.0   # dB above local floor to count as spike


class Forensics:
    """
    Detects signals suggesting AI-generated or stock audio.

    Implements: ForensicsAnalyzer protocol (core/protocols.py)

    Usage:
        service = Forensics()
        result  = service.analyze(audio_buffer)
        print(result.verdict, result.flags)
    """

    # ------------------------------------------------------------------
    # Public interface (ForensicsAnalyzer protocol)
    # ------------------------------------------------------------------

    def analyze(self, audio: AudioBuffer) -> ForensicsResult:
        """
        Run all four forensic checks and aggregate into a ForensicsResult.

        Args:
            audio: In-memory audio buffer from Ingestion.

        Returns:
            ForensicsResult with per-signal scores, human-readable flags,
            and an overall verdict.

        Raises:
            ModelInferenceError: if librosa raises an unrecoverable error.
            C2PA errors degrade gracefully — they never propagate.
        """
        raw = audio.raw

        c2pa_flag, c2pa_label              = self._check_c2pa(raw)
        ibi_variance, spectral_slop        = self._analyse_groove(raw)
        loop_score                          = self._detect_loops(raw)
        synthid_bins                        = self._check_synthid(raw)

        flags = _build_flags(
            c2pa_label=c2pa_label,
            ibi_variance=ibi_variance,
            spectral_slop=spectral_slop,
            loop_score=loop_score,
            synthid_bins=synthid_bins,
        )
        verdict = _compute_verdict(
            c2pa_flag=c2pa_flag,
            ibi_variance=ibi_variance,
            loop_score=loop_score,
            synthid_bins=synthid_bins,
        )

        return ForensicsResult(
            c2pa_flag=c2pa_flag,
            ibi_variance=ibi_variance,
            loop_score=loop_score,
            spectral_slop=spectral_slop,
            synthid_score=float(synthid_bins),
            flags=flags,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Private: C2PA manifest check  (CPU — no librosa)
    # ------------------------------------------------------------------

    def _check_c2pa(self, raw: bytes) -> tuple[bool, str]:
        """
        Read C2PA Content Credentials from raw bytes.

        Returns:
            (born_ai, label) where born_ai is True only when a certified
            AI-generation assertion is found in the manifest.
            Gracefully returns (False, description) on any error — a missing
            manifest is normal, not an exception.
        """
        try:
            import c2pa
        except ImportError:
            return False, "c2pa-python not installed"

        try:
            reader = c2pa.Reader.try_create("audio/mpeg", io.BytesIO(raw))
            if reader is None:
                return False, "No C2PA Manifest"

            data = reader.json()
            if isinstance(data, str):
                data = json.loads(data)

            for manifest in (data.get("manifests") or {}).values():
                for assertion in manifest.get("assertions", []):
                    label = assertion.get("label", "")
                    if label in (
                        "c2pa.assertions.ai-generated",
                        "c2pa.assertions.training-mining",
                    ):
                        return True, "Born-AI (Certified)"

            return False, "C2PA Present — No AI Assertion"

        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("no active manifest", "not found", "jumbf")):
                return False, "No C2PA Manifest"
            return False, f"C2PA read error: {exc}"

    # ------------------------------------------------------------------
    # Private: IBI groove analysis  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_groove(self, raw: bytes) -> tuple[float, float]:
        """
        Compute inter-beat interval variance and spectral slop ratio.

        Returns:
            (ibi_variance, spectral_slop_ratio)
            ibi_variance = -1.0 when the track is too short to analyse.

        Raises:
            ModelInferenceError: on librosa load or beat-tracking failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True
            )

            _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
            beat_times_ms = librosa.frames_to_time(beat_frames, sr=sr) * 1000.0

            if len(beat_times_ms) < 2:
                return -1.0, 0.0

            ibi = np.diff(beat_times_ms)
            ibi_variance = float(np.var(ibi))
            spectral_slop = _check_spectral_slop(
                audio, sr, CONSTANTS.SPECTRAL_SLOP_HZ, CONSTANTS.SPECTRAL_SLOP_RATIO
            )
            return ibi_variance, spectral_slop

        except Exception as exc:
            raise ModelInferenceError(
                "Groove analysis failed.",
                context={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Private: Loop detection  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _detect_loops(self, raw: bytes) -> float:
        """
        Cross-correlate 4-bar spectral fingerprints across the track.

        Returns:
            Maximum cosine similarity score (0.0–1.0).
            0.0 when the track is too short or BPM is out of range.

        Raises:
            ModelInferenceError: on librosa failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True
            )
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])

            if not (CONSTANTS.LOOP_BPM_MIN <= bpm <= CONSTANTS.LOOP_BPM_MAX):
                return 0.0

            segment_len = int(
                (60.0 / bpm) * CONSTANTS.BEATS_PER_WINDOW // CONSTANTS.BEATS_PER_WINDOW
                * CONSTANTS.BEATS_PER_WINDOW
                * (sr / CONSTANTS.SAMPLE_RATE)
            )
            # simpler: 4 bars × 4 beats/bar × seconds/beat × samples/second
            segment_len = int((60.0 / bpm) * 4 * 4 * sr)

            if segment_len > len(audio):
                return 0.0

            segments = [
                audio[i: i + segment_len]
                for i in range(0, len(audio) - segment_len, segment_len)
            ]
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

        except Exception as exc:
            raise ModelInferenceError(
                "Loop detection failed.",
                context={"original_error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Private: SynthID-style phase coherence scan  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _check_synthid(self, raw: bytes) -> int:
        """
        Scan the 18–22 kHz band for phase-coherent bins that may indicate
        an AI watermark. Loads at 44.1 kHz so Nyquist = 22.05 kHz.

        Returns:
            Number of coherent bins found (0 = no signal).

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa
            from scipy.ndimage import uniform_filter1d

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=_SYNTHID_LOAD_SR, mono=True
            )
        except Exception as exc:
            raise ModelInferenceError(
                "SynthID scan: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        if len(audio) < _SYNTHID_LOAD_SR:   # need at least 1 second
            return 0

        try:
            stft  = librosa.stft(audio, n_fft=_SYNTHID_N_FFT, hop_length=_SYNTHID_HOP)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=_SYNTHID_N_FFT)

            hf_mask = (freqs >= CONSTANTS.SYNTHID_BAND_LOW_HZ) & (
                freqs <= CONSTANTS.SYNTHID_BAND_HIGH_HZ
            )
            if not np.any(hf_mask):
                return 0

            hf_stft  = stft[hf_mask]
            hf_phase = np.angle(hf_stft)
            hf_mag   = np.abs(hf_stft)

            # Circular standard deviation of phase across time frames
            cos_mean  = np.mean(np.cos(hf_phase), axis=1)
            sin_mean  = np.mean(np.sin(hf_phase), axis=1)
            R         = np.clip(
                np.sqrt(cos_mean ** 2 + sin_mean ** 2), 1e-9, 1.0 - 1e-9
            )
            phase_std = np.sqrt(-2.0 * np.log(R))

            mean_mag_db    = 20.0 * np.log10(np.mean(hf_mag, axis=1) + 1e-9)
            local_floor_db = uniform_filter1d(mean_mag_db, size=50, mode="nearest")

            coherent = (phase_std < _SYNTHID_PHASE_STD_MAX) & (
                (mean_mag_db - local_floor_db) >= _SYNTHID_MAG_SPIKE_DB
            )
            return int(np.sum(coherent))

        except Exception as exc:
            raise ModelInferenceError(
                "SynthID scan: STFT analysis failed.",
                context={"original_error": str(exc)},
            ) from exc


# ---------------------------------------------------------------------------
# Module-level pure functions — independently testable
# ---------------------------------------------------------------------------

def _check_spectral_slop(
    audio: np.ndarray,
    sr: int,
    threshold_hz: int,
    ratio_threshold: float,
) -> float:
    """
    Return the ratio of HF energy (above threshold_hz) to total energy.

    Pure function — no side effects, no I/O.
    Returns a float in [0.0, 1.0]; the caller decides whether to flag it.
    """
    fft   = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    total = np.sum(fft) + 1e-9
    hf    = np.sum(fft[freqs >= threshold_hz])
    return float(hf / total)


def _spectral_fingerprint(segment: np.ndarray) -> np.ndarray:
    """Return an L2-normalised magnitude spectrum for a waveform segment."""
    fft  = np.abs(np.fft.rfft(segment))
    norm = np.linalg.norm(fft)
    return fft / (norm + 1e-9)


def _cross_correlate(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity of two L2-normalised fingerprint vectors.

    O(n) dot product — fingerprints from _spectral_fingerprint are already
    unit-normalised so dot product equals cosine similarity.
    """
    n = min(len(a), len(b))
    return float(np.dot(a[:n], b[:n]))


def _synthid_confidence(coherent_bins: int) -> str:
    """Map a coherent-bin count to a confidence label string."""
    if coherent_bins == 0:
        return "none"
    if coherent_bins <= CONSTANTS.SYNTHID_LOW_BINS:
        return "low"
    if coherent_bins <= CONSTANTS.SYNTHID_MEDIUM_BINS:
        return "medium"
    return "high"


def _build_flags(
    c2pa_label: str,
    ibi_variance: float,
    spectral_slop: float,
    loop_score: float,
    synthid_bins: int,
) -> list[str]:
    """
    Build the list of human-readable flag strings for the UI.

    Pure function — separated from verdict logic so each can be tested
    independently. Verdict uses numeric comparisons; flags use display text.
    """
    flags: list[str] = []

    if "Born-AI" in c2pa_label:
        flags.append(c2pa_label)

    if ibi_variance < 0:
        flags.append("Insufficient data for groove analysis")
    elif ibi_variance < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        flags.append("Perfect Quantization (AI signal)")
    elif ibi_variance > CONSTANTS.IBI_ERRATIC_MIN:
        flags.append("Erratic Humanization (AI signal)")

    if spectral_slop > CONSTANTS.SPECTRAL_SLOP_RATIO:
        flags.append(f"Spectral Slop detected ({spectral_slop:.1%} HF energy)")

    if isinstance(loop_score, float) and loop_score > CONSTANTS.LOOP_SCORE_CEILING:
        flags.append(f"Likely Stock Loop/Sample (score {loop_score:.3f})")
    elif isinstance(loop_score, float) and loop_score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        flags.append(f"Possible Repetition (score {loop_score:.3f})")

    conf = _synthid_confidence(synthid_bins)
    if conf == "high":
        flags.append(f"High-confidence AI watermark ({synthid_bins} coherent bins in 18–22 kHz)")
    elif conf == "medium":
        flags.append(f"Medium-confidence AI watermark ({synthid_bins} coherent bins in 18–22 kHz)")
    elif conf == "low":
        flags.append(f"Low-confidence HF phase anomaly ({synthid_bins} bin{'s' if synthid_bins > 1 else ''}) — monitor")

    return flags


def _compute_verdict(
    c2pa_flag: bool,
    ibi_variance: float,
    loop_score: float,
    synthid_bins: int,
) -> ForensicVerdict:
    """
    Aggregate numeric forensic scores into a final verdict.

    Rules (ordered by certainty):
      - Certified C2PA AI assertion → AI (hard evidence)
      - High-confidence SynthID     → AI (hard evidence)
      - Two or more soft signals    → AI
      - One soft signal             → Uncertain
      - No signals                  → Human

    Pure function — compares numbers against CONSTANTS, no string matching.
    """
    if c2pa_flag:
        return "AI"
    if _synthid_confidence(synthid_bins) == "high":
        return "AI"

    ai_signals = 0
    if 0.0 <= ibi_variance < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        ai_signals += 1
    if ibi_variance > CONSTANTS.IBI_ERRATIC_MIN:
        ai_signals += 1
    if loop_score > CONSTANTS.LOOP_SCORE_CEILING:
        ai_signals += 1
    if _synthid_confidence(synthid_bins) in ("medium", "low"):
        ai_signals += 1

    if ai_signals >= 2:
        return "AI"
    if ai_signals == 1:
        return "Uncertain"
    return "Human"
