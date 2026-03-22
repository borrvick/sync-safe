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
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Signal bundle — carries all per-signal scores between aggregator functions
# ---------------------------------------------------------------------------

@dataclass
class _SignalBundle:
    """All per-signal scores in one place, eliminating 15-param signatures."""
    c2pa_flag: bool = False
    c2pa_label: str = "No C2PA Manifest"
    ibi_variance: float = -1.0
    loop_score: float = 0.0
    loop_autocorr_score: float = 0.0
    centroid_instability_score: float = -1.0
    harmonic_ratio_score: float = -1.0
    synthid_bins: int = 0
    spectral_slop: float = 0.0
    kurtosis_variability: float = -1.0
    decoder_peak_score: float = 0.0
    spectral_centroid_mean: float = -1.0
    self_similarity_entropy: float = -1.0
    noise_floor_ratio: float = -1.0
    onset_strength_cv: float = -1.0
    spectral_flatness_var: float = -1.0
    subbeat_grid_deviation: float = -1.0


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
        Run all forensic checks and aggregate into a ForensicsResult.

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
        loop_autocorr_score                 = self._detect_loops_autocorr(raw)
        centroid_instability_score          = self._analyse_centroid_instability(raw)
        harmonic_ratio_score                = self._analyse_harmonic_ratio(raw)
        synthid_bins                        = self._check_synthid(raw)
        kurtosis_variability                = self._analyse_kurtosis_variability(raw)
        decoder_peak_score                  = self._detect_decoder_peaks(raw)
        spectral_centroid_mean              = self._compute_spectral_centroid_mean(raw)
        self_similarity_entropy             = self._analyse_self_similarity(raw)
        noise_floor_ratio                   = self._analyse_noise_floor(raw)
        onset_strength_cv                   = self._analyse_onset_strength(raw)
        spectral_flatness_var               = self._analyse_spectral_flatness(raw)
        subbeat_grid_deviation              = self._analyse_subbeat_grid(raw)

        bundle = _SignalBundle(
            c2pa_flag=c2pa_flag,
            c2pa_label=c2pa_label,
            ibi_variance=ibi_variance,
            loop_score=loop_score,
            loop_autocorr_score=loop_autocorr_score,
            centroid_instability_score=centroid_instability_score,
            harmonic_ratio_score=harmonic_ratio_score,
            synthid_bins=synthid_bins,
            spectral_slop=spectral_slop,
            kurtosis_variability=kurtosis_variability,
            decoder_peak_score=decoder_peak_score,
            spectral_centroid_mean=spectral_centroid_mean,
            self_similarity_entropy=self_similarity_entropy,
            noise_floor_ratio=noise_floor_ratio,
            onset_strength_cv=onset_strength_cv,
            spectral_flatness_var=spectral_flatness_var,
            subbeat_grid_deviation=subbeat_grid_deviation,
        )
        ai_probability = _compute_ai_probability(bundle)
        flags          = _build_flags(bundle)
        verdict        = _compute_verdict(bundle)

        result = ForensicsResult(
            c2pa_flag=c2pa_flag,
            ibi_variance=ibi_variance,
            loop_score=loop_score,
            loop_autocorr_score=loop_autocorr_score,
            spectral_slop=spectral_slop,
            synthid_score=float(synthid_bins),
            centroid_instability_score=centroid_instability_score,
            harmonic_ratio_score=harmonic_ratio_score,
            kurtosis_variability=kurtosis_variability,
            decoder_peak_score=decoder_peak_score,
            spectral_centroid_mean=spectral_centroid_mean,
            ai_probability=ai_probability,
            flags=flags,
            verdict=verdict,
            self_similarity_entropy=self_similarity_entropy,
            noise_floor_ratio=noise_floor_ratio,
            onset_strength_cv=onset_strength_cv,
            spectral_flatness_var=spectral_flatness_var,
            subbeat_grid_deviation=subbeat_grid_deviation,
        )

        return result

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

            segment_len = int((60.0 / bpm) * CONSTANTS.BEATS_PER_WINDOW * sr)

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
    # Private: Autocorrelation loop detection  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _detect_loops_autocorr(self, raw: bytes) -> float:
        """
        Detect regular loop repetition via onset-envelope autocorrelation.

        Returns:
            Loop autocorrelation score (0.0–1.0); 1.0 = strong regular repetition.
            Delegates all computation to the module-level pure function.

        Raises:
            ModelInferenceError: on librosa load or autocorrelation failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Loop autocorrelation: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_loop_autocorrelation_score(
            audio,
            sr,
            CONSTANTS.LOOP_PEAK_COUNT_THRESHOLD,
            CONSTANTS.LOOP_PEAK_SPACING_MAX,
        )

    # ------------------------------------------------------------------
    # Private: Spectral centroid instability  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_centroid_instability(self, raw: bytes) -> float:
        """
        Detect formant drift within sustained notes via spectral centroid CV.

        AI vocoders shift upper partials mid-note (the "glassy/hollow" artifact).
        This shows up as high coefficient-of-variation of the spectral centroid
        within each non-silent interval. Human vibrato modulates all partials
        together so the centroid stays relatively stable.

        Returns:
            Mean within-interval centroid CV across all sustained intervals.
            -1.0 when no qualifying intervals are found.
            Delegates all computation to compute_centroid_instability_score().

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Centroid instability: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_centroid_instability_score(
            audio,
            sr,
            CONSTANTS.CENTROID_TOP_DB,
            CONSTANTS.CENTROID_MIN_INTERVAL_S,
        )

    # ------------------------------------------------------------------
    # Private: Harmonic-to-noise ratio within sustained intervals  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_harmonic_ratio(self, raw: bytes) -> float:
        """
        Measure harmonic purity within sustained intervals via HPSS.

        AI generators produce unnaturally clean harmonic content — no breath,
        reed noise, or physical resonance artifacts. High harmonic_energy /
        total_energy within sustained notes is an AI signal.

        Returns:
            Mean HNR across qualifying intervals in [0.0, 1.0].
            -1.0 when no qualifying intervals are found.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(
                io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True
            )
        except Exception as exc:
            raise ModelInferenceError(
                "Harmonic ratio: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_harmonic_ratio_score(
            audio,
            sr,
            CONSTANTS.CENTROID_TOP_DB,
            CONSTANTS.CENTROID_MIN_INTERVAL_S,
        )

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

    # ------------------------------------------------------------------
    # Private: Chroma self-similarity entropy  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_self_similarity(self, raw: bytes) -> float:
        """
        Compute Shannon entropy of the chroma recurrence matrix.

        Low entropy = blocky, repetitive structure typical of AI transformer
        attention patterns. High entropy = varied, organic song structure.

        Returns:
            Entropy in bits (0.0–log2(20) ≈ 4.32). -1.0 if too short.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Self-similarity: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_self_similarity_entropy(audio, sr)

    # ------------------------------------------------------------------
    # Private: Noise floor ratio  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_noise_floor(self, raw: bytes) -> float:
        """
        Measure the ratio of quiet-moment RMS to mean RMS.

        VST renders have digital silence (ratio ≈ 0) between notes;
        real recordings always have a room/mic noise floor.

        Returns:
            Ratio in [0.0, 1.0]. -1.0 if audio is too quiet to analyse.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Noise floor: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_noise_floor_ratio(audio)

    # ------------------------------------------------------------------
    # Private: Onset strength CV  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_onset_strength(self, raw: bytes) -> float:
        """
        Measure the coefficient of variation of onset strengths.

        Low CV = uniform hit strength = AI (no performance dynamics).
        High CV = varied dynamics = human (ghost notes, accents, expression).

        Returns:
            CV (std/mean) of onset strength envelope. -1.0 if not computable.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Onset strength: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_onset_strength_cv(audio, sr)

    # ------------------------------------------------------------------
    # Private: Spectral flatness variance  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_spectral_flatness(self, raw: bytes) -> float:
        """
        Measure the variance of Wiener entropy (spectral flatness) over time.

        Low variance = uniform spectral flatness = AI synthesizer (no physical
        noise source between notes). High variance = real recording (tonal
        notes interspersed with noise — breath, bow, room).

        Returns:
            Variance of spectral flatness frames. -1.0 if not computable.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Spectral flatness: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_spectral_flatness_variance(audio)

    # ------------------------------------------------------------------
    # Private: Sub-beat grid deviation  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_subbeat_grid(self, raw: bytes) -> float:
        """
        Measure the variance of onset-to-nearest-16th-note-grid offsets.

        Low variance = onsets land precisely on the grid = AI generation
        or heavily quantized production (weak signal for modern pop/hip-hop).
        High variance = human micro-timing feel.

        Returns:
            Variance of normalised grid offsets. -1.0 if BPM undetectable.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Sub-beat grid: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_subbeat_grid_deviation(audio, sr)

    # ------------------------------------------------------------------
    # Private: Mel-band kurtosis variability  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _analyse_kurtosis_variability(self, raw: bytes) -> float:
        """
        Compute the variance of per-frame mel-band kurtosis across time.

        Neural audio codec decoders (EnCodec, DAC) introduce checkerboard
        artifacts in the mel spectrogram — occasional sharp energy spikes in
        specific mel bands. These spikes appear as high per-frame kurtosis
        values that vary wildly between frames, producing very high kurtosis
        variance. Human audio has smooth, consistent mel distributions with
        near-zero kurtosis variance.

        Source: ISMIR TISMIR 2025 (Cros Vila) — Suno: ~1508 ± 1304, Human: ~2.

        Returns:
            Variance of per-frame mel-band kurtosis (float ≥ 0.0).
            -1.0 when audio is too short to analyse.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa
            from scipy.stats import kurtosis as scipy_kurtosis

            audio, sr = librosa.load(io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Kurtosis variability: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_kurtosis_variability(audio, sr, CONSTANTS.KURTOSIS_N_MELS)

    # ------------------------------------------------------------------
    # Private: Neural decoder spectral peak fingerprint  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _detect_decoder_peaks(self, raw: bytes) -> float:
        """
        Detect the periodic spectral peak fingerprint left by transposed
        convolution layers in neural audio vocoders / codec decoders.

        Transposed convolution with stride k periodizes the spectrum of any
        bias component at intervals of f_s / k. Multiple stacked layers
        compound this, creating a comb of regularly-spaced peaks in the
        1–16 kHz range that is absent in natural recordings.

        Source: arXiv 2506.19108 — >99% accuracy on uncompressed audio.

        Returns:
            Score in [0.0, 1.0] — 1.0 = strong periodic peak pattern found.
            Loaded at 44.1 kHz to access the full 1–16 kHz analysis band.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa

            # Load at 44.1 kHz — must resolve peaks up to 16 kHz
            audio, sr = librosa.load(io.BytesIO(raw), sr=_SYNTHID_LOAD_SR, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Decoder peak detection: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        return compute_decoder_peak_score(
            audio,
            sr,
            CONSTANTS.DECODER_PEAK_WINDOW_HZ,
            CONSTANTS.DECODER_PEAK_PROMINENCE_DB,
            CONSTANTS.DECODER_PEAK_REGULARITY_MAX,
            CONSTANTS.DECODER_PEAK_MIN_COUNT,
        )

    # ------------------------------------------------------------------
    # Private: Mean spectral centroid  (GPU)
    # ------------------------------------------------------------------

    @spaces.GPU
    def _compute_spectral_centroid_mean(self, raw: bytes) -> float:
        """
        Compute the mean spectral centroid of the track in Hz.

        AI generators concentrate energy in lower frequency bands; natural
        recordings contain more high-frequency content from room acoustics,
        instrument overtones, and recording noise.

        Source: ISMIR TISMIR 2025 — Suno: 1091 ± 386 Hz, Human: 1501 ± 632 Hz.

        Returns:
            Mean spectral centroid in Hz (float > 0), or -1.0 on failure.

        Raises:
            ModelInferenceError: on librosa load failure.
        """
        try:
            import librosa

            audio, sr = librosa.load(io.BytesIO(raw), sr=CONSTANTS.SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ModelInferenceError(
                "Spectral centroid mean: audio load failed.",
                context={"original_error": str(exc)},
            ) from exc

        try:
            centroid_frames = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            return float(np.mean(centroid_frames))
        except Exception as exc:
            raise ModelInferenceError(
                "Spectral centroid mean: computation failed.",
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


def compute_loop_autocorrelation_score(
    audio: np.ndarray,
    sr: int,
    peak_count_threshold: int,
    peak_spacing_max: int,
) -> float:
    """
    Detect regular loop repetition via onset-envelope autocorrelation.

    Algorithm:
      1. Compute onset envelope from the waveform.
      2. Autocorrelate (lag 1 onward — lag 0 is always the global max).
      3. Normalise to [0, 1] and pick peaks.
      4. If peak count ≥ peak_count_threshold AND mean spacing ≤ peak_spacing_max,
         return a score combining peak density and spacing tightness; else 0.0.

    Args:
        audio:               Mono waveform array.
        sr:                  Sample rate (Hz).
        peak_count_threshold: Minimum peaks required to flag repetition.
        peak_spacing_max:    Mean peak spacing above this (frames) → not a tight loop.

    Returns:
        float in [0.0, 1.0]; 1.0 = strong regular repetition detected.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        # Autocorrelate up to half the signal length; skip lag-0 (always 1.0 after norm)
        autocorr = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        autocorr_tail = autocorr[1:]
        peak_val = np.max(autocorr_tail)
        if peak_val < 1e-9:
            return 0.0
        autocorr_norm = autocorr_tail / peak_val

        peaks = librosa.util.peak_pick(
            autocorr_norm,
            pre_max=10,
            post_max=10,
            pre_avg=10,
            post_avg=10,
            delta=0.05,
            wait=5,
        )

        if len(peaks) < peak_count_threshold:
            return 0.0

        spacings = np.diff(peaks)
        if len(spacings) == 0:
            return 0.0

        mean_spacing = float(np.mean(spacings))
        if mean_spacing > peak_spacing_max:
            return 0.0

        # Score: blend peak density and spacing tightness, each capped at 1.0
        count_score = min(float(len(peaks)) / float(peak_count_threshold * 3), 1.0)
        spacing_score = 1.0 - min(mean_spacing / float(peak_spacing_max), 1.0)
        return float((count_score + spacing_score) / 2.0)

    except Exception as exc:
        raise ModelInferenceError(
            "Loop autocorrelation detection failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_centroid_instability_score(
    audio: np.ndarray,
    sr: int,
    top_db: float,
    min_interval_s: float,
) -> float:
    """
    Detect formant drift within sustained notes via spectral centroid CV.

    Algorithm:
      1. Split the waveform into non-silent intervals using librosa.effects.split.
      2. Discard intervals shorter than min_interval_s (too brief for reliable CV).
      3. For each qualifying interval, compute the spectral centroid frame-by-frame
         and calculate the coefficient of variation (std / mean) of those centroids.
      4. Return the mean CV across all qualifying intervals.

    Why this works:
      AI vocoders synthesise each note independently and can introduce erratic
      phase shifts in upper partials mid-note — the "glassy/hollow/formant-shifting"
      artifact that audio engineers hear. This causes the spectral centroid (centre
      of mass of the spectrum) to drift within what should be a sustained tone.
      Human vibrato, by contrast, modulates all partials proportionally so the
      centroid moves far less relative to its mean.

    Score interpretation:
      Near 0.0 → stable centroid within each note (human or clean synthesis).
      High (> CENTROID_INSTABILITY_AI_MIN) → erratic formant drift (AI signal).
      -1.0    → no qualifying intervals found (e.g. full-instrumental, very quiet).

    Args:
        audio:           Mono waveform array.
        sr:              Sample rate (Hz).
        top_db:          Silence threshold for librosa.effects.split.
        min_interval_s:  Minimum interval duration in seconds to analyse.

    Returns:
        float in [0.0, 1.0], or -1.0 when no intervals qualify.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        min_interval_samples = int(min_interval_s * sr)
        intervals = librosa.effects.split(audio, top_db=top_db)

        cvs: list[float] = []
        for start, end in intervals:
            if (end - start) < min_interval_samples:
                continue
            segment  = audio[start:end]
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            mean_c   = float(np.mean(centroid))
            if mean_c < 1e-9:
                continue
            cv = float(np.std(centroid) / mean_c)
            cvs.append(cv)

        if not cvs:
            return -1.0

        return float(np.clip(np.mean(cvs), 0.0, 1.0))

    except Exception as exc:
        raise ModelInferenceError(
            "Centroid instability computation failed.",
            context={"original_error": str(exc)},
        ) from exc



def compute_harmonic_ratio_score(
    audio: np.ndarray,
    sr: int,
    top_db: float,
    min_interval_s: float,
) -> float:
    """
    Measure harmonic purity within sustained intervals via HPSS.

    Algorithm:
      1. Split the waveform into non-silent intervals (librosa.effects.split).
      2. Discard intervals shorter than min_interval_s.
      3. For each qualifying interval, run HPSS and compute
         harmonic_energy / total_energy (HNR for that interval).
      4. Return the mean HNR across all qualifying intervals.

    Why this works:
      Real acoustic instruments — saxophone, guitar, voice — have physical noise
      components: breath, reed vibration, bow pressure, room reflections. AI audio
      generators synthesise each note from a learned latent space that lacks these
      noise components, producing unnaturally high harmonic purity.
      HNR within sustained notes targets this difference without being confused
      by percussive sections (excluded by librosa.effects.split).

    Score interpretation:
      Near 1.0 → almost entirely harmonic content (AI signal if above threshold).
      Near 0.5 → mixed harmonic/noise (typical real acoustic recording).
      -1.0     → no qualifying intervals found.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        min_interval_samples = int(min_interval_s * sr)
        intervals = librosa.effects.split(audio, top_db=top_db)

        ratios: list[float] = []
        for start, end in intervals:
            if (end - start) < min_interval_samples:
                continue
            segment = audio[start:end]
            total_energy = float(np.mean(segment ** 2))
            if total_energy < 1e-9:
                continue
            y_harmonic, _ = librosa.effects.hpss(segment)
            harmonic_energy = float(np.mean(y_harmonic ** 2))
            ratios.append(harmonic_energy / total_energy)

        if not ratios:
            return -1.0

        return float(np.clip(np.mean(ratios), 0.0, 1.0))

    except Exception as exc:
        raise ModelInferenceError(
            "Harmonic ratio computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_self_similarity_entropy(
    audio: np.ndarray,
    sr: int,
    subsample: int = 4,
) -> float:
    """
    Compute Shannon entropy of the chroma recurrence matrix upper triangle.

    Algorithm:
      1. Compute chroma CQT features and subsample for efficiency.
      2. Build an affinity-mode symmetric recurrence matrix.
      3. Extract upper-triangle values (excludes self-similarity diagonal).
      4. Bin into 20 buckets over [0, 1] and compute Shannon entropy.

    Low entropy = similarity values concentrated near 0 or 1 (blocky structure
    from AI attention patterns). High entropy = spread distribution (organic
    human performance drift and variation).

    Args:
        audio:      Mono waveform.
        sr:         Sample rate.
        subsample:  Keep every Nth chroma frame to keep matrix size tractable.

    Returns:
        Shannon entropy in bits [0.0, log2(20)≈4.32]. -1.0 if too short.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma = chroma[:, ::subsample]

        if chroma.shape[1] < 4:
            return -1.0

        R = librosa.segment.recurrence_matrix(chroma, mode="affinity", sym=True)
        triu = R[np.triu_indices_from(R, k=1)]

        if len(triu) == 0:
            return -1.0

        hist, _ = np.histogram(triu, bins=20, range=(0.0, 1.0))
        hist = hist.astype(float) + 1e-10
        hist /= hist.sum()
        return float(-np.sum(hist * np.log2(hist)))

    except Exception as exc:
        raise ModelInferenceError(
            "Self-similarity entropy computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_noise_floor_ratio(audio: np.ndarray) -> float:
    """
    Compute the ratio of quiet-moment RMS energy to mean RMS energy.

    Algorithm:
      1. Compute RMS in short frames (512-sample hop).
      2. Take the 5th-percentile RMS value as the "noise floor".
      3. Return floor / mean.

    Near-zero ratio = digital silence between notes = VST render (no room noise).
    Higher ratio = consistent background noise = real microphone recording.

    Args:
        audio: Mono waveform.

    Returns:
        Ratio in [0.0, 1.0]. -1.0 if mean energy is too low to measure.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if numpy raises unexpectedly.
    """
    try:
        import librosa

        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        mean_rms = float(np.mean(rms))

        if mean_rms < 1e-9:
            return -1.0

        floor_rms = float(np.percentile(rms, 5))
        return float(np.clip(floor_rms / mean_rms, 0.0, 1.0))

    except Exception as exc:
        raise ModelInferenceError(
            "Noise floor ratio computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_onset_strength_cv(audio: np.ndarray, sr: int) -> float:
    """
    Compute the coefficient of variation of the onset strength envelope.

    Low CV = uniform hit strength = AI generation (no performance dynamics).
    High CV = varied attack strengths = human expression (ghost notes, accents).

    Args:
        audio: Mono waveform.
        sr:    Sample rate.

    Returns:
        CV (std / mean) of onset strength. -1.0 if not computable.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

        if len(onset_env) < 2:
            return -1.0

        mean = float(np.mean(onset_env))
        if mean < 1e-9:
            return -1.0

        return float(np.std(onset_env) / mean)

    except Exception as exc:
        raise ModelInferenceError(
            "Onset strength CV computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_spectral_flatness_variance(audio: np.ndarray) -> float:
    """
    Compute the variance of Wiener entropy (spectral flatness) over time.

    Low variance = AI synthesizer — no physical noise source, spectral
    flatness stays constant between notes. High variance = real recording —
    tonal note frames (low flatness) alternate with noise frames (high flatness).

    Args:
        audio: Mono waveform.

    Returns:
        Variance of spectral flatness frames. -1.0 if not computable.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        flatness = librosa.feature.spectral_flatness(y=audio)[0]

        if len(flatness) < 2:
            return -1.0

        return float(np.var(flatness))

    except Exception as exc:
        raise ModelInferenceError(
            "Spectral flatness variance computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_subbeat_grid_deviation(audio: np.ndarray, sr: int) -> float:
    """
    Measure how precisely onsets land on the 16th-note grid.

    Algorithm:
      1. Detect BPM and beat positions via librosa beat tracker.
      2. Derive 16th-note grid positions (4 subdivisions per beat).
      3. Detect all onsets in the track.
      4. For each onset, compute the distance to the nearest grid position,
         normalised by the 16th-note duration.
      5. Return the variance of those normalised offsets.

    Near-zero variance = all onsets on the grid = AI (or heavily quantized
    production — expected to be a weak signal for modern pop/hip-hop).
    High variance = micro-timing feel = human performance or live recording.

    Args:
        audio: Mono waveform.
        sr:    Sample rate.

    Returns:
        Variance of normalised grid offsets (float ≥ 0). -1.0 if BPM
        is undetectable or fewer than 4 onsets found.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        if len(beat_frames) < 2:
            return -1.0

        bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        if bpm < 40.0 or bpm > 300.0:
            return -1.0

        sixteenth = 60.0 / (bpm * 4)  # duration of one 16th note in seconds

        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        if len(onset_frames) < 4:
            return -1.0

        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        deviations = []
        for t in onset_times:
            nearest_grid = round(t / sixteenth) * sixteenth
            dev = abs(t - nearest_grid) / sixteenth
            deviations.append(dev)

        return float(np.var(deviations))

    except Exception as exc:
        raise ModelInferenceError(
            "Sub-beat grid deviation computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_kurtosis_variability(
    audio: np.ndarray,
    sr: int,
    n_mels: int,
) -> float:
    """
    Compute the variance of per-frame mel-band kurtosis across time.

    Neural audio codec decoders (EnCodec, DAC) introduce checkerboard
    artifacts in the mel spectrogram: occasional sharp energy spikes in
    specific mel bands that appear as high kurtosis values in some frames
    but not others, producing very high kurtosis variance. Human recordings
    have smooth, consistent mel distributions with near-zero variance.

    Algorithm:
      1. Compute mel power spectrogram (n_mels × n_frames).
      2. For each time frame, compute Fisher kurtosis across the n_mels bands.
      3. Return the variance of those per-frame kurtosis values.

    Source: ISMIR TISMIR 2025 (Cros Vila) — Suno: ~1508 ± 1304, Human MSD: ~2.

    Args:
        audio:  Mono waveform array.
        sr:     Sample rate (Hz).
        n_mels: Number of mel bands.

    Returns:
        Variance of per-frame mel-band kurtosis (float ≥ 0.0).
        -1.0 when audio is too short to analyse (< 1 second).

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa or scipy raises unexpectedly.
    """
    if len(audio) < sr:
        return -1.0

    try:
        import librosa
        from scipy.stats import kurtosis as scipy_kurtosis

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        # mel: (n_mels, n_frames) — compute kurtosis across the mel axis per frame
        frame_kurtosis = scipy_kurtosis(mel, axis=0, fisher=True)
        # Use nanvar: kurtosis returns NaN for frames where all mel bins are equal
        # (e.g. silent/zero frames) — exclude those rather than propagating NaN.
        return float(np.nanvar(frame_kurtosis))

    except Exception as exc:
        raise ModelInferenceError(
            "Kurtosis variability computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_decoder_peak_score(
    audio: np.ndarray,
    sr: int,
    window_hz: int,
    prominence_db: float,
    regularity_max: float,
    min_count: int,
) -> float:
    """
    Detect the periodic spectral peak fingerprint left by transposed
    convolution layers in neural audio vocoders / codec decoders.

    Transposed convolution with stride k periodizes the spectrum of any
    bias component at intervals of f_s / k.  Multiple stacked layers
    compound this, creating a comb of regularly-spaced peaks in the
    1–16 kHz range that is absent in natural recordings.

    Algorithm:
      1. Compute the average power spectrum via high-resolution rfft.
      2. Convert to dB and subtract a rolling median floor to isolate peaks.
      3. Find peaks in the 1–16 kHz band above prominence_db threshold.
      4. Accept only if at least min_count peaks found.
      5. Check inter-peak spacing regularity: CV (std/mean) < regularity_max.
      6. Score: count and regularity components combined in [0.0, 1.0].

    Source: arXiv 2506.19108 — >99% accuracy on uncompressed audio.

    Args:
        audio:           Mono waveform loaded at 44.1 kHz.
        sr:              Sample rate (Hz) — must be ≥ 32 kHz to resolve 16 kHz.
        window_hz:       Rolling median half-width in Hz for floor estimation.
        prominence_db:   Minimum peak height above floor (dB) to count.
        regularity_max:  Max CV (std/mean spacing) to accept as periodic.
        min_count:       Minimum qualifying peaks to return a non-zero score.

    Returns:
        Score in [0.0, 1.0].  0.0 when no periodic pattern found.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if numpy or scipy raises unexpectedly.
    """
    if len(audio) < sr:
        return 0.0

    try:
        from scipy.signal import find_peaks, medfilt

        # High-resolution power spectrum — 32768 bins ≈ 1.3 Hz/bin at 44.1 kHz
        n_fft = min(32768, len(audio))
        power = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Restrict to 1–16 kHz
        band = (freqs >= 1000) & (freqs <= 16000)
        if not np.any(band):
            return 0.0

        band_power = power[band]
        power_db = 10.0 * np.log10(band_power + 1e-12)

        # Rolling median floor — window in bins
        bin_hz = freqs[1] - freqs[0]
        window_bins = max(3, int(window_hz / bin_hz))
        if window_bins % 2 == 0:
            window_bins += 1
        floor_db = medfilt(power_db, kernel_size=window_bins)

        residual_db = power_db - floor_db
        peak_indices, _ = find_peaks(residual_db, height=prominence_db)

        if len(peak_indices) < min_count:
            return 0.0

        spacings = np.diff(peak_indices)
        if len(spacings) < 2:
            return 0.0

        mean_spacing = float(np.mean(spacings))
        cv = float(np.std(spacings) / (mean_spacing + 1e-9))
        if cv > regularity_max:
            return 0.0

        # Score: 50% from peak density, 50% from spacing tightness
        count_component = min(float(len(peak_indices)) / float(min_count * 4), 0.5)
        regularity_component = 0.5 * (1.0 - min(cv / regularity_max, 1.0))
        return float(count_component + regularity_component)

    except Exception as exc:
        raise ModelInferenceError(
            "Decoder peak score computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def _score_organic_signals(bundle: _SignalBundle) -> float:
    """
    Weighted sum of dampable AI probability signals.

    These signals can all be explained by organic production techniques
    (pitch stacking, heavy DSP, vocal processing) when loop repetition is
    low. Separated so _compute_ai_probability can apply damping only here
    without accidentally dampening hardware-evidence signals like noise floor.

    Pure function — no I/O, no side effects, deterministic.
    """
    score = 0.0

    # Centroid flagged only in the AI range [AI_MIN, VOCODER_MIN).
    # Above VOCODER_MIN it's extreme DSP, not AI generation.
    centroid_flagged = (
        bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN
        and bundle.centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    )

    if centroid_flagged:
        score += CONSTANTS.PROB_WEIGHT_CENTROID

    if 0.0 <= bundle.ibi_variance < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        score += CONSTANTS.PROB_WEIGHT_IBI_QUANTIZED

    if bundle.loop_score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        score += CONSTANTS.PROB_WEIGHT_LOOP_CROSS_CORR

    if centroid_flagged and bundle.loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_VERDICT_THRESHOLD:
        score += CONSTANTS.PROB_WEIGHT_AUTOCORR_CENTROID

    if bundle.harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN:
        score += CONSTANTS.PROB_WEIGHT_HARMONIC_RATIO

    synthid_conf = _synthid_confidence(bundle.synthid_bins)
    if synthid_conf == "medium":
        score += CONSTANTS.PROB_WEIGHT_SYNTHID_MEDIUM
    elif synthid_conf == "low":
        score += CONSTANTS.PROB_WEIGHT_SYNTHID_LOW

    if bundle.spectral_slop > CONSTANTS.SPECTRAL_SLOP_RATIO:
        score += CONSTANTS.PROB_WEIGHT_SPECTRAL_SLOP

    if bundle.kurtosis_variability >= CONSTANTS.KURTOSIS_VARIABILITY_AI_MIN:
        score += CONSTANTS.PROB_WEIGHT_KURTOSIS

    if bundle.decoder_peak_score >= CONSTANTS.DECODER_PEAK_SCORE_MIN:
        score += CONSTANTS.PROB_WEIGHT_DECODER_PEAK

    if 0.0 < bundle.spectral_centroid_mean <= CONSTANTS.SPECTRAL_CENTROID_MEAN_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_CENTROID_MEAN

    if 0.0 <= bundle.self_similarity_entropy < CONSTANTS.SELF_SIMILARITY_ENTROPY_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_SELF_SIMILARITY

    if 0.0 <= bundle.onset_strength_cv < CONSTANTS.ONSET_STRENGTH_CV_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_ONSET_STRENGTH

    if 0.0 <= bundle.spectral_flatness_var < CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_SPECTRAL_FLATNESS

    if 0.0 <= bundle.subbeat_grid_deviation < CONSTANTS.SUBBEAT_DEVIATION_AI_MAX:
        score += CONSTANTS.PROB_WEIGHT_SUBBEAT_GRID

    return score


def _compute_ai_probability(bundle: _SignalBundle) -> float:
    """
    Compute a weighted AI probability score in [0.0, 1.0].

    Each signal contributes an additive weight when it fires; the sum is
    clamped to 1.0. Hard-evidence overrides (C2PA, high SynthID) bypass
    probability in _compute_verdict().

    Organic production damping: halves the score when autocorr is below
    PROB_AUTOCORR_ORGANIC_THRESHOLD and centroid is below vocoder range.
    Experimental human production (e.g. Bon Iver) elevates centroid + HNR
    via pitch stacking, but has near-zero loop repetition — a pattern
    incompatible with AI generators, which always produce structured
    repetitive content (autocorr > 0.83 empirically).

    Hardware-evidence signals (noise_floor_ratio) are added AFTER damping.
    A near-zero noise floor is a physical recording-chain property that
    organic production damping does not explain away.

    Calibrated against 10 tracks — see _score_organic_signals docstring.
    Pure function — no I/O, no side effects, deterministic.
    """
    score = _score_organic_signals(bundle)

    # Organic production damping — fires on non-repetitive + non-vocoder tracks.
    # Does NOT fire for vocoder tracks (centroid ≥ VOCODER_MIN) — heavy DSP
    # still needs supervisor review.
    if (
        bundle.loop_autocorr_score < CONSTANTS.PROB_AUTOCORR_ORGANIC_THRESHOLD
        and bundle.centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    ):
        score *= CONSTANTS.PROB_ORGANIC_DAMPING_FACTOR

    # Hardware-evidence signals — bypass organic damping. A VST render has
    # digital silence (nfr ≈ 0.0); real recordings always have room/mic noise.
    # This is the primary signal for orchestral AI (AIVA, Emily Howell) which
    # has no vocals and therefore no centroid/HNR signal.
    hardware_score = 0.0
    if 0.0 <= bundle.noise_floor_ratio < CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX:
        hardware_score += CONSTANTS.PROB_WEIGHT_NOISE_FLOOR

    return float(min(score + hardware_score, 1.0))


def _synthid_confidence(coherent_bins: int) -> str:
    """Map a coherent-bin count to a confidence label string."""
    if coherent_bins == 0:
        return "none"
    if coherent_bins <= CONSTANTS.SYNTHID_LOW_BINS:
        return "low"
    if coherent_bins <= CONSTANTS.SYNTHID_MEDIUM_BINS:
        return "medium"
    return "high"


def _build_flags(bundle: _SignalBundle) -> list[str]:
    """
    Build the list of human-readable flag strings for the UI.

    Pure function — separated from verdict logic so each can be tested
    independently. Verdict uses numeric comparisons; flags use display text.
    """
    flags: list[str] = []

    if "Born-AI" in bundle.c2pa_label:
        flags.append(bundle.c2pa_label)

    if bundle.ibi_variance < 0:
        flags.append("Insufficient data for groove analysis")
    elif bundle.ibi_variance < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        flags.append("Perfect Quantization (AI signal)")
    elif bundle.ibi_variance > CONSTANTS.IBI_ERRATIC_MIN:
        # High variance = human timing imperfection — organic signal, not AI
        flags.append("Human-Feel Timing (Organic)")

    if bundle.spectral_slop > CONSTANTS.SPECTRAL_SLOP_RATIO:
        flags.append(f"Spectral Slop detected ({bundle.spectral_slop:.1%} HF energy)")

    if bundle.loop_score > CONSTANTS.LOOP_SCORE_CEILING:
        flags.append(f"Likely Stock Loop/Sample (score {bundle.loop_score:.3f})")
    elif bundle.loop_score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        flags.append(f"Possible Repetition (score {bundle.loop_score:.3f})")

    if bundle.loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_VERDICT_THRESHOLD:
        flags.append(
            f"Sample-Heavy / Loop-Based (Organic Production) "
            f"(autocorr score {bundle.loop_autocorr_score:.3f})"
        )

    if bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN:
        flags.append(
            f"Extreme Spectral Processing Detected (centroid CV {bundle.centroid_instability_score:.3f}) — "
            f"total formant replacement; consistent with vocoder, talkbox, or heavy DSP. "
            f"Not typical of AI voice generators (AI range: 0.32–0.38)"
        )
    elif bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN:
        if bundle.loop_autocorr_score < CONSTANTS.PROB_AUTOCORR_ORGANIC_THRESHOLD:
            flags.append(
                f"Spectral Processing Detected (centroid CV {bundle.centroid_instability_score:.3f}) — "
                f"elevated formant drift, but non-repetitive song structure (autocorr "
                f"{bundle.loop_autocorr_score:.3f}) suggests pitch correction / vocal stacking "
                f"rather than AI generation. Probability weight reduced."
            )
        else:
            flags.append(
                f"Formant Drift Detected (centroid CV {bundle.centroid_instability_score:.3f}) — "
                f"erratic within-note spectral shift (AI signal)"
            )

    if bundle.harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN:
        flags.append(
            f"Unnaturally Clean Harmonics (HNR {bundle.harmonic_ratio_score:.3f}) — "
            f"no breath/noise artifacts detected within sustained notes (AI signal)"
        )

    conf = _synthid_confidence(bundle.synthid_bins)
    if conf == "high":
        flags.append(f"High-confidence AI watermark ({bundle.synthid_bins} coherent bins in 18–22 kHz)")
    elif conf == "medium":
        flags.append(f"Medium-confidence AI watermark ({bundle.synthid_bins} coherent bins in 18–22 kHz)")
    elif conf == "low":
        flags.append(f"Low-confidence HF phase anomaly ({bundle.synthid_bins} bin{'s' if bundle.synthid_bins > 1 else ''}) — monitor")

    if bundle.kurtosis_variability >= CONSTANTS.KURTOSIS_VARIABILITY_AI_MIN:
        flags.append(
            f"Neural Codec Artifacts Detected (mel-kurtosis variance {bundle.kurtosis_variability:.1f}) — "
            f"checkerboard mel-band spikes consistent with EnCodec/DAC decoder synthesis"
        )

    if bundle.decoder_peak_score >= CONSTANTS.DECODER_PEAK_SCORE_MIN:
        flags.append(
            f"Decoder Spectral Fingerprint Detected (score {bundle.decoder_peak_score:.3f}) — "
            f"periodic peaks in 1–16 kHz consistent with transposed convolution strides (AI vocoder)"
        )

    if 0.0 < bundle.spectral_centroid_mean <= CONSTANTS.SPECTRAL_CENTROID_MEAN_AI_MAX:
        flags.append(
            f"Low Mean Spectral Centroid ({bundle.spectral_centroid_mean:.0f} Hz) — "
            f"energy concentrated in lower frequencies (AI generators ~1091 Hz, human recordings ~1501 Hz)"
        )

    # Structural / instrumental signals — only flag when threshold is calibrated (> 0.0)
    if CONSTANTS.SELF_SIMILARITY_ENTROPY_AI_MAX > 0.0 and 0.0 <= bundle.self_similarity_entropy < CONSTANTS.SELF_SIMILARITY_ENTROPY_AI_MAX:
        flags.append(
            f"Low Structural Entropy (self-similarity {bundle.self_similarity_entropy:.2f} bits) — "
            f"blocky, repetitive structure consistent with AI attention patterns"
        )

    if CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX > 0.0 and 0.0 <= bundle.noise_floor_ratio < CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX:
        flags.append(
            f"Digital Silence Detected (noise floor ratio {bundle.noise_floor_ratio:.4f}) — "
            f"near-zero energy between notes; consistent with VST render (no room/mic noise)"
        )

    if CONSTANTS.ONSET_STRENGTH_CV_AI_MAX > 0.0 and 0.0 <= bundle.onset_strength_cv < CONSTANTS.ONSET_STRENGTH_CV_AI_MAX:
        flags.append(
            f"Uniform Onset Dynamics (CV {bundle.onset_strength_cv:.3f}) — "
            f"abnormally consistent hit strength; no performance dynamics variation (AI signal)"
        )

    if CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX > 0.0 and 0.0 <= bundle.spectral_flatness_var < CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX:
        flags.append(
            f"Uniform Spectral Flatness (var {bundle.spectral_flatness_var:.6f}) — "
            f"no physical noise source detected between notes (AI synthesizer signal)"
        )

    if CONSTANTS.SUBBEAT_DEVIATION_AI_MAX > 0.0 and 0.0 <= bundle.subbeat_grid_deviation < CONSTANTS.SUBBEAT_DEVIATION_AI_MAX:
        flags.append(
            f"On-Grid Timing (16th-note deviation var {bundle.subbeat_grid_deviation:.4f}) — "
            f"onsets land precisely on the grid (AI or heavily quantized production)"
        )

    return flags


def _compute_verdict(bundle: _SignalBundle) -> ForensicVerdict:
    """
    Aggregate numeric forensic scores into a final verdict using a
    probability-weighted score rather than a binary signal counter.

    Rules (ordered by certainty):
      1. Hard evidence overrides — C2PA certified AI, high SynthID → "AI"
         "AI" is reserved exclusively for cryptographically proven cases.
      2. Loop-heavy + human-feel + no AI signals → "Human (Sample/Loop)"
      3. Probability-weighted score (from _compute_ai_probability):
           ≥ PROB_VERDICT_AI      → "Likely AI"  (strong signals, no metadata proof)
           ≥ PROB_VERDICT_HYBRID  → "Possible Hybrid AI Cover"
           ≥ PROB_VERDICT_UNCERTAIN → "Uncertain"
           < PROB_VERDICT_UNCERTAIN → "Human"

    Pure function — compares numbers against CONSTANTS, no string matching.
    """
    # ── Hard-evidence overrides (bypass probability) ──────────────────────────
    if bundle.c2pa_flag:
        return "AI"
    if _synthid_confidence(bundle.synthid_bins) == "high":
        return "AI"

    # Centroid flagged only in the AI range — vocoder territory (>= VOCODER_MIN)
    # is extreme DSP processing, not AI generation. Consistent with ai_probability.
    centroid_flagged = (
        bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN
        and bundle.centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    )

    # ── Human (Sample/Loop) override ──────────────────────────────────────────
    # High autocorr + human-feel timing + no hard AI evidence → organic sampled
    # production. Two paths to pass:
    #
    # Path A — no AI vocal signals (centroid clear, HNR clear): simple gate.
    # Path B — centroid in AI range BUT very high IBI (> 300ms² variance) AND
    #   HNR not strongly elevated: the extreme timing jitter is more consistent
    #   with a human artist rapping/performing over loops than AI generation.
    #   Calibrated so Careless Whisper AI Cover (ibi=162) is blocked; heavily
    #   sampled hip-hop (ibi=461) passes.
    hnr_flagged        = bundle.harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN
    in_vocoder         = bundle.centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    hnr_blocks_override = bundle.harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_SAMPLE_OVERRIDE_BLOCK

    no_ai_signals = not centroid_flagged and not hnr_flagged
    strong_human_timing = (
        bundle.ibi_variance > CONSTANTS.IBI_SAMPLE_LOOP_HUMAN_MIN_WITH_AI_SIGNALS
        and not hnr_blocks_override
    )

    if (
        bundle.loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_SAMPLE_VERDICT_THRESHOLD
        and bundle.ibi_variance > CONSTANTS.IBI_ERRATIC_MIN
        and bundle.synthid_bins == 0
        and bundle.spectral_slop <= CONSTANTS.SPECTRAL_SLOP_RATIO
        and not in_vocoder
        and (no_ai_signals or strong_human_timing)
    ):
        return "Human (Sample/Loop)"

    # ── Probability-weighted verdict ──────────────────────────────────────────
    prob = _compute_ai_probability(bundle)

    if prob >= CONSTANTS.PROB_VERDICT_AI:
        return "Likely AI"
    if prob >= CONSTANTS.PROB_VERDICT_HYBRID:
        return "Possible Hybrid AI Cover"
    if prob >= CONSTANTS.PROB_VERDICT_UNCERTAIN:
        return "Uncertain"
    return "Human"
