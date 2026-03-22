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

        ai_probability = _compute_ai_probability(
            c2pa_flag=c2pa_flag,
            ibi_variance=ibi_variance,
            loop_score=loop_score,
            loop_autocorr_score=loop_autocorr_score,
            centroid_instability_score=centroid_instability_score,
            harmonic_ratio_score=harmonic_ratio_score,
            synthid_bins=synthid_bins,
            spectral_slop=spectral_slop,
        )
        flags = _build_flags(
            c2pa_label=c2pa_label,
            ibi_variance=ibi_variance,
            spectral_slop=spectral_slop,
            loop_score=loop_score,
            loop_autocorr_score=loop_autocorr_score,
            centroid_instability_score=centroid_instability_score,
            harmonic_ratio_score=harmonic_ratio_score,
            synthid_bins=synthid_bins,
        )
        verdict = _compute_verdict(
            c2pa_flag=c2pa_flag,
            ibi_variance=ibi_variance,
            loop_score=loop_score,
            loop_autocorr_score=loop_autocorr_score,
            centroid_instability_score=centroid_instability_score,
            harmonic_ratio_score=harmonic_ratio_score,
            synthid_bins=synthid_bins,
            spectral_slop=spectral_slop,
        )

        result = ForensicsResult(
            c2pa_flag=c2pa_flag,
            ibi_variance=ibi_variance,
            loop_score=loop_score,
            loop_autocorr_score=loop_autocorr_score,
            spectral_slop=spectral_slop,
            synthid_score=float(synthid_bins),
            centroid_instability_score=centroid_instability_score,
            harmonic_ratio_score=harmonic_ratio_score,
            ai_probability=ai_probability,
            flags=flags,
            verdict=verdict,
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


def _compute_ai_probability(
    c2pa_flag: bool,
    ibi_variance: float,
    loop_score: float,
    loop_autocorr_score: float,
    centroid_instability_score: float,
    harmonic_ratio_score: float,
    synthid_bins: int,
    spectral_slop: float,
) -> float:
    """
    Compute a weighted AI probability score in [0.0, 1.0].

    Each signal contributes an additive weight when it fires; the sum is
    clamped to 1.0. Hard-evidence overrides (C2PA, high SynthID) are NOT
    included here — they bypass probability in _compute_verdict().

    Organic production damping: if autocorr is below
    PROB_AUTOCORR_ORGANIC_THRESHOLD and centroid is below vocoder range,
    the probability is halved. Experimental human production (e.g. Bon Iver)
    uses pitch stacking that elevates centroid and HNR but has near-zero loop
    repetition — a pattern incompatible with AI generators, which always
    produce structured repetitive content (autocorr > 0.83 empirically).

    Calibrated against 10 tracks:
      Human cluster (0–15%):
        Espresso — Sabrina Carpenter:  centroid=0.196, HNR=0.227 → 0.00
        Born in the USA — Springsteen: centroid=0.205, HNR=0.485 → 0.00
        Nuthin' But a G Thang — Dre:  centroid=0.242, HNR=0.563 → 0.00
        My Body — Young the Giant:     centroid=0.319, HNR=0.368 → 0.00
        Levitating — Dua Lipa:         centroid=0.299, HNR=0.503 → 0.15
      Adversarial / edge cases:
        22 (OVER S∞∞N) — Bon Iver:    centroid=0.408, HNR=0.791, autocorr=0.000 → 0.30 (damped)
        Hide and Seek — Imogen Heap:   centroid=0.677, HNR=0.789, autocorr=0.562 → 0.60 (vocoder)
      AI cluster (55–75%):
        Careless Whisper AI cover:     centroid=0.364, HNR=0.619 → 0.75
        Walk My Walk — Breaking Rust:  centroid=0.378, HNR=0.664 → 0.75
        Dust on the Wind — Velvet Sundown: centroid=0.322, HNR=0.718 → 0.75

    Pure function — no I/O, no side effects, deterministic.
    """
    score = 0.0

    # Centroid is flagged only when in the AI range [AI_MIN, VOCODER_MIN).
    # Above VOCODER_MIN the flag text already says "NOT typical of AI voice
    # generators" — extreme formant replacement is a hardware/DSP processing
    # artifact, not AI generation, so it must not add to AI probability.
    centroid_flagged = (
        centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN
        and centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    )

    if centroid_flagged:
        score += CONSTANTS.PROB_WEIGHT_CENTROID

    if 0.0 <= ibi_variance < CONSTANTS.IBI_PERFECT_QUANTIZATION_MAX:
        score += CONSTANTS.PROB_WEIGHT_IBI_QUANTIZED

    if loop_score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        score += CONSTANTS.PROB_WEIGHT_LOOP_CROSS_CORR

    # Centroid + autocorr together are stronger evidence of AI loop structure
    if centroid_flagged and loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_VERDICT_THRESHOLD:
        score += CONSTANTS.PROB_WEIGHT_AUTOCORR_CENTROID

    # Harmonic ratio — contributes only when computed (≥ 0 means real value)
    if harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN:
        score += CONSTANTS.PROB_WEIGHT_HARMONIC_RATIO

    synthid_conf = _synthid_confidence(synthid_bins)
    if synthid_conf == "medium":
        score += CONSTANTS.PROB_WEIGHT_SYNTHID_MEDIUM
    elif synthid_conf == "low":
        score += CONSTANTS.PROB_WEIGHT_SYNTHID_LOW

    if spectral_slop > CONSTANTS.SPECTRAL_SLOP_RATIO:
        score += CONSTANTS.PROB_WEIGHT_SPECTRAL_SLOP

    # Organic production damping — fires when the track has near-zero loop
    # repetition (autocorr < threshold) and centroid is below vocoder range.
    # In this regime, elevated centroid + HNR are better explained by pitch
    # processing / vocal stacking (e.g. Bon Iver) than AI generation, which
    # always produces structured repetitive content (autocorr > 0.83 empirically).
    # Does NOT fire for vocoder tracks (centroid ≥ VOCODER_MIN) — those should
    # remain flagged because a sync supervisor needs to review heavy processing.
    if (
        loop_autocorr_score < CONSTANTS.PROB_AUTOCORR_ORGANIC_THRESHOLD
        and centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    ):
        score *= CONSTANTS.PROB_ORGANIC_DAMPING_FACTOR

    return float(min(score, 1.0))


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
    loop_autocorr_score: float,
    centroid_instability_score: float,
    harmonic_ratio_score: float,
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
        # High variance = human timing imperfection — organic signal, not AI
        flags.append("Human-Feel Timing (Organic)")

    if spectral_slop > CONSTANTS.SPECTRAL_SLOP_RATIO:
        flags.append(f"Spectral Slop detected ({spectral_slop:.1%} HF energy)")

    if isinstance(loop_score, float) and loop_score > CONSTANTS.LOOP_SCORE_CEILING:
        flags.append(f"Likely Stock Loop/Sample (score {loop_score:.3f})")
    elif isinstance(loop_score, float) and loop_score > CONSTANTS.LOOP_SCORE_POSSIBLE:
        flags.append(f"Possible Repetition (score {loop_score:.3f})")

    if loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_VERDICT_THRESHOLD:
        flags.append(
            f"Sample-Heavy / Loop-Based (Organic Production) "
            f"(autocorr score {loop_autocorr_score:.3f})"
        )

    if centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN:
        flags.append(
            f"Extreme Spectral Processing Detected (centroid CV {centroid_instability_score:.3f}) — "
            f"total formant replacement; consistent with vocoder, talkbox, or heavy DSP. "
            f"Not typical of AI voice generators (AI range: 0.32–0.38)"
        )
    elif centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN:
        if loop_autocorr_score < CONSTANTS.PROB_AUTOCORR_ORGANIC_THRESHOLD:
            flags.append(
                f"Spectral Processing Detected (centroid CV {centroid_instability_score:.3f}) — "
                f"elevated formant drift, but non-repetitive song structure (autocorr "
                f"{loop_autocorr_score:.3f}) suggests pitch correction / vocal stacking "
                f"rather than AI generation. Probability weight reduced."
            )
        else:
            flags.append(
                f"Formant Drift Detected (centroid CV {centroid_instability_score:.3f}) — "
                f"erratic within-note spectral shift (AI signal)"
            )

    if harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN:
        flags.append(
            f"Unnaturally Clean Harmonics (HNR {harmonic_ratio_score:.3f}) — "
            f"no breath/noise artifacts detected within sustained notes (AI signal)"
        )

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
    loop_autocorr_score: float,
    centroid_instability_score: float,
    synthid_bins: int,
    spectral_slop: float,
    harmonic_ratio_score: float = -1.0,
) -> ForensicVerdict:
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

    harmonic_ratio_score defaults to -1.0 for backward compatibility with
    fixtures that pre-date the HNR signal.

    Pure function — compares numbers against CONSTANTS, no string matching.
    """
    # ── Hard-evidence overrides (bypass probability) ──────────────────────────
    if c2pa_flag:
        return "AI"
    if _synthid_confidence(synthid_bins) == "high":
        return "AI"

    # Centroid flagged only in the AI range — vocoder territory (>= VOCODER_MIN)
    # is extreme DSP processing, not AI generation. Consistent with ai_probability.
    centroid_flagged = (
        centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_AI_MIN
        and centroid_instability_score < CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
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
    hnr_flagged = harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_AI_MIN
    in_vocoder = centroid_instability_score >= CONSTANTS.CENTROID_INSTABILITY_VOCODER_MIN
    hnr_blocks_override = harmonic_ratio_score >= CONSTANTS.HARMONIC_RATIO_SAMPLE_OVERRIDE_BLOCK

    no_ai_signals = not centroid_flagged and not hnr_flagged
    strong_human_timing = (
        ibi_variance > CONSTANTS.IBI_SAMPLE_LOOP_HUMAN_MIN_WITH_AI_SIGNALS
        and not hnr_blocks_override
    )

    if (
        loop_autocorr_score >= CONSTANTS.LOOP_AUTOCORR_SAMPLE_VERDICT_THRESHOLD
        and ibi_variance > CONSTANTS.IBI_ERRATIC_MIN
        and synthid_bins == 0
        and spectral_slop <= CONSTANTS.SPECTRAL_SLOP_RATIO
        and not in_vocoder
        and (no_ai_signals or strong_human_timing)
    ):
        return "Human (Sample/Loop)"

    # ── Probability-weighted verdict ──────────────────────────────────────────
    prob = _compute_ai_probability(
        c2pa_flag=c2pa_flag,
        ibi_variance=ibi_variance,
        loop_score=loop_score,
        loop_autocorr_score=loop_autocorr_score,
        centroid_instability_score=centroid_instability_score,
        harmonic_ratio_score=harmonic_ratio_score,
        synthid_bins=synthid_bins,
        spectral_slop=spectral_slop,
    )

    if prob >= CONSTANTS.PROB_VERDICT_AI:
        return "Likely AI"
    if prob >= CONSTANTS.PROB_VERDICT_HYBRID:
        return "Possible Hybrid AI Cover"
    if prob >= CONSTANTS.PROB_VERDICT_UNCERTAIN:
        return "Uncertain"
    return "Human"
