"""
services/forensics/_pure.py
Pure signal-processing functions — no I/O, no side effects, deterministic.
All functions accept numpy arrays directly and are independently unit-testable.
"""
from __future__ import annotations

import numpy as np

from core.config import CONSTANTS
from core.exceptions import ModelInferenceError
from core.models import AiSegment


# ---------------------------------------------------------------------------
# STFT constants — SynthID / ultrasonic band analysis
# ---------------------------------------------------------------------------

_SYNTHID_LOAD_SR: int          = 44_100
_SYNTHID_N_FFT: int            = 8_192
_SYNTHID_HOP: int              = 2_048
_SYNTHID_PHASE_STD_MAX: float  = 0.10
_SYNTHID_MAG_SPIKE_DB: float   = 12.0


# ---------------------------------------------------------------------------
# SynthID confidence label
# ---------------------------------------------------------------------------

def _synthid_confidence(coherent_bins: int) -> str:
    """Map a coherent-bin count to a confidence label string."""
    if coherent_bins == 0:
        return "none"
    if coherent_bins <= CONSTANTS.SYNTHID_LOW_BINS:
        return "low"
    if coherent_bins <= CONSTANTS.SYNTHID_MEDIUM_BINS:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# C2PA helpers
# ---------------------------------------------------------------------------

def _classify_c2pa_origin(manifest_data: dict) -> tuple[bool, str]:
    """
    Classify the C2PA origin from a parsed manifest JSON dict.

    Pure function — no I/O, no side effects; independently testable.

    Logic:
      - Assertions labelled "ai.generated" or "c2pa.ai_generated" → (True, "ai")
      - Assertions labelled "c2pa.created" or "c2pa.edited":
          check data.softwareAgent (case-insensitive) against
          CONSTANTS.C2PA_DAW_SOFTWARE_AGENTS → (False, "daw")
          no DAW match found → (False, "unknown")
      - Manifest present but no recognised assertion → (False, "unknown")

    Returns:
        (born_ai, origin) where origin is "ai" | "daw" | "unknown".
    """
    _AI_LABELS  = ("ai.generated", "c2pa.ai_generated")
    _DAW_LABELS = ("c2pa.created", "c2pa.edited")

    has_daw_label = False
    daw_agent: str = ""

    for manifest in (manifest_data.get("manifests") or {}).values():
        for assertion in manifest.get("assertions", []):
            label: str = assertion.get("label", "")

            if any(ai_lbl in label for ai_lbl in _AI_LABELS):
                return True, "ai"

            if any(daw_lbl in label for daw_lbl in _DAW_LABELS):
                has_daw_label = True
                data_block = assertion.get("data") or {}
                agent = (data_block.get("softwareAgent") or "").lower()
                if agent:
                    daw_agent = agent

    if has_daw_label:
        for known_daw in CONSTANTS.C2PA_DAW_SOFTWARE_AGENTS:
            if known_daw in daw_agent:
                return False, "daw"
        return False, "unknown"

    return False, "unknown"


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def _check_spectral_slop(
    audio: np.ndarray,
    sr: int,
    threshold_hz: int,
    ratio_threshold: float,
) -> float:
    """Return the ratio of HF energy (above threshold_hz) to total energy."""
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
    """Cosine similarity of two L2-normalised fingerprint vectors."""
    n = min(len(a), len(b))
    return float(np.dot(a[:n], b[:n]))


# ---------------------------------------------------------------------------
# Vocal presence
# ---------------------------------------------------------------------------

def _detect_is_vocal(audio: np.ndarray, sr: int) -> bool:
    """
    Return True when the track contains significant pitched vocal content.

    Uses librosa.pyin voiced-frame detection.  A track is classified as vocal
    when ≥ VOCAL_MIN_VOICED_FRAMES frames are detected as voiced.

    Pure function — no I/O, no side effects, deterministic.
    """
    import librosa

    _, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    if voiced_flag is None:
        return False
    return int(np.sum(voiced_flag)) >= CONSTANTS.VOCAL_MIN_VOICED_FRAMES


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def compute_loop_autocorrelation_score(
    audio: np.ndarray,
    sr: int,
    peak_count_threshold: int,
    peak_spacing_max: int,
) -> float:
    """
    Detect regular loop repetition via onset-envelope autocorrelation.

    Returns:
        float in [0.0, 1.0]; 1.0 = strong regular repetition detected.

    Pure function — no I/O, no side effects, deterministic.

    Raises:
        ModelInferenceError: if librosa raises unexpectedly.
    """
    try:
        import librosa

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        autocorr  = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        autocorr_tail = autocorr[1:]
        peak_val = np.max(autocorr_tail)
        if peak_val < 1e-9:
            return 0.0
        autocorr_norm = autocorr_tail / peak_val

        peaks = librosa.util.peak_pick(
            autocorr_norm,
            pre_max=10, post_max=10, pre_avg=10, post_avg=10,
            delta=0.05, wait=5,
        )

        if len(peaks) < peak_count_threshold:
            return 0.0

        spacings = np.diff(peaks)
        if len(spacings) == 0:
            return 0.0

        mean_spacing = float(np.mean(spacings))
        if mean_spacing > peak_spacing_max:
            return 0.0

        count_score   = min(float(len(peaks)) / float(peak_count_threshold * 3), 1.0)
        spacing_score = 1.0 - min(mean_spacing / float(peak_spacing_max), 1.0)
        return float((count_score + spacing_score) / 2.0)

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Loop autocorrelation detection failed.",
            context={"original_error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Spectral analysis functions
# ---------------------------------------------------------------------------

def compute_centroid_instability_score(
    audio: np.ndarray,
    sr: int,
    top_db: float,
    min_interval_s: float,
) -> float:
    """
    Detect formant drift within sustained notes via spectral centroid CV.

    Returns:
        float in [0.0, 1.0], or -1.0 when no intervals qualify.

    Pure function — no I/O, no side effects, deterministic.
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
            cvs.append(float(np.std(centroid) / mean_c))

        if not cvs:
            return -1.0
        return float(np.clip(np.mean(cvs), 0.0, 1.0))

    except (OSError, ValueError, RuntimeError) as exc:
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

    Returns:
        Mean HNR across qualifying intervals in [0.0, 1.0]. -1.0 if none.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        min_interval_samples = int(min_interval_s * sr)
        intervals = librosa.effects.split(audio, top_db=top_db)

        ratios: list[float] = []
        for start, end in intervals:
            if (end - start) < min_interval_samples:
                continue
            segment      = audio[start:end]
            total_energy = float(np.mean(segment ** 2))
            if total_energy < 1e-9:
                continue
            y_harmonic, _ = librosa.effects.hpss(segment)
            ratios.append(float(np.mean(y_harmonic ** 2)) / total_energy)

        if not ratios:
            return -1.0
        return float(np.clip(np.mean(ratios), 0.0, 1.0))

    except (OSError, ValueError, RuntimeError) as exc:
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

    Returns:
        Entropy in bits [0.0, log2(20)≈4.32]. -1.0 if too short.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma = chroma[:, ::subsample]

        if chroma.shape[1] < 4:
            return -1.0

        R    = librosa.segment.recurrence_matrix(chroma, mode="affinity", sym=True)
        triu = R[np.triu_indices_from(R, k=1)]

        if len(triu) == 0:
            return -1.0

        hist, _ = np.histogram(triu, bins=20, range=(0.0, 1.0))
        hist = hist.astype(float) + 1e-10
        hist /= hist.sum()
        return float(-np.sum(hist * np.log2(hist)))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Self-similarity entropy computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_noise_floor_ratio(audio: np.ndarray) -> float:
    """
    Compute the ratio of quiet-moment RMS energy to mean RMS energy.

    Returns:
        Ratio in [0.0, 1.0]. -1.0 if mean energy is too low to measure.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        rms      = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        mean_rms = float(np.mean(rms))

        if mean_rms < 1e-9:
            return -1.0

        floor_rms = float(np.percentile(rms, 5))
        return float(np.clip(floor_rms / mean_rms, 0.0, 1.0))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Noise floor ratio computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_onset_strength_cv(audio: np.ndarray, sr: int) -> float:
    """
    Compute the coefficient of variation of the onset strength envelope.

    Returns:
        CV (std/mean) of onset strength. -1.0 if not computable.

    Pure function — no I/O, no side effects, deterministic.
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

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Onset strength CV computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_spectral_flatness_variance(audio: np.ndarray) -> float:
    """
    Compute the variance of Wiener entropy (spectral flatness) over time.

    Returns:
        Variance of spectral flatness frames. -1.0 if not computable.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        if len(flatness) < 2:
            return -1.0
        return float(np.var(flatness))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Spectral flatness variance computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_subbeat_grid_deviation(audio: np.ndarray, sr: int) -> float:
    """
    Measure how precisely onsets land on the 16th-note grid.

    Returns:
        Variance of normalised grid offsets. -1.0 if BPM undetectable.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        if len(beat_frames) < 2:
            return -1.0

        bpm = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        if bpm < 40.0 or bpm > 300.0:
            return -1.0

        sixteenth    = 60.0 / (bpm * 4)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        if len(onset_frames) < 4:
            return -1.0

        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        deviations  = [
            abs(t - round(t / sixteenth) * sixteenth) / sixteenth
            for t in onset_times
        ]
        return float(np.var(deviations))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Sub-beat grid deviation computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_pitch_quantization_score(audio: np.ndarray, sr: int) -> float:
    """
    Measure how closely detected pitches align to equal temperament (12-TET).

    Returns:
        Mean absolute pitch deviation in cents [0.0, 50.0]. -1.0 if too few
        voiced frames detected.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        f0 = librosa.yin(
            audio,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        voiced = f0[np.isfinite(f0) & (f0 > 0.0)]
        if len(voiced) < CONSTANTS.PITCH_QUANTIZATION_MIN_VOICED_FRAMES:
            return -1.0

        midi            = 12.0 * np.log2(voiced / 440.0) + 69.0
        deviation_cents = np.abs(midi - np.round(midi)) * 100.0
        return float(np.mean(deviation_cents))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Pitch quantization computation failed.",
            context={"original_error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Spectro-temporal monitoring signals
# ---------------------------------------------------------------------------

def compute_ultrasonic_noise_ratio(audio: np.ndarray, sr: int) -> float:
    """
    Measure energy in the 20–22 kHz band as a fraction of total energy.

    Only meaningful on audio loaded at 44.1 kHz from a ≥ 40 kHz source.

    Returns:
        Energy ratio in [0.0, 1.0].

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        n_fft  = min(65536, len(audio))
        power  = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2
        freqs  = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        total_energy = float(np.sum(power))
        if total_energy == 0.0:
            return 0.0

        band_mask   = (freqs >= CONSTANTS.ULTRASONIC_BAND_LOW_HZ) & (freqs <= CONSTANTS.ULTRASONIC_BAND_HIGH_HZ)
        band_energy = float(np.sum(power[band_mask]))
        return band_energy / total_energy

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Ultrasonic noise ratio computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_infrasonic_energy_ratio(audio: np.ndarray, sr: int) -> float:
    """
    Measure energy in the 1–20 Hz band as a fraction of total energy.

    Returns:
        Energy ratio in [0.0, 1.0]. 0.0 if silent. -1.0 if too short.

    Pure function — no I/O, no side effects, deterministic.
    """
    if len(audio) < sr:
        return -1.0

    try:
        n_fft = min(sr * 4, len(audio))
        power = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        total_energy = float(np.sum(power))
        if total_energy == 0.0:
            return 0.0

        band_mask   = (freqs > 0.0) & (freqs <= CONSTANTS.INFRASONIC_BAND_HIGH_HZ)
        band_energy = float(np.sum(power[band_mask]))
        return band_energy / total_energy

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Infrasonic energy ratio computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_phase_coherence_differential(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
) -> float:
    """
    Measure inter-channel phase coherence differential (LF coherence − HF coherence).

    Returns:
        Differential in [-1.0, 1.0]. 0.0 if either band has no energy.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        stft_l = librosa.stft(left)
        stft_r = librosa.stft(right)

        cross        = stft_l * np.conj(stft_r)
        cross_mean   = np.abs(np.mean(cross, axis=1))
        power_l_mean = np.mean(np.abs(stft_l) ** 2, axis=1)
        power_r_mean = np.mean(np.abs(stft_r) ** 2, axis=1)

        coherence = (cross_mean ** 2) / (power_l_mean * power_r_mean + 1e-10)

        freqs    = librosa.fft_frequencies(sr=sr)
        lf_mask  = (freqs >= 20.0) & (freqs <= CONSTANTS.PHASE_COHERENCE_LF_MAX_HZ)
        hf_mask  = (freqs >= CONSTANTS.PHASE_COHERENCE_HF_MIN_HZ) & (freqs <= 16_000.0)

        if not np.any(lf_mask) or not np.any(hf_mask):
            return 0.0

        return float(np.mean(coherence[lf_mask]) - np.mean(coherence[hf_mask]))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Phase coherence differential computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_plr_std(audio: np.ndarray, sr: int) -> float:
    """
    Compute the standard deviation of per-window Peak-to-Loudness Ratio.

    Returns:
        Std of per-window PLR in dB. -1.0 if too few windows.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        window_samples = int(CONSTANTS.PLR_WINDOW_SECONDS * sr)
        n_complete     = len(audio) // window_samples

        if n_complete < CONSTANTS.PLR_MIN_WINDOWS:
            return -1.0

        plr_values: list[float] = []
        for i in range(n_complete):
            window = audio[i * window_samples : (i + 1) * window_samples]
            peak   = float(np.max(np.abs(window)))
            rms    = float(np.sqrt(np.mean(window ** 2)))
            if rms < 1e-9:
                continue
            plr_values.append(
                20.0 * np.log10(peak + 1e-9) - 20.0 * np.log10(rms + 1e-9)
            )

        if len(plr_values) < CONSTANTS.PLR_MIN_WINDOWS:
            return -1.0
        return float(np.std(plr_values))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "PLR std computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_voiced_noise_floor(audio: np.ndarray, sr: int) -> float:
    """
    Measure mean spectral flatness in the 4–12 kHz band across pyin-voiced frames.

    Returns:
        Mean spectral flatness in [0, 1]. -1.0 if too few voiced frames.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa
        from scipy.stats import gmean

        hop_length = CONSTANTS.VOICED_NOISE_FLOOR_HOP_LENGTH

        _, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
            hop_length=hop_length,
        )

        if voiced_flag is None or int(np.sum(voiced_flag)) < CONSTANTS.VOCAL_MIN_VOICED_FRAMES:
            return -1.0

        n_fft     = CONSTANTS.VOICED_NOISE_FLOOR_N_FFT
        freqs     = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        band_mask = (freqs >= CONSTANTS.VOICED_NOISE_FLOOR_HZ_LOW) & (freqs <= CONSTANTS.VOICED_NOISE_FLOOR_HZ_HIGH)

        if not np.any(band_mask):
            return -1.0

        stft_mag = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        n_frames = min(stft_mag.shape[1], len(voiced_flag))

        flatness_values: list[float] = []
        for t in range(n_frames):
            if not voiced_flag[t]:
                continue
            band  = stft_mag[band_mask, t] + 1e-10
            geo   = float(gmean(band))
            arith = float(np.mean(band))
            flatness_values.append(geo / arith if arith > 0 else 0.0)

        if len(flatness_values) < CONSTANTS.VOCAL_MIN_VOICED_FRAMES:
            return -1.0
        return float(np.mean(flatness_values))

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Voiced noise floor computation failed.",
            context={"original_error": str(exc)},
        ) from exc


def compute_kurtosis_variability(
    audio: np.ndarray,
    sr: int,
    n_mels: int,
) -> float:
    """
    Compute the variance of per-frame mel-band kurtosis across time.

    Returns:
        Variance of per-frame kurtosis (float ≥ 0.0). -1.0 if too short.

    Pure function — no I/O, no side effects, deterministic.
    """
    if len(audio) < sr:
        return -1.0

    try:
        import librosa
        from scipy.stats import kurtosis as scipy_kurtosis

        mel            = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        frame_kurtosis = scipy_kurtosis(mel, axis=0, fisher=True)
        return float(np.nanvar(frame_kurtosis))

    except (OSError, ValueError, RuntimeError) as exc:
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
    Detect the periodic spectral peak fingerprint from transposed convolution layers.

    Returns:
        Score in [0.0, 1.0]. 0.0 when no periodic pattern found.

    Pure function — no I/O, no side effects, deterministic.
    """
    if len(audio) < sr:
        return 0.0

    try:
        from scipy.signal import find_peaks, medfilt

        n_fft     = min(32768, len(audio))
        power     = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2
        freqs     = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        band = (freqs >= 1000) & (freqs <= 16000)
        if not np.any(band):
            return 0.0

        band_power = power[band]
        power_db   = 10.0 * np.log10(band_power + 1e-12)

        bin_hz       = freqs[1] - freqs[0]
        window_bins  = max(3, int(window_hz / bin_hz))
        if window_bins % 2 == 0:
            window_bins += 1
        floor_db     = medfilt(power_db, kernel_size=window_bins)
        residual_db  = power_db - floor_db

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

        count_component      = min(float(len(peak_indices)) / float(min_count * 4), 0.5)
        regularity_component = 0.5 * (1.0 - min(cv / regularity_max, 1.0))
        return float(count_component + regularity_component)

    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Decoder peak score computation failed.",
            context={"original_error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Per-segment AI probability (heatmap)
# ---------------------------------------------------------------------------

def segment_ai_probabilities(
    y: np.ndarray,
    sr: int,
    window_s: int = 10,
    hop_s: int = 5,
) -> list[AiSegment]:
    """
    Compute per-window AI probability estimates for timeline heatmap display.

    Uses a fast subset of signals — noise floor, spectral flatness, centroid
    mean, harmonic ratio — that are meaningful on short clips.

    Returns:
        List of AiSegment, one per window.  Empty list if track is too short.

    Pure function — no I/O, no side effects, deterministic.
    """
    try:
        import librosa

        window_samples = window_s * sr
        hop_samples    = hop_s * sr
        total_samples  = len(y)

        if total_samples < window_samples:
            return []

        segments: list[AiSegment] = []
        start = 0

        while start + window_samples <= total_samples:
            end     = start + window_samples
            window  = y[start:end]
            start_s = start / sr
            end_s   = end / sr
            score   = 0.0

            nfr = compute_noise_floor_ratio(window)
            if nfr >= 0.0 and CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX > 0.0 and nfr <= CONSTANTS.NOISE_FLOOR_RATIO_AI_MAX:
                score += 0.25

            sfv = compute_spectral_flatness_variance(window)
            if sfv >= 0.0 and CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX > 0.0 and sfv <= CONSTANTS.SPECTRAL_FLATNESS_VAR_AI_MAX:
                score += 0.25

            centroid_frames = librosa.feature.spectral_centroid(y=window, sr=sr)[0]
            centroid_mean   = float(np.mean(centroid_frames)) if len(centroid_frames) > 0 else -1.0
            if centroid_mean >= 0.0 and CONSTANTS.SPECTRAL_CENTROID_MEAN_AI_MAX > 0.0 and centroid_mean <= CONSTANTS.SPECTRAL_CENTROID_MEAN_AI_MAX:
                score += 0.25

            hr = compute_harmonic_ratio_score(
                window, sr,
                top_db=CONSTANTS.CENTROID_TOP_DB,
                min_interval_s=CONSTANTS.CENTROID_MIN_INTERVAL_S,
            )
            if hr >= 0.0 and hr >= CONSTANTS.HARMONIC_RATIO_AI_MIN:
                score += 0.25

            segments.append(AiSegment(
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                probability=round(min(score, 1.0), 4),
            ))
            start += hop_samples

        return segments

    except ModelInferenceError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelInferenceError(
            "Per-segment AI probability computation failed.",
            context={"original_error": str(exc)},
        ) from exc
