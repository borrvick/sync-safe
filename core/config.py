"""
core/config.py
Three-layer configuration system for Sync-Safe.

Layers (ordered from least to most environment-specific):
  1. SystemConstants  — frozen dataclass; physical and algorithmic constants.
                        Never read from the environment. Change only with a
                        code review and a comment explaining the new value.

  2. Settings         — pydantic-settings BaseSettings; sourced from .env
                        file and/or environment variables. API keys, feature
                        flags, infrastructure knobs. No Streamlit imports —
                        this must work in any Python process (API server,
                        worker, test suite).

  3. ModelParams      — pydantic BaseModel; ML hyperparameters. Separate
                        from Settings so a paid tier can override model
                        choices (e.g. Whisper large) without touching infra
                        config. Serialisable to JSON for audit logging.

Usage:
    from core.config import CONSTANTS, get_settings, ModelParams

    sr = CONSTANTS.SAMPLE_RATE
    key = get_settings().lastfm_api_key
    params = ModelParams()          # defaults
    params = ModelParams(whisper_model="large")   # override
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Layer 1: System Constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemConstants:
    """
    Immutable physical and algorithmic constants.

    All magic numbers that appear in service code must be traced back to a
    field here.  Adding a new constant requires a comment explaining its
    source (paper citation, empirical measurement, industry standard, etc.).
    """

    # ---- Audio ----------------------------------------------------------------
    SAMPLE_RATE: int = 22_050           # Hz; librosa default, balances quality/memory
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB hard ceiling

    # ---- Forensics ------------------------------------------------------------
    # Inter-beat interval variance below this → "Perfect Quantization" (AI signal)
    IBI_VARIANCE_THRESHOLD: float = 0.001

    # Cross-correlation score above this → likely stock loop (sync readiness)
    LOOP_SCORE_CEILING: float = 0.98

    # Frequency above which spectral energy is checked for AI artefacts ("slop")
    SPECTRAL_SLOP_HZ: int = 16_000

    # SynthID watermark scan band (Hz)
    SYNTHID_BAND_LOW_HZ: int = 18_000
    SYNTHID_BAND_HIGH_HZ: int = 22_000

    # ---- Compliance: Sting / Ending -------------------------------------------
    # Tail energy / mean energy ratio below this → ending qualifies as a fade
    STING_TAIL_RATIO: float = 0.05

    # Onset strength spike must be at least this multiple of the mean to flag sting
    STING_SPIKE_FACTOR: float = 3.0

    # RMS must drop by at least this fraction after the onset spike → sting confirmed
    STING_RMS_DROP_RATIO: float = 0.75

    # Window (seconds from end) evaluated for sting detection
    STING_WINDOW_SECONDS: float = 2.0

    # Window (seconds from end) used for fade slope regression
    FADE_WINDOW_SECONDS: float = 10.0

    # Sting check: librosa RMS hop/frame sizes (samples)
    STING_HOP_LENGTH: int = 512
    STING_FRAME_LENGTH: int = 1024

    # Fade detection: normalised slope below this → declining energy
    FADE_SLOPE_THRESHOLD: float = -0.0005
    # Fade detection: tail-to-mean energy ratio below this → low tail energy
    FADE_RATIO_MAX: float = 0.25

    # ---- Compliance: 4-8 Bar Energy Rule --------------------------------------
    # Minimum normalised spectral-contrast delta across a 4-bar window
    ENERGY_DELTA_MIN: float = 0.10

    # Beats grouped per analysis window (4 bars × 4 beats)
    BEATS_PER_WINDOW: int = 16

    # ---- Compliance: Intro ----------------------------------------------------
    # Intro segments longer than this (seconds) are flagged
    INTRO_MAX_SECONDS: int = 15

    # ---- Discovery ------------------------------------------------------------
    MAX_SIMILAR_TRACKS: int = 5

    # ---- Forensics: IBI / Groove ----------------------------------------------
    # IBI variance below this → "Perfect Quantization" AI signal
    IBI_PERFECT_QUANTIZATION_MAX: float = 0.5
    # IBI variance above this → "Erratic Humanization" AI signal
    IBI_ERRATIC_MIN: float = 90.0

    # ---- Forensics: Loop detection --------------------------------------------
    # Loop score above this (but below CEILING) → "Possible Repetition"
    LOOP_SCORE_POSSIBLE: float = 0.90
    # BPM range outside this → skip loop detection (unreliable tempo)
    LOOP_BPM_MIN: float = 40.0
    LOOP_BPM_MAX: float = 300.0

    # ---- Forensics: Autocorrelation loop detection ----------------------------
    # Minimum peak count in normalised autocorrelation to flag as regularly looping
    LOOP_PEAK_COUNT_THRESHOLD: int = 5
    # Mean spacing between autocorrelation peaks above this (frames) → not a tight loop
    LOOP_PEAK_SPACING_MAX: int = 100
    # Autocorrelation loop score at or above this → "Sample-Heavy / Loop-Based" verdict path
    LOOP_AUTOCORR_VERDICT_THRESHOLD: float = 0.70
    # Autocorrelation score above this + organic groove + high IBI → "Human (Sample/Loop)" verdict
    LOOP_AUTOCORR_SAMPLE_VERDICT_THRESHOLD: float = 0.85
    # UI display threshold — scores above this shown as "Moderate" repetition (below = "Low")
    LOOP_AUTOCORR_DISPLAY_MODERATE_MIN: float = 0.40

    # ---- Forensics: Spectral centroid instability ----------------------------
    # Silence threshold (dB below peak) for librosa.effects.split — splits track
    # into non-silent intervals where centroid is evaluated
    CENTROID_TOP_DB: float = 30.0
    # Ignore intervals shorter than this (seconds) — too brief for reliable centroid CV
    CENTROID_MIN_INTERVAL_S: float = 0.5
    # Mean within-interval centroid CV above this → erratic formant drift (AI signal)
    # AI vocoders shift upper partials mid-note; human vibrato modulates all partials
    # together so centroid stays more stable.
    # Calibrated against 8 tracks:
    #   Human (Espresso=0.196, Springsteen=0.205, G Thang=0.242, My Body=0.319, Levitating=0.299)
    #   AI    (Careless Whisper AI=0.364, Breaking Rust AI=0.378, Velvet Sundown=0.322)
    # Threshold at 0.32 splits cleanly: highest non-AI human=0.299, lowest AI=0.322.
    CENTROID_INSTABILITY_AI_MIN: float = 0.32
    # Centroid CV above this → extreme formant replacement (vocoder/talkbox/heavy processing)
    # NOT an additional AI signal; a separate flag that contextualises very high centroid values.
    # Empirically: AI tracks top out at ~0.38; Imogen Heap vocoder = 0.677.
    # Anything above 0.50 is beyond any AI generator observed — indicates analog/digital processing.
    CENTROID_INSTABILITY_VOCODER_MIN: float = 0.50

    # ---- Forensics: Harmonic-to-noise ratio (HNR) ---------------------------
    # AI generators produce unnaturally clean harmonic content — no breath, reed
    # noise, or physical resonance. HPSS separates harmonic/percussive components;
    # harmonic_energy / total_energy within sustained intervals gives HNR.
    # Calibrated against 5 tracks (3 with live HNR values):
    #   Human: Springsteen=0.485, Levitating=0.503
    #   AI:    Breaking Rust=0.664
    # Threshold at 0.59: highest human=0.503, lowest AI=0.664.
    # Margin: 0.087 on human side, 0.074 on AI side.
    HARMONIC_RATIO_AI_MIN: float = 0.59

    # HNR threshold for blocking the Human (Sample/Loop) override when centroid
    # is also flagged. Sampled tracks from clean libraries score 0.60–0.62; true
    # AI covers score 0.66+. Set above the sampled-track cluster (0.62) and below
    # the AI cluster (0.66) to allow high-autocorr / high-IBI sampled tracks
    # through the override while still blocking AI covers.
    # Calibrated: 01a4x17A3Ks (Heavily Sampled)=0.615 → passes; Careless Whisper
    # AI Cover=0.619 → passes (blocked by IBI gate); ea5C9IVarZM (100% AI)=0.68
    # → blocked by this threshold.
    HARMONIC_RATIO_SAMPLE_OVERRIDE_BLOCK: float = 0.65

    # IBI variance required to override "Human (Sample/Loop)" when centroid IS
    # flagged (i.e. AI vocal signals are present). The idea: if centroid is in the
    # AI range but ibi_variance is very high, the human timing jitter is strong
    # enough to override the weaker centroid signal. Calibrated so that the
    # Careless Whisper AI cover (ibi=162) does NOT trigger but 01a4x17A3Ks
    # (ibi=461, genuinely heavy-sampled) does.
    IBI_SAMPLE_LOOP_HUMAN_MIN_WITH_AI_SIGNALS: float = 300.0

    # ---- Forensics: Probability weights for verdict scoring ------------------
    # Each weight is the contribution to ai_probability [0.0, 1.0] when the
    # corresponding signal fires. Weights are additive; score is clamped to 1.0.
    # Calibrated so Espresso (human + Splice) ≈ 0.00 and
    # Careless Whisper AI cover ≈ 0.55.
    PROB_WEIGHT_CENTROID: float = 0.40          # within-note formant drift
    PROB_WEIGHT_IBI_QUANTIZED: float = 0.25     # machine-grid beat timing
    PROB_WEIGHT_LOOP_CROSS_CORR: float = 0.15   # near-identical 4-bar fingerprints
    PROB_WEIGHT_AUTOCORR_CENTROID: float = 0.15 # autocorr + centroid together
    PROB_WEIGHT_HARMONIC_RATIO: float = 0.20    # unnaturally clean harmonics
    PROB_WEIGHT_SYNTHID_MEDIUM: float = 0.10    # medium-confidence watermark
    PROB_WEIGHT_SYNTHID_LOW: float = 0.05       # low-confidence watermark
    PROB_WEIGHT_SPECTRAL_SLOP: float = 0.10     # HF energy anomaly
    # New signals (2026-03-21)
    # TODO: kurtosis and decoder_peak are calibrated on uncompressed audio (ISMIR TISMIR 2025,
    # arXiv 2506.19108). YouTube MP3 encoding masks both artifacts — overlap between AI and human
    # distributions collapses to noise (AI mean=664 vs Human mean=622 for kurtosis;
    # decoder_peak=0.0 for all 54 tracks). Re-enable and recalibrate once direct file upload
    # (uncompressed WAV/FLAC) is the primary input path.
    PROB_WEIGHT_KURTOSIS: float = 0.0           # DISABLED — requires uncompressed audio
    PROB_WEIGHT_DECODER_PEAK: float = 0.0       # DISABLED — requires uncompressed audio
    PROB_WEIGHT_CENTROID_MEAN: float = 0.10     # low mean spectral centroid (AI energy concentrated low)

    # Probability thresholds for final verdict assignment
    PROB_VERDICT_AI: float = 0.70               # ≥ this → "AI"
    PROB_VERDICT_HYBRID: float = 0.45           # ≥ this → "Possible Hybrid AI Cover"
    PROB_VERDICT_UNCERTAIN: float = 0.25        # ≥ this → "Uncertain"
                                                # < this → "Human" or override

    # Organic production damping: if autocorr is below this threshold (highly
    # non-repetitive structure) AND centroid is below vocoder territory, the
    # elevated centroid+HNR are more likely from pitch processing / vocal stacking
    # than AI generation. Halve the probability contribution of those two signals.
    # AI generators always produce structured repetitive content (autocorr > 0.83
    # across all tracks tested); experimental human production (Bon Iver, avant-garde)
    # can have autocorr near zero while using heavy pitch processing.
    # Calibrated: Bon Iver 22 (OVER S∞∞N) autocorr=0.000 → correctly damped.
    PROB_AUTOCORR_ORGANIC_THRESHOLD: float = 0.30   # below → genuinely non-repetitive
    PROB_ORGANIC_DAMPING_FACTOR: float = 0.50        # multiply score by this when damped

    # ---- Forensics: Spectral slop --------------------------------------------
    # HF-to-total energy ratio above this → spectral slop flag
    SPECTRAL_SLOP_RATIO: float = 0.15

    # ---- Forensics: Mel-band kurtosis variability ----------------------------
    # Variance of per-frame mel-band kurtosis.
    # Source: ISMIR TISMIR 2025 (Cros Vila) — Suno: ~1508 ± 1304, Human MSD: ~2.
    # Codec decoder checkerboard artifacts create sharp per-frame mel-band spikes
    # (high kurtosis) that vary wildly frame-to-frame → high variance.
    # Human audio has smooth, consistent mel distributions → near-zero variance.
    # Uncompressed threshold: ~50 (Suno=1508, Human=2). On YouTube audio both collapse
    # to ~640 with heavy overlap — see calibration run 2026-03-21. Raised to 9999 to
    # effectively disable until direct-upload path is available (PROB_WEIGHT_KURTOSIS=0).
    # TODO: recalibrate on uncompressed WAV/FLAC uploads; expected threshold ~800.
    KURTOSIS_VARIABILITY_AI_MIN: float = 9999.0
    # Number of mel bands used for kurtosis computation
    KURTOSIS_N_MELS: int = 128

    # ---- Forensics: Decoder spectral peak fingerprint ------------------------
    # Neural vocoders (HiFi-GAN, EnCodec, DAC) use transposed convolution with
    # fixed strides, which periodizes bias components into the spectrum at
    # intervals of f_s / stride. Multiple layers compound.
    # Source: arXiv 2506.19108 — >99% accuracy on uncompressed audio.
    # Peak detection window half-width in Hz (used to group nearby peaks)
    DECODER_PEAK_WINDOW_HZ: int = 200
    # Minimum peak prominence above local baseline (dB) to count as a spike
    DECODER_PEAK_PROMINENCE_DB: float = 3.0
    # Periodicity tolerance: ratio of std/mean spacing below which peaks are
    # considered periodic (tight clustering = decoder artifact)
    DECODER_PEAK_REGULARITY_MAX: float = 0.25
    # Minimum number of evenly-spaced peaks to flag a periodic pattern
    DECODER_PEAK_MIN_COUNT: int = 4
    # Score above this → decoder fingerprint detected.
    # TODO: all 54 YouTube tracks scored 0.0 — MP3 encoding masks the periodic peaks.
    # Re-enable on direct uncompressed uploads. Set to 2.0 (unreachable) until then.
    DECODER_PEAK_SCORE_MIN: float = 2.0

    # ---- Forensics: Spectral centroid mean -----------------------------------
    # Mean spectral centroid across the full track in Hz.
    # Source: ISMIR TISMIR 2025 — Suno: 1091 ± 386 Hz, Human: 1501 ± 632 Hz.
    # AI generators concentrate energy lower in the spectrum; real recordings
    # have more high-frequency content from room acoustics, instrument overtones,
    # and natural noise. Threshold set conservatively above the Suno mean + 1σ:
    # 1091 + 386 = 1477 ≈ 1400 Hz. Will be refined after batch scan.
    SPECTRAL_CENTROID_MEAN_AI_MAX: float = 1400.0

    # ---- Forensics: Structural / instrumental signals (2026-03-21) -----------
    # All weights start at 0.0 — disabled pending calibration against 54-track dataset.
    # TODO: run scripts/calibrate_signals.py after implementing, set thresholds from
    # the AI vs Human distribution data, then enable weights.

    # Self-similarity entropy: Shannon entropy of chroma recurrence matrix upper-triangle.
    # Calibrated 2026-03-21: AI mean=0.296, Human mean=0.262 — WRONG direction.
    # AI has HIGHER entropy than human, opposite of hypothesis. Likely because AI-generated
    # chroma has more mid-range similarities (0.3–0.7) rather than clean high/low clusters.
    # TODO: try spatial/structural entropy (e.g. diagonal line density) instead of
    # distribution entropy — may better capture regularity of repetition intervals.
    SELF_SIMILARITY_ENTROPY_AI_MAX: float = 0.0   # DISABLED — wrong direction
    PROB_WEIGHT_SELF_SIMILARITY: float = 0.0

    # Noise floor ratio: quiet-moment RMS / mean RMS.
    # Near-zero = digital silence between notes = VST render (no room noise).
    # Real recordings always have a noise floor from room/mic/preamps.
    # Calibrated 2026-03-21 against 54 tracks:
    #   9 AI tracks score exactly 0.000 (AIVA, Emily Howell — pure VST renders)
    #   Lowest human track: cimoNqiulUE = 0.005 → threshold 0.005 gives 0 false positives.
    #   Catches 10/42 AI (24%) at 100% precision, specifically the orchestral/instrumental
    #   AI that vocal signals (centroid instability, HNR) cannot detect.
    NOISE_FLOOR_RATIO_AI_MAX: float = 0.005
    PROB_WEIGHT_NOISE_FLOOR: float = 0.30           # high precision warrants meaningful weight

    # Onset strength CV: coefficient of variation of the onset strength envelope.
    # Calibrated 2026-03-21: AI mean=1.03, Human mean=1.13 — gap too small, heavy overlap.
    # TODO: revisit with larger human dataset; currently disabled.
    ONSET_STRENGTH_CV_AI_MAX: float = 0.0          # DISABLED — insufficient separation
    PROB_WEIGHT_ONSET_STRENGTH: float = 0.0

    # Spectral flatness variance: variance of Wiener entropy over time.
    # Calibrated 2026-03-21: signal fires in the WRONG direction (AI higher than human).
    # The high AI variance is caused by digital silence in VST renders (flatness swings
    # between near-0 during notes and near-1 during zero-energy frames) — same phenomenon
    # as noise_floor_ratio, not independent. Covered by PROB_WEIGHT_NOISE_FLOOR instead.
    # TODO: re-evaluate on uncompressed uploads where noise floor can be cleanly measured.
    SPECTRAL_FLATNESS_VAR_AI_MAX: float = 0.0      # DISABLED — correlated with noise floor
    PROB_WEIGHT_SPECTRAL_FLATNESS: float = 0.0

    # Sub-beat grid deviation: variance of onset-to-nearest-16th-note offset (normalised).
    # Calibrated 2026-03-21: AI mean=0.0217, Human mean=0.0219 — essentially identical.
    # Confirmed: modern music is recorded to a click grid regardless of AI origin.
    SUBBEAT_DEVIATION_AI_MAX: float = 0.0          # DISABLED — no separation found
    PROB_WEIGHT_SUBBEAT_GRID: float = 0.0

    # ---- Forensics: SynthID watermark scan -----------------------------------
    # Coherent-bin count thresholds for low / medium / high confidence
    SYNTHID_LOW_BINS: int = 2
    SYNTHID_MEDIUM_BINS: int = 6

    # ---- NLP / Lyric Audit ----------------------------------------------------
    # Profanity classifier score above this threshold → EXPLICIT flag
    PROFANITY_SCORE_THRESHOLD: float = 0.5

    # Detoxify obscenity/threat score bands for compliance grading
    # Scores between CONFIRMED and HARD thresholds → soft (director's call)
    # Scores at or above HARD threshold → hard deal-breaker in any sync context
    EXPLICIT_CONFIRMED: float = 0.60
    EXPLICIT_POTENTIAL: float = 0.40
    EXPLICIT_HARD:      float = 0.80
    VIOLENCE_CONFIRMED: float = 0.70
    VIOLENCE_POTENTIAL: float = 0.50
    VIOLENCE_HARD:      float = 0.85
    DRUGS_TOXIC_MIN:    float = 0.75

    # Lyric authorship signals — thresholds from lyric_authorship.py
    BURSTINESS_CV_THRESHOLD: float = 0.20      # below → low variance (AI signal)
    UNIQUE_WORD_RATIO_THRESHOLD: float = 0.45  # below → low vocabulary (AI signal)
    RHYME_DENSITY_THRESHOLD: float = 0.72      # above → high rhyme density (AI signal)
    REPETITION_SCORE_THRESHOLD: float = 0.40   # above → high repetition (AI signal)
    AI_SIGNAL_COUNT_CERTAIN: int = 3           # signals ≥ this → "Likely AI"
    AI_SIGNAL_COUNT_UNCERTAIN: int = 1         # signals ≥ this → "Uncertain"


# Module-level singleton — import directly, never instantiate.
CONSTANTS = SystemConstants()


# ---------------------------------------------------------------------------
# Layer 2: Runtime Settings (environment / .env / HF Spaces secrets)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Runtime configuration sourced from environment variables or a .env file.

    On Hugging Face Spaces, set these as Repository Secrets — the platform
    exposes them as environment variables automatically. No Streamlit-specific
    code here: this class must be importable outside a Streamlit context
    (e.g. a FastAPI backend, a Celery worker, a test suite).

    Key naming convention: lowercase snake_case to match both .env files and
    HF Spaces secrets (which are case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # silently ignore unknown env vars
    )

    # ---- External API keys ----------------------------------------------------
    lastfm_api_key: str = Field(
        default="",
        description="Last.fm API key for similar-track discovery. "
                    "Get one at https://www.last.fm/api/account/create",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token for private model access (optional).",
    )

    # ---- Model selection ------------------------------------------------------
    whisper_model: str = Field(
        default="large-v3",
        description="Whisper model size. Options: tiny | base | small | medium | large-v3. "
                    "Larger models are more accurate but require more VRAM/RAM.",
    )

    # ---- Feature flags --------------------------------------------------------
    # These are designed for paid-tier gating: free tier → False, paid → True.
    enable_c2pa: bool = Field(
        default=True,
        description="Run C2PA manifest checks for AI provenance detection.",
    )
    enable_ai_detection: bool = Field(
        default=True,
        description="Run AI lyric authorship detection (RoBERTa + linguistic signals).",
    )

    # ---- Infrastructure -------------------------------------------------------
    max_upload_mb: int = Field(
        default=50,
        description="Maximum permitted file upload size in megabytes.",
        ge=1,
        le=500,
    )

    # ---- Logging --------------------------------------------------------------
    log_dir: str = Field(
        default="",
        description=(
            "Directory for daily pipeline log files. "
            "Empty string (default) resolves to <project_root>/logs/ via core/logging.py. "
            "Set to an absolute path to override."
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Reads the environment exactly once — at first call — and caches the result
    for the lifetime of the process. Call `get_settings.cache_clear()` in tests
    that need to override environment variables between runs.
    """
    return Settings()


# ---------------------------------------------------------------------------
# Layer 3: Model Hyperparameters (tunable per deployment / tier)
# ---------------------------------------------------------------------------

class ModelParams(BaseModel):
    """
    ML hyperparameters for every model used in the pipeline.

    Separate from Settings so they can be:
    - Overridden at the service level (paid tier gets Whisper large)
    - Serialised to JSON and stored alongside a result for reproducibility
    - Validated with Pydantic (range checks, type coercion)

    These are not secrets and do not need environment-variable sourcing.
    """

    # ---- Whisper (transcription) ----------------------------------------------
    whisper_model: str = Field(
        default="large-v3",
        description="Model size passed to whisper.load_model(). large-v3 is the strongest free local option.",
    )
    whisper_initial_prompt: str = Field(
        default="",
        description=(
            "Prepended to Whisper's context window. Intentionally left empty: "
            "when the first window is silence (instrumental intro), Whisper "
            "'completes' a non-empty prompt with hallucinated meta-text instead "
            "of transcribing audio. Leave empty for music use cases."
        ),
    )
    whisper_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Decoding temperature. 0.0 = greedy (fastest, most deterministic).",
    )
    whisper_fp16: bool = Field(
        default=False,
        description="Use FP16 inference. Set False for CPU/MPS safety.",
    )
    whisper_condition_on_previous_text: bool = Field(
        default=False,
        description=(
            "Whether Whisper conditions each window on the previous output. "
            "False prevents hallucination loops (repeating phrases) that are "
            "common when transcribing music. Should remain False for audio tracks."
        ),
    )
    whisper_no_speech_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Segments with a no-speech probability above this threshold are "
            "discarded. 0.6 filters instrumental sections without cutting vocals."
        ),
    )
    whisper_compression_ratio_threshold: float = Field(
        default=2.4,
        ge=1.0,
        description=(
            "Segments whose gzip compression ratio exceeds this value are "
            "treated as hallucinations and dropped. 2.4 is the Whisper default; "
            "lower to be more aggressive about dropping repetitive output."
        ),
    )
    whisper_logprob_threshold: float = Field(
        default=-1.0,
        description=(
            "Segments with an average log-probability below this threshold are "
            "dropped as low-confidence. -1.0 is the Whisper default."
        ),
    )
    whisper_language: str = Field(
        default="en",
        description=(
            "BCP-47 language code passed to Whisper. Forcing 'en' prevents "
            "Whisper from auto-detecting the language from the first 30 seconds "
            "of audio — which on isolated vocal stems with processing artifacts "
            "often misidentifies as non-English and produces garbled output."
        ),
    )

    # ---- RoBERTa (AI lyric authorship) ----------------------------------------
    roberta_model: str = Field(
        default="Hello-SimpleAI/chatgpt-detector-roberta",
        description="HuggingFace model ID for the AI-text classifier.",
    )
    roberta_chunk_words: int = Field(
        default=400,
        gt=0,
        description="Max words per chunk fed to RoBERTa (token-window safety).",
    )

    # ---- allin1 (structure analysis) ------------------------------------------
    allin1_model: str = Field(
        default="harmonix-fold0",
        description=(
            "allin1 model name passed to allin1.analyze(). "
            "'harmonix-all' is an 8-model ensemble (more accurate but 8× slower). "
            "'harmonix-fold0' … 'harmonix-fold7' are single-fold models — "
            "fold0 is the default: fast, one HF download, good accuracy for sync use."
        ),
    )

    # ---- Demucs (source separation — called by allin1) ------------------------
    demucs_model: str = Field(
        default="htdemucs",
        description="Demucs checkpoint name passed to allin1.analyze().",
    )
    demucs_device: Optional[str] = Field(
        default=None,
        description="Torch device string. None = auto-detect (MPS → CPU fallback).",
    )
