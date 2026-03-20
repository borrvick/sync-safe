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

    # ---- Forensics: Spectral slop --------------------------------------------
    # HF-to-total energy ratio above this → spectral slop flag
    SPECTRAL_SLOP_RATIO: float = 0.15

    # ---- Forensics: SynthID watermark scan -----------------------------------
    # Coherent-bin count thresholds for low / medium / high confidence
    SYNTHID_LOW_BINS: int = 2
    SYNTHID_MEDIUM_BINS: int = 6

    # ---- NLP / Lyric Audit ----------------------------------------------------
    # Profanity classifier score above this threshold → EXPLICIT flag
    PROFANITY_SCORE_THRESHOLD: float = 0.5

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
        default="logs",
        description="Directory for daily pipeline log files, relative to the project root.",
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
