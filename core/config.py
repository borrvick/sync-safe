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
    # Bytes to hash for session-state loudness cache key — enough for reliable identity (#102)
    LOUDNESS_CACHE_HASH_BYTES: int = 65_536  # 64 KB

    # ---- Forensics ------------------------------------------------------------
    # Inter-beat interval variance below this → "Perfect Quantization" (AI signal)
    IBI_VARIANCE_THRESHOLD: float = 0.001

    # Cross-correlation score above this → likely stock loop (sync readiness)
    LOOP_SCORE_CEILING: float = 0.98

    # ---- Repetition Index (blended loop signal for UI display) ----------------
    # repetition_index = WEIGHT_LOOP * loop_score + WEIGHT_AUTOCORR * loop_autocorr_score
    # Weights reflect signal hierarchy: cross-corr is more discriminative for AI loops.
    REPETITION_INDEX_WEIGHT_LOOP: float     = 0.6
    REPETITION_INDEX_WEIGHT_AUTOCORR: float = 0.4
    REPETITION_INDEX_HIGH: float            = 0.75   # >= → "High"
    REPETITION_INDEX_MODERATE: float        = 0.45   # >= → "Moderate"; < → "Low"

    # Frequency above which spectral energy is checked for AI artefacts ("slop")
    SPECTRAL_SLOP_HZ: int = 16_000

    # SynthID watermark scan band (Hz)
    SYNTHID_BAND_LOW_HZ: int = 18_000
    SYNTHID_BAND_HIGH_HZ: int = 22_000

    # ---- C2PA: DAW software agent strings -----------------------------------------
    # Case-insensitive substrings matched against c2pa.created/edited softwareAgent field.
    # Presence → c2pa_origin = "daw"; absence with manifest → "unknown".
    C2PA_DAW_SOFTWARE_AGENTS: tuple[str, ...] = (
        "adobe audition", "logic pro", "garageband", "pro tools",
        "ableton", "fl studio", "reaper", "cubase", "studio one",
        "nuendo", "reason", "bitwig",
    )

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
    # Fade severity: tail duration that maps to severity = 1.0 (#103)
    FADE_SEVERITY_MAX_SECONDS: float = 60.0
    # Fade severity: fraction of track mean RMS defining the "tail" region (#103)
    FADE_TAIL_THRESHOLD_RATIO: float = 0.10

    # ---- Compliance: 4-8 Bar Energy Rule --------------------------------------
    # Minimum normalised spectral-contrast delta across a 4-bar window
    ENERGY_DELTA_MIN: float = 0.10

    # Beats grouped per analysis window (4 bars × 4 beats)
    BEATS_PER_WINDOW: int = 16
    # Cut type detection: max seconds from track end to nearest beat → "clean_cut" (#104)
    CUT_BEAT_TOLERANCE_S: float = 0.075

    # ---- Compliance: Intro ----------------------------------------------------
    # Intro segments longer than this (seconds) are flagged
    INTRO_MAX_SECONDS: int = 15

    # Onset-energy intro detection (#105)
    # Minimum beat count before an RMS jump is eligible to signal intro end.
    INTRO_ONSET_MIN_BEATS: int = 8
    # RMS must exceed pre-onset mean by this ratio to count as a significant jump.
    INTRO_ONSET_RMS_JUMP_RATIO: float = 2.0
    # Two signals are considered "in agreement" when within this many seconds.
    INTRO_CONFIDENCE_AGREEMENT_S: float = 2.0

    # ---- Discovery ------------------------------------------------------------
    MAX_SIMILAR_TRACKS: int = 5
    ESSENTIA_MAX_MOOD_TAGS: int    = 3   # top-N mood tags passed to Last.fm tag discovery
    ESSENTIA_FEATURE_TIMEOUT_S: int = 30  # max seconds for MusicExtractor per track

    # ---- Sync-Cut Detection ---------------------------------------------------
    # Standard ad/TV edit durations (seconds) to find edit points for.
    SYNC_CUT_TARGET_DURATIONS: tuple[int, ...] = (15, 30, 60)

    # Beat snap granularity: prefer windows that start on a bar boundary (every
    # SYNC_CUT_SNAP_BARS beats). 4 = 1 bar in 4/4 time.
    SYNC_CUT_SNAP_BARS: int = 4

    # How close (seconds) a beat must be to a section boundary to count as
    # "starts/ends at section boundary" for scoring.
    SYNC_CUT_BOUNDARY_TOLERANCE_S: float = 0.5

    # Duration tolerance: candidate window must be within ±this many seconds
    # of the target duration to be considered.
    SYNC_CUT_DURATION_TOLERANCE_S: float = 3.0

    # Top-N candidates to return per target duration (#148)
    SYNC_CUT_TOP_N: int = 3

    # Bonus added to confidence when track loop structure is in the moderate range (#151).
    # Moderate repetition (REPETITION_INDEX_MODERATE ≤ loop_score < REPETITION_INDEX_HIGH)
    # means the track loops cleanly but isn't mechanical — most versatile for short spots.
    SYNC_CUT_LOOP_BONUS: float = 0.05

    # ---- Section IBI tightness (#137) ----------------------------------------
    # Std dev of inter-beat intervals within a section, in milliseconds.
    # ≤ LOCKED → quantized / grid-locked (dance/sync-ready)
    # ≥ LOOSE  → rubato or live feel (harder to hit-sync to picture)
    # Between the two → Moderate
    SECTION_IBI_LOCKED_MS: float = 5.0
    SECTION_IBI_LOOSE_MS:  float = 20.0
    # Minimum beats required inside a section for a reliable std dev estimate.
    # Sections with fewer beats return None (no tag rendered).
    SECTION_IBI_MIN_BEATS: int = 4

    # ---- Section-aware repetition (#143, #145) --------------------------------
    # Minimum section duration (seconds) to attempt fingerprinting.
    # Sections shorter than this are skipped — avoids noise in tiny blips.
    SECTION_MIN_DURATION_S: float = 0.5

    # Boundary trim applied to each section slice before fingerprinting.
    # Removes 50 ms of click/bleed at edit points (sec.start / sec.end boundaries).
    SECTION_BOUNDARY_TRIM_S: float = 0.05

    # Sub-window size for intra-section internal repetition (#145).
    # 8 beats = 2 bars — gives more windows per section than the global 16-beat window.
    INTERNAL_LOOP_BEATS_PER_WINDOW: int = 8

    # Custom duration slider bounds and step (#150)
    SYNC_CUT_SLIDER_MIN: int = 15
    SYNC_CUT_SLIDER_MAX: int = 120
    SYNC_CUT_SLIDER_STEP: int = 5

    # ---- Stem validation: mono compatibility / phase alignment ----------------
    # Pearson L/R correlation below this → warn about phase issues
    PHASE_CORRELATION_WARN: float = 0.0
    # Pearson L/R correlation below this → flag as likely anti-phase
    PHASE_CORRELATION_FAIL: float = -0.3
    # Mono sum dB loss below this (negative) → warn
    MONO_CANCELLATION_DB_WARN: float = -3.0
    # Mono sum dB loss below this (negative) → significant cancellation flag
    MONO_CANCELLATION_DB_FAIL: float = -6.0

    # ---- AI Probability Heatmap -----------------------------------------------
    # Window and hop size (seconds) for per-segment AI probability analysis.
    AI_HEATMAP_WINDOW_S: int = 10
    AI_HEATMAP_HOP_S: int    = 5

    # Grade thresholds: probability below each value earns the corresponding grade.
    # A: [0, 0.20)  B: [0.20, 0.40)  C: [0.40, 0.60)  D: [0.60, 0.80)  F: [0.80, 1.0]
    AI_GRADE_THRESHOLDS: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)

    # ---- Metadata / Split Sheet Validation ------------------------------------
    # Writer/publisher splits must sum to 100 % within this tolerance.
    # Allows for standard 2-decimal-place rounding (e.g. 33.33 + 33.33 + 33.34).
    SPLIT_TOLERANCE: float = 0.01

    # ---- MusicBrainz API --------------------------------------------------------
    # Per-request HTTP timeout for MusicBrainz recordings API calls.
    MB_TIMEOUT_S: int = 8

    # ---- Pipeline step timeout budgets (seconds) ------------------------------
    # Generous walls to prevent hung subprocesses (yt-dlp, Whisper, allin1)
    # from blocking the UI indefinitely on ZeroGPU.
    STEP_TIMEOUT_INGESTION_S: int     = 120
    STEP_TIMEOUT_STRUCTURE_S: int     = 90
    STEP_TIMEOUT_TRANSCRIPTION_S: int = 180   # Whisper on cold GPU is slow
    STEP_TIMEOUT_FORENSICS_S: int     = 60
    STEP_TIMEOUT_COMPLIANCE_S: int    = 60
    STEP_TIMEOUT_AUTHORSHIP_S: int    = 30
    STEP_TIMEOUT_THEME_MOOD_S: int    = 15

    # ---- Theme & Mood Detection (#167) ----------------------------------------
    # Minimum per-theme score to include in the ranked output list.
    THEME_MIN_CONFIDENCE: float       = 0.25
    # Tokens to scan before a keyword match to detect negation ("not", "never", …).
    THEME_NEGATION_WINDOW: int        = 4
    # Groq model used for on-demand mood summary enrichment (#169).
    THEME_GROQ_MODEL: str             = "llama-3.3-70b-versatile"
    # Max lyric characters sent to Groq — caps token cost.
    THEME_GROQ_LYRICS_CAP: int        = 2000
    STEP_TIMEOUT_DISCOVERY_S: int     = 30
    STEP_TIMEOUT_LEGAL_S: int         = 30

    # ---- DAW Export (#152) ----------------------------------------------------
    # Default timecode framerate for Premiere Pro / DaVinci Resolve marker export.
    # Editors can override via UI selectbox; this is the safe film-standard default.
    EXPORT_FRAMERATE: float           = 24.0

    # ---- Loudness & Dialogue (LUFS / ITU-R BS.1770-4) ------------------------
    # Target integrated loudness per platform (LUFS)
    LUFS_SPOTIFY: float       = -14.0
    LUFS_APPLE_MUSIC: float   = -16.0
    LUFS_YOUTUBE: float       = -14.0
    LUFS_BROADCAST: float     = -23.0   # ATSC A/85 (US) / EBU R128 (EU)

    # Minimum track duration for pyloudnorm integrated LUFS (gating requires several 400ms blocks)
    LUFS_MIN_DURATION_S: float = 3.0

    # True peak warning threshold — exceeding causes clipping on loudness-normalised playback
    TRUE_PEAK_WARN_DBFS: float = -1.0
    # Oversampling factor for inter-sample true peak (ITU-R BS.1770-4 minimum is 4×)
    TRUE_PEAK_OVERSAMPLE: int = 4
    # Minimum section duration (s) for per-section LUFS/LRA — pyloudnorm needs several 400ms gating blocks
    DIALOGUE_MIN_SECTION_DUR_S: float = 2.0

    # Loudness verdict classification thresholds (#95)
    LOUDNESS_BROADCAST_DELTA_MAX: float = 2.0    # ±LU from broadcast target → "Broadcast-ready"
    LOUDNESS_STREAMING_HOT_MIN: float   = -14.0  # above Spotify/YT target → will be turned down
    LOUDNESS_NEEDS_MASTERING_MAX: float = -20.0  # below this → too quiet for any platform

    # Gain adjustment display color thresholds — used in report UI (#94)
    GAIN_OK_THRESHOLD_DB: float   = 1.0   # |gain| ≤ this → green (negligible adjustment)
    GAIN_WARN_THRESHOLD_DB: float = 4.0   # |gain| ≤ this → amber (moderate); above → red

    # PRO confidence scoring thresholds (#118)
    PRO_CONFIDENCE_MB_SCORE_THRESHOLD: int = 80    # MusicBrainz score must exceed this for "score_ok"
    PRO_CONFIDENCE_ARTIST_OVERLAP: float   = 0.90  # token overlap ratio must meet/exceed this

    # VO headroom estimate — max dB headroom at perfect dialogue-ready score (#92)
    VO_HEADROOM_MAX_DB: float = 12.0

    # Section label normalization — max seconds before a chorus for a section
    # to be considered "pre-chorus adjacent" for timeline highlight (#136)
    PRE_CHORUS_ADJACENT_MAX_S: float = 16.0

    # Dialogue-ready score thresholds (0.0–1.0)
    # Score = fraction of energy OUTSIDE the 300–3000 Hz dialogue competition band.
    # Higher = sits more cleanly under voiceover.
    DIALOGUE_READY_HIGH: float = 0.70   # ≥ this → "Dialogue-Ready"
    DIALOGUE_READY_LOW: float  = 0.40   # < this → "Dialogue-Heavy"; between → "Mixed"

    # ---- Track Popularity (blended 0–100 score) --------------------------------
    # Tier boundaries applied to the normalised popularity_score (0–100).
    # Score is derived from whichever signals are available:
    #   Last.fm listeners, Last.fm playcount, YouTube/platform view count,
    #   YouTube like count, Spotify popularity (0–100 native).
    # Each signal is normalised independently then the max is taken so a strong
    # signal on any single platform cannot be drowned out by weak others.
    # Tiers: Emerging < Regional < Mainstream < Global
    POPULARITY_REGIONAL_MIN: int    = 25    # normalised score ≥ 25
    POPULARITY_MAINSTREAM_MIN: int  = 50    # normalised score ≥ 50
    POPULARITY_GLOBAL_MIN: int      = 75    # normalised score ≥ 75

    # Last.fm listener count ceilings for per-signal normalisation.
    # Raw listener counts above the ceiling are clamped to 100.
    LASTFM_LISTENERS_REGIONAL: int    = 10_000
    LASTFM_LISTENERS_MAINSTREAM: int  = 100_000
    LASTFM_LISTENERS_GLOBAL: int      = 1_000_000

    # YouTube / platform view count ceilings for normalisation.
    # Calibrated against known mainstream (100M+ views) and emerging (<1M) tracks.
    PLATFORM_VIEWS_REGIONAL: int    = 1_000_000     # 1M views → score ~25
    PLATFORM_VIEWS_MAINSTREAM: int  = 50_000_000    # 50M views → score ~50
    PLATFORM_VIEWS_GLOBAL: int      = 500_000_000   # 500M views → score ~75+

    # Minimum number of independent signals required to award higher tiers.
    # Protects against false highs (e.g. a meme remix with 200M views but
    # near-zero Last.fm + Spotify scores reaching Global).
    # Emerging/Regional: 1 signal sufficient (single source is credible enough).
    # Mainstream: at least 2 signals must score > 0.
    # Global: at least 2 signals must score > 0.
    POPULARITY_MIN_SIGNALS_MAINSTREAM: int = 2
    POPULARITY_MIN_SIGNALS_GLOBAL: int     = 2

    # Last.fm listener threshold below which a fuzzy retry is attempted.
    # If the first lookup returns fewer listeners than this, strip featured
    # artists and parenthetical suffixes and retry — covers "13 listeners for
    # Bruno Mars" cases caused by title decoration mismatches in Last.fm's index.
    LASTFM_LOW_LISTENER_THRESHOLD: int = 1_000

    # Estimated sync fee ranges (USD) per popularity tier — shown as guidance only.
    # Source: industry averages 2024-2026; highly variable by usage and territory.
    SYNC_COST_EMERGING: tuple[int, int]    = (500,    5_000)
    SYNC_COST_REGIONAL: tuple[int, int]    = (2_000,  25_000)
    SYNC_COST_MAINSTREAM: tuple[int, int]  = (15_000, 100_000)
    SYNC_COST_GLOBAL: tuple[int, int]      = (50_000, 500_000)

    # Sync-readiness fee modifier multipliers (#112).
    # Applied to the displayed fee range when all/some compliance checks pass.
    SYNC_READINESS_UPLIFT: float   = 1.15   # all 3 checks pass → +15%
    SYNC_READINESS_DISCOUNT: float = 0.80   # sting or intro fails → −20%

    # ---- Forensics: IBI / Groove ----------------------------------------------
    # IBI variance below this → "Perfect Quantization" AI signal.
    # FakeMusicCaps calibration 2026-03-23: AI median IBI variance (621) is
    # HIGHER than human median (179). The "AI = perfect grid" hypothesis is not
    # supported — FMC generators add humanization jitter, or librosa beat tracking
    # is less stable on AI audio. The very-low-IBI path (< 0.5) still catches
    # outlier AI tracks (AI min=0.0) and is retained at current threshold.
    # The erratic-high-IBI path (IBI_ERRATIC_MIN) was originally designed to
    # catch randomly humanized AI but FMC data shows human p90=618 overlaps
    # AI p50=621 — retained for continuity, but precision is low.
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
    # Autocorrelation score above this + organic groove + high IBI → "Likely Not AI" (organic sampled production)
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
    # Original 5-track calibration: human max=0.503, AI min=0.664 → threshold 0.59.
    # FakeMusicCaps calibration 2026-03-24 (502 AI / 27 human):
    #   AI:    p25=0.507, p50=0.679, p75=0.847, p90=0.933
    #   Human: p25=0.455, p50=0.558, p75=0.630, p90=0.778, max=0.844
    #   At 0.59: ~50% of AI fire, ~30% human FP → unacceptable false positive rate.
    #   Raised to 0.85: AI p90=0.933 catches ~10% of AI; human p90=0.778 gives
    #   ~5-10% FP rate — acceptable at PROB_WEIGHT_HARMONIC_RATIO=0.10.
    HARMONIC_RATIO_AI_MIN: float = 0.85

    # HNR threshold for blocking the organic-sampled override toward "Likely Not AI".
    # Originally 0.65 based on 5-track calibration. FMC 2026-03-24: human p90=0.778,
    # so 0.65 would block ~10% of human tracks from the organic override — acceptable.
    # Raised to match recalibrated HARMONIC_RATIO_AI_MIN (0.85) context: tracks below
    # 0.85 are not flagged as AI anyway, so the block only matters at the high end.
    # Kept at 0.65 as a conservative gate; revisit if organic override misfires.
    HARMONIC_RATIO_SAMPLE_OVERRIDE_BLOCK: float = 0.65

    # ---- Forensics: Vocal / Instrumental path routing -------------------------
    # Tracks are routed to vocal or instrumental scoring based on voiced frame
    # count from librosa.pyin. Vocal and instrumental AI have fundamentally
    # different spectral signatures — one algorithm cannot serve both accurately.
    #
    # Vocal AI (Suno/Udio): neural vocoders produce formant drift (centroid
    # instability) + unnaturally clean harmonics (HNR) → those signals are valid.
    # Instrumental AI (MusicGen, AudioLDM2): stable spectral profiles + lower HNR
    # than human counterparts → centroid/HNR are inverted or absent (FMC 2026-03-24).
    #
    # Minimum voiced frames from librosa.pyin to classify a track as vocal.
    # At SAMPLE_RATE=22050 with librosa.pyin default hop=512: each frame ≈ 23.2 ms.
    # 100 frames ≈ 2.3 seconds of detectable pitched vocal content.
    VOCAL_MIN_VOICED_FRAMES: int = 100

    # Vocal-path harmonic ratio threshold.
    # SONICS calibration 2026-03-24 (500 AI vocal: 250 Suno + 250 Udio / 27 human WAV masters):
    #   Suno median=0.688, Udio median=0.590, human median=0.558
    #   Suno p10=0.535, Udio p10=0.397, human p10=0.439, human p90=0.778
    # Threshold 0.70 sits just above Suno median → catches ~50% Suno, misses most Udio.
    # Udio at 0.590 median is below human median; no threshold separates Udio cleanly.
    # This is the best available single threshold given the bimodal AI distribution.
    HARMONIC_RATIO_AI_MIN_VOCAL: float = 0.70

    # Vocal-path probability weights.
    # centroid_instability: DISABLED (weight=0.0) — SONICS calibration 2026-03-24
    #   confirmed inversion on 500 vocal AI tracks: AI p10 avg=0.187 < human p10=0.251.
    #   The original 8-track calibration was overfitted. FMC showed same inversion for
    #   instrumental AI; SONICS confirms it extends to vocal AI as well.
    # harmonic_ratio: reduced to 0.15 — marginal Udio separation means HNR alone cannot
    #   reliably identify vocal AI. Weight reduced to limit false positive impact.
    PROB_WEIGHT_CENTROID_VOCAL: float = 0.0
    PROB_WEIGHT_HARMONIC_RATIO_VOCAL: float = 0.15

    # IBI variance required to override toward "Likely Not AI" when centroid IS
    # flagged (i.e. AI vocal signals are present). The idea: if centroid is in the
    # AI range but ibi_variance is very high, the human timing jitter is strong
    # enough to override the weaker centroid signal. Calibrated so that the
    # Careless Whisper AI cover (ibi=162) does NOT trigger but 01a4x17A3Ks
    # (ibi=461, genuinely heavy-sampled) does.
    IBI_SAMPLE_LOOP_HUMAN_MIN_WITH_AI_SIGNALS: float = 300.0

    # ---- Forensics: Probability weights for verdict scoring ------------------
    # Each weight is the contribution to ai_probability [0.0, 1.0] when the
    # corresponding signal fires. Weights are additive; score is clamped to 1.0.
    #
    # FakeMusicCaps calibration 2026-03-24 (1000 AI / 27 human, 32-bit WAVs)
    # drove several weight changes — see individual signal comments below.

    # DISABLED 2026-03-24: FMC calibration shows human centroid instability
    # (median=0.379) is HIGHER than AI (median=0.244). The signal fires on ~50%
    # of human tracks and only ~25% of AI tracks at the 0.32 threshold — inverted.
    # Original 8-track calibration was overfitted to unusually stable pop songs.
    # Kept as research history; do not re-enable without a vocal-only AI dataset.
    PROB_WEIGHT_CENTROID: float = 0.0

    # IBI near-zero (machine-perfect grid). FMC: fires on <2% of AI tracks
    # (only AI min=0.0 qualifies at threshold 0.5 ms²). Retained for that
    # rare case; weight unchanged.
    PROB_WEIGHT_IBI_QUANTIZED: float = 0.25

    # Loop signals removed from AI detection 2026-03-24.
    # FMC calibration shows human music has FAR higher loop scores than AI:
    #   loop_cross_corr: human median=0.895, AI median=0.000
    #   loop_autocorr:   human p05=0.835,   AI median=0.000
    # Human verse/chorus repetition produces higher cross-correlation and
    # autocorrelation than AI generators, which produce more varied structure.
    # Loop detection is now a standalone Sync/Sample Analysis feature —
    # not an AI-detection signal.
    PROB_WEIGHT_LOOP_CROSS_CORR: float = 0.0
    PROB_WEIGHT_AUTOCORR_CENTROID: float = 0.0

    # Harmonic ratio threshold raised 2026-03-24.
    # Original 5-track calibration: human max=0.503, AI min=0.664 → threshold 0.59.
    # FMC calibration: human max=0.844, human p75=0.630 → ~30% FP rate at 0.59.
    # Raised to 0.85 (AI p90=0.933, human p90=0.778): ~10% catch rate, ~5% FP.
    # Weight halved to reflect lower catch rate and remaining overlap.
    PROB_WEIGHT_HARMONIC_RATIO: float = 0.10

    PROB_WEIGHT_SYNTHID_MEDIUM: float = 0.10    # medium-confidence watermark
    PROB_WEIGHT_SYNTHID_LOW: float = 0.05       # low-confidence watermark

    # DEAD SIGNAL 2026-03-24: all zeros on both AI and human in FMC.
    # Retained at 0.10 only if somehow fires on pathological uploads.
    PROB_WEIGHT_SPECTRAL_SLOP: float = 0.10

    # kurtosis and decoder_peak gated on uncompressed source; weights 0.0 pending
    # calibration on uncompressed WAV/FLAC uploads (FMC showed no separation).
    PROB_WEIGHT_KURTOSIS: float = 0.0
    PROB_WEIGHT_DECODER_PEAK: float = 0.0
    PROB_WEIGHT_CENTROID_MEAN: float = 0.10     # low mean centroid — AI energy concentrated low

    # Probability threshold for "Likely AI" verdict assignment.
    # "AI" is reserved for hard-evidence signals (C2PA / SynthID) in _compute_verdict.
    PROB_VERDICT_HYBRID: float = 0.45           # ≥ this → "Likely AI"
                                                # < this → "Likely Not AI" or organic override

    # Organic production damping — retained for vocoder/experimental tracks
    # (Bon Iver, heavy DSP) that have near-zero autocorr and elevated HNR.
    # Note: FMC calibration showed AI also has near-zero autocorr (median=0.0),
    # so this damping accidentally suppresses AI scores too. With centroid and
    # loop weights now at 0.0 the damping has minimal effect on scoring; it
    # remains to protect against false positives on avant-garde human music.
    PROB_AUTOCORR_ORGANIC_THRESHOLD: float = 0.30   # below → genuinely non-repetitive
    PROB_ORGANIC_DAMPING_FACTOR: float = 0.50        # multiply score by this when damped

    # ---- Forensics: Spectral slop --------------------------------------------
    # HF-to-total energy ratio above SPECTRAL_SLOP_HZ → spectral slop flag.
    # FakeMusicCaps calibration 2026-03-23: ALL zeros for both AI (1000 tracks)
    # and Human (27 tracks). No track from either group has >15% of total energy
    # above 16kHz. PROB_WEIGHT_SPECTRAL_SLOP = 0.10 is a dead weight in practice —
    # the signal never fires. Kept for theoretical completeness; would only
    # activate on severely broken AI output with pathological HF boosting.
    SPECTRAL_SLOP_RATIO: float = 0.15

    # ---- Forensics: Mel-band kurtosis variability ----------------------------
    # Variance of per-frame mel-band kurtosis.
    # Source: ISMIR TISMIR 2025 (Cros Vila) — Suno: ~1508 ± 1304, Human MSD: ~2.
    # Codec decoder checkerboard artifacts create sharp per-frame mel-band spikes
    # (high kurtosis) that vary wildly frame-to-frame → high variance.
    # Human audio has smooth, consistent mel distributions → near-zero variance.
    # On YouTube MP3 both groups collapse to ~640 (see calibration 2026-03-21).
    # FakeMusicCaps calibration 2026-03-23 (32-bit float WAVs, 1000 AI / 27 human):
    #   AI mean=649, Human mean=652 — COMPLETE OVERLAP even on uncompressed audio.
    #   FMC generators (MusicGen, StableAudioOpen, etc.) do NOT produce the
    #   checkerboard kurtosis spikes described in ISMIR TISMIR 2025 for Suno.
    #   Either the Suno architecture is uniquely responsible, or FMC generators
    #   use a different vocoder path. This signal is not useful for the current
    #   detector corpus. Kept as research history — do not re-enable without
    #   a Suno-specific uncompressed dataset.
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
    # FakeMusicCaps calibration 2026-03-23 (32-bit float WAVs):
    #   AI: all 0.0 (1000 tracks). Human: all 0.0 (27 tracks).
    #   The periodic peak pattern described in arXiv 2506.19108 does not
    #   manifest in the FMC generator set. Possible reasons: (a) FMC generators
    #   use DAC/EnCodec vocoders whose stride artifacts fall below the 3 dB
    #   prominence threshold at 44.1kHz, or (b) the periodization is present
    #   but masked by the generators' post-processing. Kept as research history.
    DECODER_PEAK_SCORE_MIN: float = 2.0

    # ---- Forensics: Spectral centroid mean -----------------------------------
    # Mean spectral centroid across the full track in Hz.
    # Source: ISMIR TISMIR 2025 — Suno: 1091 ± 386 Hz, Human: 1501 ± 632 Hz.
    # AI generators concentrate energy lower in the spectrum; real recordings
    # have more high-frequency content from room acoustics, instrument overtones,
    # and natural noise. Threshold set conservatively above the Suno mean + 1σ:
    # 1091 + 386 = 1477 ≈ 1400 Hz.
    # FakeMusicCaps calibration 2026-03-23 (503 AI / 27 human):
    #   AI:    p25=1104, p50=1554, p75=2000, mean=1596
    #   Human: p25=1850, p50=2044, p75=2418, mean=2040, min=1163
    #   Threshold at 1400 catches ~25% of AI (below AI p25) with ~2% human
    #   false positive rate (human min=1163 is the single track below threshold).
    #   Cross-validation confirms existing threshold. Weight=0.10 is appropriate
    #   given centroid stacking risk (centroid family can reach 0.65 combined).
    SPECTRAL_CENTROID_MEAN_AI_MAX: float = 1400.0

    # ---- Forensics: Structural / instrumental signals (2026-03-21) -----------
    # Calibrated 2026-03-23 against FakeMusicCaps (1000 AI / 27 human, 32-bit float WAVs).
    # Each signal's calibration result is documented below. Signals without useful
    # separation are kept as research history (disabled, do not re-enable without
    # new evidence). Signals with clean separation are enabled below.

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

    # Pitch quantization score: mean absolute deviation of detected pitches from
    # equal temperament (12-TET) in cents (100 cents = 1 semitone).
    # Hypothesis: AI generators produce pitch-perfect output (near-zero cents deviation);
    # real instruments have natural intonation drift of 10–30 cents empirically.
    # FakeMusicCaps calibration 2026-03-23:
    #   AI:    median=13.9 cents, p25=9.0, p75=18.6, mean=14.2
    #   Human: median=12.4 cents, p25=11.4, p75=13.5, mean=12.2
    #   COMPLETE OVERLAP — no threshold separates the distributions.
    #   AI generators do NOT produce near-zero pitch deviation; they produce
    #   the same ~10–20 cent scatter as human recordings. Hypothesis disproved.
    #   Kept as research history.
    PITCH_QUANTIZATION_AI_MAX: float = 0.0         # DISABLED — no separation in FMC data
    PROB_WEIGHT_PITCH_QUANTIZATION: float = 0.0
    PITCH_QUANTIZATION_MIN_VOICED_FRAMES: int = 20 # require ≥ this many voiced frames

    # ---- Forensics: Ultrasonic noise plateau (20–22 kHz) ---------------------
    # Diffusion models start with white noise across the full 0–22 kHz spectrum
    # and must explicitly carve it away during denoising. They often don't bother
    # above 20 kHz. Human masters are shelf-filtered or LPF'd by the engineer.
    # The "tell": a flat plateau of residual noise at 20–22 kHz that no mastering
    # engineer would leave there — it wastes headroom and is inaudible.
    #
    # Gated on not compressed_source AND native_sr ≥ 40000 — YouTube strips HF
    # above 16–17 kHz, and resampling a 16 kHz file to 44.1 kHz cannot create
    # real content above its original 8 kHz Nyquist. Both checks live inside
    # _analyse_ultrasonic() so the gate is self-contained (GPU decorator rule).
    #
    # FakeMusicCaps calibration 2026-03-23 (44.1kHz 32-bit WAVs):
    #   AI: count=0 — no AI track produced a measurable score.
    #   Human: count=27, median=2.9e-6, max=0.00126.
    #   The AI count=0 result is directionally interesting (FMC generators are
    #   hard band-limited at 20 kHz and produce no ultrasonic content), but
    #   the signal as implemented produces -1.0 (not computed) for all AI tracks,
    #   not a positive score. Rewiring the absence of ultrasonic content as an
    #   AI signal would require a redesign (presence-of-content gate → binary flag).
    #   Keeping disabled pending that redesign. Do not confuse with the diffusion
    #   residue hypothesis: the FMC generators simply do not render above 20 kHz
    #   at all, rather than leaving a noise plateau.
    ULTRASONIC_BAND_LOW_HZ: int   = 20_000
    ULTRASONIC_BAND_HIGH_HZ: int  = 22_000
    ULTRASONIC_ENERGY_RATIO_AI_MIN: float = 0.0   # DISABLED — requires redesign (see above)
    PROB_WEIGHT_ULTRASONIC: float = 0.0

    # ---- Forensics: Infrasonic rumble (1–20 Hz) ------------------------------
    # Real microphone diaphragms and preamp capacitors act as natural high-pass
    # filters — it is physically impossible for a vocal or instrument to produce
    # a pure 1–20 Hz signal. AI math can drift, leaving DC bias or sub-sonic
    # "rumble" that no microphone would ever capture and no engineer would leave.
    #
    # Gated on uncompressed sources only (upload-only): MP3 encoding quantization
    # noise sits above the infrasonic band and would create false positives on
    # YouTube-sourced tracks. Gate added 2026-03-23.
    #
    # FakeMusicCaps calibration 2026-03-23 (32-bit float WAVs, 503 AI / 27 human):
    #   AI:    median=0.000151, p95=0.0062, mean=0.0042
    #   Human: median=0.000043, p75=0.00018, max=0.016
    #   AI has ~3.5× higher median than human (consistent with "math drift" theory)
    #   but distributions overlap heavily — human max (0.016) >> AI p95 (0.006).
    #   No threshold exists that doesn't produce significant false positives.
    #   Kept disabled as research history.
    INFRASONIC_BAND_HIGH_HZ: int  = 20
    INFRASONIC_ENERGY_RATIO_AI_MIN: float = 0.0   # DISABLED — no clean threshold in FMC data
    PROB_WEIGHT_INFRASONIC: float = 0.0

    # ---- Forensics: Inter-channel phase coherence ----------------------------
    # In human recordings the stereo field is intentional — phase relationships
    # between L and R are set by the engineer (panning, delays, room mics).
    # AI diffusion "carves" high and low frequencies as separate statistical
    # events, causing HF phase to wobble while LF phase stays stable.
    #
    # Metric: mean_LF_coherence − mean_HF_coherence.
    #   Positive = LF more coherent than HF = AI pattern.
    #   Near zero or negative = both bands coherent = intentional human mix.
    #
    # Gated on not compressed_source — stereo field degrades under MP3.
    # Stereo check is self-contained in _analyse_phase_coherence(); returns
    # -1.0 automatically for mono sources (YouTube always transcodes mono).
    #
    # FakeMusicCaps calibration 2026-03-23:
    #   AI: count=0 — FMC WAVs appear to be mono (or stereo but collapsing to
    #   identical channels), so the stereo gate returned -1.0 for all AI tracks.
    #   Human: count=26, median=-0.004, p25=-0.079, p75=0.140, mean=0.011.
    #   Human distribution is centred near zero with high variance — no bias
    #   toward positive differential as hypothesised.
    #   Cannot calibrate without AI stereo data. The FMC dataset does not provide
    #   usable stereo content for this signal. Kept disabled pending a stereo-only
    #   AI dataset (e.g. MusicGen-stereo or StableAudioOpen raw outputs).
    PHASE_COHERENCE_LF_MAX_HZ: int  = 2_000
    PHASE_COHERENCE_HF_MIN_HZ: int  = 8_000
    PHASE_COHERENCE_DIFFERENTIAL_AI_MIN: float = 0.0   # DISABLED — no AI stereo data in FMC
    PROB_WEIGHT_PHASE_COHERENCE: float = 0.0

    # ---- Forensics: Voiced-region noise floor --------------------------------
    # Real vocal recordings contain continuous mic/room noise even during
    # sustained notes — breath, room tone, and microphone self-noise sit in
    # the 4–12 kHz band where vocal harmonics are sparse.  Neural vocoder
    # synthesis produces spectrally clean partials: the 4–12 kHz band between
    # harmonics approaches digital silence.
    #
    # Metric: mean spectral flatness of voiced frames in the 4–12 kHz band.
    #   High flatness → noise present → human recording.
    #   Low flatness  → unnaturally clean partials → AI synthesis.
    #
    # Only meaningful for tracks flagged as vocal (is_vocal=True).
    # Returns -1.0 when is_vocal=False or too few voiced frames detected.
    #
    # FILE-UPLOAD ONLY — gated on compressed_source=False in forensics.py.
    # Calibrated 2026-03-25:
    #   AI (native WAV): stable_audio_open mean=0.008, audioldm2 mean=0.008,
    #                    MusicGen mean=0.013, max across all=0.042
    #   AI (SONICS MP3): Suno p90=0.005, Udio p90=0.005, max=0.022
    #   Human (iTunes MP3): p10=0.333, p05=0.293
    #   Human (debug WAV/YouTube): 0.54–0.69 — NOT used for threshold (codec noise)
    # YouTube AAC/Opus → WAV decoding floods the between-harmonic band; signal
    # gated to file-upload path only via compressed_source check.
    VOICED_NOISE_FLOOR_HZ_LOW:  int   = 4_000   # lower bound of analysis band
    VOICED_NOISE_FLOOR_HZ_HIGH: int   = 12_000  # upper bound of analysis band
    VOICED_NOISE_FLOOR_AI_MAX:  float = 0.10    # fires when flatness ≤ 0.10; AI max=0.042
    PROB_WEIGHT_VOICED_NOISE_FLOOR: float = 0.15  # same weight as PLR — strong separation
    # STFT parameters for voiced noise floor computation.
    # hop_length must match librosa.pyin default so voiced-frame indices align
    # with STFT frame indices. n_fft=2048 at sr=22050 → ~93 Hz frequency resolution,
    # sufficient to resolve the 4–12 kHz analysis band into ~86 bins.
    VOICED_NOISE_FLOOR_HOP_LENGTH: int = 512
    VOICED_NOISE_FLOOR_N_FFT:      int = 2048

    # ---- Forensics: Temporal loudness flatness (PLR variance) ----------------
    # Human masters "breathe" — the crest factor (peak_db − rms_db) varies
    # naturally as quiet verses and loud choruses alternate. AI generators
    # produce "frozen density": they look-ahead limit to contain wild diffusion
    # values, flattening the PLR across every window to maintain a high
    # confidence score for every second of audio.
    #
    # Metric: std of per-window PLR across the full track.
    #   Low std = frozen density = AI pattern.
    #   High std = breathing room = human pattern.
    #
    # FakeMusicCaps calibration 2026-03-23 (1000 AI / 27 human, 32-bit WAVs):
    #   AI:    p25=0.70, p50=1.04, p75=1.51, mean=1.31, min=0.14
    #   Human: p25=1.50, p50=1.80, p75=2.14, mean=1.82, min=1.17
    #   Threshold at 1.2 catches ~55% of AI (below AI p50=1.04 → AI p75=1.51)
    #   with ~4% human false positive rate (human min=1.17, p05=1.26 → only
    #   the single lowest-PLR-std track in our 27-track set falls near threshold).
    #   Heavily brick-wall mastered pop may score low regardless of origin;
    #   weight is kept conservative (0.15) to avoid false positives on that genre.
    PLR_WINDOW_SECONDS: int  = 2
    PLR_MIN_WINDOWS:    int  = 5      # require ≥ this many windows; short tracks return -1.0
    PLR_STD_AI_MAX:     float = 1.2   # below → frozen loudness density (AI signal)
    PROB_WEIGHT_PLR_FLATNESS: float = 0.15   # calibrated 2026-03-23 against FMC

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
    AUTHORSHIP_MAX_SIGNALS: float = 6.5        # 4 heuristics + up to 2 from RoBERTa + 0.5 phrase
    AI_PHRASE_WEIGHT: float = 0.5              # half-weight — softer than structural signals (#160)
    SYNC_FEE_STRONG_CONFIDENCE_THRESHOLD: int = 60  # popularity_score ≥ this → "strong" tier confidence (#116)
    # Section labels where repetition/rhyme is structurally expected — excluded
    # from per-section AI signal scoring so chorus repeats don't inflate flags (#158).
    CHORUS_OUTRO_LABELS: frozenset[str] = frozenset({
        "chorus", "refrain", "hook", "outro",
    })


# Module-level singleton — import directly, never instantiate.
CONSTANTS = SystemConstants()

# Genre-aware LRA context ranges (LU low, LU high) — soft recommendation only (#99).
# Kept as a module-level dict (not on SystemConstants) to avoid Pydantic mutable-default errors.
GENRE_LRA_RANGES: dict[str, tuple[float, float]] = {
    "pop":        (4.0,  8.0),
    "hip-hop":    (4.0,  8.0),
    "electronic": (5.0,  9.0),
    "rock":       (6.0, 10.0),
    "country":    (6.0, 10.0),
    "r&b":        (5.0,  9.0),
    "jazz":       (10.0, 16.0),
    "classical":  (12.0, 20.0),
    "cinematic":  (12.0, 18.0),
    "ambient":    (10.0, 18.0),
    "folk":       (8.0,  14.0),
}
GENRE_LRA_DEFAULT: tuple[float, float] = (6.0, 14.0)  # fallback for unrecognised genres


# Sync fee scenario multipliers (#110).
# Defined as a module-level constant (not inside SystemConstants) to avoid
# Pydantic's mutable-default error on nested dicts — same pattern as GENRE_LRA_RANGES.
SYNC_FEE_MULTIPLIERS: dict[str, dict[str, float]] = {
    "usage": {
        "Documentary":   1.0,
        "TV Scene":      1.5,
        "Ad (30s)":      2.5,
        "Trailer":       3.5,
        "Online/Social": 0.6,
    },
    "territory": {
        "US-only":   1.0,
        "Europe":    1.3,
        "Worldwide": 2.0,
    },
    "exclusivity": {
        "Non-exclusive": 1.0,
        "Exclusive":     1.8,
    },
}


# Placement profiles for compliance threshold overrides (#107).
# Defined as module-level constants (not inside SystemConstants) to avoid
# Pydantic's mutable-default error — same pattern as SYNC_FEE_MULTIPLIERS.
# Import PlacementProfile lazily to avoid circular imports with core.models.
def _build_placement_profiles() -> "dict[str, PlacementProfile]":
    from core.models import PlacementProfile  # noqa: PLC0415
    return {
        "Standard": PlacementProfile(
            name="Standard",
            intro_max_seconds=15,
            bar_energy_delta_min=0.10,
            sting_rms_drop_ratio=0.75,
            sting_spike_factor=3.0,
        ),
        "Broadcast (EBU R128)": PlacementProfile(
            name="Broadcast (EBU R128)",
            intro_max_seconds=10,
            bar_energy_delta_min=0.12,
            sting_rms_drop_ratio=0.08,
            sting_spike_factor=2.5,
        ),
        "Commercial (30s spot)": PlacementProfile(
            name="Commercial (30s spot)",
            intro_max_seconds=5,
            bar_energy_delta_min=0.15,
            sting_rms_drop_ratio=0.10,
            sting_spike_factor=3.0,
        ),
        "Trailer": PlacementProfile(
            name="Trailer",
            intro_max_seconds=8,
            bar_energy_delta_min=0.20,
            sting_rms_drop_ratio=0.05,
            sting_spike_factor=4.0,
        ),
        "Library / Background": PlacementProfile(
            name="Library / Background",
            intro_max_seconds=15,
            bar_energy_delta_min=0.08,
            sting_rms_drop_ratio=0.04,
            sting_spike_factor=1.5,
        ),
    }


PLACEMENT_PROFILES: "dict[str, PlacementProfile]" = _build_placement_profiles()
PLACEMENT_PROFILE_DEFAULT: str = "Standard"


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
    spotify_client_id: str = Field(
        default="",
        description="Spotify Web API client ID for popularity score lookup. "
                    "Create an app at https://developer.spotify.com/dashboard",
    )
    spotify_client_secret: str = Field(
        default="",
        description="Spotify Web API client secret (pairs with spotify_client_id).",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token for private model access (optional).",
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key for LLM inference (issue planning workflow). "
                    "Get one at https://console.groq.com",
    )

    # ---- MusicBrainz API (PRO lookup) -----------------------------------------
    # MusicBrainz requires a descriptive User-Agent for all API requests.
    # Format: "AppName/Version (contact_url_or_email)"
    musicbrainz_app: str = Field(
        default="sync-safe-forensic-portal/1.0 (https://github.com/borrvick/sync-safe)",
        description="User-Agent string sent to MusicBrainz API. "
                    "Set to your app name, version, and contact URL/email.",
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
