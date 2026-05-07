"""
core/report.py

TrackReport — flat, database-ready record of every data field produced by a
single pipeline run.

Design rules:
- One field per data point (no nesting). JSON blob columns hold one-to-many
  relations as serialised JSON strings; they become child DB tables later.
- All Optional fields default to None so partial pipeline runs (e.g. no
  compliance check) still produce a valid record.
- sentinel value -1.0 is preserved from ForensicsResult for uncomputed signals
  so downstream filtering can distinguish "not computed" from "zero".
- track_id = SHA-256( title | artist | duration ) truncated to 12 hex chars —
  stable across re-scans of the same track, collision risk negligible for a
  library of any realistic size.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def _track_id(title: str, artist: str, duration: float) -> str:
    payload = f"{title.lower().strip()}|{artist.lower().strip()}|{duration:.1f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _dumps(obj: Any) -> str:
    """
    Serialise an object to a compact JSON string.

    Handles:
    - None                                → "null"
    - dict (Pydantic/dataclass/plain vals) → JSON object string
    - list of Pydantic models             → JSON array of .model_dump() results
    - list of plain values (str, float, tuple, dict, …) → JSON array
    - empty list / dict                   → "[]" / "{}"
    """
    if obj is None:
        return "null"
    if isinstance(obj, dict):
        if not obj:
            return "{}"

        def _val(v: Any) -> Any:
            if hasattr(v, "model_dump"):
                return v.model_dump()
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return dataclasses.asdict(v)
            return v

        return json.dumps({k: _val(v) for k, v in obj.items()}, separators=(",", ":"))
    # list branch
    if not obj:
        return "[]"
    if hasattr(obj[0], "model_dump"):
        return json.dumps([item.model_dump() for item in obj], separators=(",", ":"))
    return json.dumps(list(obj), separators=(",", ":"))


class TrackReport(BaseModel):
    """
    Flat record containing every data point produced by the pipeline.

    Scalar fields map 1:1 to future DB columns.
    JSON blob fields hold one-to-many relations; they become child tables
    once a database is added (see project backlog).
    """

    model_config = ConfigDict(frozen=True)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    track_id: str                   # SHA-256(title|artist|duration)[:12]
    scan_timestamp: str             # ISO 8601 UTC

    # ------------------------------------------------------------------
    # Audio / Ingestion
    # ------------------------------------------------------------------
    title: str          = ""
    artist: str         = ""
    source: str         = ""        # "youtube" | "file" | "bandcamp" | etc.
    sample_rate: int    = 0
    yt_view_count: int  = 0
    yt_like_count: int  = 0

    # ------------------------------------------------------------------
    # Structure / Musicality
    # ------------------------------------------------------------------
    bpm: Optional[float]    = None
    key: str                = ""
    section_count: int      = 0
    beat_count: int         = 0
    duration_s: float       = 0.0   # last beat or last section end
    intro_s: float          = 0.0   # combined intro section length
    verse_count: int        = 0
    chorus_count: int       = 0
    bridge_count: int       = 0

    # ------------------------------------------------------------------
    # Forensics — Verdict
    # ------------------------------------------------------------------
    forensic_verdict: str       = ""
    ai_probability: float       = 0.0
    forensic_flag_count: int    = 0
    c2pa_flag: bool             = False
    c2pa_origin: str            = ""
    is_vocal: bool              = False

    # ------------------------------------------------------------------
    # Forensics — Raw Signals (all scores; -1.0 = not computed)
    # ------------------------------------------------------------------
    ibi_variance: float                     = 1.0
    loop_score: float                       = 0.0
    loop_autocorr_score: float              = 0.0
    repetition_index: Optional[float]       = None   # blended 0.6*loop + 0.4*autocorr
    spectral_slop: float                    = 0.0
    synthid_score: float                    = 0.0
    centroid_instability_score: float       = -1.0
    harmonic_ratio_score: float             = -1.0
    kurtosis_variability: float             = -1.0
    decoder_peak_score: float               = 0.0
    spectral_centroid_mean: float           = -1.0
    self_similarity_entropy: float          = -1.0
    noise_floor_ratio: float                = -1.0
    onset_strength_cv: float                = -1.0
    spectral_flatness_var: float            = -1.0
    subbeat_grid_deviation: float           = -1.0
    pitch_quantization_score: float         = -1.0
    ultrasonic_noise_ratio: float           = -1.0
    infrasonic_energy_ratio: float          = -1.0
    phase_coherence_differential: float     = -1.0
    plr_std: float                          = -1.0
    voiced_noise_floor: float               = -1.0

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------
    compliance_grade: str               = ""
    total_flag_count: int               = 0
    confirmed_flag_count: int           = 0
    potential_flag_count: int           = 0
    hard_flag_count: int                = 0
    soft_flag_count: int                = 0
    # Sting enrichment (#103, #104, #108)
    sting_flag: Optional[bool]          = None
    sting_ending_type: str              = ""
    sting_final_energy_ratio: Optional[float] = None
    sting_fade_severity: float          = 0.0
    sting_fade_tail_seconds: float      = 0.0
    sting_cut_type: Optional[str]       = None
    sting_norm_slope: float             = 0.0
    sting_onset_spike_factor: float     = 0.0
    # Energy evolution enrichment (#106, #108)
    energy_evolution_flag: Optional[bool]     = None
    stagnant_windows: int               = 0
    total_windows: int                  = 0
    energy_evolution_detail: str        = ""
    energy_evolution_ending_section: Optional[str] = None
    # Intro enrichment (#105)
    intro_flag: Optional[bool]          = None
    intro_seconds: float                = 0.0
    intro_source: str                   = ""
    intro_confidence: str               = ""

    # ------------------------------------------------------------------
    # Authorship
    # ------------------------------------------------------------------
    authorship_verdict: str             = ""
    authorship_signal_count: float      = 0.0   # float: supports 0.5-weight phrase signal (#160)
    authorship_skip_reason: Optional[str] = None
    roberta_score: Optional[float]      = None
    burstiness_score: Optional[float]   = None
    unique_word_ratio: Optional[float]  = None
    rhyme_density: Optional[float]      = None
    repetition_score: Optional[float]   = None

    # ------------------------------------------------------------------
    # Theme & Mood
    # ------------------------------------------------------------------
    mood: str                       = ""
    theme_confidence: float         = 0.0
    top_category: str               = ""        # "energy"|"emotional"|"seasonal"|"" (#167)
    mood_summary: Optional[str]     = None      # Groq one-sentence summary (#169)
    groq_enriched: bool             = False

    # ------------------------------------------------------------------
    # Audio Quality / Loudness
    # ------------------------------------------------------------------
    integrated_lufs: Optional[float]    = None
    true_peak_dbfs: Optional[float]     = None
    loudness_range_lu: Optional[float]  = None
    loudness_verdict: str               = ""    # e.g. "Broadcast-ready" (#95)
    true_peak_warning: Optional[bool]   = None
    delta_spotify: Optional[float]      = None
    delta_apple_music: Optional[float]  = None
    delta_youtube: Optional[float]      = None
    delta_broadcast: Optional[float]    = None
    gain_spotify_db: Optional[float]    = None  # (#94)
    gain_apple_music_db: Optional[float] = None
    gain_youtube_db: Optional[float]    = None
    gain_broadcast_db: Optional[float]  = None
    dialogue_score: Optional[float]     = None
    dialogue_label: str                 = ""
    vo_headroom_db: Optional[float]     = None  # (#92)

    # ------------------------------------------------------------------
    # Stem Validation
    # ------------------------------------------------------------------
    mono_compatible: Optional[bool]     = None
    phase_correlation: Optional[float]  = None
    cancellation_db: Optional[float]    = None
    mid_side_ratio: Optional[float]     = None
    stem_flag_count: int                = 0

    # ------------------------------------------------------------------
    # Popularity & Cost
    # ------------------------------------------------------------------
    popularity_score: Optional[int]     = None
    popularity_tier: str                = ""
    lastfm_listeners: int               = 0
    lastfm_playcount: int               = 0
    spotify_score: Optional[int]        = None
    sync_cost_low: Optional[int]        = None
    sync_cost_high: Optional[int]       = None

    # ------------------------------------------------------------------
    # Legal
    # ------------------------------------------------------------------
    isrc: Optional[str]             = None
    pro_match: Optional[str]        = None
    pro_confidence: Optional[str]   = None      # "High"|"Medium"|"Low" (#118)

    # ------------------------------------------------------------------
    # Metadata Validation
    # ------------------------------------------------------------------
    metadata_valid: Optional[bool]      = None
    missing_fields_count: int           = 0
    split_total: Optional[float]        = None
    split_error: Optional[float]        = None
    isrc_valid: Optional[bool]          = None

    # ------------------------------------------------------------------
    # JSON Blobs — one-to-many relations (become child DB tables later)
    # ------------------------------------------------------------------
    compliance_flags_json: str              = Field(default="[]")   # list[ComplianceFlag]
    ai_segments_json: str                   = Field(default="[]")   # list[AiSegment]
    loop_window_scores_json: str            = Field(default="[]")   # list[tuple[float,float]] (#142)
    section_similarities_json: str          = Field(default="{}")   # dict[str,SectionRepetition] (#143)
    section_internal_repetition_json: str   = Field(default="{}")   # dict[str,SectionRepetition] (#145)
    sections_json: str                      = Field(default="[]")   # list[Section]
    transcript_json: str                    = Field(default="[]")   # list[TranscriptSegment]
    similar_tracks_json: str                = Field(default="[]")   # list[TrackCandidate]
    sync_cuts_json: str                     = Field(default="[]")   # list[SyncCut]
    forensic_notes_json: str                = Field(default="[]")   # list[str]
    forensic_flags_json: str                = Field(default="[]")   # list[str]
    themes_json: str                        = Field(default="[]")   # list[str]
    theme_keywords_json: str                = Field(default="[]")   # list[str]
    theme_scores_json: str                  = Field(default="{}")   # dict[str,float] (#167)
    per_section_authorship_json: str        = Field(default="{}")   # dict[str,SectionAuthorshipResult] (#156)
    platform_metrics_json: str              = Field(default="{}")   # dict[str,int] (engagement)
    section_loudness_json: str              = Field(default="[]")   # list[dict] (#96)
    section_dialogue_json: str              = Field(default="[]")   # list[dict] (#91)
    genre_lra_context_json: str             = Field(default="null") # dict|null (#99)
    energy_stagnant_timestamps_json: str    = Field(default="[]")   # list[float] (#108)
    energy_per_window_contrasts_json: str   = Field(default="[]")   # list[float] (#108)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
