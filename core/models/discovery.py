"""
core/models/discovery.py
Discovery, popularity, placement, and sync-cut output models.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class AudioQualityResult(BaseModel):
    """
    Broadcast loudness and dialogue-readiness metrics.

    LUFS figures follow ITU-R BS.1770-4 / EBU R128.
    Dialogue score measures how much mid-frequency energy (300–3 kHz) the
    track contributes relative to the full spectrum — higher = sits cleaner
    under voiceover without competing with spoken dialogue.
    """

    model_config = ConfigDict(frozen=True)

    # LUFS / broadcast loudness
    integrated_lufs: float      # integrated programme loudness (LUFS)
    true_peak_dbfs: float       # highest inter-sample peak (dBFS)
    loudness_range_lu: float    # LRA — dynamic range in Loudness Units

    # Platform deltas (positive = louder than target, negative = quieter)
    delta_spotify: float
    delta_apple_music: float
    delta_youtube: float
    delta_broadcast: float

    # True peak warning
    true_peak_warning: bool     # True if true peak > TRUE_PEAK_WARN_DBFS

    # Gain adjustment to reach each platform target (negative = turn down, positive = turn up)
    gain_spotify_db:     float  # dB to apply to hit Spotify target (-delta_spotify)  (#94)
    gain_apple_music_db: float  # dB to apply to hit Apple Music target               (#94)
    gain_youtube_db:     float  # dB to apply to hit YouTube target                   (#94)
    gain_broadcast_db:   float  # dB to apply to hit broadcast target                 (#94)

    # Top-line loudness verdict (#95)
    loudness_verdict: str       # "Broadcast-ready" | "Streaming-hot" | "Streaming-ready" |
                                # "Needs mastering" | "Clipping risk"

    # Dialogue-readiness
    dialogue_score: float       # 0.0–1.0; fraction of energy outside 300–3 kHz band
    dialogue_label: str         # "Dialogue-Ready" | "Mixed" | "Dialogue-Heavy"

    # VO headroom estimate (#92)
    # Estimated dB of headroom available for voiceover; scaled from dialogue_score.
    # None when not yet computed (backward-compatible with older results).
    vo_headroom_db: Optional[float] = None

    # Per-section LUFS and LRA breakdown (#96)
    # Each entry: {label, start_s, end_s, integrated_lufs, lra_lu, is_peak?}
    # Empty list when sections were not provided or all were too short for measurement.
    section_loudness: list[dict[str, Any]] = Field(default_factory=list)

    # Genre-aware LRA soft recommendation (#99)
    # None when genre is unknown or not provided — no recommendation shown in that case.
    genre_lra_context: Optional[dict[str, Any]] = None

    # Per-section dialogue compatibility scores (#91)
    # Each entry: {label, start_s, end_s, dialogue_score, dialogue_label}
    # Sections shorter than DIALOGUE_MIN_SECTION_DUR_S are omitted.
    section_dialogue: list[dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PopularityResult(BaseModel):
    """
    Multi-signal popularity result for the scanned track.

    popularity_score is a normalised 0–100 value derived from whichever
    signals are available (Last.fm, Spotify, platform engagement).  tier
    is derived from popularity_score so the tier cannot be dragged down by
    a single bad Last.fm lookup.

    platform_metrics holds raw per-platform engagement counts keyed by
    field name (e.g. "view_count", "like_count", "share_count").  Values
    are source-dependent and may be absent when the scan is from a file
    upload rather than a URL.

    Designed to be re-fetched independently of the static forensic signals
    once result caching is added (issue #84) — popularity data goes stale,
    forensic data does not.
    """

    model_config = ConfigDict(frozen=True)

    listeners: int                                              # Last.fm unique listener count
    playcount: int                                              # Last.fm total scrobble count
    spotify_score: Optional[int]                               # Spotify popularity 0–100 (None if unavailable)
    platform_metrics: dict[str, int] = Field(default_factory=dict)  # raw engagement: view_count, like_count, etc.
    popularity_score: int                   # blended normalised 0–100 score
    tier: str                               # "Emerging" | "Regional" | "Mainstream" | "Global"
    sync_cost_low: int                      # estimated sync fee lower bound (USD)
    sync_cost_high: int                     # estimated sync fee upper bound (USD)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PlacementProfile(BaseModel):
    """
    Threshold overrides for a specific sync placement context (#107).

    When a profile is selected in the sidebar, compliance check functions
    use profile values instead of the global CONSTANTS, allowing stricter
    or looser rules per placement type.

    Default field values match the current CONSTANTS so that callers can
    always construct a PlacementProfile without changing behavior.
    """

    model_config = ConfigDict(frozen=True)

    name:                  str   = "Standard"
    intro_max_seconds:     int   = 15     # matches CONSTANTS.INTRO_MAX_SECONDS
    bar_energy_delta_min:  float = 0.10   # matches CONSTANTS.ENERGY_DELTA_MIN
    sting_rms_drop_ratio:  float = 0.75   # matches CONSTANTS.STING_RMS_DROP_RATIO
    sting_spike_factor:    float = 3.0    # matches CONSTANTS.STING_SPIKE_FACTOR


class TrackCandidate(BaseModel):
    """A similar track returned by the discovery service."""

    model_config = ConfigDict(frozen=True)

    title: str
    artist: str
    youtube_url: Optional[str]   = None   # None when yt-dlp URL lookup fails
    similarity: float            = 0.0    # 0.0–1.0; rank-derived from Last.fm order
    popularity_tier: Optional[str] = None  # "Emerging"|"Regional"|"Mainstream"|"Global"; None if unknown (#124)
    source: str                  = "lastfm"  # "lastfm" | "spotify" | "audio" (#129)
    sync_ready: Optional[bool]   = None   # True/False after on-demand check; None = unchecked (#127)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SyncCut(BaseModel):
    """
    A suggested edit point for a standard ad/TV format duration.

    Produced by services/sync_cut.py from allin1 structure + beat grid.
    """

    model_config = ConfigDict(frozen=True)

    duration_s: int             # target format duration (15 / 30 / 60)
    start_s: float              # recommended cut-in point (seconds from track start)
    end_s: float                # recommended cut-out point (seconds from track start)
    actual_duration_s: float    # actual edit length (end_s − start_s)
    confidence: float           # 0.0–1.0; higher = better section-boundary alignment
    note: str                   # e.g. "Chorus at 0:32 → natural ending at 1:00"
    # Top-N ranking and per-criterion breakdown (#148, #155)
    rank: int                                   = 1    # 1 = best, 2 = second-best, 3 = third-best
    score_breakdown: dict[str, bool]            = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
