"""
core/models.py
Typed domain models for the Sync-Safe pipeline.

Design rules:
- Every model uses Pydantic BaseModel for validation + serialisation.
- Categorical fields use Literal types — a typo becomes a ValidationError,
  not a silent bug that reaches the UI.
- frozen=True on immutable results: a StructureResult produced by allin1
  should never be mutated by a downstream service.
- AudioBuffer holds raw bytes for in-process use; call .to_bytesio() when
  a service needs an io.BytesIO. Exclude `raw` when serialising to JSON
  (it's binary data, not a domain value): model.model_dump(exclude={'raw'}).
- All models expose .to_dict() as a convenience alias over model_dump() so
  future API layers don't need to know the Pydantic internals.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Type aliases — used as Literal constraints throughout
# ---------------------------------------------------------------------------

IssueType    = Literal["EXPLICIT", "BRAND", "LOCATION", "VIOLENCE", "DRUGS"]
Confidence   = Literal["confirmed", "potential"]
Severity     = Literal["hard", "soft"]
EndingType   = Literal["sting", "fade", "cut"]
AIVerdict    = Literal["Likely Human", "Uncertain", "Likely AI", "Insufficient data"]
ForensicVerdict = Literal["AI", "Likely AI", "Likely Not AI", "Not AI", "Insufficient data"]
AudioSource  = Literal["youtube", "file", "bandcamp", "soundcloud", "direct"]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

class AudioBuffer(BaseModel):
    """
    In-memory audio representation passed between pipeline stages.

    `raw` holds the WAV/MP3 bytes as ingested — no resampling is done here.
    Each service is responsible for resampling to its required rate via
    librosa.load(buffer.to_bytesio(), sr=CONSTANTS.SAMPLE_RATE).

    `metadata` carries title/artist extracted at ingestion time (e.g. from
    yt-dlp's --dump-json for YouTube sources). This is the primary source
    for the LRCLib lyrics lookup since embedded audio tags are stripped
    during the yt-dlp → ffmpeg transcode.
    """

    model_config = ConfigDict(frozen=True)

    raw: bytes = Field(repr=False)                          # excluded from repr; can be 50 MB
    sample_rate: int = Field(default=22_050)
    label: str = Field(default="")                          # display name shown in the UI
    metadata: dict[str, str] = Field(default_factory=dict)  # title, artist from ingestion
    engagement: dict[str, int] = Field(default_factory=dict)  # view_count, like_count, etc. from yt-dlp
    source: AudioSource = Field(default="file")             # "youtube" = lossy MP3 transcode; "file" = direct upload

    def to_bytesio(self) -> io.BytesIO:
        """Return a fresh BytesIO cursor at position 0."""
        return io.BytesIO(self.raw)

    def to_array(self, sr: int, mono: bool = True) -> tuple["np.ndarray", int]:
        """
        Decode raw audio bytes to a numpy array at the requested sample rate.

        This is the single librosa.load call point for the entire pipeline.
        To swap librosa for a different audio backend, change this method only.

        Args:
            sr:   Target sample rate in Hz. Pass sr=None to preserve native rate.
            mono: Mix to mono when True; preserve channels when False.

        Returns:
            (y, sr) tuple — same semantics as librosa.load.

        Raises:
            ModelInferenceError: if the audio cannot be decoded.
        """
        try:
            import librosa
            y, actual_sr = librosa.load(self.to_bytesio(), sr=sr, mono=mono)
            return y, actual_sr
        except Exception as exc:
            from core.exceptions import ModelInferenceError  # local to avoid circular import
            raise ModelInferenceError(
                "AudioBuffer.to_array: decode failed.",
                context={"sr": sr, "mono": mono, "error": str(exc)},
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Serialise without the raw bytes (not meaningful in JSON)."""
        return self.model_dump(exclude={"raw"})


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class TranscriptSegment(BaseModel):
    """A single time-stamped segment from Whisper."""

    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0.0)
    end: float   = Field(ge=0.0)
    text: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Structure Analysis
# ---------------------------------------------------------------------------

class Section(BaseModel):
    """An allin1 structural segment (intro, verse, chorus, etc.)."""

    model_config = ConfigDict(frozen=True)

    label: str
    start: float = Field(ge=0.0)
    end: float   = Field(ge=0.0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class StructureResult(BaseModel):
    """Full output of the structure analysis stage."""

    model_config = ConfigDict(frozen=True)

    bpm: float | str            # float when detected; str error message when not
    key: str                    # e.g. "C# Major"
    sections: list[Section]     = Field(default_factory=list)
    beats: list[float]          = Field(default_factory=list)
    metadata: dict[str, str]    = Field(default_factory=dict)  # title, artist

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Forensics
# ---------------------------------------------------------------------------

class AiSegment(BaseModel):
    """One time-windowed AI-probability estimate within a track."""

    model_config = ConfigDict(frozen=True)

    start_s: float      # window start (seconds from track start)
    end_s: float        # window end
    probability: float  # [0.0, 1.0]; higher = more AI-like

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ForensicsResult(BaseModel):
    """Output of the AI-humanity forensics stage."""

    model_config = ConfigDict(frozen=True)

    # Individual signal scores (0.0–1.0 where 1.0 = most AI-like)
    c2pa_flag: bool         = False     # True → born-AI assertion found in manifest
    ibi_variance: float     = 1.0       # inter-beat interval variance
    loop_score: float       = 0.0       # highest cross-correlation across 4-bar windows
    loop_autocorr_score: float = 0.0    # onset autocorrelation loop repetition score
    spectral_slop: float    = 0.0       # anomalous energy above SPECTRAL_SLOP_HZ
    synthid_score: float    = 0.0       # phase coherence in 18–22 kHz band
    centroid_instability_score: float = -1.0  # mean within-interval centroid CV; -1 = not computed
    harmonic_ratio_score: float = -1.0        # mean HNR within sustained intervals; -1 = not computed
    # New signals (2026-03-21) — calibrated from ISMIR TISMIR 2025 + arXiv 2506.19108
    kurtosis_variability: float = -1.0        # variance of per-frame mel-band kurtosis; -1 = not computed
    decoder_peak_score: float = 0.0           # periodic deconvolution peak density in 1–16 kHz band
    spectral_centroid_mean: float = -1.0      # mean spectral centroid in Hz across the track
    ai_probability: float = 0.0               # weighted probability score [0.0–1.0] used for verdict
    # Structural / instrumental signals (2026-03-21) — pending calibration, weights=0 until thresholds set
    self_similarity_entropy: float = -1.0     # Shannon entropy of chroma recurrence matrix; low = repetitive AI structure
    noise_floor_ratio: float = -1.0           # quiet-frame RMS / mean RMS; near-zero = VST render (no room noise)
    onset_strength_cv: float = -1.0           # CV of onset strength envelope; low = uniform AI dynamics
    spectral_flatness_var: float = -1.0       # variance of Wiener entropy over time; low = AI synth uniformity
    subbeat_grid_deviation: float = -1.0      # variance of onset-to-nearest-16th-note offset; low = on-grid
    pitch_quantization_score: float = -1.0    # mean abs cents deviation from 12-TET; near-zero = AI pitch-perfect
    ultrasonic_noise_ratio: float = -1.0      # energy ratio in 20–22 kHz band; elevated = diffusion residue (-1 = not computed)
    infrasonic_energy_ratio: float = -1.0     # energy ratio in 1–20 Hz band; elevated = AI math drift / DC bias (-1 = not computed)
    phase_coherence_differential: float = -1.0  # LF coherence − HF coherence; positive = AI phase pattern (-1 = mono/not computed)
    plr_std: float = -1.0                        # std of per-window peak-to-loudness ratio; low = frozen density (AI) (-1 = too short)
    voiced_noise_floor: float = -1.0             # mean spectral flatness in voiced 4–12 kHz frames; low = AI clean synthesis (-1 = non-vocal/not computed)
    is_vocal: bool = False                       # True → pyin detected vocal content; routes vocal scoring path
    c2pa_origin: str = ""                        # "ai" | "daw" | "unknown" | "" (no manifest)

    ai_segments: list[AiSegment] = Field(default_factory=list)  # per-window heatmap data

    flags: list[str]        = Field(default_factory=list)  # human-readable flag labels
    forensic_notes: list[str] = Field(default_factory=list)  # secondary context shown below verdict
    verdict: ForensicVerdict = "Likely Not AI"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

class ComplianceFlag(BaseModel):
    """A single issue surfaced during lyric/structural compliance checking."""

    model_config = ConfigDict(frozen=True)

    timestamp_s: int                        # seconds from track start
    issue_type: IssueType
    text: str                               # flagged excerpt or brand name
    recommendation: str                     # supervisor action guidance
    confidence: Confidence = "confirmed"    # confirmed = NER hit; potential = keyword
    severity: Severity     = "soft"         # hard = deal-breaker in any context;
                                            # soft = placement-dependent, director's call

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class StingResult(BaseModel):
    """Result of the sting/ending-type check."""

    model_config = ConfigDict(frozen=True)

    ending_type: EndingType
    sync_ready: bool
    final_energy_ratio: float   = Field(ge=0.0)
    flag: bool                  = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EnergyEvolutionResult(BaseModel):
    """Result of the 4-8 bar energy evolution check."""

    model_config = ConfigDict(frozen=True)

    stagnant_windows: int   = 0     # count of windows below ENERGY_DELTA_MIN
    total_windows: int      = 0
    flag: bool              = False
    detail: str             = ""    # e.g. "3 of 12 windows below 10% delta"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class IntroResult(BaseModel):
    """Result of the intro-length check."""

    model_config = ConfigDict(frozen=True)

    intro_seconds: float    = 0.0
    flag: bool              = False
    source: str             = ""    # "allin1" | "whisper_fallback" | "none"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ComplianceReport(BaseModel):
    """Aggregated output of all sync readiness compliance checks."""

    model_config = ConfigDict(frozen=True)

    flags: list[ComplianceFlag]         = Field(default_factory=list)
    sting: StingResult                  = Field(default_factory=lambda: StingResult(ending_type="cut", sync_ready=False, final_energy_ratio=0.0))
    evolution: EnergyEvolutionResult    = Field(default_factory=EnergyEvolutionResult)
    intro: IntroResult                  = Field(default_factory=IntroResult)
    grade: str                          = "N/A"     # A–F or "N/A"

    @property
    def confirmed_flags(self) -> list[ComplianceFlag]:
        return [f for f in self.flags if f.confidence == "confirmed"]

    @property
    def hard_flags(self) -> list[ComplianceFlag]:
        """Confirmed flags that are absolute deal-breakers in any placement context."""
        return [f for f in self.flags if f.confidence == "confirmed" and f.severity == "hard"]

    @property
    def soft_flags(self) -> list[ComplianceFlag]:
        """Confirmed flags that are placement-dependent — sync director's call."""
        return [f for f in self.flags if f.confidence == "confirmed" and f.severity == "soft"]

    @property
    def potential_flags(self) -> list[ComplianceFlag]:
        return [f for f in self.flags if f.confidence == "potential"]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Authorship
# ---------------------------------------------------------------------------

class AuthorshipResult(BaseModel):
    """Output of the AI lyric authorship detection stage."""

    model_config = ConfigDict(frozen=True)

    verdict: AIVerdict                  = "Likely Human"
    signal_count: int                   = 0
    roberta_score: Optional[float]      = None  # None when model not run or insufficient data
    feature_notes: list[str]            = Field(default_factory=list)
    scores: dict[str, Optional[float]]  = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Theme & Mood
# ---------------------------------------------------------------------------

class ThemeMoodResult(BaseModel):
    """Output of the theme and mood analysis stage."""

    model_config = ConfigDict(frozen=True)

    themes: list[str]               = Field(default_factory=list)
    mood: str                       = ""
    confidence: float               = 0.0
    groq_enriched: bool             = False
    raw_keywords: list[str]         = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Discovery & Legal
# ---------------------------------------------------------------------------

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

    # Dialogue-readiness
    dialogue_score: float       # 0.0–1.0; fraction of energy outside 300–3 kHz band
    dialogue_label: str         # "Dialogue-Ready" | "Mixed" | "Dialogue-Heavy"

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


class TrackCandidate(BaseModel):
    """A similar track returned by the discovery service."""

    model_config = ConfigDict(frozen=True)

    title: str
    artist: str
    youtube_url: Optional[str]  = None  # None when yt-dlp URL lookup fails
    similarity: float           = 0.0   # 0.0–1.0; rank-derived from Last.fm order

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

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class MetadataValidationResult(BaseModel):
    """
    Result of pre-flight track rights metadata validation.

    Produced by services/metadata_validator.py before the scan pipeline runs.
    Stored in AnalysisResult so the compliance report can surface intake issues.
    """

    model_config = ConfigDict(frozen=True)

    valid: bool                         # True when all checks pass
    missing_fields: list[str]           = Field(default_factory=list)
    split_total: float                  = 0.0   # sum of supplied writer splits
    split_error: Optional[float]        = None  # abs deviation from 100.0, or None if not supplied
    isrc_valid: bool                    = True  # True when ISRC matches ISO 3901 or was not provided
    rejection_reason: Optional[str]     = None  # human-readable summary; None when valid

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class LegalLinks(BaseModel):
    """PRO repertory search URLs and inferred PRO match for a given track."""

    model_config = ConfigDict(frozen=True)

    ascap: str  = ""
    bmi: str    = ""
    sesac: str  = ""

    # Populated by services/pro_lookup.py — None when MusicBrainz returns no hit
    isrc: Optional[str]      = None  # e.g. "US-ABC-23-12345"
    pro_match: Optional[str] = None  # e.g. "ASCAP/BMI (US)", "PRS (UK)"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Stem / alternate mix validation
# ---------------------------------------------------------------------------

class StemValidationResult(BaseModel):
    """Output of stereo/mono compatibility and phase alignment analysis."""

    model_config = ConfigDict(frozen=True)

    mono_compatible: bool        # True if cancellation < MONO_CANCELLATION_DB_WARN
    phase_correlation: float     # Pearson L/R correlation [-1, 1]
    cancellation_db: float       # dB loss in mono sum vs stereo RMS; negative = cancellation
    mid_side_ratio: float        # Side/Mid energy ratio; -1.0 if mono or undefined
    flags: list[str]             = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Top-level pipeline result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """
    The complete output of a single pipeline run.

    This is the only object app.py needs to render the full UI.
    Serialise with .to_dict() (excludes audio.raw) for JSON storage or
    a future REST API response.
    """

    audio: AudioBuffer
    structure: Optional[StructureResult]                    = None
    forensics: Optional[ForensicsResult]                    = None
    transcript: list[TranscriptSegment]                     = Field(default_factory=list)
    compliance: Optional[ComplianceReport]                  = None
    authorship: Optional[AuthorshipResult]                  = None
    similar_tracks: list[TrackCandidate]                    = Field(default_factory=list)
    legal: Optional[LegalLinks]                             = None
    popularity: Optional[PopularityResult]                  = None
    audio_quality: Optional[AudioQualityResult]             = None
    metadata_validation: Optional[MetadataValidationResult] = None
    sync_cuts: list[SyncCut]                                = Field(default_factory=list)
    stem_validation: Optional[StemValidationResult]         = None
    theme_mood: Optional[ThemeMoodResult]                   = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to a plain dict suitable for JSON storage or an API response.
        Raw audio bytes are excluded — they are not a domain value.
        """
        data = self.model_dump(exclude={"audio": {"raw"}})
        return data

    def to_json(self) -> str:
        """Serialise to a JSON string (audio bytes excluded)."""
        return self.model_dump_json(exclude={"audio": {"raw"}})
