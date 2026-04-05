"""
core/protocols.py
Structural interfaces (Protocols) for every swappable integration.

Design rules:
- Use typing.Protocol, not ABC. Concrete classes do NOT need to inherit from
  these — they just need to match the method signatures (duck typing). This
  keeps services decoupled from the core package.
- @runtime_checkable allows isinstance() checks in tests and pipeline wiring.
- Every method signature uses domain models from core.models — never raw
  dicts or bytes — so swapping an implementation never requires changing
  call sites in pipeline.py or app.py.
- Protocols are intentionally minimal: one public method each. Anything more
  is a sign the protocol is mixing concerns.

Swap guide:
  AudioProvider      → swap yt-dlp for a direct S3 fetch or SoundCloud API
  YtDlpProvider      → swap yt-dlp + ffmpeg for Piped API, RapidAPI, or yt-dlp-server
  TranscriptionProvider → swap Whisper for Deepgram, AssemblyAI, or Azure STT
  StructureAnalyzer  → swap allin1 for a custom BPM/section detector
  ForensicsAnalyzer  → swap the librosa heuristics for a trained classifier
  ComplianceChecker  → swap sync readiness rules for a different editorial standard
  AuthorshipAnalyzer → swap RoBERTa for GPTZero API or a fine-tuned model
  TrackDiscovery     → swap Last.fm for Spotify, Soundcharts, or internal DB
  LegalLinksProvider → swap static URL templates for a live licensing API
  SyncCutProvider           → swap heuristic scorer for an ML-based edit point detector
  TagInjectorProvider       → swap mutagen for a cloud tagging API or a different tag schema
  PlatformExportProvider    → swap built-in CSV templates for a paid sync API (Songtradr, etc.)
  MetadataValidatorProvider → swap local rules for AllTrack, DDEX, or SoundExchange
  ProLookupProvider  → swap MusicBrainz for a paid metadata provider
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union, runtime_checkable
from typing import Protocol

if TYPE_CHECKING:
    import streamlit as st  # UploadedFile type hint only — no runtime dependency

from core.models import (
    AnalysisResult,
    AudioBuffer,
    AuthorshipResult,
    ComplianceReport,
    ForensicsResult,
    LegalLinks,
    MetadataValidationResult,
    StructureResult,
    SyncCut,
    ThemeMoodResult,
    TrackCandidate,
    TranscriptSegment,
)


# ---------------------------------------------------------------------------
# Audio ingestion
# ---------------------------------------------------------------------------

@runtime_checkable
class AudioProvider(Protocol):
    """
    Load audio from any source into an in-memory AudioBuffer.

    Implementations: services/ingestion/ (Ingestion class)
    Swap candidates:  S3 presigned URL fetch, SoundCloud API, local file path
    """

    def load(
        self,
        source: Union[str, "st.runtime.uploaded_file_manager.UploadedFile"],
    ) -> AudioBuffer:
        """
        Args:
            source: A YouTube URL string or a Streamlit UploadedFile object.

        Returns:
            AudioBuffer with raw WAV bytes and display label.

        Raises:
            AudioSourceError: if the source cannot be fetched or decoded.
            ValidationError:  if the URL is malformed or points to a
                              non-permitted domain.
        """
        ...


# ---------------------------------------------------------------------------
# Lyrics lookup (pre-transcription)
# ---------------------------------------------------------------------------

@runtime_checkable
class LyricsProvider(Protocol):
    """
    Fetch synced lyrics for a known track title + artist.

    Implementations: services/lyrics/ (LRCLibClient)
    Swap candidates:  Genius API, Musixmatch, local lyrics DB
    """

    def get_lyrics(self, title: str, artist: str) -> list[TranscriptSegment] | None:
        """
        Args:
            title:  Track title.
            artist: Artist name.

        Returns:
            Ordered list of time-stamped TranscriptSegment objects if synced
            lyrics are found, or None if the track is unknown or has no
            synced lyrics.
        """
        ...


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@runtime_checkable
class TranscriptionProvider(Protocol):
    """
    Transcribe audio to time-stamped text segments.

    Implementations:
        services/transcription/ — LyricsOrchestrator (LRCLib → Whisper full-mix)
        services/transcription/ — Transcription (Whisper-only, used as fallback)
    Swap candidates:  Deepgram, AssemblyAI, Azure Speech, Google STT
    """

    def transcribe(
        self,
        audio: AudioBuffer,
        title: str = "",
        artist: str = "",
    ) -> list[TranscriptSegment]:
        """
        Args:
            audio:  In-memory audio buffer.
            title:  Track title (used by LyricsOrchestrator for lyrics lookup).
            artist: Artist name (used by LyricsOrchestrator for lyrics lookup).

        Returns:
            Ordered list of TranscriptSegment objects.

        Raises:
            ModelInferenceError: on OOM, model load failure, or timeout.
        """
        ...


# ---------------------------------------------------------------------------
# Structure analysis
# ---------------------------------------------------------------------------

@runtime_checkable
class StructureAnalyzer(Protocol):
    """
    Detect tempo, key, structural sections, and beat grid.

    Implementations: services/analysis/ (allin1 + librosa)
    Swap candidates:  Essentia, madmom standalone, a custom trained model
    """

    def analyze(self, audio: AudioBuffer) -> StructureResult:
        """
        Args:
            audio: In-memory audio buffer.

        Returns:
            StructureResult with BPM, key, sections, beats, and metadata.

        Raises:
            ModelInferenceError: on allin1/librosa failure.
        """
        ...


# ---------------------------------------------------------------------------
# Forensics (AI-origin detection)
# ---------------------------------------------------------------------------

@runtime_checkable
class ForensicsAnalyzer(Protocol):
    """
    Detect signals that suggest AI-generated or stock audio.

    Implementations: services/forensics/ (Forensics class)
    Swap candidates:  A trained binary classifier, an external detection API
    """

    def analyze(self, audio: AudioBuffer) -> ForensicsResult:
        """
        Args:
            audio: In-memory audio buffer.

        Returns:
            ForensicsResult with individual signal scores and an overall verdict.

        Raises:
            ModelInferenceError: on librosa or C2PA processing failure.
        """
        ...


# ---------------------------------------------------------------------------
# Compliance checking
# ---------------------------------------------------------------------------

@runtime_checkable
class ComplianceChecker(Protocol):
    """
    Apply editorial compliance rules to audio and its transcript.

    Implementations: services/compliance/ (sync readiness rules)
    Swap candidates:  A different editorial standard, a paid compliance API
    """

    def check(
        self,
        audio: AudioBuffer,
        transcript: list[TranscriptSegment],
        sections: "list[Section]",  # forward ref avoids circular import
        beats: list[float],
    ) -> ComplianceReport:
        """
        Args:
            audio:      In-memory audio buffer.
            transcript: Whisper segments for lyric audit.
            sections:   allin1 structural sections for intro-length check.

        Returns:
            ComplianceReport with flags, sting result, energy check,
            intro check, and an A–F grade.

        Raises:
            ModelInferenceError: on Detoxify or spaCy processing failure.
        """
        ...


# ---------------------------------------------------------------------------
# Authorship analysis
# ---------------------------------------------------------------------------

@runtime_checkable
class AuthorshipAnalyzer(Protocol):
    """
    Estimate whether lyrics were written by a human or AI.

    Implementations: services/content/ (Authorship — RoBERTa + linguistic signals)
    Swap candidates:  GPTZero API, a fine-tuned model, a rules-only baseline
    """

    def analyze(self, transcript: list[TranscriptSegment]) -> AuthorshipResult:
        """
        Args:
            transcript: Whisper segments (text fields are the input to the model).

        Returns:
            AuthorshipResult with verdict, signal count, and per-signal scores.

        Raises:
            ModelInferenceError: on RoBERTa load or inference failure.
        """
        ...


# ---------------------------------------------------------------------------
# Theme & Mood analysis
# ---------------------------------------------------------------------------

@runtime_checkable
class ThemeMoodAnalyzer(Protocol):
    """
    Classify lyric themes and overall mood from a transcript.

    Implementations: services/content/_theme.py (keyword taxonomy + Groq)
    Swap candidates:  zero-shot NLI classifier, OpenAI chat completion
    """

    def analyze(self, transcript: list[TranscriptSegment]) -> ThemeMoodResult:
        """
        Args:
            transcript: Whisper/LRCLib segments (text fields are the input).

        Returns:
            ThemeMoodResult with themes, mood, confidence, and raw keywords.

        Raises:
            ModelInferenceError: on Groq API failure (keyword path never raises).
        """
        ...


# ---------------------------------------------------------------------------
# Similar-track discovery
# ---------------------------------------------------------------------------

@runtime_checkable
class TrackDiscovery(Protocol):
    """
    Find commercially similar tracks for sync licensing reference.

    Implementations: services/discovery/ (Last.fm + yt-dlp)
    Swap candidates:  Spotify recommendations API, Soundcharts, internal DB
    """

    def find_similar(self, title: str, artist: str) -> list[TrackCandidate]:
        """
        Args:
            title:  Track title string (from metadata or user input).
            artist: Artist name string.

        Returns:
            Up to CONSTANTS.MAX_SIMILAR_TRACKS TrackCandidate objects,
            ordered by similarity score descending.

        Raises:
            AudioSourceError: on Last.fm API failure or yt-dlp errors.
            ConfigurationError: if lastfm_api_key is missing.
        """
        ...


# ---------------------------------------------------------------------------
# Legal / PRO links
# ---------------------------------------------------------------------------

@runtime_checkable
class LegalLinksProvider(Protocol):
    """
    Generate PRO (Performing Rights Organisation) repertory search links.

    Implementations: services/legal/ (static URL templates, no network calls)
    Swap candidates:  A live licensing API (ASCAP ACE, BMI Repertoire, etc.)
    """

    def get_links(self, title: str, artist: str) -> LegalLinks:
        """
        Args:
            title:  Track title.
            artist: Artist name.

        Returns:
            LegalLinks with ASCAP, BMI, and SESAC search URLs.
        """
        ...


class SyncCutProvider(Protocol):
    """
    Suggest edit-point windows for standard ad/TV format durations.

    Implementations: services/sync_cut/ (SyncCutAnalyzer)
    Swap candidates:  An ML-based edit detector, a manual cue-sheet override
    """

    def suggest(
        self,
        sections: list[Section],
        beats: list[float],
        target_durations: "list[int]",
    ) -> list[SyncCut]:
        """
        Args:
            sections:          allin1 structural sections (label, start, end).
            beats:             Beat grid as seconds-from-track-start.
            target_durations:  Format lengths to target (e.g. [15, 30, 60]).

        Returns:
            One SyncCut per target duration (fewer if the track is too short).
        """
        ...


class TagInjectorProvider(Protocol):
    """
    Embed audit results into audio file tags.

    Implementations: services/tagging/ (TagInjector)
    Swap candidates:  Cloud-based tag writing service, BWF metadata standard
    """

    def inject(self, audio_bytes: bytes, result: AnalysisResult) -> bytes:
        """
        Args:
            audio_bytes: Raw audio bytes (MP3, FLAC, OGG, M4A).
            result:      Complete AnalysisResult to embed as SYNC_SAFE_* tags.

        Returns:
            Audio bytes with tags injected; returns *audio_bytes* unchanged
            if the format is unsupported or tagging fails (non-fatal).
        """
        ...


class PlatformExportProvider(Protocol):
    """
    Generate platform-specific catalog CSV bytes for a sync licensing portal.

    Implementations: services/export/ (to_platform_csv)
    Swap candidates:  Songtradr API, Musicstax, or a custom DAM integration
    """

    def export(self, result: AnalysisResult, platform: str) -> bytes:
        """
        Args:
            result:   Complete pipeline AnalysisResult.
            platform: Target schema name (e.g. "generic", "disco", "synchtank").

        Returns:
            UTF-8-with-BOM CSV bytes ready for download or API upload.

        Raises:
            ValueError: if *platform* is not a recognised schema name.
        """
        ...


class MetadataValidatorProvider(Protocol):
    """
    Validate pre-flight track rights metadata at sync intake.

    Implementations: services/validation/ (MetadataValidator)
    Swap candidates:  A paid metadata registry (AllTrack, DDEX, SoundExchange)
    """

    def validate(
        self,
        fields: dict[str, str],
        splits: list[float],
        isrc: str = "",
    ) -> MetadataValidationResult:
        """
        Args:
            fields: Mapping of required intake field names to values.
                    Expected keys: "title", "artist", "pro", "publisher".
            splits: List of writer/publisher percentage splits (must sum to 100).
            isrc:   ISRC string; empty means not yet known.

        Returns:
            MetadataValidationResult with per-field detail and rejection reason.
        """
        ...


class ProLookupProvider(Protocol):
    """
    Best-effort ISRC and PRO affiliation lookup from an external metadata source.

    Implementations: services/legal/ (ProLookup — MusicBrainz recordings API)
    Swap candidates:  Songkick, AcousticBrainz, or a paid licensing data provider.
    """

    def lookup(
        self,
        title: str,
        artist: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Args:
            title:  Track title.
            artist: Artist name.

        Returns:
            (isrc, pro_match, pro_confidence) tuple — all None when no match found or on error.
            pro_confidence is 'High'|'Medium'|'Low' when pro_match is set (#118).
        """
        ...


# ---------------------------------------------------------------------------
# yt-dlp subprocess boundary
# ---------------------------------------------------------------------------

class YtDlpProvider(Protocol):
    """
    Download audio, fetch engagement metrics, and resolve YouTube URLs.

    Implementations: services/ingestion/_ytdlp.py (YtDlpClient)
    Swap candidates:  Piped API, RapidAPI YouTube endpoint, yt-dlp-server,
                      or any paid audio-download service.

    All three methods are on one Protocol because they share the same binary
    dependency and swap together — you wouldn't replace only one of them.
    """

    def download_audio(self, url: str, sample_rate: int) -> bytes:
        """
        Download audio from a URL and transcode to WAV bytes.

        Args:
            url:         Source URL (YouTube, SoundCloud, etc.)
            sample_rate: Target WAV sample rate in Hz.

        Returns:
            Raw WAV bytes (16-bit PCM mono).

        Raises:
            AudioSourceError: if the download or transcode fails.
        """
        ...

    def fetch_engagement(self, url: str) -> dict[str, int]:
        """
        Fetch per-platform engagement metrics for a URL.

        Never raises — engagement is always supplementary.

        Returns:
            Dict of engagement field names to integer values (may be empty).
        """
        ...

    def search_url(self, artist: str, title: str) -> Optional[str]:
        """
        Resolve a YouTube watch URL for the given artist + title.

        No audio is downloaded; only metadata is fetched.

        Returns:
            YouTube watch URL string, or None on any failure.
        """
        ...
