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
  TranscriptionProvider → swap Whisper for Deepgram, AssemblyAI, or Azure STT
  StructureAnalyzer  → swap allin1 for a custom BPM/section detector
  ForensicsAnalyzer  → swap the librosa heuristics for a trained classifier
  ComplianceChecker  → swap sync readiness rules for a different editorial standard
  AuthorshipAnalyzer → swap RoBERTa for GPTZero API or a fine-tuned model
  TrackDiscovery     → swap Last.fm for Spotify, Soundcharts, or internal DB
  LegalLinksProvider → swap static URL templates for a live licensing API
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
    StructureResult,
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

    Implementations: services/ingestion.py
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

    Implementations: services/lyrics_provider.py (LRCLibClient)
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
        services/transcription.py — LyricsOrchestrator (LRCLib → Whisper full-mix)
        services/transcription.py — Transcription (Whisper-only, used as fallback)
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

    Implementations: services/analysis.py (allin1 + librosa)
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

    Implementations: services/forensics.py
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

    Implementations: services/compliance.py (sync readiness rules)
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

    Implementations: services/authorship.py (RoBERTa + linguistic signals)
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
# Similar-track discovery
# ---------------------------------------------------------------------------

@runtime_checkable
class TrackDiscovery(Protocol):
    """
    Find commercially similar tracks for sync licensing reference.

    Implementations: services/discovery.py (Last.fm + yt-dlp)
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

    Implementations: services/legal.py (static URL templates, no network calls)
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
