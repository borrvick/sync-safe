"""
core/models/audio.py
In-memory audio representation passed between pipeline stages.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ._types import AudioSource

if TYPE_CHECKING:
    import numpy as np


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
