"""
core/exceptions.py
Custom exception hierarchy for Sync-Safe.

Design rules:
- Every raise site imports from here — never raise bare Exception.
- Each class carries a `context` dict so callers get structured data,
  not string-parsed messages.
- The hierarchy is intentionally shallow: one root + four leaves.
  Adding sub-classes requires justification (a new distinct recovery path).
"""
from __future__ import annotations

from typing import Any


class SyncSafeError(Exception):
    """
    Root of the Sync-Safe exception hierarchy.

    All application errors inherit from this class so callers can catch
    the full surface with a single `except SyncSafeError` when needed,
    or be selective with the leaf classes below.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self!s}, context={self.context!r})"


class AudioSourceError(SyncSafeError):
    """
    Raised when an audio source cannot be loaded or is unusable.

    Covers:
    - YouTube download failures (network, geo-block, private video)
    - Unsupported or corrupt file formats
    - Files that exceed the configured size limit
    - Empty / zero-duration audio

    Example context keys: url, format, size_mb, http_status
    """


class ModelInferenceError(SyncSafeError):
    """
    Raised when a model fails to produce a result.

    Covers:
    - Out-of-memory errors (CUDA / MPS)
    - Model load failures (missing weights, bad checkpoint)
    - Subprocess timeouts (demucs, yt-dlp)
    - Unexpected output shapes or NaN results

    Example context keys: model_name, device, input_shape, original_error
    """


class ValidationError(SyncSafeError):
    """
    Raised when user-supplied input fails validation.

    Covers:
    - Malformed or non-YouTube URLs
    - Missing required fields in a request
    - Values outside acceptable ranges

    Example context keys: field, value, constraint
    """


class ConfigurationError(SyncSafeError):
    """
    Raised when the runtime environment is misconfigured.

    Covers:
    - Missing API keys (Last.fm, HuggingFace)
    - Invalid Settings values caught at startup
    - Required system binaries not found (ffmpeg, yt-dlp)

    Example context keys: key, source (env/secrets), suggestion
    """
