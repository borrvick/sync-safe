"""
backend/core/protocols.py
Infrastructure-layer Protocols for Effort 1 → Effort 2 swappability.

These are the only abstractions Django depends on for compute and storage.
Concrete implementations live in core/providers/.
"""
from __future__ import annotations

from typing import Any, Protocol


class MLWorkerProvider(Protocol):
    """
    Fire-and-forget compute dispatch.

    Django calls dispatch() and returns immediately.
    The worker POSTs results back via POST /api/webhooks/analysis-complete/
    when the job finishes.

    During Effort 1: StubMLWorkerProvider — logs and does nothing.
    During Effort 2: ModalMLWorkerProvider — calls Modal SDK.
    """

    def dispatch(
        self,
        job_id: str,
        source_url: str,
        config: dict[str, Any],
    ) -> None:
        """Submit a job. Must not block. Must not raise on network errors."""
        ...


class StorageProvider(Protocol):
    """
    Blob storage for future file upload support.

    Stubbed in Effort 1. Wired to R2/S3 in Effort 2 if needed.
    """

    def put(self, key: str, data: bytes, content_type: str) -> str:
        """Store bytes and return a URL."""
        ...

    def delete(self, key: str) -> None:
        """Delete a blob by key."""
        ...
