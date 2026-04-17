"""
backend/core/providers/stub.py
No-op implementations of MLWorkerProvider and StorageProvider.

Used during Effort 1 (local dev, no Modal account).
Swap for real providers in Effort 2 by changing the dependency injection site.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class StubMLWorkerProvider:
    """Logs dispatch calls and does nothing. Safe for local dev and tests."""

    def dispatch(self, job_id: str, source_url: str, config: dict[str, Any]) -> None:
        logger.info(
            "StubMLWorkerProvider.dispatch called — job_id=%s source_url=%s",
            job_id,
            source_url,
        )


class StubStorageProvider:
    """In-memory blob storage. Not persistent across requests."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, key: str, data: bytes, content_type: str) -> str:  # noqa: ARG002
        self._store[key] = data
        return f"stub://{key}"

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
