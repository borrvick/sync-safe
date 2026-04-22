"""
backend/core/providers/modal_worker.py
ModalMLWorkerProvider — production implementation of MLWorkerProvider.

Thin wrapper around modal_app.client.ModalWorkerClient. Django never imports
modal directly — modal_app/client.py is the single Modal boundary.

Django returns immediately on dispatch(); Modal POSTs results back via
POST /api/webhooks/analysis-complete/ when the job finishes.
"""
from __future__ import annotations

import logging
from typing import Any

from modal_app.client import ModalWorkerClient

logger = logging.getLogger(__name__)


class ModalMLWorkerProvider:
    """
    Production ML worker that dispatches jobs to Modal run_orchestrator.

    Constructor args:
        app_name:       Modal app name (default: "sync-safe").
        function_name:  Modal function name (default: "run_orchestrator").
    """

    def __init__(
        self,
        app_name: str = "sync-safe",
        function_name: str = "run_orchestrator",
    ) -> None:
        self._client = ModalWorkerClient(app_name=app_name, function_name=function_name)

    def dispatch(self, job_id: str, source_url: str, config: dict[str, Any]) -> None:
        """
        Spawn the Modal orchestrator asynchronously. Returns immediately.

        Raises:
            RuntimeError: if the Modal function cannot be looked up (not deployed).
        """
        self._client.dispatch(job_id=job_id, source_url=source_url, config=config)
        logger.info(
            "ModalMLWorkerProvider.dispatch — spawned job_id=%s source_url=%s",
            job_id,
            source_url,
        )
