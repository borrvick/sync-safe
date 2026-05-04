"""
modal_app/client.py
Django-side MLWorkerProvider implementations.

ModalWorkerClient  — production: spawns the Modal orchestrator asynchronously.
LocalWorkerClient  — dev/test:   runs no external process; safe when Modal
                                 credentials are not present.

Django never imports modal directly. This file is the single boundary.

Selection is settings-driven:
    USE_MODAL_WORKER=False (default) → LocalWorkerClient
    USE_MODAL_WORKER=True            → ModalWorkerClient

The existing backend/core/providers/ wrappers delegate here so Django views
have no knowledge of Modal at all.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Modal orchestrator coordinates
_APP_NAME = "sync-safe"
_FUNCTION_NAME = "run_orchestrator"


class ModalWorkerClient:
    """
    Implements MLWorkerProvider by spawning the Modal run_orchestrator function.

    dispatch() returns immediately — Modal calls back via
    POST /api/webhooks/analysis-complete/ when the job finishes.

    Constructor args:
        app_name:      Modal app name (default: "sync-safe").
        function_name: Modal function name (default: "run_orchestrator").
    """

    def __init__(
        self,
        app_name: str = _APP_NAME,
        function_name: str = _FUNCTION_NAME,
    ) -> None:
        self._app_name = app_name
        self._function_name = function_name

    def dispatch(self, job_id: str, source_url: str, config: dict[str, Any]) -> None:
        """
        Spawn the Modal orchestrator asynchronously. Returns immediately.

        Raises:
            RuntimeError: if the Modal function cannot be looked up (not deployed).
        """
        try:
            import modal  # local import — modal not installed in dev/test env
            from modal.exception import NotFoundError, RemoteError
        except ImportError as exc:
            raise RuntimeError(
                "modal package is not installed. "
                "Use LocalWorkerClient for dev or install modal for prod."
            ) from exc

        try:
            fn = modal.Function.lookup(self._app_name, self._function_name)
            fn.spawn(job_id=job_id, source_url=source_url, config=config)
            logger.info(
                "ModalWorkerClient dispatched job_id=%s source_url=%s",
                job_id,
                source_url,
            )
        except NotFoundError as exc:
            raise RuntimeError(
                f"Modal function '{self._function_name}' not found in app "
                f"'{self._app_name}'. Run: modal deploy modal_app/main.py"
            ) from exc
        except RemoteError as exc:
            raise RuntimeError(
                f"Modal remote error dispatching job_id={job_id}: {exc}"
            ) from exc


class LocalWorkerClient:
    """
    No-op implementation for local dev without Modal credentials.
    Logs the dispatch call and does nothing else.
    """

    def dispatch(self, job_id: str, source_url: str, config: dict[str, Any]) -> None:
        logger.info(
            "LocalWorkerClient.dispatch — job_id=%s source_url=%s "
            "(no-op; set USE_MODAL_WORKER=True to use Modal)",
            job_id,
            source_url,
        )
