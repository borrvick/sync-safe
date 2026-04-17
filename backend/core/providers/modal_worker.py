"""
backend/core/providers/modal_worker.py
ModalMLWorkerProvider — production implementation of MLWorkerProvider.

Calls modal.Function.lookup() to find the deployed run_analysis function and
spawns it asynchronously. Django returns immediately; Modal calls back via
POST /api/webhooks/analysis-complete/ when the job finishes.

Requires MODAL_WEBHOOK_SECRET and DJANGO_BASE_URL in the environment so the
Modal function knows where to POST results.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModalMLWorkerProvider:
    """
    Production ML worker that dispatches jobs to Modal.

    Constructor args:
        app_name:       Modal app name (default: "sync-safe-ml-worker").
        function_name:  Modal function name (default: "run-analysis").
    """

    def __init__(
        self,
        app_name: str = "sync-safe-ml-worker",
        function_name: str = "run-analysis",
    ) -> None:
        self._app_name      = app_name
        self._function_name = function_name

    def dispatch(self, job_id: str, source_url: str, config: dict[str, Any]) -> None:
        """
        Spawn the Modal function asynchronously. Returns immediately.

        Raises:
            RuntimeError: if the Modal function cannot be looked up (e.g. not deployed).
        """
        try:
            import modal  # local import — modal is not installed in dev/test env
            from modal.exception import NotFoundError, RemoteError
        except ImportError as exc:
            raise RuntimeError(
                "modal package is not installed. "
                "Add modal to requirements-django.txt or use StubMLWorkerProvider in dev."
            ) from exc

        try:
            fn = modal.Function.lookup(self._app_name, self._function_name)
            fn.spawn(job_id=job_id, source_url=source_url, config=config)
            logger.info(
                "ModalMLWorkerProvider.dispatch — spawned job_id=%s source_url=%s",
                job_id,
                source_url,
            )
        except NotFoundError as exc:
            raise RuntimeError(
                f"Modal function '{self._function_name}' not found in app '{self._app_name}'. "
                "Run: modal deploy modal_worker/app.py"
            ) from exc
        except RemoteError as exc:
            raise RuntimeError(
                f"Modal remote error while dispatching job_id={job_id}: {exc}"
            ) from exc
