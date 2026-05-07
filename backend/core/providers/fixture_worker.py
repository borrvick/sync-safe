"""
backend/core/providers/fixture_worker.py
FixtureWorkerClient — fires a synthetic webhook using a saved golden fixture.

Used when USE_FIXTURE_WORKER=True to exercise the full Django → webhook →
complete flow without spending Modal GPU credits. Spawns a background thread
so dispatch() returns immediately (same behaviour as the real Modal worker).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "golden_analysis.json"
# Short delay so the analysis record is committed before the webhook fires.
_DISPATCH_DELAY_SECONDS = 0.5


class FixtureWorkerClient:
    """
    Posts a saved golden fixture to the webhook endpoint in a background thread.

    Constructor args:
        django_base_url:  Base URL of the Django backend (default: http://localhost:8000).
        webhook_secret:   MODAL_WEBHOOK_SECRET value for HMAC header (may be empty).
    """

    def __init__(self, django_base_url: str, webhook_secret: str) -> None:
        if not _FIXTURE_PATH.exists():
            raise FileNotFoundError(
                f"Fixture file not found: {_FIXTURE_PATH}. "
                "Run 'python manage.py capture_fixture <analysis-id>' to generate it, "
                "or set USE_FIXTURE_WORKER=False to disable fixture mode."
            )
        self._webhook_url = f"{django_base_url}/api/webhooks/analysis-complete/"
        self._webhook_secret = webhook_secret

    def dispatch(self, job_id: str, source_url: str, config: dict[str, Any]) -> None:
        """
        Spawn a background thread that posts the golden fixture to the webhook.
        Returns immediately — the webhook fires after a short delay.
        """
        thread = threading.Thread(
            target=self._fire_webhook,
            args=(job_id,),
            daemon=True,
        )
        thread.start()
        logger.info("FixtureWorkerClient dispatched job_id=%s (fixture mode)", job_id)

    def _fire_webhook(self, job_id: str) -> None:
        time.sleep(_DISPATCH_DELAY_SECONDS)
        try:
            result = json.loads(_FIXTURE_PATH.read_text())
            result["job_id"] = job_id

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._webhook_secret:
                headers["X-Modal-Webhook-Secret"] = self._webhook_secret

            response = requests.post(
                self._webhook_url,
                json={"job_id": job_id, "status": "complete", "result": result},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            logger.info("FixtureWorkerClient webhook fired for job_id=%s", job_id)
        except Exception as exc:  # noqa: BLE001 — background thread, must not crash silently
            logger.exception("FixtureWorkerClient webhook failed for job_id=%s: %s", job_id, exc)
