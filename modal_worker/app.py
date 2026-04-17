"""
modal_worker/app.py
Modal ML worker — receives analysis jobs from Django and POSTs results back.

Deploy: modal deploy modal_worker/app.py
Test locally: modal run modal_worker/app.py::run_analysis --job-id test --source-url https://...
"""
from __future__ import annotations

import logging
import os
from typing import Any

import modal
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

app = modal.App("sync-safe-ml-worker")

# Base image: Python 3.11 + the packages needed for analysis.
# Heavy ML deps (Whisper, allin1) will be added in a later effort.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests==2.32.3",
    )
)


# ---------------------------------------------------------------------------
# Analysis function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    timeout=600,   # 10 min ceiling; real Whisper jobs can take ~3 min
    retries=1,     # one automatic retry on transient failure
)
def run_analysis(job_id: str, source_url: str, config: dict[str, Any]) -> None:
    """
    Entry point called by ModalMLWorkerProvider.dispatch().

    Runs analysis on source_url, then POSTs the result back to the Django
    webhook so the Analysis record is updated atomically.

    Args:
        job_id:     UUID string matching Analysis.id in the Django DB.
        source_url: YouTube URL or public audio URL to analyse.
        config:     Metadata dict passed from the API (title, artist, etc.).
    """
    django_base_url = os.environ["DJANGO_BASE_URL"]
    webhook_secret  = os.environ.get("MODAL_WEBHOOK_SECRET", "")
    webhook_url     = f"{django_base_url}/api/webhooks/analysis-complete/"

    try:
        result = _run_stub_analysis(job_id=job_id, source_url=source_url, config=config)
        _post_result(
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            job_id=job_id,
            job_status="complete",
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 — top-level boundary; guarantees webhook POST
        logger.exception("Analysis failed for job_id=%s: %s", job_id, exc)
        # Attempt to notify Django of the failure. If the webhook endpoint is also
        # unreachable, log it and re-raise the original so Modal's retry fires.
        try:
            _post_result(
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                job_id=job_id,
                job_status="failed",
                result={"error": str(exc)},
            )
        except Exception as post_exc:  # noqa: BLE001
            logger.exception(
                "Webhook POST also failed for job_id=%s: %s — re-raising for Modal retry",
                job_id,
                post_exc,
            )
            raise


# ---------------------------------------------------------------------------
# Analysis logic (stub — real ML wired in Effort 4)
# ---------------------------------------------------------------------------

def _run_stub_analysis(
    job_id: str,
    source_url: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Placeholder analysis. Returns a minimal result dict.
    Replace the body of this function with real Whisper/allin1 calls in Effort 4.
    The return shape (keys) must stay stable — the webhook receiver persists it as-is.
    """
    logger.info("Running stub analysis for job_id=%s source_url=%s", job_id, source_url)
    return {
        "job_id":     job_id,
        "source_url": source_url,
        "title":      config.get("title", ""),
        "artist":     config.get("artist", ""),
        "status":     "stub — real ML pending",
    }


# ---------------------------------------------------------------------------
# Webhook callback
# ---------------------------------------------------------------------------

def _post_result(
    webhook_url: str,
    webhook_secret: str,
    job_id: str,
    job_status: str,
    result: dict[str, Any],
) -> None:
    """
    POST analysis result back to Django webhook endpoint.

    Raises:
        requests.HTTPError: if Django returns a non-2xx response.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["X-Modal-Webhook-Secret"] = webhook_secret

    response = requests.post(
        webhook_url,
        json={"job_id": job_id, "status": job_status, "result": result},
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    logger.info("Webhook POST succeeded for job_id=%s status=%s", job_id, job_status)
