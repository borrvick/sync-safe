"""
tests/test_webhook.py
Webhook receiver tests: completion, failure, bad secret, malformed payload, unknown job.
"""
from __future__ import annotations

import uuid

import pytest
from django.contrib.auth import get_user_model
from rest_framework.response import Response
from rest_framework.test import APIClient

from apps.analyses.models import Analysis

User = get_user_model()

_WEBHOOK_URL = "/api/webhooks/analysis-complete/"


@pytest.fixture
def client() -> APIClient:
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(email="webhook@example.com", password="strongpass1")


@pytest.fixture
def pending_analysis(user):
    return Analysis.objects.create(
        user=user,
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        status=Analysis.Status.PENDING,
    )


def _post(client: APIClient, payload: dict, secret: str = "") -> Response:
    headers = {"HTTP_X_MODAL_WEBHOOK_SECRET": secret} if secret else {}
    return client.post(_WEBHOOK_URL, payload, format="json", **headers)


@pytest.fixture
def webhook_secret():
    """Set MODAL_WEBHOOK_SECRET for the duration of the test then restore it."""
    from django.conf import settings as django_settings
    old = django_settings.APP_SETTINGS.MODAL_WEBHOOK_SECRET
    django_settings.APP_SETTINGS.MODAL_WEBHOOK_SECRET = "correct-secret"
    yield "correct-secret"
    django_settings.APP_SETTINGS.MODAL_WEBHOOK_SECRET = old


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_webhook_complete(client: APIClient, pending_analysis) -> None:
    result = {"structure": {"bpm": 120.0}, "forensics": None}
    resp = _post(client, {
        "job_id": str(pending_analysis.id),
        "status": "complete",
        "result": result,
    })
    assert resp.status_code == 200
    pending_analysis.refresh_from_db()
    assert pending_analysis.status == Analysis.Status.COMPLETE
    assert pending_analysis.result_json == result
    assert pending_analysis.error == ""


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_webhook_failed(client: APIClient, pending_analysis) -> None:
    resp = _post(client, {
        "job_id": str(pending_analysis.id),
        "status": "failed",
        "error": "Modal OOM on GPU",
    })
    assert resp.status_code == 200
    pending_analysis.refresh_from_db()
    assert pending_analysis.status == Analysis.Status.FAILED
    assert pending_analysis.error == "Modal OOM on GPU"
    assert pending_analysis.result_json is None


# ---------------------------------------------------------------------------
# Security — bad secret
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_webhook_bad_secret_returns_403(client: APIClient, pending_analysis, webhook_secret) -> None:
    """When MODAL_WEBHOOK_SECRET is configured, wrong secret must return 403."""
    resp = _post(client, {
        "job_id": str(pending_analysis.id),
        "status": "complete",
        "result": {},
    }, secret="wrong-secret")
    assert resp.status_code == 403
    pending_analysis.refresh_from_db()
    assert pending_analysis.status == Analysis.Status.PENDING  # unchanged


@pytest.mark.django_db
def test_webhook_correct_secret_accepted(client: APIClient, pending_analysis, webhook_secret) -> None:
    resp = _post(client, {
        "job_id": str(pending_analysis.id),
        "status": "complete",
        "result": {},
    }, secret=webhook_secret)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Malformed payload
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_webhook_missing_job_id(client: APIClient) -> None:
    resp = _post(client, {"status": "complete", "result": {}})
    assert resp.status_code == 400


@pytest.mark.django_db
def test_webhook_invalid_status(client: APIClient, pending_analysis) -> None:
    resp = _post(client, {"job_id": str(pending_analysis.id), "status": "unknown"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Unknown job
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_webhook_unknown_job_id(client: APIClient) -> None:
    resp = _post(client, {"job_id": str(uuid.uuid4()), "status": "complete", "result": {}})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Idempotence — duplicate delivery must not clobber terminal state
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_webhook_duplicate_complete_is_ignored(client: APIClient, user) -> None:
    """A second complete callback for an already-complete job must return 200 without re-writing."""
    analysis = Analysis.objects.create(
        user=user,
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        status=Analysis.Status.COMPLETE,
        result_json={"bpm": 120.0},
    )
    resp = _post(client, {
        "job_id": str(analysis.id),
        "status": "complete",
        "result": {"bpm": 999.0},  # different payload — must be ignored
    })
    assert resp.status_code == 200
    analysis.refresh_from_db()
    assert analysis.result_json == {"bpm": 120.0}  # original result preserved


@pytest.mark.django_db
def test_webhook_duplicate_failed_is_ignored(client: APIClient, user) -> None:
    """A second failed callback for an already-failed job must return 200 without re-writing."""
    analysis = Analysis.objects.create(
        user=user,
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        status=Analysis.Status.FAILED,
        error="original error",
    )
    resp = _post(client, {
        "job_id": str(analysis.id),
        "status": "failed",
        "error": "duplicate error",
    })
    assert resp.status_code == 200
    analysis.refresh_from_db()
    assert analysis.error == "original error"  # original error preserved
