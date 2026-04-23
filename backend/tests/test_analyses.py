"""
tests/test_analyses.py
Analysis API tests: submit, list, poll, label, health — all scoped to the requesting user.
"""
from __future__ import annotations

import uuid

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from apps.analyses.models import Analysis

User = get_user_model()


@pytest.fixture
def client() -> APIClient:
    return APIClient()


@pytest.fixture
def user_a(db):
    return User.objects.create_user(email="usera@example.com", password="strongpass1")


@pytest.fixture
def user_b(db):
    return User.objects.create_user(email="userb@example.com", password="strongpass1")


def _auth_client(client: APIClient, email: str, password: str) -> APIClient:
    resp = client.post("/api/auth/login/", {"email": email, "password": password}, format="json")
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")
    return client


@pytest.fixture
def client_a(client: APIClient, user_a) -> APIClient:
    return _auth_client(client, "usera@example.com", "strongpass1")


@pytest.fixture
def client_b(user_b) -> APIClient:
    # user_b must be created before login is attempted
    c = APIClient()
    return _auth_client(c, "userb@example.com", "strongpass1")


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_submit_analysis_success(client_a: APIClient, user_a) -> None:
    resp = client_a.post(
        "/api/analyses/",
        {"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Test", "artist": "Artist"},
        format="json",
    )
    assert resp.status_code == 202
    assert resp.data["status"] == "pending"
    assert resp.data["source_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert Analysis.objects.filter(user=user_a).count() == 1


@pytest.mark.django_db
def test_submit_analysis_unauthenticated(client: APIClient) -> None:
    resp = client.post(
        "/api/analyses/",
        {"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        format="json",
    )
    assert resp.status_code == 401


@pytest.mark.django_db
def test_submit_analysis_missing_url(client_a: APIClient) -> None:
    resp = client_a.post("/api/analyses/", {"title": "No URL"}, format="json")
    assert resp.status_code == 400


@pytest.mark.django_db
def test_submit_analysis_invalid_url(client_a: APIClient) -> None:
    resp = client_a.post("/api/analyses/", {"source_url": "not-a-url"}, format="json")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# List — paginated, scoped to requesting user
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_list_analyses_only_own_records(client_a: APIClient, client_b: APIClient, user_a, user_b) -> None:
    Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa", status="pending")
    Analysis.objects.create(user=user_b, source_url="https://www.youtube.com/watch?v=bbb", status="pending")

    resp = client_a.get("/api/analyses/")
    assert resp.status_code == 200
    assert resp.data["count"] == 1
    assert resp.data["results"][0]["source_url"] == "https://www.youtube.com/watch?v=aaa"


@pytest.mark.django_db
def test_list_analyses_empty(client_a: APIClient) -> None:
    resp = client_a.get("/api/analyses/")
    assert resp.status_code == 200
    assert resp.data["count"] == 0
    assert resp.data["results"] == []


@pytest.mark.django_db
def test_list_analyses_pagination_shape(client_a: APIClient, user_a) -> None:
    for i in range(3):
        Analysis.objects.create(user=user_a, source_url=f"https://www.youtube.com/watch?v={i:03}", status="pending")

    resp = client_a.get("/api/analyses/")
    assert resp.status_code == 200
    assert "count" in resp.data
    assert "results" in resp.data
    assert resp.data["count"] == 3
    assert len(resp.data["results"]) == 3


@pytest.mark.django_db
def test_list_analyses_unauthenticated(client: APIClient) -> None:
    resp = client.get("/api/analyses/")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Poll (detail)
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_poll_own_analysis(client_a: APIClient, user_a) -> None:
    analysis = Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa", status="pending")
    resp = client_a.get(f"/api/analyses/{analysis.id}/")
    assert resp.status_code == 200
    assert str(resp.data["id"]) == str(analysis.id)
    assert resp.data["status"] == "pending"


@pytest.mark.django_db
def test_poll_complete_analysis_includes_result(client_a: APIClient, user_a) -> None:
    result = {"bpm": 120.0, "key": "C"}
    analysis = Analysis.objects.create(
        user=user_a,
        source_url="https://www.youtube.com/watch?v=aaa",
        status=Analysis.Status.COMPLETE,
        result_json=result,
    )
    resp = client_a.get(f"/api/analyses/{analysis.id}/")
    assert resp.status_code == 200
    assert resp.data["result_json"] == result
    assert resp.data["updated_at"] is not None


@pytest.mark.django_db
def test_poll_failed_analysis_includes_error(client_a: APIClient, user_a) -> None:
    analysis = Analysis.objects.create(
        user=user_a,
        source_url="https://www.youtube.com/watch?v=aaa",
        status=Analysis.Status.FAILED,
        error="Modal OOM",
    )
    resp = client_a.get(f"/api/analyses/{analysis.id}/")
    assert resp.status_code == 200
    assert resp.data["status"] == "failed"
    assert resp.data["error"] == "Modal OOM"


@pytest.mark.django_db
def test_poll_other_users_analysis_returns_404(client_a: APIClient, user_b) -> None:
    """User A must not be able to read User B's analysis — returns 404 not 403."""
    other_analysis = Analysis.objects.create(
        user=user_b, source_url="https://www.youtube.com/watch?v=bbb", status="complete"
    )
    resp = client_a.get(f"/api/analyses/{other_analysis.id}/")
    assert resp.status_code == 404


@pytest.mark.django_db
def test_poll_nonexistent_analysis(client_a: APIClient) -> None:
    resp = client_a.get(f"/api/analyses/{uuid.uuid4()}/")
    assert resp.status_code == 404


@pytest.mark.django_db
def test_poll_unauthenticated(client: APIClient, user_a) -> None:
    analysis = Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa", status="pending")
    resp = client.get(f"/api/analyses/{analysis.id}/")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Label (PATCH)
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_patch_label_success(client_a: APIClient, user_a) -> None:
    analysis = Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa", status="complete")
    resp = client_a.patch(f"/api/analyses/{analysis.id}/label/", {"label": "sync-ready"}, format="json")
    assert resp.status_code == 200
    assert resp.data["label"] == "sync-ready"
    analysis.refresh_from_db()
    assert analysis.label == "sync-ready"


@pytest.mark.django_db
def test_patch_label_clear(client_a: APIClient, user_a) -> None:
    analysis = Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa", label="old-label")
    resp = client_a.patch(f"/api/analyses/{analysis.id}/label/", {"label": ""}, format="json")
    assert resp.status_code == 200
    assert resp.data["label"] == ""


@pytest.mark.django_db
def test_patch_label_other_user_returns_404(client_a: APIClient, user_b) -> None:
    analysis = Analysis.objects.create(user=user_b, source_url="https://www.youtube.com/watch?v=bbb")
    resp = client_a.patch(f"/api/analyses/{analysis.id}/label/", {"label": "sync-ready"}, format="json")
    assert resp.status_code == 404


@pytest.mark.django_db
def test_patch_label_unauthenticated(client: APIClient, user_a) -> None:
    analysis = Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa")
    resp = client.patch(f"/api/analyses/{analysis.id}/label/", {"label": "sync-ready"}, format="json")
    assert resp.status_code == 401


@pytest.mark.django_db
def test_patch_label_missing_field_returns_400(client_a: APIClient, user_a) -> None:
    analysis = Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa")
    resp = client_a.patch(f"/api/analyses/{analysis.id}/label/", {}, format="json")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_health_check_ok(client: APIClient) -> None:
    resp = client.get("/health/")
    assert resp.status_code == 200
    assert resp.data == {"status": "ok", "db": "ok"}


@pytest.mark.django_db
def test_health_check_no_auth_required(client: APIClient) -> None:
    """Health endpoint must be public — Railway has no JWT token."""
    resp = client.get("/health/")
    assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Rate limiting (throttle class attributes — exhaustion not tested; DummyCache)
# ---------------------------------------------------------------------------

def test_analyze_throttle_scope() -> None:
    from apps.analyses.throttles import AnalyzeRateThrottle
    assert AnalyzeRateThrottle.scope == "analyze"


def test_analyze_throttle_rate_from_app_settings() -> None:
    from django.conf import settings
    from apps.analyses.throttles import AnalyzeRateThrottle
    # rate is set at class definition from AppSettings — not from DEFAULT_THROTTLE_RATES
    # (which test fixtures wipe). This verifies the env-var path is wired correctly.
    assert AnalyzeRateThrottle.rate == settings.APP_SETTINGS.ANALYZE_RATE_LIMIT
    assert settings.APP_SETTINGS.ANALYZE_RATE_LIMIT == "10/hour"  # default


@pytest.mark.django_db
def test_analyze_throttle_applied_on_post(client_a: APIClient) -> None:
    """POST /api/analyses/ must pass through AnalyzeRateThrottle (DummyCache always allows)."""
    resp = client_a.post(
        "/api/analyses/",
        {"source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        format="json",
    )
    assert resp.status_code == 202  # throttle allowed — DummyCache never blocks


@pytest.mark.django_db
def test_analyze_throttle_not_applied_on_get(client_a: APIClient) -> None:
    """GET /api/analyses/ must NOT be throttled — list is cheap."""
    resp = client_a.get("/api/analyses/")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# AnalysisQuerySet.for_user
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_for_user_returns_only_own_analyses(user_a, user_b) -> None:
    Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa")
    Analysis.objects.create(user=user_b, source_url="https://www.youtube.com/watch?v=bbb")

    qs_a = Analysis.objects.for_user(user_a)
    assert qs_a.count() == 1
    assert qs_a.first().user == user_a


@pytest.mark.django_db
def test_for_user_empty_when_no_analyses(user_a) -> None:
    assert Analysis.objects.for_user(user_a).count() == 0


# ---------------------------------------------------------------------------
# TrackLabel list endpoint
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_track_label_list_is_public(client: APIClient) -> None:
    """GET /api/analyses/labels/ must not require authentication."""
    resp = client.get("/api/analyses/labels/")
    assert resp.status_code == 200


@pytest.mark.django_db
def test_track_label_list_returns_seeded_labels(client: APIClient) -> None:
    from apps.analyses.models import TrackLabel
    TrackLabel.objects.create(slug="test-cat", name="Test Category", sort_order=0)
    resp = client.get("/api/analyses/labels/")
    slugs = [item["slug"] for item in resp.data]
    assert "test-cat" in slugs


@pytest.mark.django_db
def test_track_label_serializer_fields(client: APIClient) -> None:
    from apps.analyses.models import TrackLabel
    # film-feature is seeded by migration 0005 — use get_or_create to be idempotent
    TrackLabel.objects.get_or_create(
        slug="film-feature",
        defaults={"name": "Film (Feature)", "description": "Theatrical", "sort_order": 30},
    )
    resp = client.get("/api/analyses/labels/")
    item = next(i for i in resp.data if i["slug"] == "film-feature")
    assert item["name"] == "Film (Feature)"
    assert "description" in item
    assert item["sort_order"] == 30
