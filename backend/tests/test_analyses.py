"""
tests/test_analyses.py
Analysis API tests: submit, list, poll — all scoped to the requesting user.
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
# List — must be scoped to requesting user
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_list_analyses_only_own_records(client_a: APIClient, client_b: APIClient, user_a, user_b) -> None:
    # Create one analysis for each user
    Analysis.objects.create(user=user_a, source_url="https://www.youtube.com/watch?v=aaa", status="pending")
    Analysis.objects.create(user=user_b, source_url="https://www.youtube.com/watch?v=bbb", status="pending")

    resp = client_a.get("/api/analyses/")
    assert resp.status_code == 200
    assert len(resp.data) == 1
    assert resp.data[0]["source_url"] == "https://www.youtube.com/watch?v=aaa"


@pytest.mark.django_db
def test_list_analyses_empty(client_a: APIClient) -> None:
    resp = client_a.get("/api/analyses/")
    assert resp.status_code == 200
    assert resp.data == []


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
