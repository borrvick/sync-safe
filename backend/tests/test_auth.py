"""
tests/test_auth.py
Auth endpoint tests: register, login, token refresh, /me, resend-verification.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

User = get_user_model()


@pytest.fixture
def client() -> APIClient:
    return APIClient()


@pytest.fixture
def registered_user(db):
    return User.objects.create_user(email="user@example.com", password="strongpass1")


@pytest.fixture
def auth_client(client: APIClient, registered_user) -> APIClient:
    resp = client.post("/api/auth/login/", {"email": "user@example.com", "password": "strongpass1"}, format="json")
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")
    return client


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_register_success(client: APIClient) -> None:
    resp = client.post("/api/auth/register/", {"email": "new@example.com", "password": "strongpass1"}, format="json")
    assert resp.status_code == 201
    assert resp.data["email"] == "new@example.com"
    assert resp.data["tier"] == "free"
    assert "password" not in resp.data


@pytest.mark.django_db
def test_register_duplicate_email(client: APIClient, registered_user) -> None:
    resp = client.post("/api/auth/register/", {"email": "user@example.com", "password": "strongpass1"}, format="json")
    assert resp.status_code == 400


@pytest.mark.django_db
def test_register_weak_password(client: APIClient) -> None:
    resp = client.post("/api/auth/register/", {"email": "weak@example.com", "password": "short"}, format="json")
    assert resp.status_code == 400


@pytest.mark.django_db
def test_register_missing_email(client: APIClient) -> None:
    resp = client.post("/api/auth/register/", {"password": "strongpass1"}, format="json")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_login_success(client: APIClient, registered_user) -> None:
    resp = client.post("/api/auth/login/", {"email": "user@example.com", "password": "strongpass1"}, format="json")
    assert resp.status_code == 200
    assert "access" in resp.data
    assert "refresh" in resp.data


@pytest.mark.django_db
def test_login_wrong_password(client: APIClient, registered_user) -> None:
    resp = client.post("/api/auth/login/", {"email": "user@example.com", "password": "wrongpass"}, format="json")
    assert resp.status_code == 401


@pytest.mark.django_db
def test_login_unknown_email(client: APIClient) -> None:
    resp = client.post("/api/auth/login/", {"email": "nobody@example.com", "password": "strongpass1"}, format="json")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Token refresh + blacklist
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_token_refresh_success(client: APIClient, registered_user) -> None:
    login = client.post("/api/auth/login/", {"email": "user@example.com", "password": "strongpass1"}, format="json")
    resp = client.post("/api/auth/token/refresh/", {"refresh": login.data["refresh"]}, format="json")
    assert resp.status_code == 200
    assert "access" in resp.data


@pytest.mark.django_db
def test_token_refresh_blacklisted_replay(client: APIClient, registered_user) -> None:
    """Rotated refresh token must be rejected on second use."""
    login = client.post("/api/auth/login/", {"email": "user@example.com", "password": "strongpass1"}, format="json")
    old_refresh = login.data["refresh"]
    # First use — rotates token
    client.post("/api/auth/token/refresh/", {"refresh": old_refresh}, format="json")
    # Replay old token — must be blacklisted
    resp = client.post("/api/auth/token/refresh/", {"refresh": old_refresh}, format="json")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_me_authenticated(auth_client: APIClient, registered_user) -> None:
    resp = auth_client.get("/api/auth/me/")
    assert resp.status_code == 200
    assert resp.data["email"] == "user@example.com"
    assert str(registered_user.id) == resp.data["id"]


@pytest.mark.django_db
def test_me_unauthenticated(client: APIClient) -> None:
    resp = client.get("/api/auth/me/")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Resend verification
# ---------------------------------------------------------------------------

@pytest.mark.django_db
def test_resend_verification_success(auth_client: APIClient) -> None:
    with patch("apps.users.views.send_verification_email_for_user") as mock_send:
        resp = auth_client.post("/api/auth/resend-verification/")
    assert resp.status_code == 200
    assert resp.data["detail"] == "Verification email sent."
    mock_send.assert_called_once()


@pytest.mark.django_db
def test_resend_verification_unauthenticated(client: APIClient) -> None:
    resp = client.post("/api/auth/resend-verification/")
    assert resp.status_code == 401


@pytest.mark.django_db
def test_resend_verification_smtp_failure(auth_client: APIClient) -> None:
    """SMTP failure must return 503, not 500."""
    with patch("apps.users.views.send_verification_email_for_user", side_effect=OSError("SMTP down")):
        resp = auth_client.post("/api/auth/resend-verification/")
    assert resp.status_code == 503
    assert "detail" in resp.data
