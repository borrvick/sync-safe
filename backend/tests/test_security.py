"""
tests/test_security.py
Security configuration assertions — no I/O, no DB, no GPU.

Covers issues #219 and #260:
  - HSTS, TLS, cookie security (prod.py)
  - JWT TTLs and token blacklisting (base.py → SIMPLE_JWT)
  - Password reset timeout tightened from Django default (3 days) to 1 hour
  - CORS: no wildcard, localhost allowed
  - Webhook _verify_secret: denies when USE_MODAL_WORKER=True and no secret set
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# HSTS & TLS — prod.py (#219)
# ---------------------------------------------------------------------------

def test_prod_hsts_seconds() -> None:
    import config.settings.prod as prod
    assert prod.SECURE_HSTS_SECONDS == 31536000  # 1 year


def test_prod_hsts_include_subdomains() -> None:
    import config.settings.prod as prod
    assert prod.SECURE_HSTS_INCLUDE_SUBDOMAINS is True


def test_prod_hsts_preload_not_enabled() -> None:
    """Preload must stay False until domain is stable — that step is essentially irreversible."""
    import config.settings.prod as prod
    assert getattr(prod, "SECURE_HSTS_PRELOAD", False) is False


def test_prod_ssl_redirect() -> None:
    import config.settings.prod as prod
    assert prod.SECURE_SSL_REDIRECT is True


def test_prod_session_cookie_secure() -> None:
    import config.settings.prod as prod
    assert prod.SESSION_COOKIE_SECURE is True


def test_prod_csrf_cookie_secure() -> None:
    import config.settings.prod as prod
    assert prod.CSRF_COOKIE_SECURE is True


def test_prod_content_type_nosniff() -> None:
    import config.settings.prod as prod
    assert prod.SECURE_CONTENT_TYPE_NOSNIFF is True


# ---------------------------------------------------------------------------
# JWT configuration — #260
# ---------------------------------------------------------------------------

def test_jwt_access_token_lifetime() -> None:
    from django.conf import settings
    assert settings.SIMPLE_JWT["ACCESS_TOKEN_LIFETIME"] == timedelta(minutes=15)


def test_jwt_refresh_token_lifetime() -> None:
    from django.conf import settings
    assert settings.SIMPLE_JWT["REFRESH_TOKEN_LIFETIME"] == timedelta(days=7)


def test_jwt_rotate_refresh_tokens_enabled() -> None:
    from django.conf import settings
    assert settings.SIMPLE_JWT["ROTATE_REFRESH_TOKENS"] is True


def test_jwt_blacklist_after_rotation_enabled() -> None:
    from django.conf import settings
    assert settings.SIMPLE_JWT["BLACKLIST_AFTER_ROTATION"] is True


def test_jwt_token_blacklist_app_installed() -> None:
    from django.conf import settings
    assert "rest_framework_simplejwt.token_blacklist" in settings.INSTALLED_APPS


def test_jwt_auth_header_type_is_bearer() -> None:
    from django.conf import settings
    assert "Bearer" in settings.SIMPLE_JWT["AUTH_HEADER_TYPES"]


# ---------------------------------------------------------------------------
# Password reset timeout — #260
# ---------------------------------------------------------------------------

def test_password_reset_timeout_is_one_hour() -> None:
    """Django default is 3 days (259200s) — hardened to 1 hour (3600s)."""
    from django.conf import settings
    assert settings.PASSWORD_RESET_TIMEOUT == 3600


# ---------------------------------------------------------------------------
# CORS — #260
# ---------------------------------------------------------------------------

def test_cors_no_wildcard_origin() -> None:
    from django.conf import settings
    assert not getattr(settings, "CORS_ALLOW_ALL_ORIGINS", False)


def test_cors_localhost_allowed_in_base() -> None:
    from django.conf import settings
    assert "http://localhost:3000" in settings.CORS_ALLOWED_ORIGINS


# ---------------------------------------------------------------------------
# Webhook HMAC — _verify_secret unit tests (#219)
# ---------------------------------------------------------------------------

def test_webhook_verify_secret_denies_when_modal_enabled_and_no_secret() -> None:
    """USE_MODAL_WORKER=True + empty secret → deny (prod misconfiguration guard)."""
    from apps.analyses.webhook_views import AnalysisCompleteWebhookView

    mock_cfg = MagicMock()
    mock_cfg.APP_SETTINGS.MODAL_WEBHOOK_SECRET = ""
    mock_cfg.APP_SETTINGS.USE_MODAL_WORKER = True

    view = AnalysisCompleteWebhookView()
    request = MagicMock()
    request.headers.get.return_value = ""

    with patch("apps.analyses.webhook_views.settings", mock_cfg):
        assert view._verify_secret(request) is False


def test_webhook_verify_secret_allows_when_modal_disabled_and_no_secret() -> None:
    """USE_MODAL_WORKER=False + empty secret → allow (dev/test with no webhook auth)."""
    from apps.analyses.webhook_views import AnalysisCompleteWebhookView

    mock_cfg = MagicMock()
    mock_cfg.APP_SETTINGS.MODAL_WEBHOOK_SECRET = ""
    mock_cfg.APP_SETTINGS.USE_MODAL_WORKER = False

    view = AnalysisCompleteWebhookView()
    request = MagicMock()
    request.headers.get.return_value = ""

    with patch("apps.analyses.webhook_views.settings", mock_cfg):
        assert view._verify_secret(request) is True


def test_webhook_verify_secret_uses_constant_time_compare() -> None:
    """Secret comparison must use hmac.compare_digest, not == (timing-safe)."""
    import inspect
    import hmac as _hmac
    from apps.analyses.webhook_views import AnalysisCompleteWebhookView

    source = inspect.getsource(AnalysisCompleteWebhookView._verify_secret)
    assert "compare_digest" in source
