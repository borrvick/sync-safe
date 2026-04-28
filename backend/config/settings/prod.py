"""
config/settings/prod.py
Production overrides for Railway deployment (Effort 2).
DATABASE_URL, SECRET_KEY, and ALLOWED_HOSTS are injected by Railway as env vars.
"""
from pathlib import Path

from .base import *  # noqa: F401, F403

DEBUG = False

# Railway injects DATABASE_URL pointing at Railway Postgres automatically.
# dj-database-url in base.py parses it — no changes needed here.

# Security headers — safe to enable once behind Railway's TLS terminator
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = True
# Railway's internal health checker hits /health/ over plain HTTP — exempt it
# so the health check isn't redirected to HTTPS before Django can respond.
SECURE_REDIRECT_EXEMPT = [r"^health/$"]
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# HSTS — tells browsers to always use HTTPS for this domain for 1 year.
# Do NOT set SECURE_HSTS_PRELOAD=True until the domain is stable and registered
# in the browser preload list (that step is essentially irreversible).
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True

# Prevent browsers from MIME-sniffing the content-type (belt and suspenders —
# Django's SecurityMiddleware also sets this, but explicit is better in prod).
SECURE_CONTENT_TYPE_NOSNIFF = True

# Restrict Referer header to same origin — prevents leaking the URL path to
# third-party requests (analytics, CDN assets) embedded in future frontend pages.
SECURE_REFERRER_POLICY = "same-origin"

# Static files — whitenoise serves them directly from gunicorn (no nginx needed)
STATIC_ROOT = Path(__file__).resolve().parents[3] / "staticfiles"

# Insert whitenoise after SecurityMiddleware (index 1) so it intercepts static
# requests before the rest of the Django middleware stack.
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    *MIDDLEWARE[1:],
]

STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Email — use SMTP when EMAIL_HOST is set in Railway env vars; fall back to
# console backend so Railway logs capture emails during initial bring-up.
if APP_SETTINGS.EMAIL_HOST:
    EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
    EMAIL_HOST = APP_SETTINGS.EMAIL_HOST
    EMAIL_PORT = APP_SETTINGS.EMAIL_PORT
    EMAIL_HOST_USER = APP_SETTINGS.EMAIL_HOST_USER
    EMAIL_HOST_PASSWORD = APP_SETTINGS.EMAIL_HOST_PASSWORD
    EMAIL_USE_TLS = APP_SETTINGS.EMAIL_USE_TLS
    DEFAULT_FROM_EMAIL = APP_SETTINGS.DEFAULT_FROM_EMAIL
    ACCOUNT_EMAIL_VERIFICATION = "mandatory"
else:
    EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
    ACCOUNT_EMAIL_VERIFICATION = "none"
