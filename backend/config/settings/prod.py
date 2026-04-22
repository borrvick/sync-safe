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
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# HSTS — tells browsers to always use HTTPS for this domain for 1 year.
# Do NOT set SECURE_HSTS_PRELOAD=True until the domain is stable and registered
# in the browser preload list (that step is essentially irreversible).
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True

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
