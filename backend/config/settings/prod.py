"""
config/settings/prod.py
Production overrides for Railway deployment (Effort 2).
DATABASE_URL, SECRET_KEY, and ALLOWED_HOSTS are injected by Railway as env vars.
"""
from .base import *  # noqa: F401, F403

DEBUG = False

# Railway injects DATABASE_URL pointing at Railway Postgres automatically.
# dj-database-url in base.py parses it — no changes needed here.

# Security headers — safe to enable once behind Railway's TLS terminator
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
