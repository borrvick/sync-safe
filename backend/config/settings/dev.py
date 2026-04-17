"""
config/settings/dev.py
Local development overrides. DEBUG on, email to console, relaxed email verification.
"""
from .base import *  # noqa: F401, F403

DEBUG = True

# Faster iteration locally — skip email verification flow
ACCOUNT_EMAIL_VERIFICATION = "optional"

# Print emails to terminal instead of sending them
EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

# Allow Django's dev server
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
