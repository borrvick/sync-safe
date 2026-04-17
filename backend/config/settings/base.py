"""
config/settings/base.py
Shared settings for all environments. Dev and prod extend this.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import dj_database_url
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# pydantic-settings config model
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    SECRET_KEY: str
    DATABASE_URL: str = "sqlite:///db.sqlite3"
    DEBUG: bool = False
    ALLOWED_HOSTS: str = "localhost,127.0.0.1"

    # OAuth
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # Modal (wired in Effort 2)
    MODAL_WEBHOOK_SECRET: str = ""
    DJANGO_BASE_URL: str = "http://localhost:8000"

    class Config:
        env_file = Path(__file__).resolve().parents[3] / ".env"
        extra = "ignore"  # .env has Streamlit keys (LASTFM_API_KEY etc.) we don't need


_app_settings = AppSettings()  # type: ignore[call-arg]

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]  # repo root

SECRET_KEY = _app_settings.SECRET_KEY
DEBUG = _app_settings.DEBUG
ALLOWED_HOSTS = [h.strip() for h in _app_settings.ALLOWED_HOSTS.split(",")]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sites",
    # Third-party
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "corsheaders",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.github",
    "allauth.socialaccount.providers.google",
    # Local
    "apps.users",
    "apps.analyses",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASES = {
    "default": dj_database_url.parse(_app_settings.DATABASE_URL, conn_max_age=600),
}

# ---------------------------------------------------------------------------
# Custom User model
# ---------------------------------------------------------------------------

AUTH_USER_MODEL = "users.User"

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ---------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Django sites framework (required by allauth)
# ---------------------------------------------------------------------------

SITE_ID = 1

# ---------------------------------------------------------------------------
# django-allauth
# ---------------------------------------------------------------------------

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]

ACCOUNT_ADAPTER = "apps.users.adapters.AccountAdapter"
SOCIALACCOUNT_ADAPTER = "apps.users.adapters.SocialAccountAdapter"

ACCOUNT_LOGIN_BY_CODE_ENABLED = False
ACCOUNT_LOGIN_METHODS = {"email"}
ACCOUNT_SIGNUP_FIELDS = ["email*", "password1*", "password2*"]
ACCOUNT_EMAIL_VERIFICATION = "mandatory"
SOCIALACCOUNT_LOGIN_ON_GET = False  # CSRF protection for OAuth initiation

SOCIALACCOUNT_PROVIDERS = {
    "github": {
        "APP": {
            "client_id": _app_settings.GITHUB_CLIENT_ID,
            "secret": _app_settings.GITHUB_CLIENT_SECRET,
        }
    },
    "google": {
        "APP": {
            "client_id": _app_settings.GOOGLE_CLIENT_ID,
            "secret": _app_settings.GOOGLE_CLIENT_SECRET,
        },
        "SCOPE": ["profile", "email"],
        "AUTH_PARAMS": {"access_type": "online"},
    },
}

# ---------------------------------------------------------------------------
# Django REST Framework
# ---------------------------------------------------------------------------

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
}

# ---------------------------------------------------------------------------
# simplejwt
# ---------------------------------------------------------------------------

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,  # rotated tokens are immediately invalidated
    "AUTH_HEADER_TYPES": ("Bearer",),
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
}

# ---------------------------------------------------------------------------
# CORS (dev allows localhost:3000 for future Next.js frontend)
# ---------------------------------------------------------------------------

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# ---------------------------------------------------------------------------
# Expose app settings for use in views / tasks
# ---------------------------------------------------------------------------

APP_SETTINGS = _app_settings
