# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage 1 — dependency layer (cached unless requirements change)
# ---------------------------------------------------------------------------
# Using tag rather than digest — digest pinning breaks cross-platform builds
# (ARM64 digest from local Mac is not found on Railway's AMD64 builders).
FROM python:3.11-slim AS deps

LABEL org.opencontainers.image.source="https://github.com/benjaminorrvick/music-sync-app" \
      org.opencontainers.image.description="Sync-Safe backend — Django/DRF API" \
      org.opencontainers.image.licenses="UNLICENSED"

WORKDIR /app

# Build-time deps: libpq-dev + gcc needed to compile psycopg2. Excluded from runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-django.txt .
RUN pip install --no-cache-dir -r requirements-django.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime dep only — no gcc, no build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user; gunicorn runs as this user at runtime
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source (.dockerignore excludes .env, db.sqlite3, tests/, etc.)
COPY . .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    DJANGO_SETTINGS_MODULE=config.settings.prod \
    PORT=8000

# Collect static files and hand ownership to non-root user in a single layer.
# SECRET_KEY is required by pydantic-settings; a dummy value is safe here —
# collectstatic never touches auth or signing logic.
RUN SECRET_KEY=dummy-build-key python manage.py collectstatic --noinput \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# True exec form — gunicorn is PID 1 and receives SIGTERM directly.
# PORT is hardcoded to 8000; override at runtime with -e PORT=... if needed.
CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-file", "-", \
     "--capture-output"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c '\
import socket, os, sys;\
port = int(os.environ.get("PORT", 8000));\
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM);\
s.settimeout(3);\
s.connect(("127.0.0.1", port));\
s.close();\
sys.exit(0)' || exit 1
