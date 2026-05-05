"""
WSGI config for config project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")

_django_app = get_wsgi_application()

_HEALTH_BODY = b'{"status":"ok"}'
_HEALTH_HEADERS = [
    ("Content-Type", "application/json"),
    ("Content-Length", str(len(_HEALTH_BODY))),
]


def application(environ, start_response):
    # Intercept /health/ before Django's ALLOWED_HOSTS check runs.
    # Railway's internal health probe sends an arbitrary Host header
    # (the container's 100.64.x.x IP) that is never in ALLOWED_HOSTS,
    # causing a 400 that blocks every deployment.
    if environ.get("PATH_INFO") == "/health/":
        start_response("200 OK", _HEALTH_HEADERS)
        return [_HEALTH_BODY]
    return _django_app(environ, start_response)
