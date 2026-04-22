"""
core/views.py
Infrastructure views not tied to a specific app.
"""
from __future__ import annotations

import logging

from django.db import OperationalError, connection
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

logger = logging.getLogger(__name__)


class HealthView(APIView):
    """
    GET /health/ — Railway health check.

    Public (no auth). Must respond in < 200ms. Checks DB with a lightweight
    SELECT 1; never touches Modal or any external service.
    """

    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            db_status = "ok"
            http_status = 200
        except OperationalError:
            logger.error("Health check: database unreachable")
            db_status = "unavailable"
            http_status = 503

        overall = "ok" if db_status == "ok" else "degraded"
        return Response({"status": overall, "db": db_status}, status=http_status)
