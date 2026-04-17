"""
apps/analyses/webhook_views.py
POST /api/webhooks/analysis-complete/ — Modal → Django completion callback.

Verifies the MODAL_WEBHOOK_SECRET header before writing any data.
"""
from __future__ import annotations

import hmac
import logging

from django.conf import settings
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Analysis

logger = logging.getLogger(__name__)


class AnalysisCompleteWebhookView(APIView):
    """
    Receives completion payloads from Modal.

    Expected body:
        {
            "job_id": "<uuid>",
            "status": "complete" | "failed",
            "result": { ...AnalysisResult.to_dict() },  # present when status=complete
            "error": "...",                              # present when status=failed
        }

    Header: X-Modal-Webhook-Secret: <MODAL_WEBHOOK_SECRET>
    """

    permission_classes = [AllowAny]  # Auth is the HMAC header, not JWT

    def post(self, request: Request) -> Response:
        if not self._verify_secret(request):
            logger.warning("Webhook received with invalid secret")
            return Response({"detail": "Forbidden."}, status=status.HTTP_403_FORBIDDEN)

        job_id  = request.data.get("job_id")
        outcome = request.data.get("status")

        if not job_id or outcome not in ("complete", "failed"):
            return Response({"detail": "Invalid payload."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            analysis = Analysis.objects.get(pk=job_id)
        except Analysis.DoesNotExist:
            logger.error("Webhook for unknown job_id=%s", job_id)
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

        if outcome == "complete":
            analysis.status      = Analysis.Status.COMPLETE
            analysis.result_json = request.data.get("result")
            analysis.error       = ""
        else:
            analysis.status = Analysis.Status.FAILED
            analysis.error  = request.data.get("error", "Unknown error")

        analysis.save(update_fields=["status", "result_json", "error", "updated_at"])
        logger.info("Analysis %s → %s", job_id, outcome)
        return Response({"detail": "ok"})

    def _verify_secret(self, request: Request) -> bool:
        webhook_secret = settings.APP_SETTINGS.MODAL_WEBHOOK_SECRET
        if not webhook_secret:
            # Secret not configured — allow through in local dev (stub worker)
            return True
        provided = request.headers.get("X-Modal-Webhook-Secret", "")
        return hmac.compare_digest(provided, webhook_secret)
