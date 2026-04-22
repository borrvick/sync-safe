"""
apps/analyses/views.py
"""
from __future__ import annotations

from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import BaseThrottle
from rest_framework.views import APIView

from core.protocols import MLWorkerProvider

from .models import Analysis
from .serializers import AnalysisSerializer, LabelSerializer, SubmitAnalysisSerializer
from .throttles import AnalyzeRateThrottle

class AnalysisPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


def _build_worker() -> MLWorkerProvider:
    """
    Return the appropriate MLWorkerProvider based on settings.

    USE_MODAL_WORKER=False (default) → StubMLWorkerProvider (safe for dev/tests).
    USE_MODAL_WORKER=True            → ModalMLWorkerProvider (requires modal package).

    Called once at module load; the result is cached in _worker.
    """
    if settings.APP_SETTINGS.USE_MODAL_WORKER:
        from core.providers.modal_worker import ModalMLWorkerProvider
        return ModalMLWorkerProvider()
    from core.providers.stub import StubMLWorkerProvider
    return StubMLWorkerProvider()


# Module-level singleton: provider is stateless so one instance per process is fine.
_worker: MLWorkerProvider = _build_worker()


class AnalysisListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get_throttles(self) -> list[BaseThrottle]:
        # Only throttle POST (submit) — GET (list) is cheap.
        if self.request.method == "POST":
            return [AnalyzeRateThrottle()]
        return []

    def get(self, request: Request) -> Response:
        analyses = Analysis.objects.filter(user=request.user)
        paginator = AnalysisPagination()
        page = paginator.paginate_queryset(analyses, request)
        return paginator.get_paginated_response(AnalysisSerializer(page, many=True).data)

    def post(self, request: Request) -> Response:
        serializer = SubmitAnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        analysis = Analysis.objects.create(
            user=request.user,
            source_url=data["source_url"],
            title=data.get("title", ""),
            artist=data.get("artist", ""),
            status=Analysis.Status.PENDING,
        )

        _worker.dispatch(
            job_id=str(analysis.id),
            source_url=data["source_url"],
            config={"title": data.get("title", ""), "artist": data.get("artist", "")},
        )

        return Response(AnalysisSerializer(analysis).data, status=status.HTTP_202_ACCEPTED)


class AnalysisDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, pk: str) -> Response:
        # Cache key is user-scoped to prevent cross-user data leakage.
        cache_key = f"analysis:{pk}:{request.user.id}"
        data = cache.get(cache_key)
        if data is None:
            try:
                analysis = Analysis.objects.get(pk=pk, user=request.user)
            except Analysis.DoesNotExist:
                return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
            data = AnalysisSerializer(analysis).data
            # Only cache terminal states — pending/running change on every webhook tick.
            if analysis.status in (Analysis.Status.COMPLETE, Analysis.Status.FAILED):
                cache.set(cache_key, data, timeout=settings.APP_SETTINGS.ANALYSIS_CACHE_TTL_SECONDS)
        return Response(data)


class AnalysisLabelView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request: Request, pk: str) -> Response:
        try:
            analysis = Analysis.objects.get(pk=pk, user=request.user)
        except Analysis.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = LabelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        analysis.label = serializer.validated_data["label"]
        analysis.save(update_fields=["label", "updated_at"])
        # Invalidate detail cache so the next poll reflects the new label.
        cache.delete(f"analysis:{pk}:{request.user.id}")
        return Response(AnalysisSerializer(analysis).data)
