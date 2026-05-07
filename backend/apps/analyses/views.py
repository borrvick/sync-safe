"""
apps/analyses/views.py
"""
from __future__ import annotations

import base64

from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import BaseThrottle
from rest_framework.views import APIView

from rest_framework.permissions import AllowAny

from core.protocols import MLWorkerProvider

from .models import Analysis, TrackLabel
from .serializers import AnalysisSerializer, LabelSerializer, SubmitAnalysisSerializer, TrackLabelSerializer
from .throttles import AnalyzeRateThrottle

class AnalysisPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


def _build_worker() -> MLWorkerProvider:
    """
    Return the appropriate MLWorkerProvider based on settings.

    USE_FIXTURE_WORKER=True → FixtureWorkerClient (dev: fires golden fixture webhook).
    USE_MODAL_WORKER=True   → ModalMLWorkerProvider (prod: dispatches to Modal GPU).
    default                 → StubMLWorkerProvider (no-op; safe for tests).

    USE_FIXTURE_WORKER takes precedence over USE_MODAL_WORKER.
    Called once at module load; the result is cached in _worker.
    """
    if settings.APP_SETTINGS.USE_FIXTURE_WORKER and settings.APP_SETTINGS.USE_MODAL_WORKER:
        raise RuntimeError(
            "USE_FIXTURE_WORKER and USE_MODAL_WORKER cannot both be True. "
            "Set USE_MODAL_WORKER=False in dev or USE_FIXTURE_WORKER=False in prod."
        )
    if settings.APP_SETTINGS.USE_FIXTURE_WORKER:
        from core.providers.fixture_worker import FixtureWorkerClient
        return FixtureWorkerClient(
            django_base_url=settings.APP_SETTINGS.DJANGO_BASE_URL,
            webhook_secret=settings.APP_SETTINGS.MODAL_WEBHOOK_SECRET,
        )
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
        analyses = Analysis.objects.for_user(request.user)
        paginator = AnalysisPagination()
        page = paginator.paginate_queryset(analyses, request)
        return paginator.get_paginated_response(AnalysisSerializer(page, many=True).data)

    def post(self, request: Request) -> Response:
        serializer = SubmitAnalysisSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # File-upload path — no deduplication (different files, no shared URL key).
        if data.get("audio_file"):
            audio_file = data["audio_file"]
            audio_b64  = base64.b64encode(audio_file.read()).decode()

            analysis = Analysis.objects.create(
                user=request.user,
                source_url="",
                title=data["title"],
                artist=data["artist"],
                status=Analysis.Status.PENDING,
            )
            _worker.dispatch(
                job_id=str(analysis.id),
                source_url="",
                config={
                    "title":          data["title"],
                    "artist":         data["artist"],
                    "audio_base64":   audio_b64,
                    "audio_filename": audio_file.name,
                },
            )
            return Response(AnalysisSerializer(analysis).data, status=status.HTTP_202_ACCEPTED)

        # URL path — deduplicate against prior completed runs to avoid GPU spend.
        if not data["force_rerun"]:
            existing = (
                Analysis.objects.filter(
                    source_url=data["source_url"],
                    status=Analysis.Status.COMPLETE,
                )
                .order_by("-created_at")
                .first()
            )
            if existing is not None:
                cloned = Analysis.objects.create(
                    user=request.user,
                    source_url=existing.source_url,
                    title=data["title"] or existing.title,
                    artist=data["artist"] or existing.artist,
                    status=Analysis.Status.COMPLETE,
                    result_json=existing.result_json,
                )
                return Response(AnalysisSerializer(cloned).data, status=status.HTTP_201_CREATED)

        analysis = Analysis.objects.create(
            user=request.user,
            source_url=data["source_url"],
            title=data["title"],
            artist=data["artist"],
            status=Analysis.Status.PENDING,
        )
        _worker.dispatch(
            job_id=str(analysis.id),
            source_url=data["source_url"],
            config={"title": data["title"], "artist": data["artist"]},
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
                analysis = Analysis.objects.for_user(request.user).get(pk=pk)
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
            analysis = Analysis.objects.for_user(request.user).get(pk=pk)
        except Analysis.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = LabelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        analysis.label = serializer.validated_data["label"]
        analysis.save(update_fields=["label", "updated_at"])
        # Invalidate detail cache so the next poll reflects the new label.
        cache.delete(f"analysis:{pk}:{request.user.id}")
        return Response(AnalysisSerializer(analysis).data)


class TrackLabelListView(APIView):
    """Return the predefined sync category list. Public — no auth required."""

    permission_classes = [AllowAny]

    def get(self, request: Request) -> Response:
        labels = TrackLabel.objects.all()
        return Response(TrackLabelSerializer(labels, many=True).data)
