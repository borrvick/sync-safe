"""
apps/analyses/views.py
"""
from __future__ import annotations

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from core.providers.stub import StubMLWorkerProvider

from .models import Analysis
from .serializers import AnalysisSerializer, SubmitAnalysisSerializer

# Module-level singleton is intentional: StubMLWorkerProvider is fully stateless
# (no DB connection, no GPU handle, no resources). Swap for ModalMLWorkerProvider
# in Effort 2 via a settings-based factory — no changes to this view required.
_worker: StubMLWorkerProvider = StubMLWorkerProvider()


class AnalysisListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        analyses = Analysis.objects.filter(user=request.user)
        return Response(AnalysisSerializer(analyses, many=True).data)

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
        try:
            analysis = Analysis.objects.get(pk=pk, user=request.user)
        except Analysis.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(AnalysisSerializer(analysis).data)
