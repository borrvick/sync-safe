"""
apps/analyses/serializers.py
"""
from __future__ import annotations

from rest_framework import serializers

from .models import Analysis


class AnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analysis
        fields = ("id", "status", "source_url", "title", "artist", "result_json", "error", "created_at", "updated_at")
        read_only_fields = ("id", "status", "result_json", "error", "created_at", "updated_at")


class SubmitAnalysisSerializer(serializers.Serializer):
    source_url = serializers.URLField()
    title      = serializers.CharField(max_length=255, required=False, default="")
    artist     = serializers.CharField(max_length=255, required=False, default="")
