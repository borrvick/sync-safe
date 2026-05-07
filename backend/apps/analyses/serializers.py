"""
apps/analyses/serializers.py
"""
from __future__ import annotations

import os

from rest_framework import serializers

from .models import Analysis, TrackLabel

_ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}
_MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 MB


class AnalysisSerializer(serializers.ModelSerializer):
    """Full serializer — includes result_json. Used by detail endpoint."""

    class Meta:
        model = Analysis
        fields = (
            "id", "status", "source_url", "title", "artist", "label",
            "result_json", "error", "created_at", "updated_at",
        )
        read_only_fields = ("id", "status", "result_json", "error", "created_at", "updated_at")


class AnalysisListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list endpoint — omits result_json to reduce payload."""

    class Meta:
        model = Analysis
        fields = (
            "id", "status", "source_url", "title", "artist", "label",
            "error", "created_at", "updated_at",
        )
        read_only_fields = fields


class SubmitAnalysisSerializer(serializers.Serializer):
    source_url   = serializers.URLField(required=False, allow_blank=True, default="")
    audio_file   = serializers.FileField(required=False, allow_empty_file=False)
    title        = serializers.CharField(max_length=255, required=False, default="")
    artist       = serializers.CharField(max_length=255, required=False, default="")
    # When True, skip deduplication and always dispatch a fresh Modal job.
    force_rerun  = serializers.BooleanField(required=False, default=False)

    def validate_audio_file(self, value: object) -> object:
        ext = os.path.splitext(getattr(value, "name", ""))[1].lower()
        if ext not in _ALLOWED_AUDIO_EXTENSIONS:
            raise serializers.ValidationError(
                f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXTENSIONS))}"
            )
        if getattr(value, "size", 0) > _MAX_AUDIO_BYTES:
            raise serializers.ValidationError("File exceeds the 20 MB limit.")
        return value

    def validate(self, data: dict) -> dict:
        has_url  = bool(data.get("source_url"))
        has_file = bool(data.get("audio_file"))
        if not has_url and not has_file:
            raise serializers.ValidationError("Provide either source_url or audio_file.")
        if has_url and has_file:
            raise serializers.ValidationError("Provide source_url or audio_file, not both.")
        return data


class LabelSerializer(serializers.Serializer):
    label = serializers.CharField(max_length=100, allow_blank=True)


class TrackLabelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrackLabel
        fields = ("slug", "name", "description", "sort_order")
