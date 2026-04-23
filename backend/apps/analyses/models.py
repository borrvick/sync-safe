"""
apps/analyses/models.py
Analysis job lifecycle — pending → running → complete | failed.
TrackLabel — predefined sync category taxonomy for the label dropdown.
"""
from __future__ import annotations

import uuid
from typing import Any

from django.conf import settings
from django.db import models


class AnalysisQuerySet(models.QuerySet["Analysis"]):
    """Custom queryset that enforces user-scoped access in one place."""

    def for_user(self, user: Any) -> "AnalysisQuerySet":
        """Return only analyses owned by `user`. Use in every authenticated view."""
        return self.filter(user=user)


class Analysis(models.Model):
    class Status(models.TextChoices):
        PENDING  = "pending",  "Pending"
        RUNNING  = "running",  "Running"
        COMPLETE = "complete", "Complete"
        FAILED   = "failed",   "Failed"

    objects = AnalysisQuerySet.as_manager()

    id          = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user        = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="analyses",
    )
    status      = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )
    source_url  = models.URLField(blank=True)
    title       = models.CharField(max_length=255, blank=True)
    artist      = models.CharField(max_length=255, blank=True)
    label       = models.CharField(max_length=100, blank=True)
    # Full AnalysisResult.to_dict() payload written by webhook receiver on completion.
    # Stored opaque — never queried by column; the frontend reads it whole.
    result_json = models.JSONField(null=True, blank=True)
    error       = models.TextField(blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        display = self.title or self.source_url or str(self.id)
        return f"{display} [{self.status}]"


class TrackLabel(models.Model):
    """
    Predefined sync licensing category — populates the label dropdown in the UI.

    Managed via Django admin or data migrations; not user-editable directly.
    slug is the stored value on Analysis.label (e.g. "tv-commercial").
    """

    slug        = models.CharField(max_length=50, unique=True)
    name        = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    sort_order  = models.PositiveSmallIntegerField(default=0, db_index=True)

    class Meta:
        ordering = ["sort_order", "name"]

    def __str__(self) -> str:
        return self.name
