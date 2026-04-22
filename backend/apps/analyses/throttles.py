"""
apps/analyses/throttles.py
Per-endpoint rate throttles for the analyses API.
"""
from __future__ import annotations

from django.conf import settings as django_settings
from rest_framework.throttling import UserRateThrottle


class AnalyzeRateThrottle(UserRateThrottle):
    """
    Per-authenticated-user limit on analysis submissions.

    Rate is read from AppSettings.ANALYZE_RATE_LIMIT at class definition time so it
    is environment-configurable (env: ANALYZE_RATE_LIMIT) without a code change.
    Setting rate as a class attribute ensures DRF uses it directly and is not
    affected by test fixtures that wipe DEFAULT_THROTTLE_RATES.
    GPU quota makes this endpoint the most expensive in the system.
    """
    scope = "analyze"
    rate = django_settings.APP_SETTINGS.ANALYZE_RATE_LIMIT
