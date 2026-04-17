"""
apps/users/throttles.py
Per-endpoint rate throttles for auth surfaces.

Uses DRF's built-in cache-backed throttling — no Redis required.
Dev uses Django's in-memory cache; prod uses whatever CACHES is configured to.
"""
from __future__ import annotations

from rest_framework.throttling import AnonRateThrottle


class LoginRateThrottle(AnonRateThrottle):
    """5 login attempts per minute per IP — brute-force protection."""
    scope = "login"
    rate = "5/min"


class RegisterRateThrottle(AnonRateThrottle):
    """3 registration attempts per hour per IP — spam/abuse protection."""
    scope = "register"
    rate = "3/hour"
