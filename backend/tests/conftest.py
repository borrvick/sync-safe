"""
tests/conftest.py
Suite-wide pytest configuration.

Throttling is disabled for all tests — rate-limit behaviour is verified by
inspecting throttle class attributes (scope, rate) in unit tests, not by
actually exhausting limits across integration tests sharing the same cache.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_throttling(settings) -> None:
    """
    Use DummyCache for all tests so throttle counters are never stored.

    View-level throttle_classes take precedence over DEFAULT_THROTTLE_CLASSES,
    so overriding the global setting alone is not enough. Using DummyCache means
    every cache.set() is a no-op and every cache.get() returns None — throttle
    counters never accumulate and allow_request() always returns True.
    """
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.dummy.DummyCache",
        }
    }
    settings.REST_FRAMEWORK = {
        **settings.REST_FRAMEWORK,
        "DEFAULT_THROTTLE_CLASSES": [],
        "DEFAULT_THROTTLE_RATES": {},
    }
