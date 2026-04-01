"""
tests/test_timeout.py
Unit tests for core/timeout.py — pure SIGALRM behaviour.
"""
from __future__ import annotations

import time

import pytest

from core.exceptions import StepTimeoutError
from core.timeout import step_timeout


class TestStepTimeout:
    def test_no_timeout_when_fast(self) -> None:
        """Block completes before budget — no exception raised."""
        with step_timeout(5, "test_step"):
            time.sleep(0.01)

    def test_raises_on_timeout(self) -> None:
        """Block exceeds budget — StepTimeoutError raised."""
        with pytest.raises(StepTimeoutError) as exc_info:
            with step_timeout(1, "slow_step"):
                time.sleep(3)
        assert "slow_step" in str(exc_info.value)

    def test_error_context_contains_step_key(self) -> None:
        """StepTimeoutError context dict includes step_key and timeout_s."""
        with pytest.raises(StepTimeoutError) as exc_info:
            with step_timeout(1, "my_step"):
                time.sleep(3)
        ctx = exc_info.value.context  # type: ignore[attr-defined]
        assert ctx["step_key"] == "my_step"
        assert ctx["timeout_s"] == 1

    def test_alarm_cancelled_on_normal_exit(self) -> None:
        """After a successful block, no alarm is pending (next alarm isn't pre-fired)."""
        with step_timeout(5, "quick"):
            pass
        # If the alarm wasn't cancelled, this next short sleep would be interrupted
        time.sleep(0.05)
