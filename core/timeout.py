"""
core/timeout.py
Wall-clock timeout context manager for pipeline steps.

Uses POSIX SIGALRM — works on Linux (HF Spaces) and macOS.
Must be called from the main thread (SIGALRM restriction).

Usage:
    from core.timeout import step_timeout

    with step_timeout(seconds=60, step_key="forensics"):
        result = Forensics().analyze(audio)
"""
from __future__ import annotations

import contextlib
import signal
import time
from collections.abc import Generator

from core.exceptions import StepTimeoutError


@contextlib.contextmanager
def step_timeout(seconds: int, step_key: str) -> Generator[None, None, None]:
    """
    Context manager that raises StepTimeoutError if the block runs longer
    than `seconds` wall-clock seconds.

    Args:
        seconds:  Budget in whole seconds.
        step_key: Pipeline step name for the error context.

    Raises:
        StepTimeoutError: if the budget is exceeded.
    """
    start = time.monotonic()

    def _handler(signum: int, frame: object) -> None:
        elapsed = round(time.monotonic() - start, 1)
        raise StepTimeoutError(
            f"Step '{step_key}' exceeded {seconds}s timeout (elapsed {elapsed}s).",
            context={"step_key": step_key, "timeout_s": seconds, "elapsed_s": elapsed},
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)                          # cancel any pending alarm
        signal.signal(signal.SIGALRM, old_handler)  # restore previous handler
