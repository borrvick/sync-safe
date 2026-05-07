"""
core/timeout.py
Wall-clock timeout context manager for pipeline steps.

Uses POSIX SIGALRM on the main thread (Linux / macOS).
Falls back to a no-op yield on non-main threads (e.g. Streamlit's
ScriptRunner), because signal.signal() and signal.alarm() are restricted
to the main Python thread — calling them from any other thread raises
ValueError: "signal only works in main thread of the main interpreter".

Usage:
    from core.timeout import step_timeout

    with step_timeout(seconds=60, step_key="forensics"):
        result = Forensics().analyze(audio)
"""
from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Generator

from core.exceptions import StepTimeoutError


@contextlib.contextmanager
def step_timeout(seconds: int, step_key: str) -> Generator[None, None, None]:
    """
    Context manager that raises StepTimeoutError if the block runs longer
    than `seconds` wall-clock seconds.

    On the main thread, SIGALRM provides reliable timeout enforcement that
    interrupts blocking syscalls (subprocess.communicate, socket I/O, etc.).

    On non-main threads (Streamlit's ScriptThread, test workers), SIGALRM is
    unavailable. The block runs without a timeout guard — Streamlit's global
    ZeroGPU session limit (25 min/day) acts as the outer hard cap.

    Args:
        seconds:  Budget in whole seconds.
        step_key: Pipeline step name for the error context.

    Raises:
        StepTimeoutError: if the budget is exceeded (main thread only).
    """
    if threading.current_thread() is not threading.main_thread():
        # SIGALRM is restricted to the main thread. Streamlit runs its script
        # runner in a daemon thread, so we yield without installing a handler.
        # Tests that exercise timeout behaviour must do so from the main thread
        # or use a dedicated test helper that patches signal directly.
        yield
        return

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
        signal.alarm(0)                             # cancel any pending alarm
        signal.signal(signal.SIGALRM, old_handler)  # restore previous handler
