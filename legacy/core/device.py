"""
core/device.py
Torch device resolution for Apple Silicon / CPU environments.

Centralises device detection so services share a single source of truth.
To swap the detection order (e.g. add CUDA), change resolve_device() only.

Swap guide:
  CUDA — add an `if torch.cuda.is_available()` branch before MPS.
  HF ZeroGPU — @spaces.GPU already handles GPU allocation for decorated
                functions; do not probe CUDA here, it won't be visible.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


def resolve_device(override: str | None = None) -> str:
    """
    Return the best available torch device string.

    Detection order:
      1. override — respected as-is (use in tests or paid-tier config)
      2. MPS     — Apple Silicon GPU (torch.backends.mps.is_available())
      3. CPU     — universal fallback

    Args:
        override: If set, return this string without probing torch.

    Returns:
        One of "mps", "cpu", or the override string.
    """
    if override is not None:
        return override
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def clear_mps_cache() -> None:
    """
    Flush the MPS allocator cache on Apple Silicon.

    MPS can leave dirty state between runs; calling this before a CPU retry
    prevents the intermittent allin1 failure pattern on M-series Macs.

    Safe to call on non-MPS hosts — silently does nothing.
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
