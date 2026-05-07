"""
services/forensics/detectors/base.py
BaseAnalyzer Protocol — all concrete analyzers implement this interface.
"""
from __future__ import annotations

from typing import Protocol

from .._audio import _AudioData
from .._bundle import _SignalBundle


class BaseAnalyzer(Protocol):
    """
    Minimal interface for a forensic signal group.

    Analyzers mutate bundle in-place with their computed scores.
    On failure they should raise ModelInferenceError so the orchestrator
    can log and continue without crashing the pipeline.
    """

    def process(self, data: _AudioData, bundle: _SignalBundle) -> None:
        """Compute signals and write results into bundle."""
        ...
