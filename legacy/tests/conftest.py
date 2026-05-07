"""
tests/conftest.py
Ensure the project root is importable for pytest without an editable install.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so that `from services.…` imports work
# regardless of how pytest is invoked (IDE, CLI, tox, etc.).
sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "forensics"


def load_forensics_fixture(name: str) -> dict:
    """
    Load a forensics JSON fixture by partial filename match.

    Usage:
        data = load_forensics_fixture("Espresso")
        data = load_forensics_fixture("Springsteen")
    """
    matches = list(FIXTURE_DIR.glob(f"*{name}*"))
    if not matches:
        raise FileNotFoundError(f"No fixture matching {name!r} in {FIXTURE_DIR}")
    return json.loads(matches[0].read_text())


def all_forensics_fixtures() -> list[tuple[str, dict]]:
    """Return (stem, data) pairs for every fixture file, sorted by name."""
    return [
        (p.stem, json.loads(p.read_text()))
        for p in sorted(FIXTURE_DIR.glob("*.json"))
    ]
