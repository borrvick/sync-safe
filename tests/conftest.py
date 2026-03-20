"""
tests/conftest.py
Ensure the project root is importable for pytest without an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so that `from services.…` imports work
# regardless of how pytest is invoked (IDE, CLI, tox, etc.).
sys.path.insert(0, str(Path(__file__).parent.parent))
