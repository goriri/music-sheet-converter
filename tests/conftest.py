"""Pytest configuration and global test fixtures.

Enforces offline mode (QA_OFFLINE=1) across test suite so tests never make live
network or LLM calls. Live evaluation is reserved for scripts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce offline mode for all tests by default
os.environ["QA_OFFLINE"] = "1"


@pytest.fixture(autouse=True)
def enforce_qa_offline_per_test(monkeypatch):
    """Ensure QA_OFFLINE remains set for every test even if modified."""
    monkeypatch.setenv("QA_OFFLINE", "1")
