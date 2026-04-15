"""Benchmark configuration."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register benchmark markers."""
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
