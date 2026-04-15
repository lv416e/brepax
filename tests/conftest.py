"""Shared fixtures for the BRepAX test suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float


@pytest.fixture(autouse=True)
def _jax_64bit() -> None:
    """Enable 64-bit precision for all tests."""
    jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def disk_params() -> dict[str, Float[Array, ...]]:
    """Default two-disk configuration for testing."""
    return {
        "c1": jnp.array([0.0, 0.0]),
        "r1": jnp.array(1.0),
        "c2": jnp.array([1.5, 0.0]),
        "r2": jnp.array(1.0),
    }
