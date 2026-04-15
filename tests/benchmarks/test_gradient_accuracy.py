"""Gradient accuracy benchmarks: Method (A) vs analytical ground truth.

Measures relative error of d(union_area)/d(r1) computed by each method
against the analytical gradient from disk_disk_union_area. Results are
parameterized over strata and temperature values.
"""

import jax
import jax.numpy as jnp
import pytest

from brepax.analytical.disk_disk import disk_disk_union_area
from brepax.boolean import union_area
from brepax.primitives import Disk

# Representative configurations for each stratum
STRATA_CONFIGS = {
    "intersecting": {
        "c1": [0.0, 0.0], "r1": 1.0,
        "c2": [1.5, 0.0], "r2": 1.0,
    },
    "disjoint": {
        "c1": [0.0, 0.0], "r1": 1.0,
        "c2": [5.0, 0.0], "r2": 1.0,
    },
    "contained": {
        "c1": [0.0, 0.0], "r1": 2.0,
        "c2": [0.3, 0.0], "r2": 0.5,
    },
}

TEMPERATURE_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]


def _analytical_grad_r1(c1, r1, c2, r2):
    """Analytical d(union_area)/d(r1)."""
    return jax.grad(disk_disk_union_area, argnums=1)(c1, r1, c2, r2)


def _smoothing_grad_r1(c1, r1, c2, r2, *, k, beta, resolution):
    """Method (A) d(union_area)/d(r1)."""
    def area_fn(radius):
        a = Disk(center=c1, radius=radius)
        b = Disk(center=c2, radius=r2)
        return union_area(a, b, method="smoothing", k=k, beta=beta, resolution=resolution)
    return jax.grad(area_fn)(r1)


def _relative_error(approx, exact):
    """Relative error, safe for near-zero exact values."""
    return jnp.where(
        jnp.abs(exact) > 1e-12,
        jnp.abs(approx - exact) / jnp.abs(exact),
        jnp.abs(approx - exact),
    )


@pytest.mark.parametrize("stratum", list(STRATA_CONFIGS.keys()))
@pytest.mark.parametrize("k_beta", TEMPERATURE_VALUES)
def test_gradient_accuracy_method_a(stratum, k_beta):
    """Measure gradient accuracy of Method (A) against analytical."""
    cfg = STRATA_CONFIGS[stratum]
    c1 = jnp.array(cfg["c1"])
    r1 = jnp.array(cfg["r1"])
    c2 = jnp.array(cfg["c2"])
    r2 = jnp.array(cfg["r2"])

    # Higher resolution for small temperature to resolve boundary
    resolution = 256 if k_beta <= 0.05 else 128

    exact = _analytical_grad_r1(c1, r1, c2, r2)
    approx = _smoothing_grad_r1(c1, r1, c2, r2, k=k_beta, beta=k_beta, resolution=resolution)
    rel_err = _relative_error(approx, exact)

    # Report values for review (visible with pytest -v -s)
    print(
        f"\n  stratum={stratum}, k=beta={k_beta:.2f}, "
        f"exact={float(exact):.6f}, approx={float(approx):.6f}, "
        f"rel_err={float(rel_err):.6f}"
    )

    # Sanity: gradient must be finite
    assert jnp.isfinite(approx), f"Non-finite gradient for {stratum} at k={k_beta}"


@pytest.mark.parametrize("stratum", list(STRATA_CONFIGS.keys()))
def test_gradient_converges_with_temperature(stratum):
    """Verify that smaller temperature yields more accurate gradients."""
    cfg = STRATA_CONFIGS[stratum]
    c1 = jnp.array(cfg["c1"])
    r1 = jnp.array(cfg["r1"])
    c2 = jnp.array(cfg["c2"])
    r2 = jnp.array(cfg["r2"])

    exact = _analytical_grad_r1(c1, r1, c2, r2)

    errors = []
    for k_beta in [1.0, 0.1, 0.01]:
        resolution = 256 if k_beta <= 0.05 else 128
        approx = _smoothing_grad_r1(c1, r1, c2, r2, k=k_beta, beta=k_beta, resolution=resolution)
        errors.append(float(_relative_error(approx, exact)))

    # Error at k=0.01 should be less than error at k=1.0
    # (monotone improvement is not guaranteed at every step due to
    # discretization, but the trend should hold across this range)
    assert errors[-1] < errors[0], (
        f"Expected convergence for {stratum}: "
        f"err@k=1.0={errors[0]:.4f}, err@k=0.01={errors[-1]:.4f}"
    )
