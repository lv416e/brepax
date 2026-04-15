"""Gradient accuracy benchmarks: Method (A) vs analytical ground truth.

Measures relative error of d(union_area)/d(r1) computed by each method
against the analytical gradient from disk_disk_union_area. Results are
parameterized over strata, temperature values, and boundary proximity.

Boundary distance for each configuration is reported alongside gradient
error so that Axis 2 (boundary proximity sweep) predictions can be
cross-referenced with these interior measurements.
"""

import jax
import jax.numpy as jnp
import pytest

from brepax.analytical.disk_disk import (
    disk_disk_boundary_distance,
    disk_disk_union_area,
)
from brepax.boolean import union_area
from brepax.primitives import Disk

# --- Interior configurations (Axis 1) ---
# Each config is annotated with its boundary distance.
STRATA_CONFIGS = {
    "intersecting": {
        # boundary_distance = 0.50 (to external tangent d=r1+r2=2)
        "c1": [0.0, 0.0],
        "r1": 1.0,
        "c2": [1.5, 0.0],
        "r2": 1.0,
    },
    "disjoint": {
        # boundary_distance = 3.00 (to external tangent d=r1+r2=2)
        "c1": [0.0, 0.0],
        "r1": 1.0,
        "c2": [5.0, 0.0],
        "r2": 1.0,
    },
    "contained": {
        # boundary_distance = 1.20 (to internal tangent d=|r1-r2|=1.5)
        "c1": [0.0, 0.0],
        "r1": 2.0,
        "c2": [0.3, 0.0],
        "r2": 0.5,
    },
}

# --- Boundary proximity configurations (simplified Axis 2) ---
# Intersecting stratum with disk centers approaching external tangent.
# External tangent at d = r1 + r2 = 2.0, so eps = 2.0 - d.
BOUNDARY_PROXIMITY_CONFIGS = {
    "eps=0.50": {"c1": [0.0, 0.0], "r1": 1.0, "c2": [1.5, 0.0], "r2": 1.0},
    "eps=0.10": {"c1": [0.0, 0.0], "r1": 1.0, "c2": [1.9, 0.0], "r2": 1.0},
    "eps=0.01": {"c1": [0.0, 0.0], "r1": 1.0, "c2": [1.99, 0.0], "r2": 1.0},
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
        return union_area(
            a, b, method="smoothing", k=k, beta=beta, resolution=resolution
        )

    return jax.grad(area_fn)(r1)


def _stratum_grad_r1(c1, r1, c2, r2):
    """Method (C) d(union_area)/d(r1)."""

    def area_fn(radius):
        a = Disk(center=c1, radius=radius)
        b = Disk(center=c2, radius=r2)
        return union_area(a, b, method="stratum")

    return jax.grad(area_fn)(r1)


def _relative_error(approx, exact):
    """Relative error, safe for near-zero exact values."""
    return jnp.where(
        jnp.abs(exact) > 1e-12,
        jnp.abs(approx - exact) / jnp.abs(exact),
        jnp.abs(approx - exact),
    )


def _boundary_dist(cfg):
    """Compute boundary distance for a config dict."""
    return disk_disk_boundary_distance(
        jnp.array(cfg["c1"]),
        jnp.array(cfg["r1"]),
        jnp.array(cfg["c2"]),
        jnp.array(cfg["r2"]),
    )


# ---- Axis 1: interior gradient accuracy ----


@pytest.mark.parametrize("stratum", list(STRATA_CONFIGS.keys()))
@pytest.mark.parametrize("k_beta", TEMPERATURE_VALUES)
def test_gradient_accuracy_method_a(stratum, k_beta):
    """Measure gradient accuracy of Method (A) against analytical."""
    cfg = STRATA_CONFIGS[stratum]
    c1 = jnp.array(cfg["c1"])
    r1 = jnp.array(cfg["r1"])
    c2 = jnp.array(cfg["c2"])
    r2 = jnp.array(cfg["r2"])
    bdist = float(_boundary_dist(cfg))

    resolution = 256 if k_beta <= 0.05 else 128

    exact = _analytical_grad_r1(c1, r1, c2, r2)
    approx = _smoothing_grad_r1(
        c1, r1, c2, r2, k=k_beta, beta=k_beta, resolution=resolution
    )
    rel_err = _relative_error(approx, exact)

    print(
        f"\n  stratum={stratum}, bdist={bdist:.2f}, k=beta={k_beta:.2f}, "
        f"exact={float(exact):.6f}, approx={float(approx):.6f}, "
        f"rel_err={float(rel_err):.6f}"
    )

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
        approx = _smoothing_grad_r1(
            c1, r1, c2, r2, k=k_beta, beta=k_beta, resolution=resolution
        )
        errors.append(float(_relative_error(approx, exact)))

    assert errors[-1] < errors[0], (
        f"Expected convergence for {stratum}: "
        f"err@k=1.0={errors[0]:.4f}, err@k=0.01={errors[-1]:.4f}"
    )


# ---- Boundary proximity: Method (A) error vs distance to boundary ----


@pytest.mark.parametrize("label", list(BOUNDARY_PROXIMITY_CONFIGS.keys()))
@pytest.mark.parametrize("k_beta", [0.01, 0.1, 1.0])
def test_boundary_proximity_method_a(label, k_beta):
    """Method (A) gradient error as a function of boundary distance.

    This is a simplified Axis 2 preview using the intersecting stratum
    approaching external tangent (d -> r1 + r2). Expected: error
    increases as boundary distance decreases, especially for k > eps.
    """
    cfg = BOUNDARY_PROXIMITY_CONFIGS[label]
    c1 = jnp.array(cfg["c1"])
    r1 = jnp.array(cfg["r1"])
    c2 = jnp.array(cfg["c2"])
    r2 = jnp.array(cfg["r2"])
    bdist = float(_boundary_dist(cfg))

    resolution = 256 if k_beta <= 0.05 else 128

    exact = _analytical_grad_r1(c1, r1, c2, r2)
    approx = _smoothing_grad_r1(
        c1, r1, c2, r2, k=k_beta, beta=k_beta, resolution=resolution
    )
    rel_err = _relative_error(approx, exact)

    print(
        f"\n  {label}, bdist={bdist:.4f}, k=beta={k_beta:.2f}, "
        f"exact={float(exact):.6f}, approx={float(approx):.6f}, "
        f"rel_err={float(rel_err):.6f}"
    )

    assert jnp.isfinite(approx), f"Non-finite gradient at {label}, k={k_beta}"


# ---- Method (A) vs (C) head-to-head comparison ----

# Internal tangent configs for the comparison that matters most
INTERNAL_TANGENT_CONFIGS = {
    "int_eps=0.50": {"c1": [0.0, 0.0], "r1": 2.0, "c2": [2.0, 0.0], "r2": 0.5},
    "int_eps=0.10": {"c1": [0.0, 0.0], "r1": 2.0, "c2": [1.6, 0.0], "r2": 0.5},
    "int_eps=0.01": {"c1": [0.0, 0.0], "r1": 2.0, "c2": [1.51, 0.0], "r2": 0.5},
}


@pytest.mark.parametrize("label", list(BOUNDARY_PROXIMITY_CONFIGS.keys()))
def test_method_c_vs_a_external_tangent(label):
    """Method (C) vs Method (A) at external tangent boundary."""
    cfg = BOUNDARY_PROXIMITY_CONFIGS[label]
    c1, r1 = jnp.array(cfg["c1"]), jnp.array(cfg["r1"])
    c2, r2 = jnp.array(cfg["c2"]), jnp.array(cfg["r2"])
    bdist = float(_boundary_dist(cfg))

    exact = _analytical_grad_r1(c1, r1, c2, r2)
    approx_c = _stratum_grad_r1(c1, r1, c2, r2)
    approx_a = _smoothing_grad_r1(c1, r1, c2, r2, k=0.1, beta=0.1, resolution=128)

    err_c = float(_relative_error(approx_c, exact))
    err_a = float(_relative_error(approx_a, exact))
    ratio = err_a / max(err_c, 1e-15)

    print(
        f"\n  EXTERNAL {label}, bdist={bdist:.4f}, "
        f"err_A={err_a:.6f}, err_C={err_c:.6f}, ratio={ratio:.1f}x"
    )

    assert jnp.isfinite(approx_c)


@pytest.mark.parametrize("label", list(INTERNAL_TANGENT_CONFIGS.keys()))
def test_method_c_vs_a_internal_tangent(label):
    """Method (C) vs Method (A) at internal tangent boundary."""
    cfg = INTERNAL_TANGENT_CONFIGS[label]
    c1, r1 = jnp.array(cfg["c1"]), jnp.array(cfg["r1"])
    c2, r2 = jnp.array(cfg["c2"]), jnp.array(cfg["r2"])
    bdist = float(_boundary_dist(cfg))

    exact = _analytical_grad_r1(c1, r1, c2, r2)
    approx_c = _stratum_grad_r1(c1, r1, c2, r2)
    approx_a = _smoothing_grad_r1(c1, r1, c2, r2, k=0.1, beta=0.1, resolution=128)

    err_c = float(_relative_error(approx_c, exact))
    err_a = float(_relative_error(approx_a, exact))
    ratio = err_a / max(err_c, 1e-15)

    print(
        f"\n  INTERNAL {label}, bdist={bdist:.4f}, "
        f"err_A={err_a:.6f}, err_C={err_c:.6f}, ratio={ratio:.1f}x"
    )

    assert jnp.isfinite(approx_c)
