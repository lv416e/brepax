"""Smooth-min/max Boolean operations with temperature parameter.

Implements the naive smoothing approach to differentiable Boolean
operations. Uses log-sum-exp smooth minimum for SDF composition
and sigmoid soft indicator for area integration.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive


def smooth_min(
    a: Float[Array, "..."],
    b: Float[Array, "..."],
    k: Float[Array, ""],
) -> Float[Array, "..."]:
    """Smooth minimum via log-sum-exp.

    Approximates min(a, b) with a differentiable function.
    Converges to exact min as k -> 0.

    Args:
        a: First operand.
        b: Second operand.
        k: Temperature parameter (positive). Smaller = sharper.

    Returns:
        Smooth approximation of min(a, b).
    """
    return -k * jnp.logaddexp(-a / k, -b / k)


def sdf_union_smooth(
    a: Primitive,
    b: Primitive,
    x: Float[Array, "... 2"],
    k: Float[Array, ""],
) -> Float[Array, "..."]:
    """Smooth union SDF of two primitives.

    The union SDF is the smooth minimum of the individual SDFs.

    Args:
        a: First primitive.
        b: Second primitive.
        x: Query points.
        k: Smooth-min temperature.

    Returns:
        Smooth union SDF values at query points.
    """
    return smooth_min(a.sdf(x), b.sdf(x), k)


def _make_grid(
    domain: tuple[Float[Array, "2"], Float[Array, "2"]],
    resolution: int,
) -> tuple[Float[Array, "res res 2"], Float[Array, ""]]:
    """Create a 2D grid over the given domain.

    Returns:
        Tuple of (grid_points, cell_area) where grid_points has shape
        (resolution, resolution, 2) and cell_area is the area of one cell.
    """
    lo, hi = domain
    x = jnp.linspace(lo[0], hi[0], resolution)
    y = jnp.linspace(lo[1], hi[1], resolution)
    dx = (hi[0] - lo[0]) / (resolution - 1)
    dy = (hi[1] - lo[1]) / (resolution - 1)
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([xx, yy], axis=-1)
    return grid, dx * dy


def union_area_smoothing(
    a: Primitive,
    b: Primitive,
    *,
    k: float = 0.1,
    beta: float = 0.1,
    resolution: int = 128,
    domain: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> Float[Array, ""]:
    """Compute union area via smooth SDF + sigmoid soft indicator.

    Integrates sigmoid(-sdf_union / beta) over a 2D grid to approximate
    the area of the union. Fully differentiable w.r.t. primitive parameters.

    Args:
        a: First primitive.
        b: Second primitive.
        k: Smooth-min temperature for SDF composition.
        beta: Sigmoid sharpness for soft indicator.
        resolution: Grid resolution per axis.
        domain: Bounding box as ((x_min, y_min), (x_max, y_max)).
            If None, auto-computed from primitive parameters.

    Returns:
        Approximate union area as a scalar.
    """
    k_arr = jnp.asarray(k)
    beta_arr = jnp.asarray(beta)

    if domain is None:
        # Auto-compute domain from disk parameters with padding
        p_a = a.parameters()
        p_b = b.parameters()
        c_a, r_a = p_a["center"], p_a["radius"]
        c_b, r_b = p_b["center"], p_b["radius"]
        margin = 0.5
        lo = jnp.minimum(c_a - r_a - margin, c_b - r_b - margin)
        hi = jnp.maximum(c_a + r_a + margin, c_b + r_b + margin)
    else:
        lo = jnp.array(domain[0])
        hi = jnp.array(domain[1])

    grid, cell_area = _make_grid((lo, hi), resolution)
    sdf_vals = sdf_union_smooth(a, b, grid, k_arr)
    # Sigmoid soft indicator: 1 inside (sdf < 0), 0 outside
    indicator = jax_sigmoid(-sdf_vals / beta_arr)
    return jnp.sum(indicator) * cell_area


def jax_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Numerically stable sigmoid."""
    return jnp.where(x >= 0, 1.0 / (1.0 + jnp.exp(-x)), jnp.exp(x) / (1.0 + jnp.exp(x)))
