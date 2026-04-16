"""Stratum-aware Boolean operations with exact per-stratum gradients.

Uses jax.custom_vjp to provide correct gradients within each topological
stratum. The forward pass evaluates exact SDF Boolean operations on a grid
(no smoothing), and the backward pass uses a thin sigmoid for autodiff
within the current stratum -- eliminating smoothing bias while remaining
JAX-differentiable.

This design generalizes to any primitive pair -- no per-pair analytical
formula is required. Analytical formulas (disk_disk, sphere_sphere) serve
as ground truth for validation, not as the computation engine.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive

# --- Grid utilities ---


def _auto_domain(
    a: Primitive,
    b: Primitive,
) -> tuple[Array, Array]:
    """Compute bounding box from primitive parameters with padding."""
    p_a = a.parameters()
    p_b = b.parameters()
    c_a, r_a = p_a["center"], p_a["radius"]
    c_b, r_b = p_b["center"], p_b["radius"]
    margin = 0.5
    lo = jnp.minimum(c_a - r_a - margin, c_b - r_b - margin)
    hi = jnp.maximum(c_a + r_a + margin, c_b + r_b + margin)
    return lo, hi


def _make_grid_nd(
    lo: Array,
    hi: Array,
    resolution: int,
) -> tuple[Array, Float[Array, ""]]:
    """Create an N-dimensional grid over the given domain."""
    dim = lo.shape[0]
    axes = [jnp.linspace(lo[i], hi[i], resolution) for i in range(dim)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    grid = jnp.stack(mesh, axis=-1)
    cell_measure = jnp.prod((hi - lo) / (resolution - 1))
    return grid, cell_measure


def _exact_sdf_union(
    a: Primitive,
    b: Primitive,
    x: Float[Array, "... dim"],
) -> Float[Array, "..."]:
    """Exact union SDF: min(sdf_a, sdf_b). No smoothing."""
    return jnp.minimum(a.sdf(x), b.sdf(x))


# --- Public API ---


def union_area_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 128,
) -> Float[Array, ""]:
    """Compute 2D union area with stratum-aware exact gradients.

    Args:
        a: First primitive.
        b: Second primitive.
        resolution: Grid resolution per axis.

    Returns:
        Union area as a differentiable scalar.
    """
    # Grid domain is not a differentiable quantity
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _union_measure_with_custom_vjp(a, b, resolution, lo, hi)


def union_volume_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute 3D union volume with stratum-aware exact gradients.

    Args:
        a: First primitive.
        b: Second primitive.
        resolution: Grid resolution per axis (64^3 = 262k points).

    Returns:
        Union volume as a differentiable scalar.
    """
    # Grid domain is not a differentiable quantity
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _union_measure_with_custom_vjp(a, b, resolution, lo, hi)


def _union_measure_with_custom_vjp(
    a: Primitive,
    b: Primitive,
    resolution: int,
    lo: Array,
    hi: Array,
) -> Float[Array, ""]:
    """Dimension-agnostic union measure with stratum-aware custom_vjp.

    Forward: exact SDF Boolean + hard indicator on grid.
    Backward: thin sigmoid (beta=0.001) on exact SDF Boolean for autodiff
    within the current stratum, avoiding smoothing bias.

    Uses a closure to capture resolution/lo/hi since custom_vjp
    does not support keyword-only arguments.
    """

    @jax.custom_vjp
    def _measure(a: Primitive, b: Primitive) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_vals = _exact_sdf_union(a, b, grid)
        indicator = jnp.heaviside(-sdf_vals, 0.5)
        return jnp.sum(indicator) * cell_m

    def _measure_fwd(
        a: Primitive,
        b: Primitive,
    ) -> tuple[Float[Array, ""], tuple[Primitive, Primitive]]:
        primal = _measure(a, b)
        return primal, (a, b)

    def _measure_bwd(
        residuals: tuple[Primitive, Primitive],
        g_bar: Float[Array, ""],
    ) -> tuple[Primitive, Primitive]:
        a_, b_ = residuals

        # Thin sigmoid on exact SDF Boolean: differentiable but nearly
        # step-function. Avoids smooth-min bias of Method (A).
        def _diff_measure(
            a__: Primitive,
            b__: Primitive,
        ) -> Float[Array, ""]:
            grid, cell_m = _make_grid_nd(lo, hi, resolution)
            sdf_vals = _exact_sdf_union(a__, b__, grid)
            beta = 0.001
            indicator = jax.nn.sigmoid(-sdf_vals / beta)
            return jnp.sum(indicator) * cell_m

        grad_a, grad_b = jax.grad(
            _diff_measure,
            argnums=(0, 1),
        )(a_, b_)
        return (
            jax.tree.map(lambda x: g_bar * x, grad_a),
            jax.tree.map(lambda x: g_bar * x, grad_b),
        )

    _measure.defvjp(_measure_fwd, _measure_bwd)
    return _measure(a, b)
