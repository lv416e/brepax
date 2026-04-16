"""Stratum-aware Boolean operations with exact per-stratum gradients.

Forward: exact SDF Boolean (jnp.minimum) + heaviside indicator on grid.
Backward: custom_vjp dispatches gradient computation by stratum label:
  - disjoint: per-primitive volume gradient (no Boolean interaction)
  - contained: outer primitive's volume gradient only
  - intersecting: straight-through estimator with grid-adaptive beta

This preserves the core design principle (stratum label dispatch)
while generalizing to any primitive pair via grid-based computation.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.primitives._base import Primitive

# --- Grid utilities ---


def _primitive_bounds(p: Primitive) -> tuple[Array, Array]:
    """Estimate bounding box for a single primitive."""
    params = p.parameters()
    if "center" in params:
        c = params["center"]
        r = params.get("radius", jnp.array(1.0))
        # FiniteCylinder has height; extend bounds along axis
        h = params.get("height", jnp.array(0.0))
        extent = jnp.maximum(r, h / 2.0)
        # Use half_extents for Box
        he = params.get("half_extents", None)
        if he is not None:
            return c - he, c + he
        return c - extent, c + extent
    elif "point" in params:
        # Infinite cylinder/cone: use point +/- radius in all dims
        pt = params["point"]
        r = params["radius"]
        return pt - r - 2.0, pt + r + 2.0
    elif "normal" in params:
        # Plane: half-space, use a large default domain
        dim = params["normal"].shape[0]
        return -jnp.ones(dim) * 3.0, jnp.ones(dim) * 3.0
    else:
        dim = 3
        return -jnp.ones(dim) * 3.0, jnp.ones(dim) * 3.0


def _auto_domain(
    a: Primitive,
    b: Primitive,
) -> tuple[Array, Array]:
    """Compute bounding box from two primitives with padding."""
    lo_a, hi_a = _primitive_bounds(a)
    lo_b, hi_b = _primitive_bounds(b)
    margin = 0.5
    lo = jnp.minimum(lo_a, lo_b) - margin
    hi = jnp.maximum(hi_a, hi_b) + margin
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


def _detect_stratum_generic(
    a: Primitive,
    b: Primitive,
    grid: Array,
) -> Float[Array, ""]:
    """Detect stratum using SDF evaluation on grid. Primitive-independent.

    Returns: 0=disjoint, 1=intersecting, 2=a_contained_in_b, 3=b_contained_in_a.
    See ADR-0012 for design rationale.
    """
    sdf_a = a.sdf(grid)
    sdf_b = b.sdf(grid)

    inside_a = sdf_a < 0  # points inside A
    inside_b = sdf_b < 0  # points inside B
    overlap = inside_a & inside_b  # points inside both

    n_a = jnp.sum(inside_a)
    n_b = jnp.sum(inside_b)
    n_overlap = jnp.sum(overlap)

    # No overlap at all
    disjoint = n_overlap == 0

    # A entirely inside B (all of A's interior is also B's interior)
    a_in_b = (n_a > 0) & (n_overlap >= n_a)
    # B entirely inside A
    b_in_a = (n_b > 0) & (n_overlap >= n_b)
    contained = a_in_b | b_in_a

    return jnp.where(
        disjoint,
        0.0,
        jnp.where(contained, jnp.where(a_in_b, 2.0, 3.0), 1.0),
    )


# --- Per-stratum gradient functions ---


def _single_primitive_volume_grad(
    prim: Primitive,
    lo: Array,
    hi: Array,
    resolution: int,
) -> Primitive:
    """Gradient of a single primitive's volume.

    Uses analytical volume() if available (finite volume primitives),
    falling back to grid-based straight-through estimator for unbounded
    primitives. Analytical gradients are exact; grid-based have
    resolution-dependent error.
    """
    vol = prim.volume()
    is_finite = jnp.isfinite(vol)

    # Analytical path: jax.grad of the exact volume formula
    analytical_grad = jax.grad(lambda p: p.volume())(prim)

    # Grid path: straight-through estimator (fallback for unbounded)
    def _grid_vol(p: Primitive) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_vals = p.sdf(grid)
        cell_width = (hi[0] - lo[0]) / (resolution - 1)
        indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
        return jnp.sum(indicator) * cell_m

    grid_grad = jax.grad(_grid_vol)(prim)

    # Select: analytical for bounded, grid for unbounded
    return jax.tree.map(
        lambda a, g: jnp.where(is_finite, a, g),
        analytical_grad,
        grid_grad,
    )


def _grad_disjoint(
    a: Primitive,
    b: Primitive,
    lo: Array,
    hi: Array,
    resolution: int,
) -> tuple[Primitive, Primitive]:
    """Disjoint: union_vol = vol_a + vol_b, gradients are independent."""
    grad_a = _single_primitive_volume_grad(a, lo, hi, resolution)
    grad_b = _single_primitive_volume_grad(b, lo, hi, resolution)
    return grad_a, grad_b


def _grad_contained(
    a: Primitive,
    b: Primitive,
    lo: Array,
    hi: Array,
    resolution: int,
    label: Float[Array, ""],
) -> tuple[Primitive, Primitive]:
    """Contained: union_vol = vol_outer. Only outer primitive has gradient.

    label=2 means A is inside B (B is outer).
    label=3 means B is inside A (A is outer).
    """
    # label 3.0 = b_in_a => a is outer
    a_is_outer = label == 3.0

    grad_a_full = _single_primitive_volume_grad(a, lo, hi, resolution)
    grad_b_full = _single_primitive_volume_grad(b, lo, hi, resolution)

    # Contained: outer primitive keeps its gradient, inner gets zero.
    # tree.map operates on each primitive type independently to avoid
    # PyTree structure mismatch between heterogeneous primitives.
    grad_a = jax.tree.map(
        lambda g: jnp.where(a_is_outer, g, jnp.zeros_like(g)),
        grad_a_full,
    )
    grad_b = jax.tree.map(
        lambda g: jnp.where(a_is_outer, jnp.zeros_like(g), g),
        grad_b_full,
    )
    return grad_a, grad_b


def _grad_intersecting(
    a: Primitive,
    b: Primitive,
    lo: Array,
    hi: Array,
    resolution: int,
) -> tuple[Primitive, Primitive]:
    """Intersecting: straight-through estimator on exact SDF Boolean.

    Uses grid-adaptive beta = cell_width for the sigmoid, which ensures
    precision improves with resolution. Only the union SDF boundary
    contributes meaningful gradient.
    """

    def _union_vol(
        a_: Primitive,
        b_: Primitive,
    ) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_union = jnp.minimum(a_.sdf(grid), b_.sdf(grid))
        cell_width = (hi[0] - lo[0]) / (resolution - 1)
        indicator = jax.nn.sigmoid(-sdf_union / cell_width)
        return jnp.sum(indicator) * cell_m

    grad_a, grad_b = jax.grad(
        _union_vol,
        argnums=(0, 1),
    )(a, b)
    return grad_a, grad_b


# --- Generic Boolean measure (for intersection/subtract) ---


def _sdf_combine(
    a_: Primitive,
    b_: Primitive,
    grid: Array,
    op: str,
) -> Array:
    """Combine SDFs based on Boolean operation type."""
    if op == "intersect":
        return jnp.maximum(a_.sdf(grid), b_.sdf(grid))
    elif op == "subtract":
        return jnp.maximum(a_.sdf(grid), -b_.sdf(grid))
    else:
        return jnp.minimum(a_.sdf(grid), b_.sdf(grid))


def _grad_intersecting_generic(
    a: Primitive,
    b: Primitive,
    lo: Array,
    hi: Array,
    resolution: int,
    op: str,
) -> tuple[Primitive, Primitive]:
    """Intersecting stratum: straight-through estimator on exact SDF Boolean."""

    def _vol(a_: Primitive, b_: Primitive) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_vals = _sdf_combine(a_, b_, grid, op)
        cell_width = (hi[0] - lo[0]) / (resolution - 1)
        indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
        return jnp.sum(indicator) * cell_m

    grad_a, grad_b = jax.grad(_vol, argnums=(0, 1))(a, b)
    return grad_a, grad_b


def _boolean_measure_with_dispatch(
    a: Primitive,
    b: Primitive,
    op: str,
    resolution: int,
    lo: Array,
    hi: Array,
) -> Float[Array, ""]:
    """Grid-based Boolean measure with full stratum dispatch.

    Forward: exact SDF Boolean + heaviside on grid.
    Backward: stratum-dispatched gradient computation.

    Supports union, subtract, and intersect operations with
    per-stratum analytical gradients where applicable.
    """

    @jax.custom_vjp
    def _measure(a: Primitive, b: Primitive) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_vals = _sdf_combine(a, b, grid, op)
        indicator = jnp.heaviside(-sdf_vals, 0.5)
        return jnp.sum(indicator) * cell_m

    def _fwd(
        a: Primitive,
        b: Primitive,
    ) -> tuple[Float[Array, ""], tuple[Primitive, Primitive, Float[Array, ""]]]:
        primal = _measure(a, b)
        grid, _ = _make_grid_nd(lo, hi, resolution)
        label = _detect_stratum_generic(a, b, grid)
        return primal, (a, b, label)

    def _bwd(
        residuals: tuple[Primitive, Primitive, Float[Array, ""]],
        g_bar: Float[Array, ""],
    ) -> tuple[Primitive, Primitive]:
        a_, b_, label = residuals

        ga_vol = _single_primitive_volume_grad(a_, lo, hi, resolution)
        gb_vol = _single_primitive_volume_grad(b_, lo, hi, resolution)
        zero_a = jax.tree.map(jnp.zeros_like, ga_vol)
        zero_b = jax.tree.map(jnp.zeros_like, gb_vol)

        # Per-stratum gradients depend on operation type
        if op == "subtract":
            # disjoint: subtract=vol_a, grad=(ga, 0)
            ga_d, gb_d = ga_vol, zero_b
            # A⊂B: subtract=0, grad=(0, 0)
            ga_ab, gb_ab = zero_a, zero_b
            # B⊂A: subtract=vol_a-vol_b, grad=(ga, -gb)
            neg_gb = jax.tree.map(lambda x: -x, gb_vol)
            ga_ba, gb_ba = ga_vol, neg_gb
        elif op == "intersect":
            # disjoint: intersect=0, grad=(0, 0)
            ga_d, gb_d = zero_a, zero_b
            # A⊂B: intersect=vol_a, grad=(ga, 0)
            ga_ab, gb_ab = ga_vol, zero_b
            # B⊂A: intersect=vol_b, grad=(0, gb)
            ga_ba, gb_ba = zero_a, gb_vol
        else:  # union
            # disjoint: union=vol_a+vol_b
            ga_d, gb_d = ga_vol, gb_vol
            # A⊂B: union=vol_b
            ga_ab, gb_ab = zero_a, gb_vol
            # B⊂A: union=vol_a
            ga_ba, gb_ba = ga_vol, zero_b

        # Intersecting: always straight-through on combined SDF
        ga_i, gb_i = _grad_intersecting_generic(
            a_,
            b_,
            lo,
            hi,
            resolution,
            op,
        )

        # Contained uses direction from label (2=A⊂B, 3=B⊂A)
        contained = (label == 2.0) | (label == 3.0)
        a_in_b = label == 2.0

        def select_a(gd: Array, gi: Array, g_ab: Array, g_ba: Array) -> Array:
            gc = jnp.where(a_in_b, g_ab, g_ba)
            return jnp.where(label == 0.0, gd, jnp.where(contained, gc, gi))

        def select_b(gd: Array, gi: Array, g_ab: Array, g_ba: Array) -> Array:
            gc = jnp.where(a_in_b, g_ab, g_ba)
            return jnp.where(label == 0.0, gd, jnp.where(contained, gc, gi))

        grad_a = jax.tree.map(select_a, ga_d, ga_i, ga_ab, ga_ba)
        grad_b = jax.tree.map(select_b, gb_d, gb_i, gb_ab, gb_ba)

        return (
            jax.tree.map(lambda x: g_bar * x, grad_a),
            jax.tree.map(lambda x: g_bar * x, grad_b),
        )

    _measure.defvjp(_fwd, _bwd)
    return _measure(a, b)


# --- Public API ---


def subtract_volume_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute volume of a - b with stratum-dispatched gradients."""
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _boolean_measure_with_dispatch(a, b, "subtract", resolution, lo, hi)


def intersect_volume_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute intersection volume with stratum-dispatched gradients."""
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _boolean_measure_with_dispatch(a, b, "intersect", resolution, lo, hi)


def union_area_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 128,
) -> Float[Array, ""]:
    """Compute 2D union area with stratum-aware exact gradients."""
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _boolean_measure_with_dispatch(a, b, "union", resolution, lo, hi)


def union_volume_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute 3D union volume with stratum-dispatched gradients."""
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _boolean_measure_with_dispatch(a, b, "union", resolution, lo, hi)
