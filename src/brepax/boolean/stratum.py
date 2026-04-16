"""Stratum-aware Boolean operations with exact per-stratum gradients.

Forward: exact SDF Boolean (jnp.minimum) + heaviside indicator on grid.
Backward: custom_vjp dispatches gradient computation by stratum label:
  - disjoint: per-primitive volume gradient (no Boolean interaction)
  - contained: outer primitive's volume gradient only
  - intersecting: straight-through estimator with grid-adaptive beta

This preserves the Phase 0 design principle (stratum label dispatch)
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
        r = params["radius"]
        return c - r, c + r
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
    """Gradient of a single primitive's volume within the grid domain.

    Uses straight-through estimator with grid-adaptive beta = cell_width.
    """

    def _vol(p: Primitive) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_vals = p.sdf(grid)
        cell_width = (hi[0] - lo[0]) / (resolution - 1)
        indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
        return jnp.sum(indicator) * cell_m

    return jax.grad(_vol)(prim)


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


def _boolean_measure_straight_through(
    a: Primitive,
    b: Primitive,
    sdf_combiner: str,
    resolution: int,
    lo: Array,
    hi: Array,
) -> Float[Array, ""]:
    """Grid-based Boolean measure with straight-through estimator.

    Used for intersection and subtract where stratum dispatch is not
    yet specialized. Forward uses heaviside, backward uses sigmoid
    with grid-adaptive beta.
    """

    def _sdf_combine(
        a_: Primitive,
        b_: Primitive,
        grid: Array,
    ) -> Array:
        if sdf_combiner == "intersect":
            return jnp.maximum(a_.sdf(grid), b_.sdf(grid))
        elif sdf_combiner == "subtract":
            return jnp.maximum(a_.sdf(grid), -b_.sdf(grid))
        else:
            return jnp.minimum(a_.sdf(grid), b_.sdf(grid))

    @jax.custom_vjp
    def _measure(a: Primitive, b: Primitive) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_vals = _sdf_combine(a, b, grid)
        indicator = jnp.heaviside(-sdf_vals, 0.5)
        return jnp.sum(indicator) * cell_m

    def _fwd(
        a: Primitive,
        b: Primitive,
    ) -> tuple[Float[Array, ""], tuple[Primitive, Primitive]]:
        primal = _measure(a, b)
        return primal, (a, b)

    def _bwd(
        residuals: tuple[Primitive, Primitive],
        g_bar: Float[Array, ""],
    ) -> tuple[Primitive, Primitive]:
        a_, b_ = residuals

        def _diff(
            a__: Primitive,
            b__: Primitive,
        ) -> Float[Array, ""]:
            grid, cell_m = _make_grid_nd(lo, hi, resolution)
            sdf_vals = _sdf_combine(a__, b__, grid)
            cell_width = (hi[0] - lo[0]) / (resolution - 1)
            indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
            return jnp.sum(indicator) * cell_m

        grad_a, grad_b = jax.grad(_diff, argnums=(0, 1))(a_, b_)
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
    """Compute volume of a - b (subtract b from a) with exact SDF gradients."""
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _boolean_measure_straight_through(a, b, "subtract", resolution, lo, hi)


def intersect_volume_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute intersection volume of a and b with exact SDF gradients."""
    lo, hi = _auto_domain(a, b)
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    return _boolean_measure_straight_through(a, b, "intersect", resolution, lo, hi)


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
    return _union_measure_with_custom_vjp(a, b, resolution, lo, hi)


def union_volume_stratum(
    a: Primitive,
    b: Primitive,
    *,
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute 3D union volume with stratum-aware exact gradients."""
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

    Forward: exact SDF Boolean + heaviside indicator on grid.
    Backward: stratum label dispatch -- disjoint/contained use per-primitive
    gradients, intersecting uses straight-through estimator with
    grid-adaptive beta.
    """

    @jax.custom_vjp
    def _measure(a: Primitive, b: Primitive) -> Float[Array, ""]:
        grid, cell_m = _make_grid_nd(lo, hi, resolution)
        sdf_vals = jnp.minimum(a.sdf(grid), b.sdf(grid))
        indicator = jnp.heaviside(-sdf_vals, 0.5)
        return jnp.sum(indicator) * cell_m

    def _measure_fwd(
        a: Primitive,
        b: Primitive,
    ) -> tuple[Float[Array, ""], tuple[Primitive, Primitive, Float[Array, ""]]]:
        primal = _measure(a, b)
        # Generic stratum detection using grid sampling (ADR-0012)
        grid, _ = _make_grid_nd(lo, hi, resolution)
        label = _detect_stratum_generic(a, b, grid)
        return primal, (a, b, label)

    def _measure_bwd(
        residuals: tuple[Primitive, Primitive, Float[Array, ""]],
        g_bar: Float[Array, ""],
    ) -> tuple[Primitive, Primitive]:
        a_, b_, label = residuals

        # Compute gradient for each stratum
        ga_d, gb_d = _grad_disjoint(a_, b_, lo, hi, resolution)
        ga_i, gb_i = _grad_intersecting(a_, b_, lo, hi, resolution)
        # label 2.0=a_in_b, 3.0=b_in_a; pass label for direction
        ga_c, gb_c = _grad_contained(a_, b_, lo, hi, resolution, label)

        # Select based on stratum label (0=disjoint, 1=intersecting, 2/3=contained)
        contained = (label == 2.0) | (label == 3.0)

        def select_grad(
            gd: Primitive,
            gi: Primitive,
            gc: Primitive,
        ) -> Primitive:
            return jax.tree.map(
                lambda d, i, c: jnp.where(
                    label == 0.0,
                    d,
                    jnp.where(contained, c, i),
                ),
                gd,
                gi,
                gc,
            )

        grad_a = select_grad(ga_d, ga_i, ga_c)
        grad_b = select_grad(gb_d, gb_i, gb_c)

        return (
            jax.tree.map(lambda x: g_bar * x, grad_a),
            jax.tree.map(lambda x: g_bar * x, grad_b),
        )

    _measure.defvjp(_measure_fwd, _measure_bwd)
    return _measure(a, b)
