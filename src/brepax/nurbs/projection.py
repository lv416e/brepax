"""Closest-point projection onto a B-spline surface.

Finds the parameter values ``(u*, v*)`` that minimize the distance
from a query point to the surface, using Newton iteration on the
first-order optimality conditions.  The iteration is unrolled for
JIT compatibility, and gradients flow through via automatic
differentiation of the unrolled loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.nurbs.evaluate import evaluate_surface

_MAX_ITER = 10
_TOL_SQ = 1e-16
_COARSE_GRID = 8


def coarse_initial_guess(
    query: Float[Array, 3],
    control_points: Float[Array, "nu nv 3"],
    knots_u: Float[Array, ...],
    knots_v: Float[Array, ...],
    degree_u: int,
    degree_v: int,
    weights: Float[Array, "nu nv"] | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Find a coarse initial guess by sampling the surface on a grid.

    Evaluates the surface at ``_COARSE_GRID x _COARSE_GRID`` parameter
    samples and returns the (u, v) of the closest sample to the query
    point.  The result is stop-gradiented since argmin is not
    differentiable; gradients flow only through Newton iterations.
    """
    u_lo, u_hi = knots_u[degree_u], knots_u[-degree_u - 1]
    v_lo, v_hi = knots_v[degree_v], knots_v[-degree_v - 1]
    us = jnp.linspace(u_lo, u_hi, _COARSE_GRID)
    vs = jnp.linspace(v_lo, v_hi, _COARSE_GRID)

    # Evaluate surface at coarse grid
    def _eval(u: Float[Array, ""], v: Float[Array, ""]) -> Float[Array, 3]:
        return evaluate_surface(
            control_points, knots_u, knots_v, degree_u, degree_v, u, v, weights
        )

    # Build (G*G, 3) surface samples
    u_grid, v_grid = jnp.meshgrid(us, vs, indexing="ij")
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    samples = jax.vmap(_eval)(u_flat, v_flat)

    # Find closest sample
    dists = jnp.sum((samples - query) ** 2, axis=-1)
    best = jnp.argmin(dists)
    u0 = jax.lax.stop_gradient(u_flat[best])
    v0 = jax.lax.stop_gradient(v_flat[best])
    return u0, v0


def closest_point(
    query: Float[Array, 3],
    control_points: Float[Array, "nu nv 3"],
    knots_u: Float[Array, ...],
    knots_v: Float[Array, ...],
    degree_u: int,
    degree_v: int,
    u0: Float[Array, ""] | float = 0.5,
    v0: Float[Array, ""] | float = 0.5,
    weights: Float[Array, "nu nv"] | None = None,
    param_u_range: tuple[float, float] | None = None,
    param_v_range: tuple[float, float] | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Find the closest point on a B-spline surface to a query point.

    Minimizes ``||q - S(u, v)||^2`` via Newton iteration on the
    gradient of the squared distance.

    Args:
        query: Query point, shape ``(3,)``.
        control_points: Control point grid, shape ``(n_u, n_v, 3)``.
        knots_u: Knot vector in u-direction.
        knots_v: Knot vector in v-direction.
        degree_u: Polynomial degree in u.
        degree_v: Polynomial degree in v.
        u0: Initial guess for u parameter.
        v0: Initial guess for v parameter.
        weights: Optional weight grid for rational surfaces.
        param_u_range: Trimmed parameter range in u. Overrides knot domain.
        param_v_range: Trimmed parameter range in v. Overrides knot domain.

    Returns:
        Optimal parameters ``(u*, v*)``, each a scalar.
    """

    def _sq_dist(uv: Float[Array, 2]) -> Float[Array, ""]:
        pt = evaluate_surface(
            control_points,
            knots_u,
            knots_v,
            degree_u,
            degree_v,
            uv[0],
            uv[1],
            weights,
        )
        return 0.5 * jnp.sum((query - pt) ** 2)

    grad_fn = jax.grad(_sq_dist)
    hess_fn = jax.hessian(_sq_dist)

    uv = jnp.array(
        [
            jnp.asarray(u0, dtype=float),
            jnp.asarray(v0, dtype=float),
        ]
    )

    u_knot_lo, u_knot_hi = knots_u[degree_u], knots_u[-degree_u - 1]
    v_knot_lo, v_knot_hi = knots_v[degree_v], knots_v[-degree_v - 1]

    if param_u_range is not None:
        # Intersect with knot domain to prevent out-of-bounds evaluation
        u_lo = jnp.clip(jnp.asarray(param_u_range[0]), u_knot_lo, u_knot_hi)
        u_hi = jnp.clip(jnp.asarray(param_u_range[1]), u_knot_lo, u_knot_hi)
    else:
        u_lo, u_hi = u_knot_lo, u_knot_hi
    if param_v_range is not None:
        v_lo = jnp.clip(jnp.asarray(param_v_range[0]), v_knot_lo, v_knot_hi)
        v_hi = jnp.clip(jnp.asarray(param_v_range[1]), v_knot_lo, v_knot_hi)
    else:
        v_lo, v_hi = v_knot_lo, v_knot_hi

    def _step(uv: Float[Array, 2]) -> Float[Array, 2]:
        g = grad_fn(uv)
        h = hess_fn(uv)
        delta = jnp.linalg.solve(h + 1e-6 * jnp.eye(2), g)
        uv_new = uv - delta
        return jnp.array(
            [
                jnp.clip(uv_new[0], u_lo, u_hi),
                jnp.clip(uv_new[1], v_lo, v_hi),
            ]
        )

    for _ in range(_MAX_ITER):
        uv_next = _step(uv)
        converged = jnp.sum((uv_next - uv) ** 2) < _TOL_SQ
        uv = jnp.where(converged, uv, uv_next)

    return uv[0], uv[1]


__all__ = [
    "closest_point",
]
