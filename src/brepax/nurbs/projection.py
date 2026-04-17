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

_MAX_ITER = 20
_TOL_SQ = 1e-16


def closest_point(
    query: Float[Array, 3],
    control_points: Float[Array, "nu nv 3"],
    knots_u: Float[Array, ...],
    knots_v: Float[Array, ...],
    degree_u: int,
    degree_v: int,
    u0: Float[Array, ""] | float = 0.5,
    v0: Float[Array, ""] | float = 0.5,
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

    u_lo, u_hi = knots_u[degree_u], knots_u[-degree_u - 1]
    v_lo, v_hi = knots_v[degree_v], knots_v[-degree_v - 1]

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
