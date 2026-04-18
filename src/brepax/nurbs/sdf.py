"""Signed distance from a query point to a B-spline surface.

Computes the distance to the closest point on the surface and
determines sign from the surface normal orientation.  The entire
computation is differentiable w.r.t. control points via JAX
automatic differentiation through the unrolled Newton projection.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.nurbs.evaluate import evaluate_surface_derivs
from brepax.nurbs.projection import closest_point


def bspline_sdf(
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
) -> Float[Array, ""]:
    """Signed distance from a query point to a B-spline surface.

    Projects the query point onto the surface via Newton iteration,
    computes the Euclidean distance, and determines sign from the
    surface normal (positive outside, negative inside when the normal
    points outward).

    The result is differentiable w.r.t. ``control_points`` through
    the unrolled Newton iteration.  Supports both non-rational and
    rational (NURBS) surfaces via optional weights.

    Args:
        query: Query point, shape ``(3,)``.
        control_points: Control point grid, shape ``(n_u, n_v, 3)``.
        knots_u: Knot vector in u-direction.
        knots_v: Knot vector in v-direction.
        degree_u: Polynomial degree in u.
        degree_v: Polynomial degree in v.
        u0: Initial u parameter guess for projection.
        v0: Initial v parameter guess for projection.
        weights: Optional weight grid for rational surfaces.

    Returns:
        Signed distance (scalar).  Positive when the query lies on
        the side of the surface normal, negative on the opposite side.

    Examples:
        >>> import jax.numpy as jnp
        >>> pts = jnp.array([[[0,0,0],[1,0,0]],
        ...                  [[0,1,0],[1,1,0]]], dtype=float)
        >>> knots = jnp.array([0., 0., 1., 1.])
        >>> d = bspline_sdf(
        ...     jnp.array([0.5, 0.5, 1.0]), pts,
        ...     knots, knots, 1, 1,
        ... )
    """
    u_opt, v_opt = closest_point(
        query,
        control_points,
        knots_u,
        knots_v,
        degree_u,
        degree_v,
        u0,
        v0,
        weights,
        param_u_range,
        param_v_range,
    )

    point, du, dv = evaluate_surface_derivs(
        control_points,
        knots_u,
        knots_v,
        degree_u,
        degree_v,
        u_opt,
        v_opt,
        weights,
    )

    diff = query - point
    dist = jnp.sqrt(jnp.sum(diff**2) + 1e-20)

    normal = jnp.cross(du, dv)
    normal = normal / (jnp.linalg.norm(normal) + 1e-10)

    sign = jnp.sign(jnp.dot(diff, normal))
    return sign * dist


__all__ = [
    "bspline_sdf",
]
