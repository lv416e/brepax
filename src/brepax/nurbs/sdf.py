"""Signed distance from a query point to a B-spline surface.

Computes the distance to the closest point on the surface via Newton
projection, and determines sign from precomputed coarse grid normals
for consistency with PMC cell classification.  The entire computation
is differentiable w.r.t. control points via JAX automatic
differentiation through the unrolled Newton projection.
"""

from __future__ import annotations

import jax
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
    sign_flip: float = 1.0,
    coarse_positions: Float[Array, "g 3"] | None = None,
    coarse_normals: Float[Array, "g 3"] | None = None,
) -> Float[Array, ""]:
    """Signed distance from a query point to a B-spline surface.

    Distance is computed via Newton closest-point projection.  Sign
    is determined by precomputed coarse grid normals when available,
    ensuring consistency between PMC reconstruction and volume
    evaluation.  Falls back to closest-point normal when coarse
    grid data is not provided.

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
        param_u_range: Trimmed parameter range in u.
        param_v_range: Trimmed parameter range in v.
        sign_flip: Orientation correction for OCCT REVERSED faces.
        coarse_positions: Precomputed surface sample positions.
        coarse_normals: Precomputed orientation-corrected normals.

    Returns:
        Signed distance (scalar).

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

    point = evaluate_surface_derivs(
        control_points,
        knots_u,
        knots_v,
        degree_u,
        degree_v,
        u_opt,
        v_opt,
        weights,
    )[0]

    diff = query - point
    dist = jnp.sqrt(jnp.sum(diff**2) + 1e-20)

    if coarse_positions is not None and coarse_normals is not None:
        # Sign from coarse grid nearest-normal (same source as PMC)
        dists_sq = jnp.sum((query - coarse_positions) ** 2, axis=-1)
        best = jax.lax.stop_gradient(jnp.argmin(dists_sq))
        nearest_nrm = coarse_normals[best]
        sign = jnp.sign(jnp.dot(query - coarse_positions[best], nearest_nrm))
    else:
        # Fallback: Newton closest-point normal
        _, du, dv = evaluate_surface_derivs(
            control_points,
            knots_u,
            knots_v,
            degree_u,
            degree_v,
            u_opt,
            v_opt,
            weights,
        )
        normal = jnp.cross(du, dv)
        normal = normal / (jnp.linalg.norm(normal) + 1e-10)
        sign = jnp.sign(jnp.dot(diff, normal)) * sign_flip

    return sign * dist


__all__ = [
    "bspline_sdf",
]
