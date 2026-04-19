"""Generalized winding number for inside/outside classification.

Computes the winding number of a query point with respect to a
closed triangulated surface.  The winding number is 1 inside the
solid and 0 outside, providing robust inside/outside classification
without requiring half-space definitions.

The solid angle per triangle uses the Van Oosterom & Strackee (1983)
formula, which is differentiable via ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg_eval import integrate_sdf_volume, make_grid_3d
from brepax.primitives._base import Primitive


def triangle_solid_angle(
    point: Float[Array, 3],
    v0: Float[Array, 3],
    v1: Float[Array, 3],
    v2: Float[Array, 3],
) -> Float[Array, ""]:
    """Signed solid angle of a triangle as seen from a point.

    Uses the Van Oosterom & Strackee (1983) formula.  The sign
    depends on the triangle's winding order relative to the point.

    Args:
        point: Query point, shape ``(3,)``.
        v0: First triangle vertex.
        v1: Second triangle vertex.
        v2: Third triangle vertex.

    Returns:
        Signed solid angle (scalar).
    """
    a = v0 - point
    b = v1 - point
    c = v2 - point

    la = jnp.linalg.norm(a) + 1e-20
    lb = jnp.linalg.norm(b) + 1e-20
    lc = jnp.linalg.norm(c) + 1e-20

    numer = jnp.dot(a, jnp.cross(b, c))
    denom = la * lb * lc + jnp.dot(a, b) * lc + jnp.dot(a, c) * lb + jnp.dot(b, c) * la
    return 2.0 * jnp.arctan2(numer, denom)


def winding_number(
    point: Float[Array, 3],
    triangles: Float[Array, "n 3 3"],
) -> Float[Array, ""]:
    """Generalized winding number at a query point.

    Sums the signed solid angles of all triangles.  Returns ~1.0
    for points inside a closed surface and ~0.0 for points outside.

    Args:
        point: Query point, shape ``(3,)``.
        triangles: Triangle vertices, shape ``(n, 3, 3)``.

    Returns:
        Winding number (scalar).
    """

    def _solid_angle(tri: Float[Array, "3 3"]) -> Float[Array, ""]:
        return triangle_solid_angle(point, tri[0], tri[1], tri[2])

    total = jnp.sum(jax.vmap(_solid_angle)(triangles))
    return total / (4.0 * jnp.pi)


def gwn_signed_sdf(
    query: Float[Array, 3],
    primitives: tuple[Primitive, ...],
    gwn_triangles: Float[Array, "n 3 3"],
) -> Float[Array, ""]:
    """SDF with sign from winding number and distance from primitives.

    Sign is determined by the generalized winding number (robust for
    all surface types including BSpline).  Unsigned distance is the
    minimum of per-face distances (accurate and differentiable).
    Sign is ``stop_gradient`` since it is locally constant; gradients
    flow only through the unsigned distance.

    Args:
        query: Query point, shape ``(3,)``.
        primitives: Tuple of primitives (each has ``.sdf()``).
        gwn_triangles: Precomputed triangle mesh for GWN sign.

    Returns:
        Signed distance (scalar).  Negative inside, positive outside.
    """
    # Unsigned distance: min of abs(per-face SDF)
    dists = jnp.stack([jnp.abs(p.sdf(query)) for p in primitives])
    # Differentiable soft-min via negative logsumexp
    temperature = 0.01
    unsigned = -temperature * jax.nn.logsumexp(-dists / temperature)
    unsigned = jnp.maximum(unsigned, 1e-20)

    # Sign from GWN (stop_gradient: locally constant)
    w = winding_number(query, gwn_triangles)
    sign = jax.lax.stop_gradient(jnp.where(w > 0.5, -1.0, 1.0))

    return sign * unsigned


def gwn_signed_volume(
    primitives: tuple[Primitive, ...],
    gwn_triangles: Float[Array, "n 3 3"],
    *,
    resolution: int = 32,
    lo: Float[Array, 3] | None = None,
    hi: Float[Array, 3] | None = None,
) -> Float[Array, ""]:
    """Differentiable volume via GWN-signed minimum distance SDF.

    Combines winding number (for sign) with per-face unsigned
    distance (for distance value) to produce a correct SDF that
    works for all surface types.  Volume is computed by sigmoid
    integration on a 3D grid, identical to the CSG-Stump path.

    Args:
        primitives: Tuple of primitives with ``.sdf()`` methods.
        gwn_triangles: Precomputed triangle mesh for GWN sign.
        resolution: Grid points per axis.
        lo: Grid lower bound.
        hi: Grid upper bound.

    Returns:
        Volume estimate (scalar), differentiable w.r.t. primitive
        parameters.
    """
    if lo is None or hi is None:
        all_verts = gwn_triangles.reshape(-1, 3)
        margin = 0.5
        if lo is None:
            lo = jnp.min(all_verts, axis=0) - margin
        if hi is None:
            hi = jnp.max(all_verts, axis=0) + margin

    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)

    grid, _ = make_grid_3d(lo, hi, resolution)
    flat_pts = grid.reshape(-1, 3)

    sdf_vals = jax.vmap(lambda q: gwn_signed_sdf(q, primitives, gwn_triangles))(
        flat_pts
    )

    return integrate_sdf_volume(sdf_vals, lo, hi, resolution)


__all__ = [
    "gwn_signed_sdf",
    "gwn_signed_volume",
    "triangle_solid_angle",
    "winding_number",
]
