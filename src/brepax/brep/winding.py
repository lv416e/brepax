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


__all__ = [
    "triangle_solid_angle",
    "winding_number",
]
