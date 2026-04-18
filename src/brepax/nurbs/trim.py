"""Differentiable trim indicator for non-rectangular face boundaries.

Computes signed distance from a parametric point ``(u, v)`` to the
trim polygon boundary in 2D parameter space.  The signed distance is
then passed through a sigmoid to produce a smooth 0-to-1 trim
indicator, analogous to how 3D SDF + sigmoid underpins all BRepAX
metrics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def signed_distance_polygon(
    point: Float[Array, 2],
    vertices: Float[Array, "n 2"],
    mask: Float[Array, ...],
) -> Float[Array, ""]:
    """Signed distance from a 2D point to a closed polygon boundary.

    Negative inside, positive outside (consistent with BRepAX SDF
    convention where ``sigmoid(-d/eps)`` yields 1 inside).

    The polygon is formed by connecting consecutive valid vertices
    (where ``mask > 0``), with the last valid vertex connecting back
    to the first.

    Args:
        point: Query point in 2D parameter space, shape ``(2,)``.
        vertices: Polygon vertices (padded), shape ``(max_n, 2)``.
        mask: Validity mask, shape ``(max_n,)``.  1.0 for valid
            vertices, 0.0 for padding.

    Returns:
        Signed distance (scalar).  Negative inside the polygon,
        positive outside.

    Examples:
        >>> import jax.numpy as jnp
        >>> tri = jnp.array([[0.,0.], [1.,0.], [0.,1.], [0.,0.]])
        >>> m = jnp.array([1., 1., 1., 0.])
        >>> d = signed_distance_polygon(jnp.array([0.2, 0.2]), tri, m)
    """
    # Edge endpoints: v1[i] -> v2[i]
    v1 = vertices
    v2 = jnp.roll(vertices, -1, axis=0)

    edge = v2 - v1
    w = point - v1

    # Project point onto each edge segment, clamp to [0, 1]
    edge_len_sq = jnp.sum(edge**2, axis=-1) + 1e-20
    t = jnp.clip(jnp.sum(w * edge, axis=-1) / edge_len_sq, 0.0, 1.0)

    closest = v1 + t[:, None] * edge
    dist_sq = jnp.sum((point - closest) ** 2, axis=-1)

    # Mask invalid edges (padding)
    dist_sq = jnp.where(mask > 0.5, dist_sq, 1e30)

    unsigned_dist = jnp.sqrt(jnp.min(dist_sq) + 1e-20)

    # Sign via winding number (sum of signed angles)
    d1 = v1 - point
    d2 = v2 - point
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    dot = d1[:, 0] * d2[:, 0] + d1[:, 1] * d2[:, 1]
    angles = jnp.arctan2(cross, dot)
    angles = jnp.where(mask > 0.5, angles, 0.0)
    winding = jnp.sum(angles) / (2.0 * jnp.pi)

    # Inside: winding ~ +1 or -1 (nonzero), outside: winding ~ 0
    sign = jnp.where(jnp.abs(winding) > 0.5, -1.0, 1.0)
    return sign * unsigned_dist


def trim_indicator(
    point: Float[Array, 2],
    vertices: Float[Array, "n 2"],
    mask: Float[Array, ...],
    sharpness: float = 200.0,
) -> Float[Array, ""]:
    """Differentiable trim indicator: 1 inside the polygon, 0 outside.

    Uses ``sigmoid(-d / epsilon)`` where ``d`` is the signed distance
    to the polygon boundary and ``epsilon = 1 / sharpness``.

    Args:
        point: Query point in 2D parameter space, shape ``(2,)``.
        vertices: Polygon vertices (padded), shape ``(max_n, 2)``.
        mask: Validity mask, shape ``(max_n,)``.
        sharpness: Controls sigmoid transition width.  Higher values
            give a sharper boundary.  Default 200.0 is suitable for
            parametric domains of order 1.

    Returns:
        Scalar in ``[0, 1]``.  Approximately 1.0 well inside the
        polygon, 0.0 well outside.

    Examples:
        >>> import jax.numpy as jnp
        >>> tri = jnp.array([[0.,0.], [1.,0.], [0.,1.], [0.,0.]])
        >>> m = jnp.array([1., 1., 1., 0.])
        >>> val = trim_indicator(jnp.array([0.2, 0.2]), tri, m)
    """
    d = signed_distance_polygon(point, vertices, mask)
    return jax.nn.sigmoid(-d * sharpness)


__all__ = [
    "signed_distance_polygon",
    "trim_indicator",
]
