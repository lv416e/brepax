"""Unsigned distance from a 3D point to a padded closed polyline.

Used as the ``d_partial(p)`` component of a trim-aware surface signed
distance: the unsigned 3D distance from the query to the surface-space
image of a closed trim loop.  Sign is determined separately by the
enclosing solid / trim indicator in parameter space; this module is
purely the 3D unsigned distance reduction over the polyline.

The closed-loop convention matches ``nurbs/trim.py``: vertices are
padded to a fixed length, ``mask`` carries per-vertex validity, and
the last valid vertex connects back to the first.

Gradients stay finite at degenerate inputs (query on a vertex or on a
segment, or a zero-length segment) via the same safe-square pattern as
``primitives/foot.py``: the ``where`` guards the squared quantity, and
``sqrt`` only runs on values bounded below by 1 at the degenerate
point.  The older ``sqrt(sq + eps)`` form in the 2D sibling is avoided
because it biases the forward value and still leaks NaN through the
untaken ``where`` branch in the VJP.

Masked-out segments contribute a large finite sentinel (``1e30``)
rather than ``jnp.inf`` so that an all-masked reduction (an empty
trim loop) stays finite in backward; this matches the convention in
``nurbs/trim.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

_EPS_SQ = 1e-24


def point_segment_distance_3d(
    point: Float[Array, 3],
    a: Float[Array, 3],
    b: Float[Array, 3],
) -> Float[Array, ""]:
    """Unsigned distance from a point to the segment ``a -> b`` in 3D.

    Degenerate ``a == b`` collapses to the point-to-vertex distance,
    with a finite gradient maintained by the safe-square pattern.
    """
    ab = b - a
    ap = point - a
    ab_sq = jnp.dot(ab, ab)
    ab_sq_safe = jnp.where(ab_sq > _EPS_SQ, ab_sq, 1.0)
    t = jnp.where(
        ab_sq > _EPS_SQ,
        jnp.clip(jnp.dot(ap, ab) / ab_sq_safe, 0.0, 1.0),
        0.0,
    )
    foot = a + t * ab
    diff = point - foot
    diff_sq = jnp.dot(diff, diff)
    diff_sq_safe = jnp.where(diff_sq > _EPS_SQ, diff_sq, 1.0)
    return jnp.where(diff_sq > _EPS_SQ, jnp.sqrt(diff_sq_safe), 0.0)


def polyline_unsigned_distance(
    point: Float[Array, 3],
    vertices: Float[Array, "n 3"],
    mask: Float[Array, ...],
) -> Float[Array, ""]:
    """Unsigned distance to a padded closed 3D polyline.

    Vertices are padded to a fixed length; ``mask`` is 1.0 on valid
    positions and 0.0 on padding.  The polyline is closed by wrapping
    the last valid vertex back to the first, so for ``mask =
    [1, 1, 1, 0]`` the segments are ``(v0->v1)``, ``(v1->v2)``,
    ``(v2->v0)``.  Segments touching padding contribute a large finite
    sentinel so the ``jnp.min`` backward stays finite even when the
    mask is all zero.

    Args:
        point: Query point, shape ``(3,)``.
        vertices: Padded loop vertices, shape ``(n, 3)``.
        mask: Per-vertex validity, shape ``(n,)``.  1.0 valid, 0.0
            padding.

    Returns:
        Scalar unsigned distance to the nearest valid segment.
    """
    v1 = vertices
    v2 = jnp.roll(vertices, -1, axis=0)
    # Close the loop: the segment leaving the last valid vertex must
    # reconnect to vertex 0, not to the first padding slot.
    num_valid = jnp.sum(mask).astype(jnp.int32)
    last_valid_idx = jnp.maximum(0, num_valid - 1)
    v2 = v2.at[last_valid_idx].set(vertices[0])

    per_segment = jax.vmap(point_segment_distance_3d, in_axes=(None, 0, 0))(
        point, v1, v2
    )

    # Mask invalid segments (either endpoint is padding).  The segment
    # starting at slot i uses v1[i] and v2[i]; v1 is valid iff mask[i]
    # and v2 is valid iff it is vertex 0 (set above) or mask[(i+1)%n].
    segment_mask = mask * jnp.roll(mask, -1, axis=0)
    segment_mask = segment_mask.at[last_valid_idx].set(mask[last_valid_idx])

    masked = jnp.where(segment_mask > 0.5, per_segment, 1e30)
    return jnp.min(masked)


__all__ = [
    "point_segment_distance_3d",
    "polyline_unsigned_distance",
]
