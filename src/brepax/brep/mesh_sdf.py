"""Mesh-based signed distance field for BSpline-heavy models.

Provides a drop-in SDF callable built from a triangle mesh, replacing
CSG-Stump SDF where BSpline half-space sign determination fails.
Sign is determined by the generalized winding number; unsigned distance
uses closed-form point-to-triangle projection.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.winding import winding_number


def point_triangle_distance(
    point: Float[Array, 3],
    v0: Float[Array, 3],
    v1: Float[Array, 3],
    v2: Float[Array, 3],
) -> Float[Array, ""]:
    """Unsigned distance from a point to a triangle.

    Uses Voronoi region classification with barycentric coordinates.
    All branching via ``jnp.where`` / ``jnp.clip`` for JIT compatibility.
    Covers all 7 cases (3 vertices, 3 edges, 1 face) implicitly:
    edge distances degenerate to vertex distances at clip boundaries.

    Args:
        point: Query point, shape ``(3,)``.
        v0: First triangle vertex, shape ``(3,)``.
        v1: Second triangle vertex, shape ``(3,)``.
        v2: Third triangle vertex, shape ``(3,)``.

    Returns:
        Unsigned distance (non-negative scalar).

    Examples:
        >>> import jax.numpy as jnp
        >>> v0 = jnp.array([0.0, 0.0, 0.0])
        >>> v1 = jnp.array([1.0, 0.0, 0.0])
        >>> v2 = jnp.array([0.0, 1.0, 0.0])
        >>> float(point_triangle_distance(jnp.array([0.25, 0.25, 1.0]), v0, v1, v2))
        1.0
    """
    ab = v1 - v0
    ac = v2 - v0

    # Project point onto triangle plane
    normal = jnp.cross(ab, ac)
    n_dot_n = jnp.dot(normal, normal) + 1e-30
    ap = point - v0
    signed_height = jnp.dot(ap, normal)
    proj = point - (signed_height / n_dot_n) * normal

    # Barycentric coordinates of projected point
    v0_to_proj = proj - v0
    d00 = jnp.dot(ab, ab)
    d01 = jnp.dot(ab, ac)
    d11 = jnp.dot(ac, ac)
    d20 = jnp.dot(v0_to_proj, ab)
    d21 = jnp.dot(v0_to_proj, ac)
    denom = d00 * d11 - d01 * d01 + 1e-30
    bary_v = (d11 * d20 - d01 * d21) / denom
    bary_w = (d00 * d21 - d01 * d20) / denom
    inside = (bary_v >= 0.0) & (bary_w >= 0.0) & (bary_v + bary_w <= 1.0)

    face_dist_sq = signed_height**2 / n_dot_n

    # Edge distances (vertex cases emerge when parameter clamps to 0 or 1)
    def _edge_dist_sq(
        p: Float[Array, 3], a: Float[Array, 3], edge: Float[Array, 3]
    ) -> Float[Array, ""]:
        t = jnp.dot(p - a, edge) / (jnp.dot(edge, edge) + 1e-30)
        t = jnp.clip(t, 0.0, 1.0)
        diff = p - (a + t * edge)
        return jnp.sum(diff**2)

    d_ab = _edge_dist_sq(point, v0, ab)
    d_bc = _edge_dist_sq(point, v1, v2 - v1)
    d_ca = _edge_dist_sq(point, v2, v0 - v2)
    min_edge_sq = jnp.minimum(d_ab, jnp.minimum(d_bc, d_ca))

    dist_sq = jnp.where(inside, face_dist_sq, min_edge_sq)
    return jnp.sqrt(dist_sq + 1e-20)


def _single_point_sdf(
    point: Float[Array, 3],
    triangles: Float[Array, "m 3 3"],
) -> Float[Array, ""]:
    """SDF at a single query point against a triangle mesh."""

    # Unsigned distance: min over all triangles
    def _dist_to_tri(tri: Float[Array, "3 3"]) -> Float[Array, ""]:
        return point_triangle_distance(point, tri[0], tri[1], tri[2])

    all_dists = jax.vmap(_dist_to_tri)(triangles)
    idx = jax.lax.stop_gradient(jnp.argmin(all_dists))
    min_dist = all_dists[idx]

    # Sign from winding number: > 0.5 means inside (negative SDF)
    wn = winding_number(point, triangles)
    sign = jnp.where(wn > 0.5, -1.0, 1.0)
    sign = jax.lax.stop_gradient(sign)

    return sign * min_dist


def mesh_sdf(
    query_points: Float[Array, "n 3"],
    triangles: Float[Array, "m 3 3"],
    *,
    chunk_size: int = 1024,
) -> Float[Array, " n"]:
    """Signed distance from query points to a triangle mesh.

    Sign is determined by the generalized winding number (inside
    negative, outside positive).  Unsigned distance is the minimum
    point-to-triangle distance.  The argmin index and sign are
    ``stop_gradient``-ed; the distance to the selected triangle
    is differentiable w.r.t. vertex positions.

    Evaluation is chunked to avoid OOM on large grids.

    Args:
        query_points: Query positions, shape ``(n, 3)``.
        triangles: Mesh triangles, shape ``(m, 3, 3)``.
        chunk_size: Number of query points per chunk (default 1024).

    Returns:
        Signed distance values, shape ``(n,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> tris = jnp.array([[[0,0,0],[1,0,0],[0,1,0]]], dtype=float)
        >>> pts = jnp.array([[0.25, 0.25, 1.0]], dtype=float)
        >>> float(mesh_sdf(pts, tris))
        1.0
    """
    n = query_points.shape[0]

    @partial(jax.jit, static_argnums=())
    def _chunk_fn(chunk: Float[Array, "c 3"]) -> Float[Array, " c"]:
        return jax.vmap(lambda p: _single_point_sdf(p, triangles))(chunk)

    # Pad to multiple of chunk_size for static shapes
    remainder = n % chunk_size
    pad_size = jnp.where(remainder == 0, 0, chunk_size - remainder)
    padded = jnp.pad(query_points, ((0, pad_size), (0, 0)))
    n_padded = padded.shape[0]
    chunks = padded.reshape(n_padded // chunk_size, chunk_size, 3)

    results = jax.lax.map(_chunk_fn, chunks)
    flat: Float[Array, " n"] = results.reshape(-1)[:n]
    return flat


def make_mesh_sdf(
    triangles: Float[Array, "m 3 3"],
    *,
    chunk_size: int = 1024,
) -> Callable[..., Float[Array, ...]]:
    """Create an SDF callable from a triangle mesh.

    Returns a function with the same interface as primitive SDF
    methods: accepts points of shape ``(..., 3)`` and returns
    SDF values of shape ``(...)``.  Drop-in replacement for
    CSG-Stump SDF in existing metrics.

    Args:
        triangles: Mesh triangles, shape ``(m, 3, 3)``.
        chunk_size: Chunk size for batched evaluation (default 1024).

    Returns:
        SDF callable: ``f(points: (..., 3)) -> (...)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.mesh_sdf import make_mesh_sdf
        >>> tris = jnp.array([[[0,0,0],[1,0,0],[0,1,0]]], dtype=float)
        >>> sdf_fn = make_mesh_sdf(tris)
        >>> grid = jnp.zeros((4, 4, 4, 3))
        >>> sdf_fn(grid).shape
        (4, 4, 4)
    """

    def sdf_fn(points: Float[Array, "... 3"]) -> Float[Array, ...]:
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        sdf_flat = mesh_sdf(flat, triangles, chunk_size=chunk_size)
        return sdf_flat.reshape(shape)

    return sdf_fn


__all__ = [
    "make_mesh_sdf",
    "mesh_sdf",
    "point_triangle_distance",
]
