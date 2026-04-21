"""Analytical foot-of-perpendicular on primitive surfaces.

Each function returns the closest point on an infinite analytical
surface (plane, sphere, cylinder, cone, torus) to a query point.
Closed form given the primitive parameters; fully differentiable
w.r.t. both the query and the primitive parameters.

Scope is the untrimmed surface only; trim-boundary interaction is
layered on via Marschner composition in a separate module.  Cone
and torus inner-region degeneracies (apex, tube axis) are handled
with an epsilon guard on the radial direction.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

_EPS = 1e-12


def foot_on_plane(
    query: Float[Array, 3],
    normal: Float[Array, 3],
    offset: Float[Array, ""],
) -> Float[Array, 3]:
    """Closest point on an infinite plane ``normal . x = offset``."""
    signed = jnp.dot(query, normal) - offset
    return query - signed * normal


def foot_on_sphere(
    query: Float[Array, 3],
    center: Float[Array, 3],
    radius: Float[Array, ""],
) -> Float[Array, 3]:
    """Closest point on a sphere of ``radius`` around ``center``.

    At the center (degenerate) the result is ``center + radius * e_z``;
    an arbitrary direction is required to keep the gradient finite.
    """
    v = query - center
    norm = jnp.linalg.norm(v)
    direction = jnp.where(norm > _EPS, v / (norm + _EPS), jnp.array([0.0, 0.0, 1.0]))
    return center + radius * direction  # type: ignore[no-any-return]


def foot_on_cylinder(
    query: Float[Array, 3],
    point: Float[Array, 3],
    axis: Float[Array, 3],
    radius: Float[Array, ""],
) -> Float[Array, 3]:
    """Closest point on an infinite cylinder of ``radius`` about the axis.

    On the axis itself the radial direction is ill-defined; a canonical
    fallback is used to keep the computation differentiable.
    """
    v = query - point
    axial_len = jnp.dot(v, axis)
    radial = v - axial_len * axis
    radial_norm = jnp.linalg.norm(radial)
    direction = jnp.where(
        radial_norm > _EPS,
        radial / (radial_norm + _EPS),
        jnp.array([1.0, 0.0, 0.0]) - axis[0] * axis,
    )
    return point + axial_len * axis + radius * direction  # type: ignore[no-any-return]


def foot_on_cone(
    query: Float[Array, 3],
    apex: Float[Array, 3],
    axis: Float[Array, 3],
    angle: Float[Array, ""],
) -> Float[Array, 3]:
    """Closest point on the half-cone extending forward of ``apex``.

    The cone is the surface ``r(h) = h * tan(angle)`` for ``h >= 0``
    along ``axis``.  For queries that project to ``h < 0`` the foot is
    clamped to ``apex``.
    """
    v = query - apex
    h = jnp.dot(v, axis)
    radial = v - h * axis
    radial_norm = jnp.linalg.norm(radial)
    radial_dir = jnp.where(
        radial_norm > _EPS,
        radial / (radial_norm + _EPS),
        jnp.array([1.0, 0.0, 0.0]) - axis[0] * axis,
    )
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    t = h * cos_a + radial_norm * sin_a
    t = jnp.maximum(t, 0.0)
    return apex + t * (cos_a * axis + sin_a * radial_dir)  # type: ignore[no-any-return]


def foot_on_torus(
    query: Float[Array, 3],
    center: Float[Array, 3],
    axis: Float[Array, 3],
    major_radius: Float[Array, ""],
    minor_radius: Float[Array, ""],
) -> Float[Array, 3]:
    """Closest point on the torus tube around the major ring.

    Project to the tube-center ring first, then shift outward by
    ``minor_radius``.  On the central axis (``radial == 0``) a canonical
    in-plane direction is used; on the tube center itself
    (``dq == 0``) the fallback is any in-plane axis.
    """
    v = query - center
    h = jnp.dot(v, axis)
    radial = v - h * axis
    radial_norm = jnp.linalg.norm(radial)
    radial_dir = jnp.where(
        radial_norm > _EPS,
        radial / (radial_norm + _EPS),
        jnp.array([1.0, 0.0, 0.0]) - axis[0] * axis,
    )
    tube_center = center + major_radius * radial_dir
    dq = query - tube_center
    dq_norm = jnp.linalg.norm(dq)
    dq_dir = jnp.where(dq_norm > _EPS, dq / (dq_norm + _EPS), radial_dir)
    return tube_center + minor_radius * dq_dir  # type: ignore[no-any-return]


__all__ = [
    "foot_on_cone",
    "foot_on_cylinder",
    "foot_on_plane",
    "foot_on_sphere",
    "foot_on_torus",
]
