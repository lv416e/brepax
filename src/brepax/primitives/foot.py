"""Analytical foot-of-perpendicular on primitive surfaces.

Each function returns the closest point on an infinite analytical
surface (plane, sphere, cylinder, cone, torus) to a query point.
Closed form given the primitive parameters; fully differentiable
w.r.t. both the query and the primitive parameters.

Scope is the untrimmed surface only; trim-boundary interaction is
layered on via Marschner composition in a separate module.  Cone
and torus inner-region degeneracies (apex, tube axis) are handled
with an epsilon guard on the radial direction.

Zero-denominator handling uses a safe-square-then-sqrt pattern that
keeps both forward *and* gradient finite at the degenerate boundary.
``jnp.linalg.norm(v)`` has infinite derivative at ``v = 0``; since
``jnp.where`` evaluates both branches in the VJP, a naive guard on
``norm`` still leaks NaN into ``jax.grad``.  The pattern used here
switches on the squared norm before ``sqrt``, so the ``sqrt`` argument
is bounded below by 1 when the query is degenerate:

    sq = sum(v * v)
    is_ok = sq > eps_sq
    safe_sq = where(is_ok, sq, 1.0)
    norm = sqrt(safe_sq)   # always >= 1 at the degenerate point
    direction = where(is_ok, v / norm, fallback)
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

_EPS_SQ = 1e-24


def _axis_orthogonal(axis: Float[Array, 3]) -> Float[Array, 3]:
    """A unit vector perpendicular to ``axis``, robust for any unit axis.

    Picks a seed orthogonal in the coordinate direction least aligned
    with ``axis``, then takes the cross product.  ``|cross(axis, seed)|``
    is at least ``sqrt(3)/2`` because ``|axis[0]| > 0.5`` picks the y
    seed and ``|axis[0]| <= 0.5`` picks the x seed; neither is parallel
    to the chosen axis.
    """
    seed = jnp.where(
        jnp.abs(axis[0]) > 0.5,
        jnp.array([0.0, 1.0, 0.0]),
        jnp.array([1.0, 0.0, 0.0]),
    )
    v = jnp.cross(axis, seed)
    return v / jnp.linalg.norm(v)  # type: ignore[no-any-return]


def _safe_unit(
    v: Float[Array, 3], fallback: Float[Array, 3]
) -> tuple[Float[Array, ""], Float[Array, 3]]:
    """Return (``norm``, ``unit``) where ``norm = ||v||`` when finite.

    The unit vector is ``v / ||v||`` away from the origin and
    ``fallback`` at the origin.  Gradients stay finite at ``v = 0`` by
    running ``sqrt`` on a shifted squared norm.  ``norm`` is returned
    as the true magnitude (zero at the origin) so callers can combine
    it in further distance calculations without re-computing it.
    """
    sq = jnp.sum(v * v)
    is_ok = sq > _EPS_SQ
    safe_sq = jnp.where(is_ok, sq, 1.0)
    safe_norm = jnp.sqrt(safe_sq)
    unit = jnp.where(is_ok, v / safe_norm, fallback)
    norm = jnp.where(is_ok, safe_norm, 0.0)
    return norm, unit


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
    _, direction = _safe_unit(v, jnp.array([0.0, 0.0, 1.0]))
    return center + radius * direction


def foot_on_cylinder(
    query: Float[Array, 3],
    point: Float[Array, 3],
    axis: Float[Array, 3],
    radius: Float[Array, ""],
) -> Float[Array, 3]:
    """Closest point on an infinite cylinder of ``radius`` about the axis.

    On the axis itself the radial direction is ill-defined; a canonical
    unit vector orthogonal to ``axis`` is used to keep the computation
    differentiable.
    """
    v = query - point
    axial_len = jnp.dot(v, axis)
    radial = v - axial_len * axis
    _, direction = _safe_unit(radial, _axis_orthogonal(axis))
    return point + axial_len * axis + radius * direction


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
    radial_norm, radial_dir = _safe_unit(radial, _axis_orthogonal(axis))
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    t = h * cos_a + radial_norm * sin_a
    t = jnp.maximum(t, 0.0)
    return apex + t * (cos_a * axis + sin_a * radial_dir)


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
    (``dq == 0``) the fallback is the radial direction.
    """
    v = query - center
    h = jnp.dot(v, axis)
    radial = v - h * axis
    _, radial_dir = _safe_unit(radial, _axis_orthogonal(axis))
    tube_center = center + major_radius * radial_dir
    dq = query - tube_center
    _, dq_dir = _safe_unit(dq, radial_dir)
    return tube_center + minor_radius * dq_dir


__all__ = [
    "foot_on_cone",
    "foot_on_cylinder",
    "foot_on_plane",
    "foot_on_sphere",
    "foot_on_torus",
]
