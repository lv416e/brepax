"""Trim-aware signed distance to a trimmed primitive surface.

Implements the signed blend from ADR-0018:

    d_T(p) = chi_T(pi_S(p)) * d_s(p)
          + (1 - chi_T(pi_S(p))) * d_partial(p)

with ``d_s`` the signed distance to the untrimmed surface and
``d_partial`` the unsigned 3D distance to the trim-boundary polyline.
Dropping ``abs`` from the literal Marschner formula gives a result
that is already correctly signed for the CSG-Stump PMC without any
shell-level winding pass: outside the trim ``chi_T -> 0`` so
``d_T -> d_partial >= 0`` regardless of which side of the mathematical
surface the query is on, and the phantom region outside the trim is
classified as outside the primitive.

This module intentionally stops at the composition itself.  The
per-primitive ``pi_S`` (returning a foot and its UV parameter), the
``d_s`` evaluation, and the trim loops in UV and 3D are the caller's
responsibility — the goal is to keep the composition free of
surface-type dispatch so each primitive wrapper can supply its own
frame extraction without growing this module.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from brepax.brep.polyline import polyline_unsigned_distance
from brepax.nurbs.trim import trim_indicator


def trim_aware_sdf(
    point: Float[Array, 3],
    d_s: Float[Array, ""],
    foot_uv: Float[Array, 2],
    polygon_uv: Float[Array, "n 2"],
    polygon_mask: Float[Array, ...],
    polyline_3d: Float[Array, "n 3"],
    polyline_mask: Float[Array, ...],
    sharpness: float = 200.0,
) -> Float[Array, ""]:
    """Signed distance to a trimmed primitive, per ADR-0018.

    Args:
        point: Query point in 3D, shape ``(3,)``.
        d_s: Signed distance from ``point`` to the untrimmed primitive
            surface (scalar).  Caller evaluates the primitive's own
            ``.sdf()``.
        foot_uv: UV parameter of the foot-of-perpendicular from
            ``point`` to the untrimmed surface, shape ``(2,)``.
        polygon_uv: Trim polygon vertices in the surface's UV
            parameter space, padded to ``n`` slots, shape ``(n, 2)``.
        polygon_mask: Per-vertex validity for ``polygon_uv``, shape
            ``(n,)``.  1.0 for valid vertices, 0.0 for padding.
        polyline_3d: Same trim loop evaluated in 3D as
            ``S(polygon_uv[i])``, shape ``(n, 3)``.
        polyline_mask: Per-vertex validity for ``polyline_3d``, shape
            ``(n,)``.  Typically the same mask as ``polygon_mask``.
        sharpness: Sigmoid sharpness for the trim indicator; higher
            values give a sharper in/out transition.  Default 200.0
            matches ``nurbs/trim.trim_indicator``.

    Returns:
        Signed scalar distance.  Negative strictly inside the trimmed
        primitive, positive outside.
    """
    chi = trim_indicator(foot_uv, polygon_uv, polygon_mask, sharpness=sharpness)
    d_partial = polyline_unsigned_distance(point, polyline_3d, polyline_mask)
    return chi * d_s + (1.0 - chi) * d_partial


__all__ = [
    "trim_aware_sdf",
]
