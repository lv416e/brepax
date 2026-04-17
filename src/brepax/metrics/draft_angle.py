"""Differentiable draft angle violation via SDF surface integration.

Draft angle is the angle between a surface normal and the mold pull
direction.  Surfaces with draft angle below a manufacturing threshold
cause ejection problems.  This module computes the surface area that
violates a minimum draft angle requirement.

The surface normal is estimated from the SDF gradient via central
finite differences (avoiding NaN at degenerate SDF points).  The
violation condition and surface membership both use sigmoid indicators,
consistent with the volume, surface area, and wall thickness metrics.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg_eval import make_grid_3d

_FD_EPS = 1e-4


def _grid_normals(
    sdf: Float[Array, "R R R"],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, "R R R 3"]:
    """Central finite-difference gradient of SDF on a structured grid.

    Avoids NaN from jax.grad at degenerate SDF points (cylinder axis,
    box center) by using numerical differentiation.
    """
    dx = (hi - lo) / resolution
    components = []
    for axis in range(3):
        # Shift the SDF grid by +/- 1 cell in the given axis
        fwd = jnp.roll(sdf, -1, axis=axis)
        bwd = jnp.roll(sdf, 1, axis=axis)
        components.append((fwd - bwd) / (2.0 * dx[axis]))
    return jnp.stack(components, axis=-1)


def integrate_sdf_draft_angle_violation(
    sdf: Float[Array, "R R R"],
    normals: Float[Array, "R R R 3"],
    mold_direction: Float[Array, 3],
    min_angle: Float[Array, ""] | float,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate draft angle violation over the surface.

    For each surface point, the draft angle is the angle between the
    surface normal and the mold pull direction.  Points where the
    draft angle is less than ``min_angle`` contribute to the violation
    surface area.

    Args:
        sdf: Pre-evaluated SDF values with shape ``(R, R, R)``.
        normals: SDF gradient vectors with shape ``(R, R, R, 3)``.
        mold_direction: Unit mold pull direction ``(3,)``.
        min_angle: Minimum acceptable draft angle in radians.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Scalar surface area with insufficient draft angle.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Box
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> sdf = box.sdf(grid)
        >>> normals = _grid_normals(sdf, lo, hi, 64)
        >>> d = jnp.array([0.0, 0.0, 1.0])
        >>> violation = integrate_sdf_draft_angle_violation(
        ...     sdf, normals, d, 0.1, lo, hi, 64,
        ... )
    """
    min_angle = jnp.asarray(min_angle)
    mold_direction = mold_direction / (jnp.linalg.norm(mold_direction) + 1e-10)

    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf / cell_width)
    surface_delta = indicator * (1.0 - indicator) / cell_width

    # Absolute dot product: n and -n represent the same surface
    normal_norm = jnp.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
    unit_normals = normals / normal_norm
    cos_angle = jnp.abs(jnp.sum(unit_normals * mold_direction, axis=-1))

    # draft_angle = arcsin(|n.d|): violation when |n.d| < sin(min_angle)
    sin_threshold = jnp.sin(min_angle)
    violation = jax.nn.sigmoid((sin_threshold - cos_angle) / (cell_width * 0.1))

    return jnp.sum(surface_delta * violation) * cell_vol


def draft_angle_violation(
    sdf_fn: Callable[..., Float[Array, ...]],
    mold_direction: Float[Array, 3],
    min_angle: Float[Array, ""] | float,
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Surface area with draft angle below the manufacturing threshold.

    Draft angle is the angle between the surface tangent plane and the
    mold pull direction.  A surface normal perpendicular to the pull
    direction has zero draft angle (worst case for ejection).  A normal
    parallel to the pull direction has 90 degrees of draft (ideal).

    This function computes the surface area where the draft angle is
    less than ``min_angle``, providing a differentiable manufacturing
    constraint.  Minimize ``draft_angle_violation(sdf, d, min_angle)``
    to ensure all surfaces have sufficient draft for mold ejection.

    Assumes ``sdf_fn`` returns a proper signed distance field
    (``||grad(f)|| = 1``).

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        mold_direction: Unit mold pull direction ``(3,)``.
        min_angle: Minimum acceptable draft angle in radians.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar surface area with insufficient draft angle,
        differentiable w.r.t. shape parameters and ``mold_direction``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> d = jnp.array([0.0, 0.0, 1.0])
        >>> violation = draft_angle_violation(
        ...     box.sdf, d, 0.1, lo=lo, hi=hi, resolution=64,
        ... )
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    normals = _grid_normals(sdf_vals, lo, hi, resolution)
    return integrate_sdf_draft_angle_violation(
        sdf_vals, normals, mold_direction, min_angle, lo, hi, resolution
    )


__all__ = [
    "draft_angle_violation",
    "integrate_sdf_draft_angle_violation",
]
