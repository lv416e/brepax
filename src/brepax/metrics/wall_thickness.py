"""Differentiable wall thickness metrics via SDF grid integration.

For a proper signed distance field, the absolute value at any interior
point equals the distance to the nearest surface boundary.  This module
provides two complementary metrics built on that property:

- :func:`thin_wall_volume` counts the volume of material closer to a
  surface than a given threshold -- directly useful as a DFM constraint
  (minimize to enforce minimum wall thickness).
- :func:`min_wall_thickness` estimates the minimum wall thickness as a
  differentiable scalar via soft-minimum over interior SDF values.

Both integrate the same sigmoid framework used by volume and surface area
metrics, with sharpness ``1 / cell_width``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg_eval import make_grid_3d


def integrate_sdf_thin_wall_volume(
    sdf: Float[Array, ...],
    threshold: Float[Array, ""] | float,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate SDF values to compute volume of thin-wall material.

    Selects interior points whose distance to the nearest surface is
    less than ``threshold``.  Both the interior membership and the
    distance test use sigmoid indicators with sharpness ``1 / cell_width``.

    This assumes the input is a proper signed distance field where
    ``||grad(f)|| = 1``.

    Args:
        sdf: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        threshold: Distance threshold.  Points inside the shape
            and within this distance of the boundary contribute.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Scalar volume of material thinner than ``threshold``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Box
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> sdf = box.sdf(grid)
        >>> vol = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
    """
    threshold = jnp.asarray(threshold)
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    # Interior membership: sigmoid(d/eps) where d = -sdf
    inside = jax.nn.sigmoid(-sdf / cell_width)
    # Thin-wall membership: points within threshold of boundary
    # sdf + threshold > 0 means |sdf| < threshold for interior points
    thin = jax.nn.sigmoid((sdf + threshold) / cell_width)
    return jnp.sum(inside * thin) * cell_vol


def thin_wall_volume(
    sdf_fn: Callable[..., Float[Array, ...]],
    threshold: Float[Array, ""] | float,
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Volume of material with wall thickness below a given threshold.

    For each interior point, the SDF value gives the distance to the
    nearest surface.  This function counts the volume of material
    where that distance is less than ``threshold``, i.e. the material
    that would violate a minimum wall thickness requirement.

    Useful as a manufacturing constraint: minimize
    ``thin_wall_volume(sdf, 1.5)`` to ensure no wall is thinner
    than 1.5 units.  Assumes ``sdf_fn`` returns a proper signed
    distance field (``||grad(f)|| = 1``).

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        threshold: Minimum acceptable wall thickness.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar volume of thin-wall material, differentiable w.r.t.
        both the SDF function's parameters and ``threshold``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> vol = thin_wall_volume(box.sdf, 0.5, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    return integrate_sdf_thin_wall_volume(sdf_vals, threshold, lo, hi, resolution)


def min_wall_thickness(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Differentiable estimate of minimum wall thickness.

    For a convex shape, the minimum wall thickness equals twice the
    maximum inscribed distance (the SDF value at the deepest interior
    point).  This function returns a differentiable approximation via
    soft-max (log-sum-exp) over interior SDF values.

    For shapes with varying wall thickness (e.g. a box with holes
    near an edge), this returns a global estimate that may not reflect
    the thinnest local section.  Use :func:`thin_wall_volume` with an
    explicit threshold for manufacturing constraint enforcement.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).
        temperature: Soft-max temperature; lower values approximate
            the true maximum more closely but may have sharper gradients.

    Returns:
        Scalar estimate of minimum wall thickness (twice the maximum
        inscribed distance), differentiable w.r.t. the SDF function's
        parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(
        ...     center=jnp.zeros(3),
        ...     half_extents=jnp.array([2.0, 1.5, 1.0]),
        ... )
        >>> lo, hi = jnp.array([-4.0]*3), jnp.array([4.0]*3)
        >>> thickness = min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    # Interior distance to nearest surface (positive inside, 0 outside)
    interior_dist = jnp.clip(-sdf_vals, 0.0, None)

    # Weight by interior membership to suppress exterior contributions
    weight = jax.nn.sigmoid(-sdf_vals / cell_width)

    # Soft-max of interior distances: temperature * logsumexp(dist/T, b=w)
    flat_dist = interior_dist.ravel()
    flat_weight = weight.ravel()
    max_dist = temperature * jax.nn.logsumexp(flat_dist / temperature, b=flat_weight)
    return 2.0 * max_dist


__all__ = [
    "integrate_sdf_thin_wall_volume",
    "min_wall_thickness",
    "thin_wall_volume",
]
