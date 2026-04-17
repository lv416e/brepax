"""Differentiable surface area via SDF boundary integral.

Approximates surface area as the integral of a sigmoid-derivative
delta function over a 3D grid.  For a signed distance field *f*,
the surface area is:

    A = integral of delta(f(x)) dx

where delta is approximated by sigma(-f/eps) * (1 - sigma(-f/eps)) / eps
with eps = cell_width (geometric mean of axis spacings).  This is the
derivative of the sigmoid Heaviside used in volume integration, ensuring
consistent sharpness scaling across metrics.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg_eval import make_grid_3d


def integrate_sdf_surface_area(
    sdf: Float[Array, ...],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate SDF values on a grid to compute surface area.

    Uses a sigmoid-derivative delta function with sharpness
    ``1 / cell_width``, matching the convention in
    :func:`~brepax.brep.csg_eval.integrate_sdf_volume`.

    Args:
        sdf: Pre-evaluated SDF values on a cell-centered grid.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Scalar surface area estimate.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Sphere
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> sdf = sphere.sdf(grid)
        >>> area = integrate_sdf_surface_area(sdf, lo, hi, 64)
    """
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    indicator = jax.nn.sigmoid(-sdf / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width
    return jnp.sum(delta) * cell_vol


def surface_area(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute differentiable surface area of a shape defined by its SDF.

    Evaluates the SDF on a cell-centered grid and integrates a
    sigmoid-derivative delta function to approximate the area of the
    zero level-set.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar surface area estimate, differentiable w.r.t. the SDF
        function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Sphere
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> area = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    return integrate_sdf_surface_area(sdf_vals, lo, hi, resolution)


__all__ = [
    "integrate_sdf_surface_area",
    "surface_area",
]
