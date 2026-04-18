"""Differentiable curvature field metrics via SDF Laplacian integration.

For a proper signed distance field where ``||grad(f)|| = 1``, the mean
curvature at the zero level-set equals the Laplacian of the SDF:

    kappa = div(grad(f) / ||grad(f)||) = laplacian(f)

This equals the sum of principal curvatures (kappa_1 + kappa_2).  This
module provides two metrics built on this property:

- :func:`mean_curvature` computes the surface-area-weighted average of
  the Laplacian, useful as a shape descriptor.
- :func:`max_curvature` estimates the maximum curvature over the surface
  via soft-max, useful as a DFM constraint.

Both use the same sigmoid delta function as :func:`surface_area` for
surface membership weighting.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg_eval import make_grid_3d


def _ad_laplacian(
    sdf_fn: Callable[..., Float[Array, ...]],
    grid: Float[Array, "R R R 3"],
) -> Float[Array, "R R R"]:
    """Laplacian of SDF via AD Hessian: trace(H) at each grid point.

    Uses forward-over-reverse AD (``jax.jacfwd(jax.grad(f))``) for
    efficient computation of the 3x3 Hessian of a scalar-valued SDF.
    """
    spatial_shape = grid.shape[:-1]
    flat = grid.reshape(-1, 3)

    def _single_laplacian(x: Float[Array, 3]) -> Float[Array, ""]:
        hessian = jax.jacfwd(jax.grad(sdf_fn))(x)
        return jnp.trace(hessian)

    lap_flat = jax.vmap(_single_laplacian)(flat)
    # SDF singularities (e.g. sphere center) produce non-finite Hessian values;
    # these occur far from the surface and are irrelevant to curvature integration.
    lap_flat = jnp.where(jnp.isfinite(lap_flat), lap_flat, 0.0)
    return lap_flat.reshape(spatial_shape)


def integrate_sdf_mean_curvature(
    sdf: Float[Array, ...],
    laplacian: Float[Array, ...],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate SDF values on a grid to compute surface-weighted mean curvature.

    Computes the surface-area-weighted average of the Laplacian:

        integral(kappa * delta(f) dV) / integral(delta(f) dV)

    where delta is the sigmoid-derivative surface delta function.

    For a sphere of radius R, the result approaches 2/R (sum of principal
    curvatures 1/R + 1/R).  For a plane, it approaches 0.

    Args:
        sdf: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        laplacian: Pre-computed Laplacian values on the same grid,
            with shape ``(R, R, R)``.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Scalar surface-area-weighted mean curvature.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Sphere
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> sdf = sphere.sdf(grid)
        >>> from brepax.metrics.curvature import _ad_laplacian
        >>> lap = _ad_laplacian(sphere.sdf, grid)
        >>> kappa = integrate_sdf_mean_curvature(sdf, lap, lo, hi, 64)
    """
    dx = (hi - lo) / resolution
    cell_vol = jnp.prod(dx)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width

    delta_sum = jnp.sum(delta) * cell_vol
    weighted_curvature = jnp.sum(laplacian * delta) * cell_vol

    return weighted_curvature / (delta_sum + 1e-20)


def mean_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute differentiable surface-weighted mean curvature.

    For a proper SDF, the Laplacian at the zero level-set equals the
    sum of principal curvatures.  This function returns the
    surface-area-weighted average of the Laplacian, providing a
    differentiable shape descriptor.

    For a sphere of radius R, returns approximately 2/R.
    For a plane, returns approximately 0.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar surface-area-weighted mean curvature, differentiable
        w.r.t. the SDF function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Sphere
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> kappa = mean_curvature(sphere.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    laplacian = _ad_laplacian(sdf_fn, grid)
    return integrate_sdf_mean_curvature(sdf_vals, laplacian, lo, hi, resolution)


def integrate_sdf_max_curvature(
    sdf: Float[Array, ...],
    laplacian: Float[Array, ...],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Integrate SDF values on a grid to estimate maximum surface curvature.

    Uses a normalized soft-max (log-mean-exp) of the absolute Laplacian
    weighted by the surface delta function.  The normalization ensures
    the estimate is invariant to grid resolution.

    Args:
        sdf: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        laplacian: Pre-computed Laplacian values on the same grid,
            with shape ``(R, R, R)``.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.
        temperature: Soft-max temperature; lower values approximate
            the true maximum more closely.

    Returns:
        Scalar estimate of maximum curvature over the surface.
    """
    dx = (hi - lo) / resolution
    cell_vol = jnp.prod(dx)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width

    abs_curvature = jnp.abs(laplacian)

    # Restrict to within one cell width of the surface; this suppresses
    # off-surface Laplacian singularities (e.g. 2/||x|| at sphere center)
    # while retaining differentiability through the sigmoid boundary.
    near_surface = jnp.abs(sdf) < cell_width
    safe_curv = jnp.where(near_surface, abs_curvature, 0.0)

    flat_curv = safe_curv.ravel()
    flat_delta = delta.ravel()

    # Softmax weighted by curvature magnitude and surface proximity
    log_w = flat_curv / temperature + jnp.log(flat_delta + 1e-30)
    softmax_w = jax.nn.softmax(log_w)
    return jnp.sum(flat_curv * softmax_w)


def max_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Differentiable estimate of maximum surface curvature.

    Estimates the maximum of the absolute Laplacian over the surface
    via normalized soft-max weighted by the surface delta function.
    Useful as a DFM constraint: "maximum curvature should not exceed
    threshold".

    For a sphere of radius R, returns approximately 2/R.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).
        temperature: Soft-max temperature; lower values approximate
            the true maximum more closely but may have sharper gradients.

    Returns:
        Scalar estimate of maximum surface curvature, differentiable
        w.r.t. the SDF function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Sphere
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> kappa_max = max_curvature(sphere.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    laplacian = _ad_laplacian(sdf_fn, grid)
    return integrate_sdf_max_curvature(
        sdf_vals, laplacian, lo, hi, resolution, temperature
    )


__all__ = [
    "integrate_sdf_max_curvature",
    "integrate_sdf_mean_curvature",
    "max_curvature",
    "mean_curvature",
]
