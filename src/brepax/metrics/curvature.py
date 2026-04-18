"""Differentiable curvature metrics via AD Hessian at SDF zero-crossings.

For a proper signed distance field where ``||grad(f)|| = 1``, the mean
curvature at the zero level-set equals the Laplacian of the SDF:

    kappa = div(grad(f) / ||grad(f)||) = laplacian(f)

This equals the sum of principal curvatures (kappa_1 + kappa_2).  This
module provides two metrics built on this property:

- :func:`mean_curvature` computes the average of the Laplacian over
  detected surface points, useful as a shape descriptor.
- :func:`max_curvature` estimates the maximum curvature over the surface
  via soft-max, useful as a DFM constraint.

Surface points are found by zero-crossing detection on a grid: for each
pair of adjacent grid points where the SDF changes sign, linear
interpolation finds the approximate crossing location. The AD Hessian
is then evaluated only at these surface points rather than the full grid.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from brepax.brep.csg_eval import make_grid_3d


def _find_surface_points(
    sdf_vals: Float[Array, "R R R"],
    grid: Float[Array, "R R R 3"],
) -> tuple[Float[Array, "N 3"], Bool[Array, " N"]]:
    """Find approximate surface points via zero-crossing detection.

    For each of the three spatial axes, compares adjacent grid points
    and identifies sign changes in the SDF. At each sign change, linear
    interpolation estimates the zero-crossing location.

    The output has a fixed size of ``3 * (R-1) * R * R`` for JIT
    compatibility; a boolean mask indicates which entries are valid
    surface points.

    Args:
        sdf_vals: Pre-evaluated SDF values with shape ``(R, R, R)``.
        grid: Grid coordinates with shape ``(R, R, R, 3)``.

    Returns:
        Tuple of ``(points, mask)`` where ``points`` has shape
        ``(N, 3)`` and ``mask`` has shape ``(N,)``.
    """
    all_points = []
    all_masks = []

    for axis in range(3):
        sdf_lo = jnp.take(
            sdf_vals, indices=jnp.arange(sdf_vals.shape[axis] - 1), axis=axis
        )
        sdf_hi = jnp.take(
            sdf_vals, indices=jnp.arange(1, sdf_vals.shape[axis]), axis=axis
        )

        sign_change = (sdf_lo * sdf_hi) < 0

        t = sdf_lo / (sdf_lo - sdf_hi + 1e-20)
        t = jnp.clip(t, 0.0, 1.0)

        grid_lo = jnp.take(grid, indices=jnp.arange(grid.shape[axis] - 1), axis=axis)
        grid_hi = jnp.take(grid, indices=jnp.arange(1, grid.shape[axis]), axis=axis)
        interp_pts = grid_lo * (1 - t[..., None]) + grid_hi * t[..., None]

        flat_pts = interp_pts.reshape(-1, 3)
        flat_mask = sign_change.ravel()
        all_points.append(flat_pts)
        all_masks.append(flat_mask)

    points = jnp.concatenate(all_points, axis=0)
    mask = jnp.concatenate(all_masks, axis=0)
    return points, mask


def _evaluate_curvature_at_points(
    sdf_fn: Callable[..., Float[Array, ...]],
    points: Float[Array, "N 3"],
    mask: Bool[Array, " N"],
) -> Float[Array, " N"]:
    """Evaluate Laplacian (sum of principal curvatures) at surface points.

    Computes ``trace(jacfwd(grad(sdf_fn))(x))`` at each point via vmap.
    Non-finite values and masked-out points are replaced with zero.

    Args:
        sdf_fn: Signed distance function ``(3,) -> ()``.
        points: Candidate surface points of shape ``(N, 3)``.
        mask: Boolean mask indicating valid surface points.

    Returns:
        Curvature values of shape ``(N,)``, zeroed for invalid entries.
    """

    def _single_laplacian(x: Float[Array, " 3"]) -> Float[Array, ""]:
        hessian = jax.jacfwd(jax.grad(sdf_fn))(x)
        return jnp.trace(hessian)

    all_curvatures = jax.vmap(_single_laplacian)(points)
    safe = jnp.where(mask & jnp.isfinite(all_curvatures), all_curvatures, 0.0)
    return safe


def integrate_sdf_mean_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    sdf_vals: Float[Array, "R R R"],
    grid: Float[Array, "R R R 3"],
) -> Float[Array, ""]:
    """Compute mean curvature via zero-crossing surface points.

    Detects surface points where the SDF changes sign between adjacent
    grid cells, then evaluates the AD Hessian Laplacian at each surface
    point.  Returns the average curvature over all detected surface
    points.

    For a sphere of radius R, the result approaches 2/R (sum of principal
    curvatures 1/R + 1/R).  For a plane, it approaches 0.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(3,)`` and returning a scalar SDF value.
        sdf_vals: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        grid: Grid coordinates with shape ``(R, R, R, 3)``.

    Returns:
        Scalar mean curvature over surface points.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Sphere
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> sdf = sphere.sdf(grid)
        >>> kappa = integrate_sdf_mean_curvature(sphere.sdf, sdf, grid)
    """
    points, mask = _find_surface_points(sdf_vals, grid)
    curvatures = _evaluate_curvature_at_points(sdf_fn, points, mask)
    n_valid = jnp.sum(mask)
    return jnp.sum(curvatures) / (n_valid + 1e-10)


def mean_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute differentiable mean curvature via zero-crossing detection.

    Finds surface points where the SDF changes sign between adjacent
    grid cells, evaluates the AD Hessian Laplacian at each point, and
    returns the average.

    For a sphere of radius R, returns approximately 2/R.
    For a plane, returns approximately 0.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar mean curvature, differentiable w.r.t. the SDF
        function's parameters.

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
    return integrate_sdf_mean_curvature(sdf_fn, sdf_vals, grid)


def integrate_sdf_max_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    sdf_vals: Float[Array, "R R R"],
    grid: Float[Array, "R R R 3"],
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Estimate maximum surface curvature via zero-crossing surface points.

    Detects surface points where the SDF changes sign, evaluates the AD
    Hessian Laplacian at each, and returns a soft-max estimate of the
    maximum absolute curvature.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(3,)`` and returning a scalar SDF value.
        sdf_vals: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        grid: Grid coordinates with shape ``(R, R, R, 3)``.
        temperature: Soft-max temperature; lower values approximate
            the true maximum more closely.

    Returns:
        Scalar estimate of maximum curvature over the surface.
    """
    points, mask = _find_surface_points(sdf_vals, grid)
    curvatures = _evaluate_curvature_at_points(sdf_fn, points, mask)
    abs_curv = jnp.abs(curvatures)

    log_w = abs_curv / temperature + jnp.log(mask.astype(float) + 1e-30)
    softmax_w = jax.nn.softmax(log_w)
    return jnp.sum(abs_curv * softmax_w)


def max_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Differentiable estimate of maximum surface curvature.

    Finds surface points via zero-crossing detection, evaluates the AD
    Hessian Laplacian, and returns a soft-max estimate of the maximum
    absolute curvature.

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
    return integrate_sdf_max_curvature(sdf_fn, sdf_vals, grid, temperature)


__all__ = [
    "integrate_sdf_max_curvature",
    "integrate_sdf_mean_curvature",
    "max_curvature",
    "mean_curvature",
]
