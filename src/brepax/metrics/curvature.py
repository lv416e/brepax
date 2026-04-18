"""Differentiable curvature metrics via sigmoid framework + Newton refinement.

For a proper signed distance field where ``||grad(f)|| = 1``, the mean
curvature at the zero level-set equals the Laplacian of the SDF:

    kappa = div(grad(f) / ||grad(f)||) = laplacian(f)

This equals the sum of principal curvatures (kappa_1 + kappa_2).  This
module provides two metrics built on this property:

- :func:`mean_curvature` computes the delta-weighted average of the
  Laplacian over the surface, useful as a shape descriptor.
- :func:`max_curvature` estimates the maximum curvature over the surface
  via soft-max, useful as a DFM constraint.

Surface contributions are weighted by a sigmoid-derivative delta function
(consistent with :func:`~brepax.metrics.surface_area.surface_area`).
Grid points are Newton-refined toward the SDF=0 surface before evaluating
the AD Hessian, improving accuracy without requiring zero-crossing
detection or boolean masks.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg_eval import make_grid_3d


def _newton_refine(
    sdf_fn: Callable[..., Float[Array, ...]],
    points: Float[Array, "N 3"],
    n_steps: int = 2,
) -> Float[Array, "N 3"]:
    """Project points toward the SDF=0 surface via Newton steps.

    Each step moves a point along the SDF gradient direction to reduce
    the residual SDF value.  Points far from the surface will not
    converge in a few steps, but their sigmoid delta weight is
    negligible so they do not affect the result.

    Args:
        sdf_fn: Signed distance function ``(3,) -> ()``.
        points: Initial points of shape ``(N, 3)``.
        n_steps: Number of Newton iterations (default 2).

    Returns:
        Refined points of shape ``(N, 3)``.
    """

    def _step(x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        f = sdf_fn(x)
        g = jax.grad(sdf_fn)(x)
        g_norm_sq = jnp.sum(g**2) + 1e-10
        refined: Float[Array, " 3"] = x - f * g / g_norm_sq
        return refined

    step_vmapped = jax.vmap(_step)
    result = points
    for _ in range(n_steps):
        result = step_vmapped(result)
    return result


def mean_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute differentiable mean curvature via sigmoid-weighted Laplacian.

    Evaluates the SDF on a cell-centered grid, computes a sigmoid-derivative
    delta to identify the surface, Newton-refines grid points toward the
    zero level-set, and returns the delta-weighted average of the AD Hessian
    Laplacian at the refined points.

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

    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width

    flat_grid = grid.reshape(-1, 3)
    refined = _newton_refine(sdf_fn, flat_grid, n_steps=2)

    curvatures = jax.vmap(lambda x: jnp.trace(jax.jacfwd(jax.grad(sdf_fn))(x)))(refined)
    curvatures = jnp.where(jnp.isfinite(curvatures), curvatures, 0.0)
    curvatures = curvatures.reshape(grid.shape[:-1])

    delta_sum = jnp.sum(delta) * cell_vol
    weighted = jnp.sum(curvatures * delta) * cell_vol
    return weighted / (delta_sum + 1e-20)


def max_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Differentiable estimate of maximum surface curvature.

    Uses the same sigmoid delta framework as :func:`mean_curvature` but
    returns a soft-max estimate of the maximum absolute curvature over
    the surface, weighted by the delta function.

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

    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width

    flat_grid = grid.reshape(-1, 3)
    refined = _newton_refine(sdf_fn, flat_grid, n_steps=2)

    curvatures = jax.vmap(lambda x: jnp.trace(jax.jacfwd(jax.grad(sdf_fn))(x)))(refined)
    curvatures = jnp.where(jnp.isfinite(curvatures), curvatures, 0.0)
    curvatures = curvatures.reshape(grid.shape[:-1])

    abs_curv = jnp.abs(curvatures)
    flat_delta = delta.ravel()
    flat_abs_curv = abs_curv.ravel()

    # Delta-weighted soft-max: use log(delta) to focus on surface points
    log_w = flat_abs_curv / temperature + jnp.log(flat_delta + 1e-30)
    softmax_w = jax.nn.softmax(log_w)
    return jnp.sum(flat_abs_curv * softmax_w)


__all__ = [
    "max_curvature",
    "mean_curvature",
]
