"""Differentiable center of mass and moment of inertia via SDF integration.

Both metrics extend the sigmoid indicator framework used by volume and
surface area: the indicator ``sigma(-f/eps)`` weights each grid point
by its interior membership, and position-dependent terms produce the
mass-weighted integrals for center of mass and inertia tensor.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg_eval import make_grid_3d


def integrate_sdf_center_of_mass(
    sdf: Float[Array, ...],
    grid: Float[Array, "R R R 3"],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, 3]:
    """Integrate SDF values on a grid to compute center of mass.

    Args:
        sdf: Pre-evaluated SDF values, shape ``(R, R, R)``.
        grid: Grid point coordinates, shape ``(R, R, R, 3)``.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Center of mass, shape ``(3,)``.
    """
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    indicator = jax.nn.sigmoid(-sdf / cell_width)
    volume = jnp.sum(indicator) * cell_vol
    weighted_pos = jnp.sum(grid * indicator[..., None], axis=(0, 1, 2)) * cell_vol
    return weighted_pos / (volume + 1e-20)


def center_of_mass(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, 3]:
    """Compute differentiable center of mass of a shape.

    Assumes unit density.  The center of mass is the volume-weighted
    average position of interior points.

    Args:
        sdf_fn: Signed distance function.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis.

    Returns:
        Center of mass, shape ``(3,)``, differentiable w.r.t. the
        SDF function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Sphere
        >>> s = Sphere(center=jnp.array([1., 2., 3.]), radius=jnp.array(1.))
        >>> com = center_of_mass(s.sdf, lo=jnp.zeros(3), hi=jnp.ones(3)*6)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    return integrate_sdf_center_of_mass(sdf_vals, grid, lo, hi, resolution)


def integrate_sdf_moment_of_inertia(
    sdf: Float[Array, ...],
    grid: Float[Array, "R R R 3"],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
    com: Float[Array, 3] | None = None,
) -> Float[Array, "3 3"]:
    """Integrate SDF values on a grid to compute moment of inertia tensor.

    Computes the inertia tensor about the center of mass (or a given
    point) assuming unit density.

    Args:
        sdf: Pre-evaluated SDF values, shape ``(R, R, R)``.
        grid: Grid point coordinates, shape ``(R, R, R, 3)``.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.
        com: Center of mass to compute about.  If None, computed
            from the SDF.

    Returns:
        Inertia tensor, shape ``(3, 3)``.
    """
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    indicator = jax.nn.sigmoid(-sdf / cell_width)

    if com is None:
        volume = jnp.sum(indicator) * cell_vol
        weighted_pos = jnp.sum(grid * indicator[..., None], axis=(0, 1, 2)) * cell_vol
        com = weighted_pos / (volume + 1e-20)

    # Position relative to center of mass
    r = grid - com
    # I_ij = integral of (||r||^2 delta_ij - r_i r_j) * indicator dV
    r_sq = jnp.sum(r**2, axis=-1)
    # Diagonal: I_ii = integral of (r_j^2 + r_k^2) * indicator
    # Off-diagonal: I_ij = -integral of r_i * r_j * indicator
    inertia = jnp.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                integrand = (r_sq - r[..., i] ** 2) * indicator
            else:
                integrand = -r[..., i] * r[..., j] * indicator
            inertia = inertia.at[i, j].set(jnp.sum(integrand) * cell_vol)

    return inertia


def moment_of_inertia(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, "3 3"]:
    """Compute differentiable moment of inertia tensor of a shape.

    Returns the 3x3 inertia tensor about the center of mass,
    assuming unit density.

    Args:
        sdf_fn: Signed distance function.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis.

    Returns:
        Inertia tensor, shape ``(3, 3)``, differentiable w.r.t.
        the SDF function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> b = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> I = moment_of_inertia(b.sdf, lo=-jnp.ones(3)*3, hi=jnp.ones(3)*3)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    return integrate_sdf_moment_of_inertia(sdf_vals, grid, lo, hi, resolution)


__all__ = [
    "center_of_mass",
    "integrate_sdf_center_of_mass",
    "integrate_sdf_moment_of_inertia",
    "moment_of_inertia",
]
