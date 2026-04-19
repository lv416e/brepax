"""Grid-based Poisson solver on SDF-defined domains.

Solves -div(alpha * grad u) = f on {x : sdf(x) < 0} with u = 0
on the boundary via conjugate gradient with ersatz material
interpolation.  The full solve is differentiable w.r.t. geometry
parameters via JAX AD through ``custom_linear_solve``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def solve_poisson_2d(
    sdf: Float[Array, "N N"],
    h: float,
    *,
    source: float = 1.0,
    alpha_min: float = 1e-3,
    eps: float | None = None,
    cg_maxiter: int = 1000,
) -> Float[Array, "N N"]:
    """Solve -div(alpha * grad u) = source on the interior of an SDF domain.

    Uses the ersatz material approach: conductivity ``alpha`` equals
    1 inside the domain and ``alpha_min`` outside, interpolated by
    the sigmoid domain indicator.  A penalty term suppresses u
    outside the domain.

    Args:
        sdf: Signed distance field on a 2D grid, shape ``(N, N)``.
            Negative inside the domain.
        h: Grid spacing.
        source: Constant source term (default 1.0).
        alpha_min: Minimum conductivity outside domain.
        eps: Sigmoid sharpness (default ``h``, the grid spacing).
        cg_maxiter: Maximum CG iterations.

    Returns:
        Solution field u, shape ``(N, N)``.
    """
    n = sdf.shape[0]
    if eps is None:
        eps = h
    indicator = jax.nn.sigmoid(-sdf / eps)
    alpha = alpha_min + (1.0 - alpha_min) * indicator
    # Penalty drives u to zero outside the domain
    exterior_penalty = (1.0 / h**2) * (1.0 - indicator)

    alpha_pad = jnp.pad(alpha, 1, constant_values=alpha_min)
    # Face-averaged conductivities are constant during the solve
    a_r = (alpha + alpha_pad[1:-1, 2:]) / 2.0
    a_l = (alpha + alpha_pad[1:-1, :-2]) / 2.0
    a_u = (alpha + alpha_pad[2:, 1:-1]) / 2.0
    a_d = (alpha + alpha_pad[:-2, 1:-1]) / 2.0

    def _matvec(u_flat: Float[Array, " m"]) -> Float[Array, " m"]:
        u = u_flat.reshape(n, n)
        u_pad = jnp.pad(u, 1)
        # -div(alpha * grad u) + penalty * (1-H) * u
        diffusion = (
            -(
                a_r * (u_pad[1:-1, 2:] - u)
                + a_l * (u_pad[1:-1, :-2] - u)
                + a_u * (u_pad[2:, 1:-1] - u)
                + a_d * (u_pad[:-2, 1:-1] - u)
            )
            / h**2
        )
        return (diffusion + exterior_penalty * u).ravel()

    rhs = (indicator * source).ravel()
    u_flat, _ = jax.scipy.sparse.linalg.cg(_matvec, rhs, maxiter=cg_maxiter)  # type: ignore[no-untyped-call]
    result: Float[Array, "N N"] = u_flat.reshape(n, n)
    return result


def average_field(
    u: Float[Array, "N N"],
    sdf: Float[Array, "N N"],
    h: float,
    eps: float | None = None,
) -> Float[Array, ""]:
    """Volume-averaged field value inside the SDF domain."""
    if eps is None:
        eps = h
    indicator = jax.nn.sigmoid(-sdf / eps)
    return jnp.sum(u * indicator) / (jnp.sum(indicator) + 1e-10)


def annulus_sdf(
    grid: Float[Array, "N N 2"],
    r_inner: Float[Array, ""],
    r_outer: float,
) -> Float[Array, "N N"]:
    """SDF for an annular domain (ring)."""
    r = jnp.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2 + 1e-20)
    return jnp.maximum(r - r_outer, r_inner - r)


def annulus_analytical(
    grid: Float[Array, "N N 2"],
    r_inner: float,
    r_outer: float,
) -> Float[Array, "N N"]:
    """Analytical solution for -nabla^2 u = 1 on an annulus, u=0 on boundaries."""
    r = jnp.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2 + 1e-20)
    c = (r_outer**2 - r_inner**2) / (4.0 * jnp.log(r_outer / r_inner))
    u = (r_outer**2 - r**2) / 4.0 + c * jnp.log(r / r_outer)
    sdf = jnp.maximum(r - r_outer, r_inner - r)
    return jnp.where(sdf < 0, u, 0.0)


__all__ = [
    "annulus_analytical",
    "annulus_sdf",
    "average_field",
    "solve_poisson_2d",
]
