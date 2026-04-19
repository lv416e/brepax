"""PINN solver for Poisson equation on SDF-defined domains.

Solves -nabla^2 u = f on domains defined by a signed distance
field with u = 0 on the boundary.  Uses soft boundary penalty
instead of hard BC (u = -sdf * nn) because the hard BC product
rule introduces a Laplacian(SDF) singularity (1/r for circular
boundaries).

Includes helpers for disk and annulus domains with analytical
solutions for verification.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class DiskPoissonPINN(eqx.Module):
    """PINN for Poisson on the unit disk with soft BC.

    The network directly outputs u(x,y).  Boundary conditions are
    enforced via a penalty term in the loss, not via SDF multiplication.

    Attributes:
        net: MLP mapping ``(2,) -> (1,)``.
    """

    net: eqx.nn.MLP

    def __init__(self, *, width: int = 32, depth: int = 3, key: PRNGKeyArray) -> None:
        self.net = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, ...]:
        """Evaluate u(x) directly from the network."""
        flat = x.reshape(-1, 2)
        out = jax.vmap(lambda xi: self.net(xi).squeeze())(flat)
        return out.reshape(x.shape[:-1])


def disk_sdf(pts: Float[Array, "... 2"]) -> Float[Array, ...]:
    """SDF for the unit disk: negative inside, positive outside."""
    r = jnp.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2)
    return r - 1.0


def disk_analytical(pts: Float[Array, "... 2"]) -> Float[Array, ...]:
    """Analytical solution for -nabla^2 u = 1, u|_bdy = 0 on unit disk."""
    r_sq = pts[..., 0] ** 2 + pts[..., 1] ** 2
    return (1.0 - r_sq) / 4.0


def _laplacian_at_point(
    net_fn: Callable[[Float[Array, 2]], Float[Array, ""]],
    xi: Float[Array, 2],
) -> Float[Array, ""]:
    """Trace of the Hessian of net_fn at a single point."""
    hess = jax.hessian(net_fn)(xi)
    result: Float[Array, ""] = hess[0, 0] + hess[1, 1]
    return result


def disk_pinn_loss(
    model: DiskPoissonPINN,
    interior_pts: Float[Array, "n 2"],
    boundary_pts: Float[Array, "m 2"],
    *,
    source: float = 1.0,
    bc_weight: float = 100.0,
) -> Float[Array, ""]:
    """Combined PDE residual + boundary penalty loss.

    PDE loss: mean((-nabla^2 u - f)^2) over interior points.
    BC loss: bc_weight * mean(u(boundary)^2).

    Args:
        model: The PINN model.
        interior_pts: Collocation points inside the domain, shape ``(n, 2)``.
        boundary_pts: Points on the boundary, shape ``(m, 2)``.
        source: Source term value (f in -nabla^2 u = f).
        bc_weight: Penalty weight for boundary condition violation.

    Returns:
        Scalar loss value.
    """

    def u_scalar(xi: Float[Array, 2]) -> Float[Array, ""]:
        return model.net(xi).squeeze()

    def residual(xi: Float[Array, 2]) -> Float[Array, ""]:
        lap = _laplacian_at_point(u_scalar, xi)
        return lap + source

    # PDE residual on interior
    pde_residuals = jax.vmap(residual)(interior_pts)
    pde_loss = jnp.mean(pde_residuals**2)

    # BC penalty on boundary
    u_bdy = model(boundary_pts)
    bc_loss = jnp.mean(u_bdy**2)

    return pde_loss + bc_weight * bc_loss


def sample_disk_interior(
    n_points: int,
    key: PRNGKeyArray,
) -> Float[Array, "n 2"]:
    """Sample uniform points inside the unit disk.

    Uses rejection sampling from the bounding box [-1, 1]^2.
    Oversamples by 2x to guarantee enough interior points
    (disk area / box area = pi/4 ~ 0.785).
    """
    candidates = jax.random.uniform(key, (n_points * 2, 2), minval=-1.0, maxval=1.0)
    r_sq = candidates[:, 0] ** 2 + candidates[:, 1] ** 2
    mask = r_sq < 1.0
    indices = jnp.where(mask, size=n_points, fill_value=0)[0]
    return candidates[indices]


def sample_disk_boundary(
    n_points: int,
    key: PRNGKeyArray,
) -> Float[Array, "n 2"]:
    """Sample points uniformly on the unit circle boundary."""
    theta = jax.random.uniform(key, (n_points,), minval=0.0, maxval=2.0 * jnp.pi)
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)


def train_disk_pinn(
    model: DiskPoissonPINN,
    interior_pts: Float[Array, "n 2"],
    boundary_pts: Float[Array, "m 2"],
    *,
    source: float = 1.0,
    bc_weight: float = 100.0,
    n_steps: int = 5000,
    lr: float = 1e-3,
) -> DiskPoissonPINN:
    """Train the PINN via Adam.

    Args:
        model: Initial PINN model.
        interior_pts: Interior collocation points, shape ``(n, 2)``.
        boundary_pts: Boundary collocation points, shape ``(m, 2)``.
        source: Source term for -nabla^2 u = source.
        bc_weight: Penalty weight for boundary conditions.
        n_steps: Number of optimizer steps.
        lr: Learning rate.

    Returns:
        Trained model.
    """
    params = eqx.filter(model, eqx.is_array)
    m_state = jax.tree.map(jnp.zeros_like, params)
    v_state = jax.tree.map(jnp.zeros_like, params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    @eqx.filter_jit
    def _step(
        mdl: DiskPoissonPINN,
        m_s: DiskPoissonPINN,
        v_s: DiskPoissonPINN,
        t: Float[Array, ""],
    ) -> tuple[DiskPoissonPINN, DiskPoissonPINN, DiskPoissonPINN, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(disk_pinn_loss)(
            mdl, interior_pts, boundary_pts, source=source, bc_weight=bc_weight
        )
        m_new = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, m_s, grads)
        v_new = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g**2, v_s, grads)
        t_f = t + 1.0
        m_hat = jax.tree.map(lambda m: m / (1 - beta1**t_f), m_new)
        v_hat = jax.tree.map(lambda v: v / (1 - beta2**t_f), v_new)
        updates = jax.tree.map(
            lambda m, v: -lr * m / (jnp.sqrt(v) + eps_adam), m_hat, v_hat
        )
        mdl = eqx.apply_updates(mdl, updates)
        return mdl, m_new, v_new, loss

    t = jnp.array(0.0)
    for _ in range(n_steps):
        model, m_state, v_state, _ = _step(model, m_state, v_state, t)
        t = t + 1.0

    return model


def evaluate_disk_pinn(
    model: DiskPoissonPINN,
    n_eval: int = 50,
) -> dict[str, float]:
    """Evaluate trained model against analytical solution.

    Returns:
        Dictionary with l2_error, linf_error, max_pred, max_exact.
    """
    x = jnp.linspace(-1.0, 1.0, n_eval)
    xx, yy = jnp.meshgrid(x, x)
    grid_pts = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    sdf_vals = disk_sdf(grid_pts)
    mask = sdf_vals < 0.0

    pred = model(grid_pts)
    exact = disk_analytical(grid_pts)

    pred_interior = pred[mask]
    exact_interior = exact[mask]

    l2 = float(jnp.sqrt(jnp.mean((pred_interior - exact_interior) ** 2)))
    linf = float(jnp.max(jnp.abs(pred_interior - exact_interior)))
    return {
        "l2_error": l2,
        "linf_error": linf,
        "max_pred": float(jnp.max(pred_interior)),
        "max_exact": float(jnp.max(exact_interior)),
    }


def sample_annulus_interior(
    r_inner: float,
    r_outer: float,
    n_points: int,
    key: PRNGKeyArray,
) -> Float[Array, "n 2"]:
    """Sample uniform points inside an annulus via rejection."""
    margin = 0.01
    candidates = jax.random.uniform(
        key, (n_points * 4, 2), minval=-r_outer, maxval=r_outer
    )
    r = jnp.sqrt(candidates[:, 0] ** 2 + candidates[:, 1] ** 2)
    mask = (r > r_inner + margin) & (r < r_outer - margin)
    indices = jnp.where(mask, size=n_points, fill_value=0)[0]
    return candidates[indices]


def sample_annulus_boundary(
    r_inner: float,
    r_outer: float,
    n_points_per_boundary: int,
    key: PRNGKeyArray,
) -> Float[Array, "n 2"]:
    """Sample points on both inner and outer boundaries of an annulus."""
    k1, k2 = jax.random.split(key)
    theta1 = jax.random.uniform(k1, (n_points_per_boundary,), maxval=2.0 * jnp.pi)
    theta2 = jax.random.uniform(k2, (n_points_per_boundary,), maxval=2.0 * jnp.pi)
    outer = jnp.stack([r_outer * jnp.cos(theta1), r_outer * jnp.sin(theta1)], axis=-1)
    inner = jnp.stack([r_inner * jnp.cos(theta2), r_inner * jnp.sin(theta2)], axis=-1)
    return jnp.concatenate([outer, inner])


def annulus_analytical(
    pts: Float[Array, "... 2"], r_inner: float, r_outer: float
) -> Float[Array, ...]:
    """Analytical solution for -nabla^2 u = 1 on an annulus, u=0 on boundaries."""
    r = jnp.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2 + 1e-20)
    c = (r_outer**2 - r_inner**2) / (4.0 * jnp.log(r_outer / r_inner))
    u = (r_outer**2 - r**2) / 4.0 + c * jnp.log(r / r_outer)
    sdf = jnp.maximum(r - r_outer, r_inner - r)
    return jnp.where(sdf < 0, u, 0.0)


__all__ = [
    "DiskPoissonPINN",
    "annulus_analytical",
    "disk_analytical",
    "disk_pinn_loss",
    "disk_sdf",
    "evaluate_disk_pinn",
    "sample_annulus_boundary",
    "sample_annulus_interior",
    "sample_disk_boundary",
    "sample_disk_interior",
    "train_disk_pinn",
]
