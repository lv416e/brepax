"""Mold pull-direction optimization via differentiable undercut analysis.

Given a shape defined by its SDF, estimates the volume of material that
would prevent mold removal in a given pull direction. The undercut
criterion is SDF-based: a surface point is undercut when its outward
normal opposes the pull direction (dot(normal, direction) < 0).

Both the undercut volume and the optimization loop are fully
differentiable, enabling gradient-based search over the unit sphere S^2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

_FD_EPS = 1e-4


def _estimate_normals(
    shape_sdf: Callable[[Float[Array, 3]], Float[Array, ""]],
    grid: Float[Array, "n 3"],
) -> Float[Array, "n 3"]:
    """Central finite-difference SDF gradient for surface normal estimation.

    Uses finite differences instead of jax.grad to avoid NaN at
    degenerate SDF points (e.g., on a cylinder axis where
    norm([0,0,0]) gradient is undefined).
    """
    offsets = _FD_EPS * jnp.eye(3)
    components = []
    for i in range(3):
        fwd = jax.vmap(shape_sdf)(grid + offsets[i])
        bwd = jax.vmap(shape_sdf)(grid - offsets[i])
        components.append((fwd - bwd) / (2.0 * _FD_EPS))
    return jnp.stack(components, axis=-1)


@dataclass
class MoldDirectionResult:
    """Result of mold direction optimization.

    Attributes:
        direction: Optimized pull direction (unit vector).
        trajectory: Direction at each step, shape (steps+1, 3).
        losses: Undercut volume at each step.
        converged: Whether loss change dropped below tolerance.
    """

    direction: Float[Array, 3]
    trajectory: Float[Array, "n 3"]
    losses: list[float]
    converged: bool


def undercut_volume(
    shape_sdf: Callable[[Float[Array, 3]], Float[Array, ""]],
    direction: Float[Array, 3],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Estimate undercut volume for a mold pull direction.

    Undercut is defined as the set of interior points whose SDF spatial
    gradient (outward normal) opposes the pull direction. Both the
    interior membership and the undercut condition are smoothed with
    sigmoid indicators scaled by the grid cell width, ensuring the
    estimate improves with resolution.

    Normals are estimated via central finite differences to avoid NaN
    gradients at degenerate SDF points (cylinder axis, box center, etc.).

    Args:
        shape_sdf: Signed distance function for a single query point.
            Must accept shape (3,) and return scalar.
        direction: Unit pull direction on S^2.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis.

    Returns:
        Scalar estimated undercut volume.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(
        ...     center=jnp.array([0.0, 0.0, 0.0]),
        ...     half_extents=jnp.array([1.0, 1.0, 0.5]),
        ... )
        >>> vol = undercut_volume(
        ...     box.sdf,
        ...     jnp.array([0.0, 0.0, 1.0]),
        ...     lo=jnp.array([-2.0, -2.0, -2.0]),
        ...     hi=jnp.array([2.0, 2.0, 2.0]),
        ...     resolution=32,
        ... )
    """
    direction = direction / jnp.linalg.norm(direction)

    axes = [jnp.linspace(lo[i], hi[i], resolution) for i in range(3)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    grid = jnp.stack(mesh, axis=-1).reshape(-1, 3)

    cell_vol = jnp.prod((hi - lo) / (resolution - 1))
    cell_width = (hi[0] - lo[0]) / (resolution - 1)

    sdf_vals = jax.vmap(shape_sdf)(grid)
    normals = _estimate_normals(shape_sdf, grid)

    # Surface delta: concentrates weight at sdf=0 so the metric
    # reflects surface undercut rather than interior bulk.
    inside = jax.nn.sigmoid(-sdf_vals / cell_width)
    surface_weight = inside * (1.0 - inside) * 4.0

    # Undercut severity via softplus: counts how strongly each surface
    # normal opposes the pull.  Unlike sigmoid, softplus breaks
    # centrosymmetric cancellation (softplus(a)+softplus(-a) = |a| + C
    # rather than constant 1), giving meaningful signal even for
    # symmetric shapes with localized non-convex features.
    dot_nd = jnp.sum(normals * direction, axis=-1)
    severity = jax.nn.softplus(-dot_nd / cell_width)

    return jnp.sum(surface_weight * severity) * cell_vol


def optimize_mold_direction(
    shape_sdf: Callable[[Float[Array, 3]], Float[Array, ""]],
    initial_direction: Float[Array, 3],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
    steps: int = 100,
    lr: float = 0.01,
    tol: float = 1e-6,
) -> MoldDirectionResult:
    """Find the mold pull direction that minimizes undercut volume.

    Uses projected gradient descent on S^2: after each gradient step
    the direction vector is re-normalized to the unit sphere.

    Args:
        shape_sdf: Signed distance function for a single query point.
        initial_direction: Starting pull direction (will be normalized).
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis.
        steps: Maximum optimization steps.
        lr: Learning rate for gradient descent.
        tol: Convergence tolerance on absolute loss change.

    Returns:
        MoldDirectionResult with optimized direction and diagnostics.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(
        ...     center=jnp.array([0.0, 0.0, 0.0]),
        ...     half_extents=jnp.array([1.0, 1.0, 0.5]),
        ... )
        >>> result = optimize_mold_direction(
        ...     box.sdf,
        ...     jnp.array([1.0, 1.0, 1.0]),
        ...     lo=jnp.array([-2.0, -2.0, -2.0]),
        ...     hi=jnp.array([2.0, 2.0, 2.0]),
        ...     resolution=32,
        ...     steps=50,
        ... )
    """
    d = initial_direction / jnp.linalg.norm(initial_direction)

    def _loss(d: Float[Array, 3]) -> Float[Array, ""]:
        return undercut_volume(shape_sdf, d, lo=lo, hi=hi, resolution=resolution)

    trajectory = [d]
    losses: list[float] = []
    converged = False

    for step in range(steps):
        val, grad = jax.value_and_grad(_loss)(d)
        losses.append(float(val))

        if step > 0 and abs(losses[-1] - losses[-2]) < tol:
            converged = True
            break

        # Projected gradient descent: step then re-normalize to S^2
        d = d - lr * grad
        d = d / jnp.linalg.norm(d)
        trajectory.append(d)

    return MoldDirectionResult(
        direction=d,
        trajectory=jnp.stack(trajectory),
        losses=losses,
        converged=converged,
    )
