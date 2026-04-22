"""``closest_point_and_foot`` roundtrip and gradient checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from brepax.nurbs.evaluate import evaluate_surface
from brepax.nurbs.projection import closest_point_and_foot


def _make_curved_patch() -> tuple:
    """Cubic patch with a raised center — same shape used in sdf tests."""
    pts = jnp.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            pts = pts.at[i, j, 0].set(i / 3.0)
            pts = pts.at[i, j, 1].set(j / 3.0)
    pts = pts.at[1, 1, 2].set(0.5)
    pts = pts.at[1, 2, 2].set(0.5)
    pts = pts.at[2, 1, 2].set(0.5)
    pts = pts.at[2, 2, 2].set(0.5)
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return pts, knots, knots, 3, 3


def test_foot_is_on_surface_at_uv() -> None:
    """``foot == S(u_opt, v_opt)`` must hold exactly."""
    pts, ku, kv, du, dv = _make_curved_patch()
    query = jnp.array([0.5, 0.5, 1.0])
    foot, u_opt, v_opt = closest_point_and_foot(query, pts, ku, kv, du, dv, 0.5, 0.5)
    expected = evaluate_surface(pts, ku, kv, du, dv, u_opt, v_opt)
    assert jnp.allclose(foot, expected, atol=1e-6)


def test_foot_is_closer_than_initial_guess() -> None:
    """The converged foot must not be worse than a lattice of alternatives."""
    pts, ku, kv, du, dv = _make_curved_patch()
    query = jnp.array([0.3, 0.7, 2.0])
    foot, _, _ = closest_point_and_foot(query, pts, ku, kv, du, dv, 0.5, 0.5)
    dist_foot = float(jnp.linalg.norm(foot - query))

    alts = [
        evaluate_surface(pts, ku, kv, du, dv, jnp.asarray(u), jnp.asarray(v))
        for u in (0.1, 0.3, 0.5, 0.7, 0.9)
        for v in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    min_alt = min(float(jnp.linalg.norm(p - query)) for p in alts)
    assert dist_foot <= min_alt + 1e-6


def test_grad_through_control_points_finite() -> None:
    """Gradient of the foot w.r.t. control points flows without NaN."""
    pts, ku, kv, du, dv = _make_curved_patch()
    query = jnp.array([0.3, 0.7, 2.0])

    def loss(cp: jnp.ndarray) -> jnp.ndarray:
        foot, _, _ = closest_point_and_foot(query, cp, ku, kv, du, dv, 0.5, 0.5)
        return jnp.sum((foot - query) ** 2)

    g = jax.grad(loss)(pts)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0
