"""Unit tests for BSplineSurface primitive."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from brepax.primitives import BSplineSurface


def _flat_patch():
    """Bilinear patch on the xy-plane."""
    pts = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    knots = jnp.array([0.0, 0.0, 1.0, 1.0])
    return BSplineSurface(
        control_points=pts,
        knots_u=knots,
        knots_v=knots,
        degree_u=1,
        degree_v=1,
    )


def _dome_patch():
    """Cubic patch with raised center."""
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
    return BSplineSurface(
        control_points=pts,
        knots_u=knots,
        knots_v=knots,
        degree_u=3,
        degree_v=3,
    )


@pytest.mark.slow
class TestBSplineSurfacePrimitive:
    """Tests for BSplineSurface as a Primitive."""

    def test_sdf_single_point(self) -> None:
        """SDF evaluates correctly for a single point."""
        surf = _flat_patch()
        dist = surf.sdf(jnp.array([0.5, 0.5, 1.0]))
        assert jnp.isclose(jnp.abs(dist), 1.0, rtol=0.05)

    def test_sdf_batch(self) -> None:
        """SDF evaluates correctly for batched points."""
        surf = _flat_patch()
        points = jnp.array(
            [
                [0.5, 0.5, 1.0],
                [0.5, 0.5, 2.0],
                [0.5, 0.5, 0.5],
            ]
        )
        dists = surf.sdf(points)
        assert dists.shape == (3,)
        assert jnp.all(jnp.isfinite(dists))

    def test_sdf_grid_shape(self) -> None:
        """SDF handles 3D grid input shape."""
        surf = _flat_patch()
        grid = jnp.zeros((4, 4, 4, 3))
        dists = surf.sdf(grid)
        assert dists.shape == (4, 4, 4)

    def test_parameters_contains_control_points(self) -> None:
        """Parameters dict includes control_points."""
        surf = _flat_patch()
        params = surf.parameters()
        assert "control_points" in params
        assert params["control_points"].shape == (2, 2, 3)

    def test_volume_is_infinite(self) -> None:
        """Unbounded single surface returns inf volume."""
        surf = _flat_patch()
        assert jnp.isinf(surf.volume())

    def test_jit_compatible(self) -> None:
        """SDF works under jax.jit."""
        surf = _flat_patch()

        @jax.jit
        def compute(x):
            return surf.sdf(x)

        result = compute(jnp.array([0.5, 0.5, 1.0]))
        assert jnp.isfinite(result)

    def test_grad_wrt_control_points(self) -> None:
        """jax.grad flows through SDF to control points."""
        surf = _dome_patch()
        query = jnp.array([0.5, 0.5, 1.0])

        def loss(s):
            return s.sdf(query)

        grad = eqx.filter_grad(loss)(surf)
        assert jnp.all(jnp.isfinite(grad.control_points))

    def test_dome_sdf_closer(self) -> None:
        """Dome surface is closer to query than flat z=0."""
        surf = _dome_patch()
        dist = surf.sdf(jnp.array([0.5, 0.5, 1.0]))
        assert jnp.abs(dist) < 1.0
