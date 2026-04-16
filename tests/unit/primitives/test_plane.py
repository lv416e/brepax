"""Unit tests for the Plane primitive."""

import equinox as eqx
import jax.numpy as jnp

from brepax.primitives import Plane


class TestPlane:
    """Tests for Plane SDF evaluation."""

    def _xy_plane(self) -> Plane:
        """Plane at z=0 with normal pointing up (+z)."""
        return Plane(
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=jnp.array(0.0),
        )

    def test_sdf_above_is_positive(self) -> None:
        plane = self._xy_plane()
        result = plane.sdf(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isclose(result, 1.0)

    def test_sdf_below_is_negative(self) -> None:
        plane = self._xy_plane()
        result = plane.sdf(jnp.array([0.0, 0.0, -1.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_on_plane_is_zero(self) -> None:
        plane = self._xy_plane()
        result = plane.sdf(jnp.array([5.0, 3.0, 0.0]))
        assert jnp.isclose(result, 0.0, atol=1e-7)

    def test_sdf_with_offset(self) -> None:
        plane = Plane(
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=jnp.array(2.0),
        )
        result = plane.sdf(jnp.array([0.0, 0.0, 3.0]))
        assert jnp.isclose(result, 1.0)

    def test_sdf_tilted_normal(self) -> None:
        n = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
        plane = Plane(normal=n, offset=jnp.array(0.0))
        # Point along the normal direction at distance 1
        result = plane.sdf(n)
        assert jnp.isclose(result, 1.0, atol=1e-6)

    def test_sdf_is_jit_compatible(self) -> None:
        plane = self._xy_plane()
        jitted = eqx.filter_jit(plane.sdf)
        result = jitted(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isclose(result, 1.0)

    def test_sdf_is_vmap_compatible(self) -> None:
        plane = self._xy_plane()
        points = jnp.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = eqx.filter_vmap(plane.sdf)(points)
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.array([-1.0, 0.0, 1.0]))

    def test_sdf_is_differentiable(self) -> None:
        plane = self._xy_plane()
        grad_fn = eqx.filter_grad(lambda x: plane.sdf(x).sum())
        grad = grad_fn(jnp.array([5.0, 3.0, 1.0]))
        assert grad.shape == (3,)
        # Gradient of dot(x, normal) - offset w.r.t. x is the normal
        assert jnp.allclose(grad, jnp.array([0.0, 0.0, 1.0]), atol=1e-5)

    def test_parameters(self) -> None:
        plane = self._xy_plane()
        params = plane.parameters()
        assert "normal" in params
        assert "offset" in params
