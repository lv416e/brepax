"""Unit tests for the Cone primitive."""

import equinox as eqx
import jax.numpy as jnp

from brepax.primitives import Cone


class TestCone:
    """Tests for Cone SDF evaluation."""

    def _z_cone(self) -> Cone:
        """Cone along +z from origin with 45-degree half-angle."""
        return Cone(
            apex=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            angle=jnp.array(jnp.pi / 4),
        )

    def test_sdf_on_surface_is_zero(self) -> None:
        cone = self._z_cone()
        # At 45 degrees, point (1, 0, 1) is on the surface
        result = cone.sdf(jnp.array([1.0, 0.0, 1.0]))
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_sdf_inside_is_negative(self) -> None:
        cone = self._z_cone()
        # Point on axis ahead of apex is inside
        result = cone.sdf(jnp.array([0.0, 0.0, 2.0]))
        assert result < 0.0

    def test_sdf_outside_is_positive(self) -> None:
        cone = self._z_cone()
        # Point far from axis is outside
        result = cone.sdf(jnp.array([5.0, 0.0, 1.0]))
        assert result > 0.0

    def test_sdf_behind_apex_is_positive(self) -> None:
        cone = self._z_cone()
        # Point behind apex
        result = cone.sdf(jnp.array([0.0, 0.0, -1.0]))
        assert result > 0.0

    def test_sdf_is_jit_compatible(self) -> None:
        cone = self._z_cone()
        jitted = eqx.filter_jit(cone.sdf)
        result = jitted(jnp.array([0.0, 0.0, 2.0]))
        assert result < 0.0

    def test_sdf_is_vmap_compatible(self) -> None:
        cone = self._z_cone()
        points = jnp.array([[0.0, 0.0, 2.0], [5.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        result = eqx.filter_vmap(cone.sdf)(points)
        assert result.shape == (3,)

    def test_sdf_is_differentiable(self) -> None:
        cone = self._z_cone()
        grad_fn = eqx.filter_grad(lambda x: cone.sdf(x).sum())
        grad = grad_fn(jnp.array([1.0, 0.0, 2.0]))
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_parameters(self) -> None:
        cone = self._z_cone()
        params = cone.parameters()
        assert "apex" in params
        assert "axis" in params
        assert "angle" in params
