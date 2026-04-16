"""Unit tests for the Cylinder primitive."""

import equinox as eqx
import jax.numpy as jnp

from brepax.primitives import Cylinder


class TestCylinder:
    """Tests for Cylinder SDF evaluation."""

    def _z_axis_cylinder(self) -> Cylinder:
        """Cylinder along z-axis at origin with radius 1."""
        return Cylinder(
            point=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(1.0),
        )

    def test_sdf_on_axis_is_negative_radius(self) -> None:
        cyl = self._z_axis_cylinder()
        result = cyl.sdf(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_on_surface_is_zero(self) -> None:
        cyl = self._z_axis_cylinder()
        result = cyl.sdf(jnp.array([1.0, 0.0, 0.0]))
        assert jnp.isclose(result, 0.0, atol=1e-7)

    def test_sdf_outside_is_positive(self) -> None:
        cyl = self._z_axis_cylinder()
        result = cyl.sdf(jnp.array([2.0, 0.0, 0.0]))
        assert jnp.isclose(result, 1.0)

    def test_sdf_independent_of_axis_position(self) -> None:
        cyl = self._z_axis_cylinder()
        # Points at same radial distance but different z should have same SDF
        r1 = cyl.sdf(jnp.array([1.5, 0.0, 0.0]))
        r2 = cyl.sdf(jnp.array([1.5, 0.0, 100.0]))
        assert jnp.isclose(r1, r2)

    def test_sdf_tilted_axis(self) -> None:
        axis = jnp.array([1.0, 1.0, 0.0]) / jnp.sqrt(2.0)
        cyl = Cylinder(
            point=jnp.array([0.0, 0.0, 0.0]),
            axis=axis,
            radius=jnp.array(1.0),
        )
        # Point perpendicular to the axis at distance 1
        result = cyl.sdf(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_sdf_is_jit_compatible(self) -> None:
        cyl = self._z_axis_cylinder()
        jitted = eqx.filter_jit(cyl.sdf)
        result = jitted(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_is_vmap_compatible(self) -> None:
        cyl = self._z_axis_cylinder()
        points = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        result = eqx.filter_vmap(cyl.sdf)(points)
        assert result.shape == (3,)

    def test_sdf_is_differentiable(self) -> None:
        cyl = self._z_axis_cylinder()
        grad_fn = eqx.filter_grad(lambda x: cyl.sdf(x).sum())
        grad = grad_fn(jnp.array([2.0, 0.0, 0.0]))
        assert grad.shape == (3,)
        # Gradient should point radially outward in xy plane
        assert jnp.isclose(grad[0], 1.0, atol=1e-5)
        assert jnp.isclose(grad[1], 0.0, atol=1e-5)
        assert jnp.isclose(grad[2], 0.0, atol=1e-5)

    def test_parameters(self) -> None:
        cyl = self._z_axis_cylinder()
        params = cyl.parameters()
        assert "point" in params
        assert "axis" in params
        assert "radius" in params
