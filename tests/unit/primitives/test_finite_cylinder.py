"""Unit tests for the FiniteCylinder primitive."""

import equinox as eqx
import jax.numpy as jnp

from brepax.primitives import FiniteCylinder


class TestFiniteCylinder:
    """Tests for FiniteCylinder SDF evaluation."""

    def _z_cylinder(self) -> FiniteCylinder:
        """Cylinder along z-axis, radius=1, height=4, centered at origin."""
        return FiniteCylinder(
            center=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(1.0),
            height=jnp.array(4.0),
        )

    def test_sdf_at_center_is_negative(self) -> None:
        cyl = self._z_cylinder()
        result = cyl.sdf(jnp.array([0.0, 0.0, 0.0]))
        assert result < 0.0

    def test_sdf_on_radial_surface_is_zero(self) -> None:
        cyl = self._z_cylinder()
        result = cyl.sdf(jnp.array([1.0, 0.0, 0.0]))
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_sdf_on_cap_is_zero(self) -> None:
        cyl = self._z_cylinder()
        result = cyl.sdf(jnp.array([0.0, 0.0, 2.0]))
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_sdf_outside_radially(self) -> None:
        cyl = self._z_cylinder()
        result = cyl.sdf(jnp.array([2.0, 0.0, 0.0]))
        assert jnp.isclose(result, 1.0, atol=1e-6)

    def test_sdf_outside_axially(self) -> None:
        cyl = self._z_cylinder()
        result = cyl.sdf(jnp.array([0.0, 0.0, 3.0]))
        assert jnp.isclose(result, 1.0, atol=1e-6)

    def test_sdf_is_jit_compatible(self) -> None:
        cyl = self._z_cylinder()
        jitted = eqx.filter_jit(cyl.sdf)
        result = jitted(jnp.array([0.0, 0.0, 0.0]))
        assert result < 0.0

    def test_sdf_is_vmap_compatible(self) -> None:
        cyl = self._z_cylinder()
        points = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        result = eqx.filter_vmap(cyl.sdf)(points)
        assert result.shape == (3,)

    def test_sdf_is_differentiable(self) -> None:
        cyl = self._z_cylinder()
        grad_fn = eqx.filter_grad(lambda x: cyl.sdf(x).sum())
        grad = grad_fn(jnp.array([2.0, 0.0, 0.0]))
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_volume(self) -> None:
        cyl = self._z_cylinder()
        expected = jnp.pi * 1.0**2 * 4.0
        assert jnp.isclose(cyl.volume(), expected)

    def test_parameters(self) -> None:
        cyl = self._z_cylinder()
        params = cyl.parameters()
        assert "center" in params
        assert "axis" in params
        assert "radius" in params
        assert "height" in params
