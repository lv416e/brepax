"""Unit tests for the Box primitive."""

import equinox as eqx
import jax.numpy as jnp

from brepax.primitives import Box


class TestBox:
    """Tests for Box SDF evaluation."""

    def _unit_box(self) -> Box:
        """Unit cube centered at origin."""
        return Box(
            center=jnp.array([0.0, 0.0, 0.0]),
            half_extents=jnp.array([1.0, 1.0, 1.0]),
        )

    def test_sdf_at_center_is_negative(self) -> None:
        box = self._unit_box()
        result = box.sdf(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_on_face_is_zero(self) -> None:
        box = self._unit_box()
        result = box.sdf(jnp.array([1.0, 0.0, 0.0]))
        assert jnp.isclose(result, 0.0, atol=1e-7)

    def test_sdf_outside_face_is_positive(self) -> None:
        box = self._unit_box()
        result = box.sdf(jnp.array([2.0, 0.0, 0.0]))
        assert jnp.isclose(result, 1.0)

    def test_sdf_at_corner(self) -> None:
        box = self._unit_box()
        result = box.sdf(jnp.array([2.0, 2.0, 2.0]))
        expected = jnp.sqrt(3.0)
        assert jnp.isclose(result, expected, atol=1e-6)

    def test_sdf_rectangular_box(self) -> None:
        box = Box(
            center=jnp.array([0.0, 0.0, 0.0]),
            half_extents=jnp.array([2.0, 1.0, 0.5]),
        )
        result = box.sdf(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, -0.5)

    def test_sdf_is_jit_compatible(self) -> None:
        box = self._unit_box()
        jitted = eqx.filter_jit(box.sdf)
        result = jitted(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_is_vmap_compatible(self) -> None:
        box = self._unit_box()
        points = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = eqx.filter_vmap(box.sdf)(points)
        assert result.shape == (3,)

    def test_sdf_is_differentiable(self) -> None:
        box = self._unit_box()
        grad_fn = eqx.filter_grad(lambda x: box.sdf(x).sum())
        grad = grad_fn(jnp.array([2.0, 0.0, 0.0]))
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_volume(self) -> None:
        box = self._unit_box()
        assert jnp.isclose(box.volume(), 8.0)

    def test_volume_rectangular(self) -> None:
        box = Box(
            center=jnp.array([0.0, 0.0, 0.0]),
            half_extents=jnp.array([2.0, 1.0, 0.5]),
        )
        assert jnp.isclose(box.volume(), 8.0)

    def test_parameters(self) -> None:
        box = self._unit_box()
        params = box.parameters()
        assert "center" in params
        assert "half_extents" in params
