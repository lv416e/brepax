"""Unit tests for the Torus primitive."""

import equinox as eqx
import jax.numpy as jnp

from brepax.primitives import Torus


class TestTorus:
    """Tests for Torus SDF evaluation."""

    def _xy_torus(self) -> Torus:
        """Torus in xy-plane, R=2, r=0.5."""
        return Torus(
            center=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            major_radius=jnp.array(2.0),
            minor_radius=jnp.array(0.5),
        )

    def test_sdf_on_surface_is_zero(self) -> None:
        torus = self._xy_torus()
        # Point on outer equator: (2.5, 0, 0)
        result = torus.sdf(jnp.array([2.5, 0.0, 0.0]))
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_sdf_inside_is_negative(self) -> None:
        torus = self._xy_torus()
        # Point at tube center: (2, 0, 0)
        result = torus.sdf(jnp.array([2.0, 0.0, 0.0]))
        assert jnp.isclose(result, -0.5, atol=1e-6)

    def test_sdf_outside_is_positive(self) -> None:
        torus = self._xy_torus()
        result = torus.sdf(jnp.array([5.0, 0.0, 0.0]))
        assert result > 0.0

    def test_sdf_at_center_is_positive(self) -> None:
        torus = self._xy_torus()
        # Center of torus hole
        result = torus.sdf(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, 1.5, atol=1e-6)

    def test_sdf_is_jit_compatible(self) -> None:
        torus = self._xy_torus()
        jitted = eqx.filter_jit(torus.sdf)
        result = jitted(jnp.array([2.0, 0.0, 0.0]))
        assert jnp.isclose(result, -0.5, atol=1e-6)

    def test_sdf_is_vmap_compatible(self) -> None:
        torus = self._xy_torus()
        points = jnp.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = eqx.filter_vmap(torus.sdf)(points)
        assert result.shape == (3,)

    def test_sdf_is_differentiable(self) -> None:
        torus = self._xy_torus()
        grad_fn = eqx.filter_grad(lambda x: torus.sdf(x).sum())
        grad = grad_fn(jnp.array([3.0, 0.0, 0.0]))
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_volume(self) -> None:
        torus = self._xy_torus()
        expected = 2.0 * jnp.pi**2 * 2.0 * 0.5**2
        assert jnp.isclose(torus.volume(), expected)

    def test_parameters(self) -> None:
        torus = self._xy_torus()
        params = torus.parameters()
        assert "center" in params
        assert "major_radius" in params
        assert "minor_radius" in params
