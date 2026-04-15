"""Unit tests for the Sphere primitive."""

import equinox as eqx
import jax.numpy as jnp

from brepax.primitives import Sphere


class TestSphere:
    """Tests for Sphere SDF evaluation."""

    def test_sdf_at_center_is_negative_radius(self) -> None:
        sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        result = sphere.sdf(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_on_surface_is_zero(self) -> None:
        sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        result = sphere.sdf(jnp.array([1.0, 0.0, 0.0]))
        assert jnp.isclose(result, 0.0, atol=1e-7)

    def test_sdf_outside_is_positive(self) -> None:
        sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        result = sphere.sdf(jnp.array([2.0, 0.0, 0.0]))
        assert result > 0.0

    def test_sdf_off_center(self) -> None:
        sphere = Sphere(center=jnp.array([1.0, 2.0, 3.0]), radius=jnp.array(0.5))
        result = sphere.sdf(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.isclose(result, -0.5)

    def test_sdf_is_jit_compatible(self) -> None:
        sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        jitted = eqx.filter_jit(sphere.sdf)
        result = jitted(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.isclose(result, -1.0)

    def test_sdf_is_vmap_compatible(self) -> None:
        sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        points = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = eqx.filter_vmap(sphere.sdf)(points)
        assert result.shape == (3,)

    def test_sdf_is_differentiable(self) -> None:
        sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        grad_fn = eqx.filter_grad(lambda x: sphere.sdf(x).sum())
        grad = grad_fn(jnp.array([2.0, 0.0, 0.0]))
        assert grad.shape == (3,)
        # Gradient should point radially outward
        assert jnp.isclose(grad[0], 1.0, atol=1e-5)
        assert jnp.isclose(grad[1], 0.0, atol=1e-5)
        assert jnp.isclose(grad[2], 0.0, atol=1e-5)

    def test_parameters(self) -> None:
        sphere = Sphere(center=jnp.array([1.0, 2.0, 3.0]), radius=jnp.array(0.5))
        params = sphere.parameters()
        assert "center" in params
        assert "radius" in params
        assert params["center"].shape == (3,)
        assert params["radius"].shape == ()
