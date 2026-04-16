"""Unit tests for generalized stratum-aware 3D Boolean operations."""

import jax
import jax.numpy as jnp

from brepax.analytical.sphere_sphere import sphere_sphere_union_volume
from brepax.boolean import union_volume
from brepax.primitives import Sphere


class TestStratumVolume3D:
    """Tests for union_volume with generalized Method (C)."""

    def test_disjoint_spheres_volume(self) -> None:
        a = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        b = Sphere(center=jnp.array([5.0, 0.0, 0.0]), radius=jnp.array(1.0))
        vol = union_volume(a, b, method="stratum", resolution=64)
        expected = (4.0 / 3.0) * jnp.pi * 2.0
        assert jnp.isclose(vol, expected, rtol=0.05)

    def test_overlapping_spheres_volume(self) -> None:
        a = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        b = Sphere(center=jnp.array([1.0, 0.0, 0.0]), radius=jnp.array(1.0))
        vol = union_volume(a, b, method="stratum", resolution=64)
        expected = sphere_sphere_union_volume(
            a.center,
            a.radius,
            b.center,
            b.radius,
        )
        assert jnp.isclose(vol, expected, rtol=0.1)

    def test_gradient_wrt_radius(self) -> None:
        b = Sphere(center=jnp.array([1.0, 0.0, 0.0]), radius=jnp.array(1.0))

        def f(r1):
            a = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=r1)
            return union_volume(a, b, method="stratum", resolution=48)

        grad = jax.grad(f)(jnp.array(1.0))
        assert jnp.isfinite(grad)
        assert grad > 0.0

    def test_gradient_wrt_center(self) -> None:
        b = Sphere(center=jnp.array([1.5, 0.0, 0.0]), radius=jnp.array(1.0))

        def f(c1):
            a = Sphere(center=c1, radius=jnp.array(1.0))
            return union_volume(a, b, method="stratum", resolution=48)

        grad = jax.grad(f)(jnp.array([0.0, 0.0, 0.0]))
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))
