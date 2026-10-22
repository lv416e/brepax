"""Tests for Boolean operations on heterogeneous primitive pairs.

Verifies that union/subtract/intersect work for all primitive
combinations, not just same-type pairs (ADR-0012).
"""

import jax
import jax.numpy as jnp
import numpy as np

from brepax.boolean import intersect_volume, subtract_volume, union_volume
from brepax.primitives import Cylinder, Plane, Sphere, Torus


class TestHeterogeneousPairs:
    """Boolean operations between different primitive types."""

    def test_torus_sphere_union_scaling(self) -> None:
        """Finite domains enclose the torus and preserve disjoint volume derivatives."""
        scales = jnp.array([1e-3, 1.0, 1e3])

        def evaluate(scale):
            sphere = Sphere(
                center=scale * jnp.array([4.0, 0.0, 0.0]), radius=scale * 0.4
            )

            def volume(major_radius):
                torus = Torus(
                    center=jnp.zeros(3),
                    axis=jnp.array([0.0, 0.0, 1.0]),
                    major_radius=major_radius,
                    minor_radius=scale * 0.5,
                )
                return union_volume(torus, sphere, resolution=48)

            return jax.value_and_grad(volume)(scale * 1.5)

        values, gradients = jax.jit(jax.vmap(evaluate))(scales)
        reference_volume = 2.0 * jnp.pi**2 * 1.5 * 0.5**2 + 4.0 * jnp.pi * 0.4**3 / 3.0
        np.testing.assert_allclose(values / scales**3, reference_volume, rtol=0.05)
        np.testing.assert_allclose(values / scales**3, values[1], rtol=1e-12)
        np.testing.assert_allclose(
            gradients / scales**2, 2.0 * jnp.pi**2 * 0.5**2, rtol=1e-12
        )

    def test_sphere_cylinder_union(self) -> None:
        s = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        c = Cylinder(
            point=jnp.array([2.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(0.5),
        )
        vol = union_volume(s, c, resolution=48)
        assert jnp.isfinite(vol)
        assert vol > 0.0

    def test_sphere_cylinder_subtract(self) -> None:
        s = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(2.0))
        c = Cylinder(
            point=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(0.5),
        )
        vol = subtract_volume(s, c, resolution=48)
        sphere_vol = (4.0 / 3.0) * jnp.pi * 2.0**3
        assert vol < sphere_vol
        assert vol > 0.0

    def test_sphere_plane_intersect(self) -> None:
        s = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))
        p = Plane(normal=jnp.array([0.0, 0.0, 1.0]), offset=jnp.array(0.0))
        # Intersection = upper hemisphere
        vol = intersect_volume(s, p, resolution=48)
        half_sphere = (2.0 / 3.0) * jnp.pi * 1.0**3
        assert jnp.isclose(vol, half_sphere, rtol=0.15)

    def test_cylinder_plane_subtract(self) -> None:
        c = Cylinder(
            point=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(1.0),
        )
        p = Plane(normal=jnp.array([0.0, 0.0, 1.0]), offset=jnp.array(0.0))
        # Subtract upper half-space from cylinder
        vol = subtract_volume(c, p, resolution=48)
        assert jnp.isfinite(vol)
        assert vol > 0.0

    def test_heterogeneous_union_gradient(self) -> None:
        c = Cylinder(
            point=jnp.array([1.5, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=jnp.array(0.5),
        )

        def f(r):
            s = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=r)
            return union_volume(s, c, resolution=48)

        grad = jax.grad(f)(jnp.array(1.0))
        assert jnp.isfinite(grad)
        assert grad > 0.0

    def test_heterogeneous_subtract_gradient(self) -> None:
        s = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(2.0))

        def f(hole_r):
            c = Cylinder(
                point=jnp.array([0.0, 0.0, 0.0]),
                axis=jnp.array([0.0, 0.0, 1.0]),
                radius=hole_r,
            )
            return subtract_volume(s, c, resolution=48)

        grad = jax.grad(f)(jnp.array(0.5))
        assert jnp.isfinite(grad)
        assert grad < 0.0
