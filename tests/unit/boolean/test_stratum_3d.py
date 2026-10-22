"""Unit tests for generalized stratum-aware 3D Boolean operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.analytical.sphere_sphere import sphere_sphere_union_volume
from brepax.boolean import intersect_volume, subtract_volume, union_volume
from brepax.primitives import Box, Sphere


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


@pytest.mark.parametrize(
    ("half_extents", "separation"),
    [([10.0, 0.1, 0.1], 30.0), ([100.0, 0.01, 0.01], 300.0)],
)
def test_thin_boxes_domain_resolution(half_extents, separation) -> None:
    """Long axes must not make padding overwhelm the short-axis grid spacing."""
    a = Box(center=jnp.zeros(3), half_extents=jnp.array(half_extents))
    b = Box(
        center=jnp.array([separation, 0.0, 0.0]), half_extents=jnp.array(half_extents)
    )
    np.testing.assert_allclose(
        union_volume(a, b, resolution=64), a.volume() + b.volume(), rtol=0.05
    )


@pytest.mark.parametrize("measure", [union_volume, intersect_volume, subtract_volume])
def test_boolean_volume_grid_similarity(measure) -> None:
    """Axis cycles and unit changes co-transform volume and length gradients."""
    a = Sphere(center=jnp.array([-0.35, 0.1, -0.2]), radius=jnp.array(0.8))
    b = Sphere(center=jnp.array([0.8, 0.45, 0.15]), radius=jnp.array(0.65))
    rotation = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    shift = jnp.array([2.3, -1.7, 0.6])

    def volume(a_, b_):
        return measure(a_, b_, method="stratum", resolution=40)

    evaluate = jax.jit(jax.value_and_grad(volume, argnums=(0, 1)))
    value, gradients = evaluate(a, b)
    scales = jnp.array([1e-6, 1.0, 1e3])

    def evaluate_transformed(scale):
        return evaluate(
            Sphere(
                center=scale * (rotation @ a.center + shift), radius=scale * a.radius
            ),
            Sphere(
                center=scale * (rotation @ b.center + shift), radius=scale * b.radius
            ),
        )

    transformed_value, transformed_gradients = jax.jit(jax.vmap(evaluate_transformed))(
        scales
    )
    np.testing.assert_allclose(transformed_value / scales**3, value, rtol=1e-12)
    for gradient, transformed in zip(gradients, transformed_gradients, strict=True):
        np.testing.assert_allclose(
            transformed.center / scales[:, None] ** 2,
            jnp.broadcast_to(rotation @ gradient.center, (len(scales), 3)),
            rtol=1e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transformed.radius / scales**2, gradient.radius, rtol=1e-11
        )
