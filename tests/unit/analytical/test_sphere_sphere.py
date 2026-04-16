"""Unit tests for analytical two-sphere solutions."""

import jax
import jax.numpy as jnp

from brepax.analytical.sphere_sphere import (
    sphere_sphere_boundary_distance,
    sphere_sphere_stratum_label,
    sphere_sphere_union_volume,
)


class TestSphereSphereUnionVolume:
    """Tests for sphere_sphere_union_volume analytical formula."""

    def test_disjoint_spheres(self) -> None:
        c1, r1 = jnp.array([0.0, 0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([5.0, 0.0, 0.0]), jnp.array(1.0)
        vol = sphere_sphere_union_volume(c1, r1, c2, r2)
        expected = (4.0 / 3.0) * jnp.pi * (1.0**3 + 1.0**3)
        assert jnp.isclose(vol, expected, atol=1e-10)

    def test_identical_spheres(self) -> None:
        c = jnp.array([0.0, 0.0, 0.0])
        r = jnp.array(1.0)
        vol = sphere_sphere_union_volume(c, r, c, r)
        expected = (4.0 / 3.0) * jnp.pi * 1.0**3
        assert jnp.isclose(vol, expected, atol=1e-10)

    def test_contained_sphere(self) -> None:
        c1, r1 = jnp.array([0.0, 0.0, 0.0]), jnp.array(2.0)
        c2, r2 = jnp.array([0.0, 0.0, 0.0]), jnp.array(0.5)
        vol = sphere_sphere_union_volume(c1, r1, c2, r2)
        expected = (4.0 / 3.0) * jnp.pi * 2.0**3
        assert jnp.isclose(vol, expected, atol=1e-10)

    def test_overlapping_equal_radii(self) -> None:
        c1, r1 = jnp.array([0.0, 0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([1.0, 0.0, 0.0]), jnp.array(1.0)
        vol = sphere_sphere_union_volume(c1, r1, c2, r2)
        # Union must be less than sum and greater than single sphere
        single = (4.0 / 3.0) * jnp.pi
        assert vol < 2 * single
        assert vol > single

    def test_is_differentiable_wrt_radius(self) -> None:
        c1, r1 = jnp.array([0.0, 0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([1.5, 0.0, 0.0]), jnp.array(1.0)
        grad_r1 = jax.grad(sphere_sphere_union_volume, argnums=1)(c1, r1, c2, r2)
        assert jnp.isfinite(grad_r1)
        # Increasing radius should increase volume
        assert grad_r1 > 0

    def test_is_differentiable_wrt_center(self) -> None:
        c1, r1 = jnp.array([0.0, 0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([1.5, 0.0, 0.0]), jnp.array(1.0)
        grad_c1 = jax.grad(sphere_sphere_union_volume, argnums=0)(c1, r1, c2, r2)
        assert grad_c1.shape == (3,)
        assert jnp.all(jnp.isfinite(grad_c1))

    def test_gradient_finite_for_disjoint(self) -> None:
        c1, r1 = jnp.array([0.0, 0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([5.0, 0.0, 0.0]), jnp.array(1.0)
        grad_r1 = jax.grad(sphere_sphere_union_volume, argnums=1)(c1, r1, c2, r2)
        assert jnp.isfinite(grad_r1)

    def test_gradient_finite_for_contained(self) -> None:
        c1, r1 = jnp.array([0.0, 0.0, 0.0]), jnp.array(2.0)
        c2, r2 = jnp.array([0.3, 0.0, 0.0]), jnp.array(0.5)
        grad_r1 = jax.grad(sphere_sphere_union_volume, argnums=1)(c1, r1, c2, r2)
        assert jnp.isfinite(grad_r1)


class TestSphereSphereStratumLabel:
    """Tests for stratum classification."""

    def test_disjoint(self) -> None:
        label = sphere_sphere_stratum_label(
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array(1.0),
            jnp.array([5.0, 0.0, 0.0]),
            jnp.array(1.0),
        )
        assert int(label) == 0

    def test_intersecting(self) -> None:
        label = sphere_sphere_stratum_label(
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array(1.0),
            jnp.array([1.5, 0.0, 0.0]),
            jnp.array(1.0),
        )
        assert int(label) == 1

    def test_contained(self) -> None:
        label = sphere_sphere_stratum_label(
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array(2.0),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array(0.5),
        )
        assert int(label) == 2


class TestSphereSphereeBoundaryDistance:
    """Tests for boundary distance computation."""

    def test_at_external_tangent(self) -> None:
        dist = sphere_sphere_boundary_distance(
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array(1.0),
            jnp.array([2.0, 0.0, 0.0]),
            jnp.array(1.0),
        )
        assert jnp.isclose(dist, 0.0, atol=1e-10)

    def test_at_internal_tangent(self) -> None:
        dist = sphere_sphere_boundary_distance(
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array(2.0),
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array(1.0),
        )
        assert jnp.isclose(dist, 0.0, atol=1e-10)
