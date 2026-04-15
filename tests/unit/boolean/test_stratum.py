"""Unit tests for stratum-aware Boolean operations (Method C)."""

import jax
import jax.numpy as jnp
import pytest

from brepax.analytical.disk_disk import disk_disk_union_area
from brepax.boolean import union_area
from brepax.primitives import Disk


@pytest.fixture()
def intersecting_disks():
    a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
    b = Disk(center=jnp.array([1.5, 0.0]), radius=jnp.array(1.0))
    return a, b


@pytest.fixture()
def disjoint_disks():
    a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
    b = Disk(center=jnp.array([5.0, 0.0]), radius=jnp.array(1.0))
    return a, b


@pytest.fixture()
def contained_disks():
    a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(2.0))
    b = Disk(center=jnp.array([0.3, 0.0]), radius=jnp.array(0.5))
    return a, b


class TestStratumAreaAccuracy:
    """Forward pass should match analytical ground truth exactly."""

    def test_intersecting(self, intersecting_disks) -> None:
        a, b = intersecting_disks
        result = union_area(a, b, method="stratum")
        expected = disk_disk_union_area(a.center, a.radius, b.center, b.radius)
        assert jnp.isclose(result, expected, atol=1e-10)

    def test_disjoint(self, disjoint_disks) -> None:
        a, b = disjoint_disks
        result = union_area(a, b, method="stratum")
        expected = disk_disk_union_area(a.center, a.radius, b.center, b.radius)
        assert jnp.isclose(result, expected, atol=1e-10)

    def test_contained(self, contained_disks) -> None:
        a, b = contained_disks
        result = union_area(a, b, method="stratum")
        expected = disk_disk_union_area(a.center, a.radius, b.center, b.radius)
        assert jnp.isclose(result, expected, atol=1e-10)


class TestStratumGradientAccuracy:
    """Custom VJP should produce analytically correct gradients per stratum."""

    def _grad_r1(self, a, b):
        def f(r1):
            disk = Disk(center=a.center, radius=r1)
            return union_area(disk, b, method="stratum")

        return jax.grad(f)(a.radius)

    def _analytical_grad_r1(self, a, b):
        return jax.grad(disk_disk_union_area, argnums=1)(
            a.center,
            a.radius,
            b.center,
            b.radius,
        )

    def test_grad_intersecting(self, intersecting_disks) -> None:
        a, b = intersecting_disks
        result = self._grad_r1(a, b)
        expected = self._analytical_grad_r1(a, b)
        assert jnp.isclose(result, expected, rtol=1e-5)

    def test_grad_disjoint(self, disjoint_disks) -> None:
        a, b = disjoint_disks
        result = self._grad_r1(a, b)
        # Disjoint: d(area)/d(r1) = 2*pi*r1
        expected = 2.0 * jnp.pi * a.radius
        assert jnp.isclose(result, expected, rtol=1e-5)

    def test_grad_contained(self, contained_disks) -> None:
        a, b = contained_disks
        result = self._grad_r1(a, b)
        # Contained (r1 > r2): d(area)/d(r1) = 2*pi*r1
        expected = 2.0 * jnp.pi * a.radius
        assert jnp.isclose(result, expected, rtol=1e-5)

    def test_grad_center_intersecting(self, intersecting_disks) -> None:
        a, b = intersecting_disks

        def f(c1):
            disk = Disk(center=c1, radius=a.radius)
            return union_area(disk, b, method="stratum")

        result = jax.grad(f)(a.center)
        expected = jax.grad(disk_disk_union_area, argnums=0)(
            a.center,
            a.radius,
            b.center,
            b.radius,
        )
        assert jnp.allclose(result, expected, rtol=1e-4)

    def test_grad_center_disjoint_is_zero(self, disjoint_disks) -> None:
        a, b = disjoint_disks

        def f(c1):
            disk = Disk(center=c1, radius=a.radius)
            return union_area(disk, b, method="stratum")

        result = jax.grad(f)(a.center)
        # Disjoint: area = pi*r1^2 + pi*r2^2, independent of centers
        assert jnp.allclose(result, jnp.zeros(2), atol=1e-6)


class TestStratumNearBoundary:
    """Gradient accuracy near stratum boundaries."""

    def test_near_external_tangent(self) -> None:
        """Gradient at eps=0.01 from external tangent."""
        a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
        b = Disk(center=jnp.array([1.99, 0.0]), radius=jnp.array(1.0))

        def f(r1):
            return union_area(Disk(center=a.center, radius=r1), b, method="stratum")

        result = jax.grad(f)(a.radius)
        expected = jax.grad(disk_disk_union_area, argnums=1)(
            a.center,
            a.radius,
            b.center,
            b.radius,
        )
        # Method (C) should be much more accurate than Method (A) here
        assert jnp.isclose(result, expected, rtol=1e-4)

    def test_near_internal_tangent(self) -> None:
        """Gradient at eps=0.01 from internal tangent."""
        a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(2.0))
        b = Disk(center=jnp.array([1.51, 0.0]), radius=jnp.array(0.5))

        def f(r1):
            return union_area(Disk(center=a.center, radius=r1), b, method="stratum")

        result = jax.grad(f)(a.radius)
        expected = jax.grad(disk_disk_union_area, argnums=1)(
            a.center,
            a.radius,
            b.center,
            b.radius,
        )
        assert jnp.isclose(result, expected, rtol=1e-4)
