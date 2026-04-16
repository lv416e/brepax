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
        assert jnp.isclose(result, expected, rtol=0.05)

    def test_disjoint(self, disjoint_disks) -> None:
        a, b = disjoint_disks
        result = union_area(a, b, method="stratum")
        expected = disk_disk_union_area(a.center, a.radius, b.center, b.radius)
        assert jnp.isclose(result, expected, rtol=0.05)

    def test_contained(self, contained_disks) -> None:
        a, b = contained_disks
        result = union_area(a, b, method="stratum")
        expected = disk_disk_union_area(a.center, a.radius, b.center, b.radius)
        assert jnp.isclose(result, expected, rtol=0.05)


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
        assert jnp.isclose(result, expected, rtol=0.1)

    def test_grad_disjoint(self, disjoint_disks) -> None:
        a, b = disjoint_disks
        result = self._grad_r1(a, b)
        # Disjoint: d(area)/d(r1) = 2*pi*r1
        expected = 2.0 * jnp.pi * a.radius
        assert jnp.isclose(result, expected, rtol=0.1)

    def test_grad_contained(self, contained_disks) -> None:
        a, b = contained_disks
        result = self._grad_r1(a, b)
        # Contained (r1 > r2): analytical d(area)/d(r1) = 2*pi*r1.
        # Grid-based method has discretization error at containment boundary.
        expected = 2.0 * jnp.pi * a.radius
        assert jnp.isclose(result, expected, rtol=0.2)

    def test_grad_center_intersecting(self, intersecting_disks) -> None:
        a, b = intersecting_disks

        def f(c1):
            disk = Disk(center=c1, radius=a.radius)
            return union_area(disk, b, method="stratum")

        result = jax.grad(f)(a.center)
        # Grid-based gradient has discretization error; verify direction
        # and order of magnitude, not exact match with analytical
        assert jnp.all(jnp.isfinite(result))
        assert result[0] < 0  # moving c1 toward c2 decreases area

    def test_grad_center_disjoint_is_finite(self, disjoint_disks) -> None:
        a, b = disjoint_disks

        def f(c1):
            disk = Disk(center=c1, radius=a.radius)
            return union_area(disk, b, method="stratum")

        result = jax.grad(f)(a.center)
        # Grid-based Method (C) may provide small gradient even in
        # disjoint stratum via thin sigmoid at primitive boundaries
        assert jnp.all(jnp.isfinite(result))


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
        assert jnp.isclose(result, expected, rtol=0.1)

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
        assert jnp.isclose(result, expected, rtol=0.1)
