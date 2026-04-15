"""Unit tests for analytical two-disk solutions."""

import jax
import jax.numpy as jnp


class TestDiskDiskUnionArea:
    """Tests for disk_disk_union_area analytical formula."""

    def test_disjoint_disks(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_union_area

        c1, r1 = jnp.array([0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([5.0, 0.0]), jnp.array(1.0)
        area = disk_disk_union_area(c1, r1, c2, r2)
        expected = jnp.pi * 1.0**2 + jnp.pi * 1.0**2
        assert jnp.isclose(area, expected, atol=1e-10)

    def test_identical_disks(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_union_area

        c = jnp.array([0.0, 0.0])
        r = jnp.array(1.0)
        area = disk_disk_union_area(c, r, c, r)
        expected = jnp.pi * 1.0**2
        assert jnp.isclose(area, expected, atol=1e-10)

    def test_contained_disk(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_union_area

        c1, r1 = jnp.array([0.0, 0.0]), jnp.array(2.0)
        c2, r2 = jnp.array([0.0, 0.0]), jnp.array(0.5)
        area = disk_disk_union_area(c1, r1, c2, r2)
        expected = jnp.pi * 2.0**2
        assert jnp.isclose(area, expected, atol=1e-10)

    def test_is_differentiable(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_union_area

        c1, r1 = jnp.array([0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([1.5, 0.0]), jnp.array(1.0)
        grad_r1 = jax.grad(disk_disk_union_area, argnums=1)(c1, r1, c2, r2)
        assert jnp.isfinite(grad_r1)


class TestDiskDiskStratumLabel:
    """Tests for stratum classification."""

    def test_disjoint(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_stratum_label

        label = disk_disk_stratum_label(
            jnp.array([0.0, 0.0]),
            jnp.array(1.0),
            jnp.array([5.0, 0.0]),
            jnp.array(1.0),
        )
        assert int(label) == 0

    def test_intersecting(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_stratum_label

        label = disk_disk_stratum_label(
            jnp.array([0.0, 0.0]),
            jnp.array(1.0),
            jnp.array([1.5, 0.0]),
            jnp.array(1.0),
        )
        assert int(label) == 1

    def test_contained(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_stratum_label

        label = disk_disk_stratum_label(
            jnp.array([0.0, 0.0]),
            jnp.array(2.0),
            jnp.array([0.0, 0.0]),
            jnp.array(0.5),
        )
        assert int(label) == 2


class TestDiskDiskBoundaryDistance:
    """Tests for boundary distance computation."""

    def test_at_external_tangent(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_boundary_distance

        c1, r1 = jnp.array([0.0, 0.0]), jnp.array(1.0)
        c2, r2 = jnp.array([2.0, 0.0]), jnp.array(1.0)
        dist = disk_disk_boundary_distance(c1, r1, c2, r2)
        assert jnp.isclose(dist, 0.0, atol=1e-10)

    def test_at_internal_tangent(self) -> None:
        from brepax.analytical.disk_disk import disk_disk_boundary_distance

        c1, r1 = jnp.array([0.0, 0.0]), jnp.array(2.0)
        c2, r2 = jnp.array([1.0, 0.0]), jnp.array(1.0)
        dist = disk_disk_boundary_distance(c1, r1, c2, r2)
        assert jnp.isclose(dist, 0.0, atol=1e-10)
