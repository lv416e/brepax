"""Unit tests for smooth-min Boolean operations."""

import jax
import jax.numpy as jnp
import pytest

from brepax.analytical.disk_disk import disk_disk_union_area
from brepax.boolean import union_area
from brepax.boolean.smoothing import sdf_union_smooth, smooth_min
from brepax.primitives import Disk


@pytest.fixture()
def two_disks():
    """Overlapping two-disk configuration."""
    a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
    b = Disk(center=jnp.array([1.5, 0.0]), radius=jnp.array(1.0))
    return a, b


@pytest.fixture()
def disjoint_disks():
    """Non-overlapping two-disk configuration."""
    a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
    b = Disk(center=jnp.array([5.0, 0.0]), radius=jnp.array(1.0))
    return a, b


class TestSmoothMin:
    """Tests for the smooth minimum operator."""

    def test_approaches_exact_min_for_small_k(self) -> None:
        a = jnp.array(1.0)
        b = jnp.array(2.0)
        result = smooth_min(a, b, jnp.array(0.001))
        assert jnp.isclose(result, 1.0, atol=0.01)

    def test_symmetric(self) -> None:
        a = jnp.array(1.0)
        b = jnp.array(2.0)
        k = jnp.array(0.1)
        assert jnp.isclose(smooth_min(a, b, k), smooth_min(b, a, k))

    def test_equal_inputs_return_input(self) -> None:
        v = jnp.array(3.0)
        k = jnp.array(0.1)
        # smooth_min(v, v) = -k * log(2 * exp(-v/k)) = v - k*log(2)
        expected = v - k * jnp.log(2.0)
        assert jnp.isclose(smooth_min(v, v, k), expected)

    def test_differentiable(self) -> None:
        grad_fn = jax.grad(lambda a: smooth_min(a, jnp.array(2.0), jnp.array(0.1)))
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)

    def test_batched_via_vmap(self) -> None:
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([3.0, 2.0, 1.0])
        k = jnp.array(0.1)
        result = jax.vmap(lambda ai, bi: smooth_min(ai, bi, k))(a, b)
        assert result.shape == (3,)


class TestSdfUnionSmooth:
    """Tests for smooth union SDF evaluation."""

    def test_negative_inside_union(self, two_disks) -> None:
        a, b = two_disks
        # Point at origin is inside disk a
        x = jnp.array([0.0, 0.0])
        sdf = sdf_union_smooth(a, b, x, jnp.array(0.1))
        assert sdf < 0.0

    def test_positive_outside_union(self, two_disks) -> None:
        a, b = two_disks
        x = jnp.array([5.0, 5.0])
        sdf = sdf_union_smooth(a, b, x, jnp.array(0.1))
        assert sdf > 0.0


class TestUnionAreaSmoothing:
    """Tests for grid-based smooth union area computation."""

    def test_converges_to_analytical_for_small_k(self, two_disks) -> None:
        a, b = two_disks
        analytical = disk_disk_union_area(
            a.center,
            a.radius,
            b.center,
            b.radius,
        )
        # Small k/beta should approximate the true area
        smooth = union_area(
            a,
            b,
            method="smoothing",
            k=0.01,
            beta=0.01,
            resolution=256,
        )
        assert jnp.isclose(smooth, analytical, rtol=0.05)

    def test_disjoint_equals_sum_of_areas(self, disjoint_disks) -> None:
        a, b = disjoint_disks
        expected = jnp.pi * a.radius**2 + jnp.pi * b.radius**2
        smooth = union_area(
            a,
            b,
            method="smoothing",
            k=0.01,
            beta=0.01,
            resolution=128,
        )
        assert jnp.isclose(smooth, expected, rtol=0.05)

    def test_is_differentiable_wrt_radius(self, two_disks) -> None:
        a, b = two_disks

        def area_fn(r1):
            disk = Disk(center=a.center, radius=r1)
            return union_area(
                disk,
                b,
                method="smoothing",
                k=0.1,
                beta=0.1,
                resolution=64,
            )

        grad = jax.grad(area_fn)(a.radius)
        assert jnp.isfinite(grad)
        # Increasing radius should increase area
        assert grad > 0.0

    def test_is_differentiable_wrt_center(self, two_disks) -> None:
        a, b = two_disks

        def area_fn(c1):
            disk = Disk(center=c1, radius=a.radius)
            return union_area(
                disk,
                b,
                method="smoothing",
                k=0.1,
                beta=0.1,
                resolution=64,
            )

        grad = jax.grad(area_fn)(a.center)
        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))

    def test_unified_api_dispatch(self, two_disks) -> None:
        a, b = two_disks
        result = union_area(a, b, method="smoothing", resolution=32)
        assert jnp.isfinite(result)
        assert result > 0.0

    def test_unknown_method_rejected_by_type_check(self, two_disks) -> None:
        a, b = two_disks
        # beartype enforces Literal type before reaching ValueError branch
        with pytest.raises((ValueError, TypeError, Exception)):
            union_area(a, b, method="bogus")  # type: ignore[arg-type]

    def test_unimplemented_toi_raises(self, two_disks) -> None:
        a, b = two_disks
        with pytest.raises(NotImplementedError):
            union_area(a, b, method="toi")
