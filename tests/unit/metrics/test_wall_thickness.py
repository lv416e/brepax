"""Unit tests for wall thickness metrics."""

import jax
import jax.numpy as jnp

from brepax.brep.csg_eval import make_grid_3d
from brepax.metrics.wall_thickness import (
    integrate_sdf_thin_wall_volume,
    min_wall_thickness,
    thin_wall_volume,
)
from brepax.primitives import Box, Sphere


class TestIntegrateSdfThinWallVolume:
    """Tests for the low-level grid integration function."""

    def test_thin_plate_full_violation(self) -> None:
        """A thin plate with threshold > half-thickness has ~full volume violation."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([5.0, 5.0, 0.25]),
        )
        lo, hi = jnp.array([-7.0] * 3), jnp.array([7.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        total_vol = 10.0 * 10.0 * 0.5  # 50.0
        # threshold=0.5 > half-thickness=0.25: all interior within threshold
        violation = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
        ratio = float(violation) / total_vol
        assert ratio > 0.8, f"Expected most volume violated, got ratio={ratio:.3f}"

    def test_thick_cube_small_violation_ratio(self) -> None:
        """A thick cube with small threshold has low violation ratio."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([5.0, 5.0, 5.0]),
        )
        lo, hi = jnp.array([-7.0] * 3), jnp.array([7.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        total_vol = 10.0**3  # 1000.0
        # threshold=0.5: thin shell near surface ~30% of volume
        violation = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
        ratio = float(violation) / total_vol
        assert ratio < 0.35, f"Expected shell ratio, got ratio={ratio:.3f}"

    def test_threshold_zero_smaller_than_positive(self) -> None:
        """Zero threshold gives less violation than positive threshold."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.ones(3),
        )
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        # Sigmoid smoothing gives nonzero at threshold=0 (surface delta),
        # but it must be less than a positive threshold
        vol_zero = integrate_sdf_thin_wall_volume(sdf, 0.0, lo, hi, 64)
        vol_half = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
        assert float(vol_zero) < float(vol_half), (
            f"zero={float(vol_zero):.4f} should be less than half={float(vol_half):.4f}"
        )


class TestThinWallVolume:
    """Tests for the high-level thin_wall_volume function."""

    def test_sphere_thin_shell(self) -> None:
        """Thin-wall volume of a sphere increases with threshold."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(2.0))
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        vol_small = thin_wall_volume(sphere.sdf, 0.3, lo=lo, hi=hi, resolution=48)
        vol_large = thin_wall_volume(sphere.sdf, 1.0, lo=lo, hi=hi, resolution=48)
        assert float(vol_large) > float(vol_small), (
            f"Larger threshold should give larger violation: "
            f"small={float(vol_small):.3f}, large={float(vol_large):.3f}"
        )

    def test_differentiable_wrt_threshold(self) -> None:
        """jax.grad w.r.t. threshold is positive (more threshold = more volume)."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([2.0, 1.5, 1.0]),
        )
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def vol_of_threshold(t: jnp.ndarray) -> jnp.ndarray:
            return thin_wall_volume(box.sdf, t, lo=lo, hi=hi, resolution=48)

        grad_t = jax.grad(vol_of_threshold)(jnp.array(0.5))
        assert jnp.isfinite(grad_t)
        assert float(grad_t) > 0.0, (
            f"Expected positive gradient, got {float(grad_t):.4f}"
        )

    def test_differentiable_wrt_shape(self) -> None:
        """jax.grad of thin_wall_volume w.r.t. box half_extents works."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def vol_of_extents(he: jnp.ndarray) -> jnp.ndarray:
            box = Box(center=jnp.zeros(3), half_extents=he)
            return thin_wall_volume(box.sdf, 0.5, lo=lo, hi=hi, resolution=48)

        he = jnp.array([2.0, 1.5, 1.0])
        grad_he = jax.grad(vol_of_extents)(he)
        assert jnp.all(jnp.isfinite(grad_he)), f"Non-finite gradient: {grad_he}"

    def test_jit_compatible(self) -> None:
        """thin_wall_volume works under jax.jit."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        @jax.jit
        def compute(t: jnp.ndarray) -> jnp.ndarray:
            return thin_wall_volume(box.sdf, t, lo=lo, hi=hi, resolution=32)

        result = compute(jnp.array(0.5))
        assert jnp.isfinite(result)
        assert float(result) > 0.0


class TestMinWallThickness:
    """Tests for the min_wall_thickness diagnostic function."""

    def test_thin_plate(self) -> None:
        """Thin plate (2.0 thick) returns ~2.0."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([3.0, 3.0, 1.0]),
        )
        lo, hi = jnp.array([-5.0] * 3), jnp.array([5.0] * 3)
        thickness = min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=64)
        # Thin dimension = 2.0 (2 * 1.0)
        assert jnp.isclose(thickness, 2.0, rtol=0.05), (
            f"thickness={float(thickness):.4f}, expected ~2.0"
        )

    def test_cube_gives_smallest_dimension(self) -> None:
        """Box half_extents=(2, 1.5, 1): min wall thickness ~2 (2*1)."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([2.0, 1.5, 1.0]),
        )
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        thickness = min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=64)
        # Smallest dimension = 2*1.0 = 2.0
        assert jnp.isclose(thickness, 2.0, rtol=0.05), (
            f"thickness={float(thickness):.4f}, expected ~2.0"
        )

    def test_sphere_gives_diameter(self) -> None:
        """Sphere r=1: min wall thickness ~2 (diameter)."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        thickness = min_wall_thickness(sphere.sdf, lo=lo, hi=hi, resolution=64)
        assert jnp.isclose(thickness, 2.0, rtol=0.05), (
            f"thickness={float(thickness):.4f}, expected ~2.0"
        )

    def test_differentiable(self) -> None:
        """jax.grad w.r.t. box half_extents works."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def thickness_of_extents(he: jnp.ndarray) -> jnp.ndarray:
            box = Box(center=jnp.zeros(3), half_extents=he)
            return min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=48)

        he = jnp.array([2.0, 1.5, 1.0])
        grad_he = jax.grad(thickness_of_extents)(he)
        assert jnp.all(jnp.isfinite(grad_he)), f"Non-finite gradient: {grad_he}"
        # Thinnest direction (z, half_extent=1.0) should have largest gradient
        assert float(grad_he[2]) > float(grad_he[0]), (
            f"z-gradient should dominate: {grad_he}"
        )
