"""Unit tests for the surface area metric."""

import jax
import jax.numpy as jnp

from brepax.brep.csg_eval import make_grid_3d
from brepax.metrics.surface_area import integrate_sdf_surface_area, surface_area
from brepax.primitives import Box, Sphere


class TestIntegrateSdfSurfaceArea:
    """Tests for the low-level grid integration function."""

    def test_sphere_r1_res64(self) -> None:
        """Sphere r=1 surface area approximates 4*pi."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = sphere.sdf(grid)
        area = integrate_sdf_surface_area(sdf, lo, hi, 64)
        expected = 4.0 * jnp.pi
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )

    def test_box_unit_res64(self) -> None:
        """Unit box surface area approximates 6."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([0.5, 0.5, 0.5]),
        )
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        area = integrate_sdf_surface_area(sdf, lo, hi, 64)
        expected = 6.0
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )

    def test_sphere_r2_scales(self) -> None:
        """Sphere r=2 surface area approximates 16*pi."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(2.0))
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = sphere.sdf(grid)
        area = integrate_sdf_surface_area(sdf, lo, hi, 64)
        expected = 4.0 * jnp.pi * 4.0
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )


class TestSurfaceArea:
    """Tests for the high-level surface_area function."""

    def test_sphere_matches_analytical(self) -> None:
        """surface_area(sphere.sdf) approximates 4*pi*r^2."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        area = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=64)
        expected = 4.0 * jnp.pi
        assert jnp.isclose(area, expected, rtol=0.05)

    def test_box_matches_analytical(self) -> None:
        """surface_area(box.sdf) approximates 2*(wh+wl+hl)."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([2.0, 1.5, 1.0]),
        )
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        area = surface_area(box.sdf, lo=lo, hi=hi, resolution=64)
        # 2*(4*3 + 4*2 + 3*2) = 2*(12+8+6) = 52
        expected = 52.0
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )

    def test_resolution_convergence(self) -> None:
        """Higher resolution reduces error."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        expected = 4.0 * jnp.pi

        area_32 = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=32)
        area_64 = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=64)

        err_32 = jnp.abs(area_32 - expected) / expected
        err_64 = jnp.abs(area_64 - expected) / expected
        assert err_64 < err_32, (
            f"res=64 error ({float(err_64):.4f}) should be less than "
            f"res=32 error ({float(err_32):.4f})"
        )

    def test_differentiable_wrt_radius(self) -> None:
        """jax.grad of surface area w.r.t. sphere radius works."""
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def area_of_radius(r: jnp.ndarray) -> jnp.ndarray:
            sphere = Sphere(center=jnp.zeros(3), radius=r)
            return surface_area(sphere.sdf, lo=lo, hi=hi, resolution=48)

        r = jnp.array(1.0)
        grad_r = jax.grad(area_of_radius)(r)
        # d/dr(4*pi*r^2) = 8*pi*r = 8*pi at r=1
        expected_grad = 8.0 * jnp.pi
        assert jnp.isfinite(grad_r)
        assert jnp.isclose(grad_r, expected_grad, rtol=0.15), (
            f"grad={float(grad_r):.4f}, expected={float(expected_grad):.4f}"
        )

    def test_differentiable_wrt_center(self) -> None:
        """Surface area gradient w.r.t. center is near zero for symmetric domain."""
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def area_of_center(c: jnp.ndarray) -> jnp.ndarray:
            sphere = Sphere(center=c, radius=jnp.array(1.0))
            return surface_area(sphere.sdf, lo=lo, hi=hi, resolution=48)

        c = jnp.zeros(3)
        grad_c = jax.grad(area_of_center)(c)
        # Centered sphere in symmetric domain: gradient should be ~0
        assert jnp.allclose(grad_c, 0.0, atol=0.5), (
            f"grad_c={grad_c}, expected near zero"
        )

    def test_jit_compatible(self) -> None:
        """surface_area works under jax.jit."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)

        @jax.jit
        def compute(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
            return surface_area(sphere.sdf, lo=lo, hi=hi, resolution=32)

        area = compute(lo, hi)
        assert jnp.isfinite(area)
        assert float(area) > 0.0
