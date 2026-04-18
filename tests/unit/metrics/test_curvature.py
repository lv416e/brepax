"""Unit tests for curvature field metrics."""

import jax
import jax.numpy as jnp

from brepax.brep.csg_eval import make_grid_3d
from brepax.metrics.curvature import (
    integrate_sdf_max_curvature,
    integrate_sdf_mean_curvature,
    max_curvature,
    mean_curvature,
)
from brepax.primitives import Box, Plane, Sphere


class TestIntegrateSdfMeanCurvature:
    """Tests for the low-level mean curvature integration."""

    def test_sphere_r1_curvature(self) -> None:
        """Sphere r=1: mean curvature = 2/R = 2.0."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        res = 32
        grid, _ = make_grid_3d(lo, hi, res)
        sdf = sphere.sdf(grid)
        kappa = integrate_sdf_mean_curvature(sphere.sdf, sdf, grid)
        expected = 2.0
        assert jnp.isclose(kappa, expected, rtol=0.10), (
            f"kappa={float(kappa):.4f}, expected={expected:.4f}"
        )

    def test_sphere_r2_curvature(self) -> None:
        """Sphere r=2: mean curvature = 2/R = 1.0."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(2.0))
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        res = 32
        grid, _ = make_grid_3d(lo, hi, res)
        sdf = sphere.sdf(grid)
        kappa = integrate_sdf_mean_curvature(sphere.sdf, sdf, grid)
        expected = 1.0
        assert jnp.isclose(kappa, expected, rtol=0.10), (
            f"kappa={float(kappa):.4f}, expected={expected:.4f}"
        )

    def test_plane_curvature_near_zero(self) -> None:
        """Plane: mean curvature = 0."""
        plane = Plane(
            normal=jnp.array([0.0, 0.0, 1.0]),
            offset=jnp.array(0.0),
        )
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        res = 32
        grid, _ = make_grid_3d(lo, hi, res)
        sdf = plane.sdf(grid)
        kappa = integrate_sdf_mean_curvature(plane.sdf, sdf, grid)
        assert jnp.isclose(kappa, 0.0, atol=0.1), (
            f"kappa={float(kappa):.4f}, expected ~0.0"
        )


class TestMeanCurvature:
    """Tests for the high-level mean_curvature function."""

    def test_sphere_matches_analytical(self) -> None:
        """mean_curvature(sphere.sdf) approximates 2/R."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        kappa = mean_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)
        expected = 2.0
        assert jnp.isclose(kappa, expected, rtol=0.10), (
            f"kappa={float(kappa):.4f}, expected={expected:.4f}"
        )

    def test_box_curvature_finite(self) -> None:
        """Box: curvature is finite (zero on faces, finite at edges)."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([1.0, 1.0, 1.0]),
        )
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        kappa = mean_curvature(box.sdf, lo=lo, hi=hi, resolution=32)
        assert jnp.isfinite(kappa)

    def test_differentiable_wrt_radius(self) -> None:
        """jax.grad of mean_curvature w.r.t. sphere radius is finite."""
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def kappa_of_radius(r: jnp.ndarray) -> jnp.ndarray:
            sphere = Sphere(center=jnp.zeros(3), radius=r)
            return mean_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)

        r = jnp.array(1.0)
        grad_r = jax.grad(kappa_of_radius)(r)
        assert jnp.isfinite(grad_r), f"Non-finite gradient: {grad_r}"

    def test_jit_compatible(self) -> None:
        """mean_curvature works under jax.jit."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)

        @jax.jit
        def compute(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
            return mean_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)

        kappa = compute(lo, hi)
        assert jnp.isfinite(kappa)


class TestIntegrateSdfMaxCurvature:
    """Tests for the low-level max curvature integration."""

    def test_sphere_r1_max_curvature(self) -> None:
        """Sphere r=1: max curvature ~ 2/R = 2.0 (uniform curvature)."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        res = 32
        grid, _ = make_grid_3d(lo, hi, res)
        sdf = sphere.sdf(grid)
        kappa_max = integrate_sdf_max_curvature(sphere.sdf, sdf, grid)
        expected = 2.0
        assert jnp.isclose(kappa_max, expected, rtol=0.10), (
            f"kappa_max={float(kappa_max):.4f}, expected={expected:.4f}"
        )


class TestMaxCurvature:
    """Tests for the high-level max_curvature function."""

    def test_sphere_matches_analytical(self) -> None:
        """max_curvature(sphere.sdf) approximates 2/R for uniform curvature."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        kappa_max = max_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)
        expected = 2.0
        assert jnp.isclose(kappa_max, expected, rtol=0.10), (
            f"kappa_max={float(kappa_max):.4f}, expected={expected:.4f}"
        )

    def test_differentiable_wrt_radius(self) -> None:
        """jax.grad of max_curvature w.r.t. sphere radius is finite."""
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def kappa_of_radius(r: jnp.ndarray) -> jnp.ndarray:
            sphere = Sphere(center=jnp.zeros(3), radius=r)
            return max_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)

        r = jnp.array(1.0)
        grad_r = jax.grad(kappa_of_radius)(r)
        assert jnp.isfinite(grad_r), f"Non-finite gradient: {grad_r}"

    def test_jit_compatible(self) -> None:
        """max_curvature works under jax.jit."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)

        @jax.jit
        def compute(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
            return max_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)

        kappa_max = compute(lo, hi)
        assert jnp.isfinite(kappa_max)
        assert float(kappa_max) > 0.0
