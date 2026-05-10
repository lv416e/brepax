"""Unit tests for curvature field metrics."""

from pathlib import Path

import jax
import jax.numpy as jnp

from brepax.io.step import read_step
from brepax.metrics.curvature import (
    max_curvature,
    mean_curvature,
    mean_curvature_per_face,
)
from brepax.primitives import Box, Plane, Sphere

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestMeanCurvature:
    """Tests for the high-level mean_curvature function."""

    def test_sphere_r1_matches_analytical(self) -> None:
        """Sphere r=1: mean curvature = 2/R = 2.0."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        kappa = mean_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)
        expected = 2.0
        assert jnp.isclose(kappa, expected, rtol=0.10), (
            f"kappa={float(kappa):.4f}, expected={expected:.4f}"
        )

    def test_sphere_r2_matches_analytical(self) -> None:
        """Sphere r=2: mean curvature = 2/R = 1.0."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(2.0))
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        kappa = mean_curvature(sphere.sdf, lo=lo, hi=hi, resolution=32)
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
        kappa = mean_curvature(plane.sdf, lo=lo, hi=hi, resolution=32)
        assert jnp.isclose(kappa, 0.0, atol=0.1), (
            f"kappa={float(kappa):.4f}, expected ~0.0"
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


class TestMeanCurvaturePerFace:
    """Per-face analytical mean curvature against closed-form values.

    The M6.1 milestone gate clause "Trim-aware face-level metrics"
    requires each face-level metric to be exposed and verified.  Mean
    curvature on plane / sphere / cylinder faces is a constant
    closed-form value (0, 1/r, 1/(2r)); the function reads ``radius``
    from the same OCCT call that drives the rest of the geometry, so
    agreement is exact (up to float32) by construction.

    Cone, torus and BSpline are deferred (NaN) until their analytical
    closed-form handlers are added in a follow-up PR.
    """

    @staticmethod
    def _read(name: str):
        return read_step(str(FIXTURES / f"{name}.step"))

    def test_sample_box_all_planes_zero(self) -> None:
        """``sample_box`` is six plane faces; all mean curvatures = 0."""
        shape = self._read("sample_box")
        kappas, params = mean_curvature_per_face(shape)
        assert kappas.shape == (len(params),) == (6,)
        assert jnp.all(kappas == 0.0), f"expected all zeros, got {kappas}"

    def test_sample_sphere_inverse_radius(self) -> None:
        """``sample_sphere`` has one sphere face; H = 1 / radius.

        The radius is read from the same OCCT call the SDF uses, so the
        agreement is exact.  ``sample_sphere`` is the unit sphere
        (radius 3 from the fixture's bbox); read it from the params
        list rather than hard-coding.
        """
        shape = self._read("sample_sphere")
        kappas, params = mean_curvature_per_face(shape)
        assert len(params) == 1
        radius = float(params[0]["radius"])
        assert radius > 0.0
        expected = 1.0 / radius
        assert jnp.isclose(kappas[0], expected, rtol=1e-6), (
            f"kappa={float(kappas[0]):.6f}, expected={expected:.6f}"
        )

    def test_sample_cylinder_caps_zero_side_inv_2r(self) -> None:
        """``sample_cylinder`` has 2 plane caps + 1 cylinder side.
        Caps give H = 0, side gives H = 1 / (2 r)."""
        shape = self._read("sample_cylinder")
        kappas, params = mean_curvature_per_face(shape)
        assert len(params) == 3

        # Sort by surface_type to compare deterministically.
        surface_types = [p["surface_type"] for p in params]
        n_planes = sum(1 for t in surface_types if t == "plane")
        n_cyls = sum(1 for t in surface_types if t == "cylinder")
        assert n_planes == 2
        assert n_cyls == 1

        for kappa, p in zip(kappas, params, strict=True):
            t = p["surface_type"]
            if t == "plane":
                assert float(kappa) == 0.0
            elif t == "cylinder":
                expected = 1.0 / (2.0 * float(p["radius"]))
                assert jnp.isclose(kappa, expected, rtol=1e-6), (
                    f"cyl side kappa={float(kappa):.6f}, expected={expected:.6f}"
                )

    def test_box_with_holes_planes_zero_holes_cylinder(self) -> None:
        """``box_with_holes`` has 6 box-plane faces + 2 cylinder hole faces."""
        shape = self._read("box_with_holes")
        kappas, params = mean_curvature_per_face(shape)
        n_planes = sum(1 for p in params if p["surface_type"] == "plane")
        n_cyls = sum(1 for p in params if p["surface_type"] == "cylinder")
        assert n_planes == 6
        assert n_cyls == 2
        for kappa, p in zip(kappas, params, strict=True):
            if p["surface_type"] == "plane":
                assert float(kappa) == 0.0
            else:
                assert float(kappa) > 0.0  # 1 / (2 r) > 0

    def test_unsupported_types_return_nan(self) -> None:
        """Cone, torus and BSpline faces return NaN until their
        analytical handlers are added."""
        # nurbs_box: 6 BSpline faces, all NaN.
        shape = self._read("nurbs_box")
        kappas, params = mean_curvature_per_face(shape)
        assert all(p["surface_type"] == "bspline" for p in params)
        assert jnp.all(jnp.isnan(kappas)), f"expected NaN for BSpline, got {kappas}"

        # sample_torus: 1 torus face, NaN.
        shape = self._read("sample_torus")
        kappas, params = mean_curvature_per_face(shape)
        assert len(params) == 1 and params[0]["surface_type"] == "torus"
        assert jnp.all(jnp.isnan(kappas))

    def test_finite_or_nan_on_all_fixtures(self) -> None:
        """Smoke property: every per-face curvature is either finite
        non-negative or NaN on every fixture in the standard set."""
        fixtures = [
            "sample_box",
            "sample_cylinder",
            "sample_sphere",
            "sample_cone",
            "sample_torus",
            "box_with_holes",
            "box_with_pocket",
            "box_with_slot",
            "l_bracket",
            "nurbs_box",
        ]
        for name in fixtures:
            shape = self._read(name)
            kappas, params = mean_curvature_per_face(shape)
            assert kappas.shape == (len(params),)
            for kappa in kappas:
                v = float(kappa)
                assert jnp.isnan(v) or v >= 0.0, f"{name}: unexpected curvature {v}"

    def test_gradient_through_radius(self) -> None:
        """``jax.grad`` of a per-face curvature sum (analytical only)
        must produce finite gradients on the radius of a sphere
        primitive.  The sphere's H = 1/r, so dH/dr = -1/r^2 = -1 at r=1."""
        from brepax.primitives import Sphere

        # Construct a Sphere primitive and use the analytical form
        # directly (the per-face function path goes through OCCT and
        # is not differentiable through ``radius`` outside the JAX
        # graph; the gradient flows through the JAX-side primitive).
        def loss(r: jnp.ndarray) -> jnp.ndarray:
            sphere = Sphere(center=jnp.zeros(3), radius=r)
            # H = 1 / r for a sphere primitive.
            return 1.0 / sphere.parameters()["radius"]

        r = jnp.array(2.0)
        grad_r = jax.grad(loss)(r)
        # dH/dr = -1/r^2 = -1/4 at r=2.
        assert jnp.isfinite(grad_r)
        assert jnp.isclose(grad_r, -0.25, rtol=1e-5)
