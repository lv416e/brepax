"""Unit tests for B-spline surface evaluation."""

import jax
import jax.numpy as jnp

from brepax.nurbs.evaluate import (
    bspline_basis,
    evaluate_surface,
    evaluate_surface_derivs,
)


class TestBsplineBasis:
    """Tests for Cox-de Boor basis function evaluation."""

    def test_partition_of_unity(self) -> None:
        """Basis functions sum to 1 at any interior parameter value."""
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
        for t_val in [0.1, 0.25, 0.5, 0.75, 0.99]:
            vals = bspline_basis(jnp.array(t_val), knots, degree=3, n_basis=5)
            assert jnp.isclose(jnp.sum(vals), 1.0, atol=1e-10), (
                f"t={t_val}: sum={float(jnp.sum(vals))}"
            )

    def test_endpoints_clamped(self) -> None:
        """First basis = 1 at t=0, last basis = 1 at t=1 for clamped knots."""
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        vals_start = bspline_basis(jnp.array(0.0), knots, degree=3, n_basis=4)
        assert jnp.isclose(vals_start[0], 1.0)
        assert jnp.allclose(vals_start[1:], 0.0, atol=1e-10)

        vals_end = bspline_basis(jnp.array(1.0), knots, degree=3, n_basis=4)
        assert jnp.isclose(vals_end[-1], 1.0)
        assert jnp.allclose(vals_end[:-1], 0.0, atol=1e-10)

    def test_degree_one_linear(self) -> None:
        """Degree 1 with uniform knots gives linear interpolation."""
        knots = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0])
        vals = bspline_basis(jnp.array(0.25), knots, degree=1, n_basis=3)
        assert jnp.isclose(vals[0], 0.5, atol=1e-6)
        assert jnp.isclose(vals[1], 0.5, atol=1e-6)
        assert jnp.isclose(vals[2], 0.0, atol=1e-6)

    def test_non_negative(self) -> None:
        """Basis functions are non-negative."""
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
        for t_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            vals = bspline_basis(jnp.array(t_val), knots, degree=3, n_basis=5)
            assert jnp.all(vals >= -1e-10), f"t={t_val}: {vals}"


class TestEvaluateSurface:
    """Tests for B-spline surface evaluation."""

    def test_flat_patch_corners(self) -> None:
        """Bilinear patch evaluates to control points at corners."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        p00 = evaluate_surface(pts, knots, knots, 1, 1, jnp.array(0.0), jnp.array(0.0))
        p11 = evaluate_surface(pts, knots, knots, 1, 1, jnp.array(1.0), jnp.array(1.0))
        assert jnp.allclose(p00, jnp.array([0.0, 0.0, 0.0]), atol=1e-10)
        assert jnp.allclose(p11, jnp.array([1.0, 1.0, 0.0]), atol=1e-10)

    def test_flat_patch_midpoint(self) -> None:
        """Bilinear patch midpoint is average of corners."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        mid = evaluate_surface(pts, knots, knots, 1, 1, jnp.array(0.5), jnp.array(0.5))
        assert jnp.allclose(mid, jnp.array([1.0, 1.0, 0.0]), atol=1e-10)

    def test_curved_patch_midpoint(self) -> None:
        """Cubic patch with raised center control point lifts the surface."""
        pts = jnp.zeros((4, 4, 3))
        for i in range(4):
            for j in range(4):
                pts = pts.at[i, j, 0].set(i / 3.0)
                pts = pts.at[i, j, 1].set(j / 3.0)
        pts = pts.at[1, 1, 2].set(1.0)
        pts = pts.at[1, 2, 2].set(1.0)
        pts = pts.at[2, 1, 2].set(1.0)
        pts = pts.at[2, 2, 2].set(1.0)

        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        mid = evaluate_surface(pts, knots, knots, 3, 3, jnp.array(0.5), jnp.array(0.5))
        assert float(mid[2]) > 0.3, f"Expected lifted z, got {float(mid[2]):.4f}"

    def test_jit_compatible(self) -> None:
        """Surface evaluation works under jax.jit."""
        pts = jnp.zeros((4, 4, 3))
        knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        @jax.jit
        def eval_at(u, v):
            return evaluate_surface(pts, knots, knots, 3, 3, u, v)

        result = eval_at(jnp.array(0.5), jnp.array(0.5))
        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))


class TestEvaluateSurfaceDerivs:
    """Tests for surface derivative computation."""

    def test_flat_patch_zero_z_derivative(self) -> None:
        """Flat xy-plane patch has zero z-derivative."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        _, du, dv = evaluate_surface_derivs(
            pts, knots, knots, 1, 1, jnp.array(0.5), jnp.array(0.5)
        )
        assert jnp.isclose(du[2], 0.0, atol=1e-6)
        assert jnp.isclose(dv[2], 0.0, atol=1e-6)

    def test_bilinear_patch_tangents(self) -> None:
        """Bilinear patch tangent vectors are constant."""
        pts = jnp.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            ]
        )
        knots = jnp.array([0.0, 0.0, 1.0, 1.0])
        _, du, dv = evaluate_surface_derivs(
            pts, knots, knots, 1, 1, jnp.array(0.3), jnp.array(0.7)
        )
        assert jnp.allclose(du, jnp.array([0.0, 2.0, 0.0]), atol=1e-4)
        assert jnp.allclose(dv, jnp.array([2.0, 0.0, 0.0]), atol=1e-4)
