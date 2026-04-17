"""Unit tests for B-spline SDF computation and differentiability."""

import jax
import jax.numpy as jnp

from brepax.nurbs.sdf import bspline_sdf


def _make_flat_patch():
    """Bilinear patch on the xy-plane: z=0, x in [0,1], y in [0,1]."""
    pts = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    knots = jnp.array([0.0, 0.0, 1.0, 1.0])
    return pts, knots, knots, 1, 1


def _make_curved_patch():
    """Cubic patch with raised center: dome shape."""
    pts = jnp.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            pts = pts.at[i, j, 0].set(i / 3.0)
            pts = pts.at[i, j, 1].set(j / 3.0)
    pts = pts.at[1, 1, 2].set(0.5)
    pts = pts.at[1, 2, 2].set(0.5)
    pts = pts.at[2, 1, 2].set(0.5)
    pts = pts.at[2, 2, 2].set(0.5)
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return pts, knots, knots, 3, 3


class TestBsplineSdf:
    """Tests for B-spline SDF computation."""

    def test_flat_patch_sign_above(self) -> None:
        """Point above flat patch has nonzero signed distance."""
        pts, ku, kv, du, dv = _make_flat_patch()
        dist = bspline_sdf(jnp.array([0.5, 0.5, 1.0]), pts, ku, kv, du, dv)
        assert jnp.isclose(jnp.abs(dist), 1.0, rtol=0.05), f"dist={float(dist)}"

    def test_flat_patch_sign_flips(self) -> None:
        """Points on opposite sides of flat patch have opposite signs."""
        pts, ku, kv, du, dv = _make_flat_patch()
        above = bspline_sdf(jnp.array([0.5, 0.5, 1.0]), pts, ku, kv, du, dv)
        below = bspline_sdf(jnp.array([0.5, 0.5, -1.0]), pts, ku, kv, du, dv)
        assert float(above) * float(below) < 0.0, (
            f"Expected opposite signs: above={float(above)}, below={float(below)}"
        )

    def test_flat_patch_distance_accuracy(self) -> None:
        """Distance to flat patch equals z-coordinate of query."""
        pts, ku, kv, du, dv = _make_flat_patch()
        dist = bspline_sdf(jnp.array([0.5, 0.5, 2.0]), pts, ku, kv, du, dv)
        assert jnp.isclose(jnp.abs(dist), 2.0, rtol=0.05)

    def test_curved_patch_closer_to_dome(self) -> None:
        """Point above dome center is closer than z-height suggests."""
        pts, ku, kv, du, dv = _make_curved_patch()
        dist = bspline_sdf(jnp.array([0.5, 0.5, 1.0]), pts, ku, kv, du, dv)
        # Dome rises to ~0.25 at center, so distance < 1.0
        assert float(dist) > 0.0
        assert float(dist) < 1.0, f"Expected < 1.0, got {float(dist)}"


class TestBsplineSdfGradient:
    """Tests for SDF gradient w.r.t. control points."""

    def test_gradient_wrt_control_points_finite(self) -> None:
        """jax.grad of SDF w.r.t. control points produces finite values."""
        pts, ku, kv, du, dv = _make_flat_patch()
        query = jnp.array([0.5, 0.5, 1.0])

        def sdf_of_pts(p):
            return bspline_sdf(query, p, ku, kv, du, dv)

        grad_pts = jax.grad(sdf_of_pts)(pts)
        assert jnp.all(jnp.isfinite(grad_pts)), f"Non-finite gradient: {grad_pts}"

    def test_gradient_matches_finite_difference(self) -> None:
        """Autodiff gradient matches central finite difference."""
        pts, ku, kv, du, dv = _make_curved_patch()
        query = jnp.array([0.5, 0.5, 1.0])

        def sdf_of_pts(p):
            return bspline_sdf(query, p, ku, kv, du, dv)

        grad_ad = jax.grad(sdf_of_pts)(pts)

        eps = 1e-5
        # Check gradient for one control point: pts[2, 2, 2] (z of interior)
        pts_fwd = pts.at[2, 2, 2].add(eps)
        pts_bwd = pts.at[2, 2, 2].add(-eps)
        grad_fd = (sdf_of_pts(pts_fwd) - sdf_of_pts(pts_bwd)) / (2.0 * eps)
        grad_ad_val = grad_ad[2, 2, 2]

        assert jnp.isclose(grad_ad_val, grad_fd, rtol=0.05), (
            f"AD={float(grad_ad_val):.6f}, FD={float(grad_fd):.6f}"
        )

    def test_raising_control_point_reduces_distance(self) -> None:
        """Raising a control point toward the query reduces SDF."""
        pts, ku, kv, du, dv = _make_curved_patch()
        query = jnp.array([0.5, 0.5, 1.0])

        def sdf_of_pts(p):
            return bspline_sdf(query, p, ku, kv, du, dv)

        grad_pts = jax.grad(sdf_of_pts)(pts)
        # Interior z-components should have negative gradient
        # (raising the surface toward query decreases distance)
        assert float(grad_pts[1, 1, 2]) < 0.0, (
            f"Expected negative z-grad, got {float(grad_pts[1, 1, 2])}"
        )


def _make_saddle_patch():
    """Cubic saddle surface: z ~ (x-0.5)^2 - (y-0.5)^2."""
    pts = jnp.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            x, y = i / 3.0, j / 3.0
            pts = pts.at[i, j, 0].set(x)
            pts = pts.at[i, j, 1].set(y)
            pts = pts.at[i, j, 2].set(0.3 * ((x - 0.5) ** 2 - (y - 0.5) ** 2))
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return pts, knots, knots, 3, 3


def _make_bumpy_patch():
    """Cubic patch with alternating peaks and valleys."""
    pts = jnp.zeros((4, 4, 3))
    heights = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, -0.3, 0.0],
            [0.0, -0.3, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    for i in range(4):
        for j in range(4):
            pts = pts.at[i, j, 0].set(i / 3.0)
            pts = pts.at[i, j, 1].set(j / 3.0)
            pts = pts.at[i, j, 2].set(heights[i, j])
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return pts, knots, knots, 3, 3


class TestNonConvexSdf:
    """Tests for SDF on non-convex B-spline surfaces."""

    def test_saddle_sign_flips(self) -> None:
        """Points on opposite sides of saddle have opposite signs."""
        pts, ku, kv, du, dv = _make_saddle_patch()
        above = bspline_sdf(jnp.array([0.5, 0.5, 0.5]), pts, ku, kv, du, dv)
        below = bspline_sdf(jnp.array([0.5, 0.5, -0.5]), pts, ku, kv, du, dv)
        assert float(above) * float(below) < 0.0, (
            f"Expected opposite signs: above={float(above)}, below={float(below)}"
        )

    def test_saddle_gradient_finite(self) -> None:
        """jax.grad on saddle surface produces finite values."""
        pts, ku, kv, du, dv = _make_saddle_patch()

        def sdf_fn(p):
            return bspline_sdf(jnp.array([0.5, 0.5, 0.5]), p, ku, kv, du, dv)

        grad = jax.grad(sdf_fn)(pts)
        assert jnp.all(jnp.isfinite(grad)), "Non-finite gradient"

    def test_saddle_gradient_fd_match(self) -> None:
        """AD gradient matches finite difference on saddle surface."""
        pts, ku, kv, du, dv = _make_saddle_patch()

        def sdf_fn(p):
            return bspline_sdf(jnp.array([0.5, 0.5, 0.5]), p, ku, kv, du, dv)

        grad_ad = jax.grad(sdf_fn)(pts)
        eps = 1e-5
        pts_fwd = pts.at[2, 2, 2].add(eps)
        pts_bwd = pts.at[2, 2, 2].add(-eps)
        grad_fd = (sdf_fn(pts_fwd) - sdf_fn(pts_bwd)) / (2.0 * eps)
        assert jnp.isclose(grad_ad[2, 2, 2], grad_fd, rtol=0.10), (
            f"AD={float(grad_ad[2, 2, 2]):.6f}, FD={float(grad_fd):.6f}"
        )

    def test_bumpy_gradient_finite(self) -> None:
        """jax.grad on bumpy surface produces finite values."""
        pts, ku, kv, du, dv = _make_bumpy_patch()

        def sdf_fn(p):
            return bspline_sdf(jnp.array([0.5, 0.5, 0.5]), p, ku, kv, du, dv)

        grad = jax.grad(sdf_fn)(pts)
        assert jnp.all(jnp.isfinite(grad)), "Non-finite gradient"

    def test_bumpy_gradient_fd_match(self) -> None:
        """AD gradient matches finite difference on bumpy surface."""
        pts, ku, kv, du, dv = _make_bumpy_patch()

        def sdf_fn(p):
            return bspline_sdf(jnp.array([0.3, 0.3, 0.5]), p, ku, kv, du, dv)

        grad_ad = jax.grad(sdf_fn)(pts)
        eps = 1e-5
        pts_fwd = pts.at[1, 1, 2].add(eps)
        pts_bwd = pts.at[1, 1, 2].add(-eps)
        grad_fd = (sdf_fn(pts_fwd) - sdf_fn(pts_bwd)) / (2.0 * eps)
        assert jnp.isclose(grad_ad[1, 1, 2], grad_fd, rtol=0.10), (
            f"AD={float(grad_ad[1, 1, 2]):.6f}, FD={float(grad_fd):.6f}"
        )

    def test_bumpy_distance_vs_brute_force(self) -> None:
        """SDF distance matches brute-force parameter-space search."""
        from brepax.nurbs.evaluate import evaluate_surface

        pts, ku, kv, du, dv = _make_bumpy_patch()
        query = jnp.array([0.3, 0.3, 0.5])

        # Brute-force: sample parameter space densely
        us = jnp.linspace(0.0, 1.0, 50)
        vs = jnp.linspace(0.0, 1.0, 50)
        min_dist = jnp.inf
        for u_val in us:
            for v_val in vs:
                pt = evaluate_surface(pts, ku, kv, du, dv, u_val, v_val)
                d = jnp.linalg.norm(query - pt)
                min_dist = jnp.minimum(min_dist, d)

        sdf_val = bspline_sdf(query, pts, ku, kv, du, dv)
        assert jnp.isclose(jnp.abs(sdf_val), min_dist, rtol=0.05), (
            f"SDF={float(jnp.abs(sdf_val)):.4f}, brute={float(min_dist):.4f}"
        )
