"""Unit tests for parametric trim bounds in B-spline SDF and projection."""

import jax
import jax.numpy as jnp
import pytest

from brepax.nurbs.projection import closest_point
from brepax.nurbs.sdf import bspline_sdf
from brepax.nurbs.trim import signed_distance_polygon, trim_indicator
from brepax.primitives import BSplineSurface


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


class TestClosestPointTrim:
    """Tests for closest_point with parametric trim bounds."""

    def test_no_trim_uses_full_domain(self) -> None:
        """Without trim bounds, projection uses the full knot domain."""
        pts, ku, kv, du, dv = _make_flat_patch()
        query = jnp.array([0.5, 0.5, 1.0])
        u, v = closest_point(query, pts, ku, kv, du, dv)
        assert jnp.isclose(u, 0.5, atol=0.05)
        assert jnp.isclose(v, 0.5, atol=0.05)

    def test_trim_clamps_projection(self) -> None:
        """With narrow trim bounds, projection is confined to the sub-region."""
        pts, ku, kv, du, dv = _make_flat_patch()
        # Query at center of full domain, but trim restricts to [0, 0.3]x[0, 0.3]
        query = jnp.array([0.5, 0.5, 1.0])
        u, v = closest_point(
            query,
            pts,
            ku,
            kv,
            du,
            dv,
            param_u_range=(0.0, 0.3),
            param_v_range=(0.0, 0.3),
        )
        assert float(u) <= 0.3 + 1e-6
        assert float(v) <= 0.3 + 1e-6

    def test_trim_does_not_affect_interior_point(self) -> None:
        """Trim bounds that contain the optimal point do not change the result."""
        pts, ku, kv, du, dv = _make_flat_patch()
        query = jnp.array([0.2, 0.2, 1.0])
        u_full, v_full = closest_point(query, pts, ku, kv, du, dv)
        u_trim, v_trim = closest_point(
            query,
            pts,
            ku,
            kv,
            du,
            dv,
            param_u_range=(0.0, 0.5),
            param_v_range=(0.0, 0.5),
        )
        assert jnp.isclose(u_full, u_trim, atol=1e-4)
        assert jnp.isclose(v_full, v_trim, atol=1e-4)


class TestBsplineSdfTrim:
    """Tests for bspline_sdf with parametric trim bounds."""

    def test_trimmed_sdf_differs_for_exterior_point(self) -> None:
        """SDF with tight trim should differ from untrimmed for a far query."""
        pts, ku, kv, du, dv = _make_flat_patch()
        # Query near u=0.8, v=0.8 -- outside the trimmed region
        query = jnp.array([0.8, 0.8, 1.0])
        dist_full = bspline_sdf(query, pts, ku, kv, du, dv)
        dist_trim = bspline_sdf(
            query,
            pts,
            ku,
            kv,
            du,
            dv,
            param_u_range=(0.0, 0.3),
            param_v_range=(0.0, 0.3),
        )
        # Trimmed SDF should be larger (closest surface point is farther)
        assert float(jnp.abs(dist_trim)) > float(jnp.abs(dist_full))

    def test_trimmed_sdf_unchanged_for_interior_point(self) -> None:
        """SDF with wide trim is unchanged for a point inside the region."""
        pts, ku, kv, du, dv = _make_flat_patch()
        query = jnp.array([0.2, 0.2, 1.0])
        dist_full = bspline_sdf(query, pts, ku, kv, du, dv)
        dist_trim = bspline_sdf(
            query,
            pts,
            ku,
            kv,
            du,
            dv,
            param_u_range=(0.0, 0.5),
            param_v_range=(0.0, 0.5),
        )
        assert jnp.isclose(dist_full, dist_trim, atol=1e-4)


class TestBSplineSurfaceTrim:
    """Tests for BSplineSurface primitive with trim bounds."""

    def test_default_trim_is_none(self) -> None:
        """Without trim bounds, param ranges default to None."""
        pts, ku, kv, du, dv = _make_flat_patch()
        surf = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
        )
        assert surf.param_u_range is None
        assert surf.param_v_range is None

    def test_trim_bounds_stored(self) -> None:
        """Trim bounds are stored correctly on the primitive."""
        pts, ku, kv, du, dv = _make_flat_patch()
        surf = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
            param_u_range=(0.1, 0.9),
            param_v_range=(0.2, 0.8),
        )
        assert surf.param_u_range == (0.1, 0.9)
        assert surf.param_v_range == (0.2, 0.8)

    @pytest.mark.slow
    def test_sdf_respects_trim(self) -> None:
        """BSplineSurface.sdf() with trim bounds restricts the projection."""
        pts, ku, kv, du, dv = _make_flat_patch()
        query = jnp.array([0.8, 0.8, 1.0])

        surf_full = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
        )
        surf_trim = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
            param_u_range=(0.0, 0.3),
            param_v_range=(0.0, 0.3),
        )
        dist_full = surf_full.sdf(query)
        dist_trim = surf_trim.sdf(query)
        assert float(jnp.abs(dist_trim)) > float(jnp.abs(dist_full))


# -- Level 2: 2D signed distance to polygon + trim indicator --


def _unit_square() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unit square polygon with vertices at (0,0), (1,0), (1,1), (0,1)."""
    verts = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return verts, jnp.ones(4)


def _triangle() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unit triangle: (0,0), (1,0), (0,1)."""
    verts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return verts, jnp.ones(3)


def _l_shape() -> tuple[jnp.ndarray, jnp.ndarray]:
    """L-shaped (non-convex) polygon."""
    verts = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
            [0.0, 1.0],
        ]
    )
    return verts, jnp.ones(6)


class TestSignedDistancePolygon:
    """Tests for signed_distance_polygon in 2D parameter space."""

    def test_inside_is_negative(self) -> None:
        verts, mask = _unit_square()
        d = signed_distance_polygon(jnp.array([0.5, 0.5]), verts, mask)
        assert float(d) < 0.0

    def test_outside_is_positive(self) -> None:
        verts, mask = _unit_square()
        d = signed_distance_polygon(jnp.array([1.5, 0.5]), verts, mask)
        assert float(d) > 0.0

    def test_on_edge_is_zero(self) -> None:
        verts, mask = _unit_square()
        d = signed_distance_polygon(jnp.array([0.5, 0.0]), verts, mask)
        assert abs(float(d)) < 1e-6

    def test_distance_to_nearest_edge(self) -> None:
        """Distance from (0.1, 0.5) to the left edge (x=0) is 0.1."""
        verts, mask = _unit_square()
        d = signed_distance_polygon(jnp.array([0.1, 0.5]), verts, mask)
        assert jnp.isclose(d, -0.1, atol=1e-6)

    def test_triangle_inside(self) -> None:
        verts, mask = _triangle()
        d = signed_distance_polygon(jnp.array([0.2, 0.2]), verts, mask)
        assert float(d) < 0.0

    def test_triangle_outside_hypotenuse(self) -> None:
        verts, mask = _triangle()
        d = signed_distance_polygon(jnp.array([0.8, 0.8]), verts, mask)
        assert float(d) > 0.0

    def test_nonconvex_inside_lower_arm(self) -> None:
        verts, mask = _l_shape()
        d = signed_distance_polygon(jnp.array([0.8, 0.25]), verts, mask)
        assert float(d) < 0.0

    def test_nonconvex_inside_upper_arm(self) -> None:
        verts, mask = _l_shape()
        d = signed_distance_polygon(jnp.array([0.25, 0.75]), verts, mask)
        assert float(d) < 0.0

    def test_nonconvex_outside_concave_corner(self) -> None:
        """Point in the concave notch of the L is outside."""
        verts, mask = _l_shape()
        d = signed_distance_polygon(jnp.array([0.75, 0.75]), verts, mask)
        assert float(d) > 0.0

    def test_padding_does_not_affect_result(self) -> None:
        verts, mask = _triangle()
        d_orig = signed_distance_polygon(jnp.array([0.2, 0.2]), verts, mask)
        # Pad to 6 vertices
        padded = jnp.concatenate([verts, jnp.zeros((3, 2))])
        mask_padded = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        d_padded = signed_distance_polygon(jnp.array([0.2, 0.2]), padded, mask_padded)
        assert jnp.isclose(d_orig, d_padded, atol=1e-6)


class TestTrimIndicator:
    """Tests for the sigmoid-based trim indicator function."""

    def test_inside_is_one(self) -> None:
        verts, mask = _unit_square()
        val = trim_indicator(jnp.array([0.5, 0.5]), verts, mask)
        assert float(val) > 0.99

    def test_outside_is_zero(self) -> None:
        verts, mask = _unit_square()
        val = trim_indicator(jnp.array([1.5, 0.5]), verts, mask)
        assert float(val) < 0.01

    def test_boundary_is_half(self) -> None:
        """On the polygon edge, the indicator should be ~0.5."""
        verts, mask = _unit_square()
        val = trim_indicator(jnp.array([0.5, 0.0]), verts, mask)
        assert 0.4 < float(val) < 0.6

    def test_jit_compatible(self) -> None:
        verts, mask = _unit_square()
        jitted = jax.jit(lambda p: trim_indicator(p, verts, mask))
        val = jitted(jnp.array([0.5, 0.5]))
        assert float(val) > 0.99

    def test_vmap_compatible(self) -> None:
        verts, mask = _unit_square()
        points = jnp.array([[0.5, 0.5], [1.5, 0.5], [0.5, 0.0]])
        vals = jax.vmap(lambda p: trim_indicator(p, verts, mask))(points)
        assert float(vals[0]) > 0.99
        assert float(vals[1]) < 0.01
        assert 0.4 < float(vals[2]) < 0.6

    def test_grad_nonzero_near_boundary(self) -> None:
        """Gradient should be nonzero near the polygon boundary."""
        verts, mask = _unit_square()
        grad_fn = jax.grad(lambda p: trim_indicator(p, verts, mask))
        g = grad_fn(jnp.array([0.01, 0.5]))
        assert jnp.any(jnp.abs(g) > 1.0)

    def test_grad_zero_far_inside(self) -> None:
        """Gradient vanishes far from the boundary (sigmoid saturated)."""
        verts, mask = _unit_square()
        grad_fn = jax.grad(lambda p: trim_indicator(p, verts, mask))
        g = grad_fn(jnp.array([0.5, 0.5]))
        assert jnp.all(jnp.abs(g) < 1e-4)


class TestBSplineSurfaceTrimPolygon:
    """Tests for BSplineSurface with trim polygon fields."""

    def test_default_trim_polygon_is_none(self) -> None:
        pts, ku, kv, du, dv = _make_flat_patch()
        surf = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
        )
        assert surf.trim_polygon is None
        assert surf.trim_mask is None

    def test_trim_polygon_stored(self) -> None:
        pts, ku, kv, du, dv = _make_flat_patch()
        tri = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mask = jnp.ones(3)
        surf = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
            trim_polygon=tri,
            trim_mask=mask,
        )
        assert surf.trim_polygon is not None
        assert surf.trim_polygon.shape == (3, 2)
        assert surf.trim_mask is not None

    @pytest.mark.slow
    def test_sdf_unchanged_with_trim_polygon(self) -> None:
        """SDF method is not affected by trim_polygon (it's stored for metrics)."""
        pts, ku, kv, du, dv = _make_flat_patch()
        query = jnp.array([0.5, 0.5, 1.0])
        surf_no_trim = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
        )
        tri = jnp.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.3]])
        mask = jnp.ones(3)
        surf_with_trim = BSplineSurface(
            control_points=pts,
            knots_u=ku,
            knots_v=kv,
            degree_u=du,
            degree_v=dv,
            trim_polygon=tri,
            trim_mask=mask,
        )
        d1 = surf_no_trim.sdf(query)
        d2 = surf_with_trim.sdf(query)
        assert jnp.isclose(d1, d2, atol=1e-6)
