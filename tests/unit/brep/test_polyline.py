"""Unsigned distance from a 3D point to a padded closed polyline."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.brep.polyline import point_segment_distance_3d, polyline_unsigned_distance


class TestPointSegmentDistance3D:
    """Distances against closed-form ground truth on a single segment."""

    def test_foot_interior(self) -> None:
        # Segment on the x-axis from (0,0,0) to (2,0,0); query above midpoint.
        d = point_segment_distance_3d(
            jnp.array([1.0, 3.0, 4.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([2.0, 0.0, 0.0]),
        )
        assert jnp.isclose(d, 5.0, atol=1e-6)

    def test_foot_clamped_at_start(self) -> None:
        # Query projects before a; closest point is a itself.
        d = point_segment_distance_3d(
            jnp.array([-3.0, 4.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([2.0, 0.0, 0.0]),
        )
        assert jnp.isclose(d, 5.0, atol=1e-6)

    def test_foot_clamped_at_end(self) -> None:
        # Query projects past b; closest point is b itself.
        d = point_segment_distance_3d(
            jnp.array([5.0, 0.0, 4.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([2.0, 0.0, 0.0]),
        )
        expected = math.hypot(3.0, 4.0)
        assert jnp.isclose(d, expected, atol=1e-6)

    def test_on_segment_returns_zero(self) -> None:
        d = point_segment_distance_3d(
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([2.0, 0.0, 0.0]),
        )
        assert jnp.isclose(d, 0.0, atol=1e-6)

    def test_degenerate_segment(self) -> None:
        # a == b: distance collapses to point-to-vertex distance.
        d = point_segment_distance_3d(
            jnp.array([3.0, 4.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
        )
        assert jnp.isclose(d, 5.0, atol=1e-6)

    def test_grad_finite_on_segment(self) -> None:
        # Query exactly on the segment — sqrt derivative is a kink but
        # the safe-square pattern must keep the backward finite.
        def loss(params):
            d = point_segment_distance_3d(params["p"], params["a"], params["b"])
            return d

        g = jax.grad(loss)(
            {
                "p": jnp.array([1.0, 0.0, 0.0]),
                "a": jnp.array([0.0, 0.0, 0.0]),
                "b": jnp.array([2.0, 0.0, 0.0]),
            }
        )
        assert jnp.all(jnp.isfinite(g["p"]))
        assert jnp.all(jnp.isfinite(g["a"]))
        assert jnp.all(jnp.isfinite(g["b"]))

    def test_grad_finite_degenerate_segment(self) -> None:
        # Zero-length segment with a finite-distance query.
        def loss(params):
            return point_segment_distance_3d(
                jnp.array([3.0, 4.0, 0.0]), params["a"], params["b"]
            )

        g = jax.grad(loss)({"a": jnp.zeros(3), "b": jnp.zeros(3)})
        assert jnp.all(jnp.isfinite(g["a"]))
        assert jnp.all(jnp.isfinite(g["b"]))


def _triangle_loop() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Right triangle (0,0,0) - (1,0,0) - (0,1,0) padded to 5 slots."""
    verts = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
    return verts, mask


def _unit_square_loop() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unit square at z=0 with no padding."""
    verts = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    mask = jnp.array([1.0, 1.0, 1.0, 1.0])
    return verts, mask


class TestPolylineUnsignedDistance:
    """Closed-form checks for simple trim-loop fixtures."""

    def test_vertex_distance_is_zero(self) -> None:
        # Distance from a vertex to the loop must be zero.
        verts, mask = _triangle_loop()
        d = polyline_unsigned_distance(verts[1], verts, mask)
        assert jnp.isclose(d, 0.0, atol=1e-6)

    def test_square_above_center(self) -> None:
        # Query at z=3 above the centroid; nearest loop edge midpoint
        # is at radial distance 0.5 in the plane, so distance = sqrt(0.25 + 9).
        verts, mask = _unit_square_loop()
        d = polyline_unsigned_distance(jnp.array([0.5, 0.5, 3.0]), verts, mask)
        expected = math.hypot(0.5, 3.0)
        assert jnp.isclose(d, expected, atol=1e-6)

    def test_triangle_outside_closest_to_edge(self) -> None:
        # Query outside, closest to the hypotenuse x+y=1.
        verts, mask = _triangle_loop()
        # Point (1,1,0) → nearest point on hypotenuse is (0.5,0.5,0)
        d = polyline_unsigned_distance(jnp.array([1.0, 1.0, 0.0]), verts, mask)
        expected = math.hypot(0.5, 0.5)
        assert jnp.isclose(d, expected, atol=1e-6)

    def test_closing_segment_is_used(self) -> None:
        # Triangle loop — the (v2 -> v0) closing segment is the x=0 edge.
        # A query at (-1, 0.5, 0) is closest to it at distance 1.
        verts, mask = _triangle_loop()
        d = polyline_unsigned_distance(jnp.array([-1.0, 0.5, 0.0]), verts, mask)
        assert jnp.isclose(d, 1.0, atol=1e-6)

    def test_padding_ignored(self) -> None:
        # Padding-filled slots with arbitrary values must not leak into
        # the min-reduction.
        verts = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [999.0, 999.0, 999.0],
                [999.0, 999.0, 999.0],
            ]
        )
        mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
        d = polyline_unsigned_distance(jnp.array([1.0, 1.0, 0.0]), verts, mask)
        expected = math.hypot(0.5, 0.5)
        assert jnp.isclose(d, expected, atol=1e-6)

    @pytest.mark.parametrize(
        "query",
        [
            np.array([0.5, 0.5, 3.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([-1.0, 0.5, 0.0]),
        ],
    )
    def test_reversed_order_gives_same_distance(self, query: np.ndarray) -> None:
        # Unsigned distance is independent of loop traversal direction.
        verts, mask = _triangle_loop()
        n_valid = int(mask.sum())
        valid = verts[:n_valid]
        padding = verts[n_valid:]
        reversed_loop = jnp.concatenate([valid[::-1], padding], axis=0)
        q = jnp.asarray(query)
        d_fwd = polyline_unsigned_distance(q, verts, mask)
        d_rev = polyline_unsigned_distance(q, reversed_loop, mask)
        assert jnp.isclose(d_fwd, d_rev, atol=1e-6)

    def test_grad_through_point_finite(self) -> None:
        verts, mask = _triangle_loop()

        def loss(p: jnp.ndarray) -> jnp.ndarray:
            return polyline_unsigned_distance(p, verts, mask)

        g = jax.grad(loss)(jnp.array([0.3, 0.3, 0.5]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_vertices_finite(self) -> None:
        _, mask = _triangle_loop()
        verts0 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        def loss(v: jnp.ndarray) -> jnp.ndarray:
            return polyline_unsigned_distance(jnp.array([1.0, 1.0, 0.0]), v, mask)

        g = jax.grad(loss)(verts0)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_finite_on_vertex(self) -> None:
        # Query coincides with vertex 1 — distance is a kink.
        verts, mask = _triangle_loop()

        def loss(p: jnp.ndarray) -> jnp.ndarray:
            return polyline_unsigned_distance(p, verts, mask)

        g = jax.grad(loss)(verts[1])
        assert jnp.all(jnp.isfinite(g))
