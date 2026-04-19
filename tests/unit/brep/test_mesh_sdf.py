"""Tests for mesh-based SDF computation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.mesh_sdf import make_mesh_sdf, mesh_sdf, point_triangle_distance


# Canonical right triangle in XY plane: (0,0,0), (1,0,0), (0,1,0)
@pytest.fixture()
def right_triangle():
    v0 = jnp.array([0.0, 0.0, 0.0])
    v1 = jnp.array([1.0, 0.0, 0.0])
    v2 = jnp.array([0.0, 1.0, 0.0])
    return v0, v1, v2


class TestPointTriangleDistance:
    """Verify all 7 Voronoi regions: 1 face, 3 edges, 3 vertices."""

    def test_face_region_above_centroid(self, right_triangle):
        v0, v1, v2 = right_triangle
        point = jnp.array([0.25, 0.25, 2.0])
        d = point_triangle_distance(point, v0, v1, v2)
        assert jnp.allclose(d, 2.0, atol=1e-6)

    def test_face_region_below(self, right_triangle):
        v0, v1, v2 = right_triangle
        point = jnp.array([0.25, 0.25, -3.0])
        d = point_triangle_distance(point, v0, v1, v2)
        assert jnp.allclose(d, 3.0, atol=1e-6)

    def test_point_on_triangle(self, right_triangle):
        v0, v1, v2 = right_triangle
        point = jnp.array([0.2, 0.3, 0.0])
        d = point_triangle_distance(point, v0, v1, v2)
        assert d < 1e-6

    def test_edge_ab(self, right_triangle):
        """Point closest to edge v0-v1 (y < 0 side)."""
        v0, v1, v2 = right_triangle
        point = jnp.array([0.5, -1.0, 0.0])
        d = point_triangle_distance(point, v0, v1, v2)
        assert jnp.allclose(d, 1.0, atol=1e-6)

    def test_edge_ac(self, right_triangle):
        """Point closest to edge v0-v2 (x < 0 side)."""
        v0, v1, v2 = right_triangle
        point = jnp.array([-2.0, 0.5, 0.0])
        d = point_triangle_distance(point, v0, v1, v2)
        assert jnp.allclose(d, 2.0, atol=1e-6)

    def test_edge_bc(self, right_triangle):
        """Point closest to hypotenuse v1-v2."""
        v0, v1, v2 = right_triangle
        # Midpoint of hypotenuse is (0.5, 0.5, 0), normal outward is (1,1,0)/sqrt(2)
        point = jnp.array([0.5 + 1.0 / jnp.sqrt(2.0), 0.5 + 1.0 / jnp.sqrt(2.0), 0.0])
        d = point_triangle_distance(point, v0, v1, v2)
        assert jnp.allclose(d, 1.0, atol=1e-5)

    def test_vertex_v0(self, right_triangle):
        """Point closest to vertex v0."""
        v0, v1, v2 = right_triangle
        point = jnp.array([-1.0, -1.0, 0.0])
        d = point_triangle_distance(point, v0, v1, v2)
        expected = jnp.sqrt(2.0)
        assert jnp.allclose(d, expected, atol=1e-6)

    def test_vertex_v1(self, right_triangle):
        """Point closest to vertex v1."""
        v0, v1, v2 = right_triangle
        point = jnp.array([2.0, -1.0, 0.0])
        d = point_triangle_distance(point, v0, v1, v2)
        expected = jnp.sqrt(1.0 + 1.0)
        assert jnp.allclose(d, expected, atol=1e-6)

    def test_vertex_v2(self, right_triangle):
        """Point closest to vertex v2."""
        v0, v1, v2 = right_triangle
        point = jnp.array([-1.0, 2.0, 0.0])
        d = point_triangle_distance(point, v0, v1, v2)
        expected = jnp.sqrt(1.0 + 1.0)
        assert jnp.allclose(d, expected, atol=1e-6)

    def test_3d_off_plane(self, right_triangle):
        """Point above an edge: combines plane + edge distance."""
        v0, v1, v2 = right_triangle
        point = jnp.array([0.5, -1.0, 1.0])
        d = point_triangle_distance(point, v0, v1, v2)
        # Closest point on edge AB at (0.5, 0, 0), distance = sqrt(1+1) = sqrt(2)
        expected = jnp.sqrt(2.0)
        assert jnp.allclose(d, expected, atol=1e-6)

    def test_jit_compatible(self, right_triangle):
        v0, v1, v2 = right_triangle
        point = jnp.array([0.25, 0.25, 1.0])
        jitted = jax.jit(point_triangle_distance)
        d = jitted(point, v0, v1, v2)
        assert jnp.allclose(d, 1.0, atol=1e-6)

    def test_vmap_compatible(self, right_triangle):
        v0, v1, v2 = right_triangle
        points = jnp.array(
            [
                [0.25, 0.25, 1.0],
                [0.25, 0.25, 2.0],
                [0.25, 0.25, 3.0],
            ]
        )
        batched = jax.vmap(point_triangle_distance, in_axes=(0, None, None, None))
        dists = batched(points, v0, v1, v2)
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(dists, expected, atol=1e-6)

    def test_grad_finite(self, right_triangle):
        v0, v1, v2 = right_triangle
        point = jnp.array([0.25, 0.25, 1.0])
        grad_fn = jax.grad(point_triangle_distance)
        g = grad_fn(point, v0, v1, v2)
        assert jnp.all(jnp.isfinite(g))
        # Gradient w.r.t. point should point away from triangle (upward)
        assert g[2] > 0.0

    def test_grad_wrt_vertex(self, right_triangle):
        v0, v1, v2 = right_triangle
        point = jnp.array([0.25, 0.25, 1.0])
        grad_fn = jax.grad(point_triangle_distance, argnums=1)
        g = grad_fn(point, v0, v1, v2)
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0.0)


def _make_box_triangles(half: float = 1.0) -> jnp.ndarray:
    """Unit box [-half, half]^3 as 12 triangles with outward normals."""
    h = half
    verts = jnp.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],  # 0-3: z=-h
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],  # 4-7: z=+h
        ]
    )
    # 6 faces, 2 triangles each, outward-facing winding
    faces = jnp.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # -z
            [4, 5, 6],
            [4, 6, 7],  # +z
            [0, 1, 5],
            [0, 5, 4],  # -y
            [2, 3, 7],
            [2, 7, 6],  # +y
            [0, 4, 7],
            [0, 7, 3],  # -x
            [1, 2, 6],
            [1, 6, 5],  # +x
        ]
    )
    return verts[faces]  # (12, 3, 3)


@pytest.fixture()
def box_mesh():
    return _make_box_triangles(1.0)


class TestMeshSdf:
    """Verify mesh_sdf sign and distance on a unit box."""

    def test_inside_negative(self, box_mesh):
        """Center of box should have negative SDF."""
        pts = jnp.array([[0.0, 0.0, 0.0]])
        sdf = mesh_sdf(pts, box_mesh)
        assert sdf[0] < 0.0

    def test_inside_distance(self, box_mesh):
        """Center of [-1,1]^3 box: distance to nearest face = 1.0."""
        pts = jnp.array([[0.0, 0.0, 0.0]])
        sdf = mesh_sdf(pts, box_mesh)
        assert jnp.allclose(jnp.abs(sdf[0]), 1.0, atol=0.05)

    def test_outside_positive(self, box_mesh):
        """Point outside box should have positive SDF."""
        pts = jnp.array([[3.0, 0.0, 0.0]])
        sdf = mesh_sdf(pts, box_mesh)
        assert sdf[0] > 0.0

    def test_outside_distance(self, box_mesh):
        """Point 2 units from +x face: SDF = 2.0."""
        pts = jnp.array([[3.0, 0.0, 0.0]])
        sdf = mesh_sdf(pts, box_mesh)
        assert jnp.allclose(sdf[0], 2.0, atol=0.05)

    def test_batch_points(self, box_mesh):
        """Multiple points: signs and distances."""
        pts = jnp.array(
            [
                [0.0, 0.0, 0.0],  # inside
                [3.0, 0.0, 0.0],  # outside +x
                [-3.0, 0.0, 0.0],  # outside -x
            ]
        )
        sdf = mesh_sdf(pts, box_mesh)
        assert sdf[0] < 0.0
        assert sdf[1] > 0.0
        assert sdf[2] > 0.0

    def test_near_surface(self, box_mesh):
        """Point just outside +x face: small positive SDF."""
        pts = jnp.array([[1.01, 0.0, 0.0]])
        sdf = mesh_sdf(pts, box_mesh)
        assert 0.0 < sdf[0] < 0.1


class TestMakeMeshSdf:
    """Verify the drop-in SDF wrapper."""

    def test_grid_shape(self, box_mesh):
        """make_mesh_sdf should handle (R, R, R, 3) → (R, R, R)."""
        sdf_fn = make_mesh_sdf(box_mesh)
        grid = jnp.zeros((4, 4, 4, 3))
        result = sdf_fn(grid)
        assert result.shape == (4, 4, 4)

    def test_1d_shape(self, box_mesh):
        """make_mesh_sdf should handle (N, 3) → (N,)."""
        sdf_fn = make_mesh_sdf(box_mesh)
        pts = jnp.zeros((10, 3))
        result = sdf_fn(pts)
        assert result.shape == (10,)

    def test_consistency_with_mesh_sdf(self, box_mesh):
        """Wrapper should match direct mesh_sdf call."""
        sdf_fn = make_mesh_sdf(box_mesh)
        pts = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        direct = mesh_sdf(pts, box_mesh)
        via_wrapper = sdf_fn(pts)
        assert jnp.allclose(direct, via_wrapper)
