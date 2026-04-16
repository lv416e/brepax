"""Tests for face adjacency graph construction."""

from pathlib import Path

import pytest

from brepax.brep import (
    build_adjacency_graph,
    face_degree,
    neighbors,
    shared_edges,
)
from brepax.io import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def box_graph():
    shape = read_step(FIXTURES / "sample_box.step")
    return build_adjacency_graph(shape)


@pytest.fixture()
def cylinder_graph():
    shape = read_step(FIXTURES / "sample_cylinder.step")
    return build_adjacency_graph(shape)


@pytest.fixture()
def sphere_graph():
    shape = read_step(FIXTURES / "sample_sphere.step")
    return build_adjacency_graph(shape)


class TestBoxAdjacency:
    """A box has 6 faces, 12 edges; each face touches 4 neighbors."""

    def test_face_count(self, box_graph):
        assert box_graph.n_faces == 6

    def test_edge_count(self, box_graph):
        assert box_graph.n_edges == 12

    def test_each_face_has_four_neighbors(self, box_graph):
        for fi in range(box_graph.n_faces):
            assert face_degree(box_graph, fi) == 4

    def test_neighbors_are_symmetric(self, box_graph):
        for fi in range(box_graph.n_faces):
            for ni in neighbors(box_graph, fi):
                assert fi in neighbors(box_graph, ni)

    def test_shared_edges_exist(self, box_graph):
        for fi in range(box_graph.n_faces):
            for ni in neighbors(box_graph, fi):
                edges = shared_edges(box_graph, fi, ni)
                assert len(edges) == 1

    def test_no_self_adjacency(self, box_graph):
        for fi in range(box_graph.n_faces):
            assert fi not in neighbors(box_graph, fi)


class TestCylinderAdjacency:
    """A cylinder has 3 faces (lateral + 2 caps) and seam edges."""

    def test_face_count(self, cylinder_graph):
        assert cylinder_graph.n_faces == 3

    def test_lateral_connects_to_both_caps(self, cylinder_graph):
        # Find the lateral face (highest degree due to seam)
        degrees = {
            fi: face_degree(cylinder_graph, fi) for fi in range(cylinder_graph.n_faces)
        }
        lateral = max(degrees, key=degrees.get)
        cap_ids = [fi for fi in range(cylinder_graph.n_faces) if fi != lateral]
        for cap in cap_ids:
            assert cap in neighbors(cylinder_graph, lateral)

    def test_caps_connect_to_lateral(self, cylinder_graph):
        degrees = {
            fi: face_degree(cylinder_graph, fi) for fi in range(cylinder_graph.n_faces)
        }
        lateral = max(degrees, key=degrees.get)
        cap_ids = [fi for fi in range(cylinder_graph.n_faces) if fi != lateral]
        for cap in cap_ids:
            assert lateral in neighbors(cylinder_graph, cap)

    def test_lateral_has_seam_edge(self, cylinder_graph):
        # Lateral face is adjacent to itself via seam
        degrees = {
            fi: face_degree(cylinder_graph, fi) for fi in range(cylinder_graph.n_faces)
        }
        lateral = max(degrees, key=degrees.get)
        assert lateral in neighbors(cylinder_graph, lateral)


class TestSphereAdjacency:
    """A sphere has 1 face with seam edges only."""

    def test_face_count(self, sphere_graph):
        assert sphere_graph.n_faces == 1

    def test_self_adjacent_via_seam(self, sphere_graph):
        assert 0 in neighbors(sphere_graph, 0)

    def test_edge_count_positive(self, sphere_graph):
        assert sphere_graph.n_edges > 0


class TestQueryAPIs:
    """Test query helper functions."""

    def test_neighbors_returns_sorted(self, box_graph):
        for fi in range(box_graph.n_faces):
            n = neighbors(box_graph, fi)
            assert n == sorted(n)

    def test_shared_edges_returns_sorted(self, box_graph):
        n = neighbors(box_graph, 0)
        for ni in n:
            edges = shared_edges(box_graph, 0, ni)
            assert edges == sorted(edges)

    def test_shared_edges_empty_for_non_neighbors(self, box_graph):
        # Find a pair of faces that are NOT neighbors
        n0 = set(neighbors(box_graph, 0))
        non_neighbor = next(
            fi for fi in range(box_graph.n_faces) if fi != 0 and fi not in n0
        )
        assert shared_edges(box_graph, 0, non_neighbor) == []

    def test_face_degree_matches_neighbors(self, box_graph):
        for fi in range(box_graph.n_faces):
            assert face_degree(box_graph, fi) == len(neighbors(box_graph, fi))

    def test_invalid_face_id_returns_empty(self, box_graph):
        assert neighbors(box_graph, 999) == []
        assert face_degree(box_graph, 999) == 0
        assert shared_edges(box_graph, 999, 0) == []
