"""Tests for shape metadata extraction."""

from __future__ import annotations

from pathlib import Path

from brepax.brep.convert import ShapeMetadata, shape_metadata
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestShapeMetadata:
    """Tests for shape_metadata()."""

    def test_box_face_count(self):
        shape = read_step(FIXTURES / "sample_box.step")
        meta = shape_metadata(shape)
        assert meta.n_faces == 6

    def test_box_edge_count(self):
        shape = read_step(FIXTURES / "sample_box.step")
        meta = shape_metadata(shape)
        assert meta.n_edges == 24

    def test_box_vertex_count(self):
        shape = read_step(FIXTURES / "sample_box.step")
        meta = shape_metadata(shape)
        assert meta.n_vertices == 48

    def test_box_face_types(self):
        shape = read_step(FIXTURES / "sample_box.step")
        meta = shape_metadata(shape)
        assert meta.face_types == {"planar": 6}

    def test_box_bounding_box(self):
        shape = read_step(FIXTURES / "sample_box.step")
        meta = shape_metadata(shape)
        # Box is 10 x 20 x 30, OCCT bbox includes small tolerance.
        assert meta.bbox_min[0] < 0.01
        assert meta.bbox_min[1] < 0.01
        assert meta.bbox_min[2] < 0.01
        assert meta.bbox_max[0] > 9.99
        assert meta.bbox_max[1] > 19.99
        assert meta.bbox_max[2] > 29.99

    def test_cylinder_face_types(self):
        shape = read_step(FIXTURES / "sample_cylinder.step")
        meta = shape_metadata(shape)
        assert "cylindrical" in meta.face_types
        assert "planar" in meta.face_types

    def test_cylinder_face_count(self):
        shape = read_step(FIXTURES / "sample_cylinder.step")
        meta = shape_metadata(shape)
        # Cylinder has 3 faces: lateral surface + top cap + bottom cap.
        assert meta.n_faces == 3

    def test_returns_dataclass(self):
        shape = read_step(FIXTURES / "sample_box.step")
        meta = shape_metadata(shape)
        assert isinstance(meta, ShapeMetadata)
