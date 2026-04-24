"""Extract Marschner-composition inputs from OCCT cylinder faces."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    GeomAbs_Cylinder,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopExp_Explorer,
    TopoDS,
)
from brepax.brep.trim_frame import (
    CylinderTrimFrame,
    extract_cylinder_trim_frame,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def box_with_holes_cylinder_faces() -> list:
    """All cylinder faces of box_with_holes (two through-holes)."""
    shape = read_step(str(FIXTURES / "box_with_holes.step"))
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder:
            faces.append(face)
        exp.Next()
    assert len(faces) == 2, f"expected 2 cylinder faces, got {len(faces)}"
    return faces


class TestExtractSucceeds:
    def test_returns_cylinder_trim_frame(self, box_with_holes_cylinder_faces) -> None:
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[0])
        assert isinstance(tf, CylinderTrimFrame)

    def test_both_hole_faces_extracted(self, box_with_holes_cylinder_faces) -> None:
        for face in box_with_holes_cylinder_faces:
            assert extract_cylinder_trim_frame(face) is not None


class TestFrameInvariants:
    """axis, x_dir, y_dir must form an orthonormal triad."""

    @pytest.mark.parametrize("face_idx", [0, 1])
    def test_axis_is_unit(self, box_with_holes_cylinder_faces, face_idx) -> None:
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[face_idx])
        assert jnp.isclose(jnp.linalg.norm(tf.axis), 1.0, atol=1e-9)

    @pytest.mark.parametrize("face_idx", [0, 1])
    def test_frame_is_orthonormal(
        self, box_with_holes_cylinder_faces, face_idx
    ) -> None:
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[face_idx])
        assert jnp.isclose(jnp.linalg.norm(tf.x_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.y_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.y_dir), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.axis), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.y_dir, tf.axis), 0.0, atol=1e-9)

    @pytest.mark.parametrize("face_idx", [0, 1])
    def test_radius_is_positive(self, box_with_holes_cylinder_faces, face_idx) -> None:
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[face_idx])
        assert float(tf.radius) > 0.0

    def test_radii_match_fixture(self, box_with_holes_cylinder_faces) -> None:
        # box_with_holes has holes of radius 4 and 3 as reported by OCCT.
        radii = sorted(
            float(extract_cylinder_trim_frame(f).radius)
            for f in box_with_holes_cylinder_faces
        )
        assert jnp.isclose(radii[0], 3.0, atol=1e-6)
        assert jnp.isclose(radii[1], 4.0, atol=1e-6)


class TestSignFlip:
    """sign_flip captures face orientation, +/-1 regardless of surface direction."""

    @pytest.mark.parametrize("face_idx", [0, 1])
    def test_sign_flip_matches_face_orientation(
        self, box_with_holes_cylinder_faces, face_idx
    ) -> None:
        face = box_with_holes_cylinder_faces[face_idx]
        tf = extract_cylinder_trim_frame(face)
        expected = 1.0 if face.Orientation() == TopAbs_FORWARD else -1.0
        assert jnp.isclose(tf.sign_flip, expected)


class TestPolylineOnCylinder:
    """The 3D polyline must lie on the cylinder it was extracted from."""

    @pytest.mark.parametrize("face_idx", [0, 1])
    def test_polyline_vertices_are_on_cylinder(
        self, box_with_holes_cylinder_faces, face_idx
    ) -> None:
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[face_idx])
        # Radial distance from the axis line must equal radius.
        delta = tf.polyline_3d - tf.origin
        axial = (delta @ tf.axis)[:, None]
        perp = delta - axial * tf.axis
        radial_dist = jnp.linalg.norm(perp, axis=-1)
        valid = tf.mask > 0.5
        err = jnp.where(valid, jnp.abs(radial_dist - tf.radius), 0.0)
        assert float(jnp.max(err)) < 1e-6

    @pytest.mark.parametrize("face_idx", [0, 1])
    def test_polyline_matches_frame_parameterisation(
        self, box_with_holes_cylinder_faces, face_idx
    ) -> None:
        # S(u, v) = origin + v * axis + r * (cos u * x_dir + sin u * y_dir)
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[face_idx])
        us = tf.polygon_uv[:, 0]
        vs = tf.polygon_uv[:, 1]
        expected = (
            tf.origin
            + vs[:, None] * tf.axis
            + tf.radius
            * (jnp.cos(us)[:, None] * tf.x_dir + jnp.sin(us)[:, None] * tf.y_dir)
        )
        np.testing.assert_allclose(
            np.asarray(tf.polyline_3d), np.asarray(expected), atol=1e-9
        )


class TestMaskAndShape:
    def test_default_max_vertices_is_64(self, box_with_holes_cylinder_faces) -> None:
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[0])
        assert tf.polygon_uv.shape == (64, 2)
        assert tf.polyline_3d.shape == (64, 3)
        assert tf.mask.shape == (64,)

    def test_mask_valid_count_positive(self, box_with_holes_cylinder_faces) -> None:
        tf = extract_cylinder_trim_frame(box_with_holes_cylinder_faces[0])
        # Full-revolution cylinder face has ~4 edges x 8 samples = 32 valid points
        # (two seam edges + top circle + bottom circle, split into 8 samples each).
        n_valid = int(tf.mask.sum())
        assert n_valid >= 16
        assert n_valid <= 64


class TestRejectsNonCylinder:
    def test_plane_face_returns_none(self) -> None:
        shape = read_step(str(FIXTURES / "sample_box.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        assert extract_cylinder_trim_frame(face) is None


class TestCapacityExceeded:
    def test_raises_when_exceeded(self, box_with_holes_cylinder_faces) -> None:
        with pytest.raises(ValueError, match="exceeds max_vertices"):
            extract_cylinder_trim_frame(
                box_with_holes_cylinder_faces[0], max_vertices=4
            )
