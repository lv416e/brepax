"""Extract Marschner-composition inputs from OCCT plane faces."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    GeomAbs_Plane,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
)
from brepax.brep.trim_frame import PlaneTrimFrame, extract_plane_trim_frame
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def box_plane_faces() -> list:
    """All 6 plane faces of the 10x20x30 axis-aligned box."""
    shape = read_step(str(FIXTURES / "sample_box.step"))
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane:
            faces.append(face)
        exp.Next()
    assert len(faces) == 6, f"expected 6 plane faces, got {len(faces)}"
    return faces


class TestExtractSucceeds:
    """Basic call-level guarantees."""

    def test_returns_plane_trim_frame(self, box_plane_faces) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[0])
        assert isinstance(tf, PlaneTrimFrame)

    def test_all_six_box_faces_extracted(self, box_plane_faces) -> None:
        for face in box_plane_faces:
            assert extract_plane_trim_frame(face) is not None


class TestFrameInvariants:
    """Geometric invariants on the extracted frame."""

    @pytest.mark.parametrize("face_idx", range(6))
    def test_normal_is_unit(self, box_plane_faces, face_idx) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[face_idx])
        assert jnp.isclose(jnp.linalg.norm(tf.normal), 1.0, atol=1e-9)

    @pytest.mark.parametrize("face_idx", range(6))
    def test_frame_is_orthonormal(self, box_plane_faces, face_idx) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[face_idx])
        assert jnp.isclose(jnp.linalg.norm(tf.frame_u), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.frame_v), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.frame_u, tf.frame_v), 0.0, atol=1e-9)
        # frame_u and frame_v span the plane; both perpendicular to normal.
        assert jnp.isclose(jnp.dot(tf.frame_u, tf.normal), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.frame_v, tf.normal), 0.0, atol=1e-9)

    @pytest.mark.parametrize("face_idx", range(6))
    def test_offset_matches_origin_normal_product(
        self, box_plane_faces, face_idx
    ) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[face_idx])
        assert jnp.isclose(tf.offset, jnp.dot(tf.normal, tf.origin), atol=1e-9)


class TestPolylineOnPlane:
    """The 3D polyline must lie on the plane it was extracted from."""

    @pytest.mark.parametrize("face_idx", range(6))
    def test_polyline_vertices_are_on_plane(self, box_plane_faces, face_idx) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[face_idx])
        # For every valid polyline vertex, n . (p - origin) = 0.
        signed = (tf.polyline_3d - tf.origin) @ tf.normal
        valid = tf.mask > 0.5
        masked = jnp.where(valid, jnp.abs(signed), 0.0)
        assert float(jnp.max(masked)) < 1e-6

    @pytest.mark.parametrize("face_idx", range(6))
    def test_polyline_matches_frame_parameterisation(
        self, box_plane_faces, face_idx
    ) -> None:
        # polyline_3d[i] must match origin + u * frame_u + v * frame_v
        tf = extract_plane_trim_frame(box_plane_faces[face_idx])
        expected = (
            tf.origin
            + tf.polygon_uv[:, 0:1] * tf.frame_u[None, :]
            + tf.polygon_uv[:, 1:2] * tf.frame_v[None, :]
        )
        np.testing.assert_allclose(
            np.asarray(tf.polyline_3d), np.asarray(expected), atol=1e-9
        )


class TestMaskAndShape:
    """Padding and mask conventions match the trim-aware SDF consumer."""

    def test_default_max_vertices_is_64(self, box_plane_faces) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[0])
        assert tf.polygon_uv.shape == (64, 2)
        assert tf.polyline_3d.shape == (64, 3)
        assert tf.mask.shape == (64,)

    def test_mask_valid_count_is_positive(self, box_plane_faces) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[0])
        n_valid = int(tf.mask.sum())
        # Box face has 4 edges x 8 samples = 32 valid vertices.
        assert n_valid == 32

    def test_mask_values_are_zero_or_one(self, box_plane_faces) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[0])
        assert jnp.all((tf.mask == 0.0) | (tf.mask == 1.0))

    def test_padding_beyond_mask_is_zero(self, box_plane_faces) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[0])
        n_valid = int(tf.mask.sum())
        # Padding polygon_uv is zeroed by convention; padding polyline_3d
        # equals origin (since padding uv = (0, 0)).
        pad_poly = tf.polygon_uv[n_valid:]
        np.testing.assert_allclose(
            np.asarray(pad_poly), np.zeros_like(pad_poly), atol=0.0
        )

    def test_max_vertices_override_propagates(self, box_plane_faces) -> None:
        tf = extract_plane_trim_frame(box_plane_faces[0], max_vertices=128)
        assert tf.polygon_uv.shape == (128, 2)
        assert tf.polyline_3d.shape == (128, 3)
        assert tf.mask.shape == (128,)


class TestRejectsNonPlane:
    """Non-plane input must return ``None``."""

    def test_sphere_face_returns_none(self) -> None:
        shape = read_step(str(FIXTURES / "sample_sphere.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        assert extract_plane_trim_frame(face) is None


class TestCapacityExceeded:
    """A too-small ``max_vertices`` must raise, not silently truncate."""

    def test_raises_when_exceeded(self, box_plane_faces) -> None:
        with pytest.raises(ValueError, match="exceeds max_vertices"):
            extract_plane_trim_frame(box_plane_faces[0], max_vertices=8)
