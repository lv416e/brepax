"""Extract Marschner-composition inputs from OCCT torus faces."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    BRepPrimAPI_MakeTorus,
    GeomAbs_Plane,
    GeomAbs_Torus,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopExp_Explorer,
    TopoDS,
)
from brepax.brep.trim_frame import TorusTrimFrame, extract_torus_trim_frame
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def sample_torus_face() -> object:
    """The single torus face of sample_torus.step."""
    shape = read_step(str(FIXTURES / "sample_torus.step"))
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Torus:
            return face
        exp.Next()
    raise AssertionError("no torus face in sample_torus.step")


def _torus_face(major: float, minor: float) -> object:
    shape = BRepPrimAPI_MakeTorus(major, minor).Shape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Torus:
            return face
        exp.Next()
    raise AssertionError("no torus face in generated solid")


class TestExtractSucceeds:
    def test_returns_torus_trim_frame(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        assert isinstance(tf, TorusTrimFrame)


class TestFrameInvariants:
    def test_axis_is_unit(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        assert jnp.isclose(jnp.linalg.norm(tf.axis), 1.0, atol=1e-9)

    def test_frame_is_orthonormal(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        assert jnp.isclose(jnp.linalg.norm(tf.x_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.y_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.y_dir), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.axis), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.y_dir, tf.axis), 0.0, atol=1e-9)

    def test_radii_positive(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        assert float(tf.major_radius) > 0.0
        assert float(tf.minor_radius) > 0.0


class TestSignFlip:
    def test_sign_flip_matches_face_orientation(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        expected = 1.0 if sample_torus_face.Orientation() == TopAbs_FORWARD else -1.0
        assert jnp.isclose(tf.sign_flip, expected)


class TestPolylineOnTorus:
    def test_polyline_vertices_are_on_torus(self, sample_torus_face) -> None:
        # For every valid polyline vertex p:
        #   project onto axis plane: h = axis . (p - center)
        #   radial in-plane: r_in = ||(p - center) - h * axis||
        #   distance from tube-centre ring: sqrt((r_in - R)^2 + h^2) == minor_radius
        tf = extract_torus_trim_frame(sample_torus_face)
        delta = tf.polyline_3d - tf.center
        axial = delta @ tf.axis
        perp = delta - axial[:, None] * tf.axis
        radial_in_plane = jnp.linalg.norm(perp, axis=-1)
        tube_dist = jnp.sqrt((radial_in_plane - tf.major_radius) ** 2 + axial**2)
        valid = tf.mask > 0.5
        err = jnp.where(valid, jnp.abs(tube_dist - tf.minor_radius), 0.0)
        assert float(jnp.max(err)) < 1e-6

    def test_polyline_matches_frame_parameterisation(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        us = tf.polygon_uv[:, 0]
        vs = tf.polygon_uv[:, 1]
        tube_radius_at_v = (tf.major_radius + tf.minor_radius * jnp.cos(vs))[:, None]
        equator_dir = jnp.cos(us)[:, None] * tf.x_dir + jnp.sin(us)[:, None] * tf.y_dir
        axial_offset = (tf.minor_radius * jnp.sin(vs))[:, None] * tf.axis
        expected = tf.center + tube_radius_at_v * equator_dir + axial_offset
        np.testing.assert_allclose(
            np.asarray(tf.polyline_3d), np.asarray(expected), atol=1e-9
        )


class TestMaskAndShape:
    def test_default_max_vertices_is_64(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        assert tf.polygon_uv.shape == (64, 2)
        assert tf.polyline_3d.shape == (64, 3)
        assert tf.mask.shape == (64,)

    def test_mask_values_are_bitmask(self, sample_torus_face) -> None:
        tf = extract_torus_trim_frame(sample_torus_face)
        assert jnp.all((tf.mask == 0.0) | (tf.mask == 1.0))
        assert int(tf.mask.sum()) > 0


class TestRejectsNonTorus:
    def test_plane_face_returns_none(self) -> None:
        shape = read_step(str(FIXTURES / "sample_box.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        plane_face = None
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            if BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane:
                plane_face = face
                break
            exp.Next()
        assert plane_face is not None
        assert extract_torus_trim_frame(plane_face) is None


class TestCapacityExceeded:
    def test_raises_when_exceeded(self, sample_torus_face) -> None:
        with pytest.raises(ValueError, match="exceeds max_vertices"):
            extract_torus_trim_frame(sample_torus_face, max_vertices=4)


class TestPropertyBased:
    @given(
        major=st.floats(min_value=2.0, max_value=50.0, allow_nan=False),
        minor=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=25, deadline=None)
    def test_radii_roundtrip(self, major: float, minor: float) -> None:
        # Keep minor < major to stay well inside the ring torus regime.
        face = _torus_face(major, minor)
        tf = extract_torus_trim_frame(face)
        assert tf is not None
        assert jnp.isclose(tf.major_radius, major, atol=1e-5)
        assert jnp.isclose(tf.minor_radius, minor, atol=1e-5)

    @given(
        major=st.floats(min_value=2.0, max_value=50.0, allow_nan=False),
        minor=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=25, deadline=None)
    def test_frame_is_orthonormal(self, major: float, minor: float) -> None:
        face = _torus_face(major, minor)
        tf = extract_torus_trim_frame(face)
        assert tf is not None
        assert jnp.isclose(jnp.linalg.norm(tf.axis), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.x_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.y_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.y_dir), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.axis), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.y_dir, tf.axis), 0.0, atol=1e-9)

    @given(
        major=st.floats(min_value=2.0, max_value=50.0, allow_nan=False),
        minor=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=25, deadline=None)
    def test_polyline_on_torus(self, major: float, minor: float) -> None:
        face = _torus_face(major, minor)
        tf = extract_torus_trim_frame(face)
        assert tf is not None
        delta = tf.polyline_3d - tf.center
        axial = delta @ tf.axis
        perp = delta - axial[:, None] * tf.axis
        radial_in_plane = jnp.linalg.norm(perp, axis=-1)
        tube_dist = jnp.sqrt((radial_in_plane - tf.major_radius) ** 2 + axial**2)
        valid = tf.mask > 0.5
        err = jnp.where(valid, jnp.abs(tube_dist - tf.minor_radius), 0.0)
        assert float(jnp.max(err)) < max(1e-6, 1e-8 * major)
