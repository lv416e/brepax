"""Extract Marschner-composition inputs from OCCT sphere faces."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    GeomAbs_Sphere,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopExp_Explorer,
    TopoDS,
)
from brepax.brep.trim_frame import SphereTrimFrame, extract_sphere_trim_frame
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def sample_sphere_face() -> object:
    """The single sphere face of sample_sphere.step (radius 3 at origin)."""
    shape = read_step(str(FIXTURES / "sample_sphere.step"))
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Sphere:
            return face
        exp.Next()
    raise AssertionError("no sphere face in sample_sphere.step")


class TestExtractSucceeds:
    def test_returns_sphere_trim_frame(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        assert isinstance(tf, SphereTrimFrame)


class TestFrameInvariants:
    def test_axis_is_unit(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        assert jnp.isclose(jnp.linalg.norm(tf.axis), 1.0, atol=1e-9)

    def test_frame_is_orthonormal(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        assert jnp.isclose(jnp.linalg.norm(tf.x_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.y_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.y_dir), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.axis), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.y_dir, tf.axis), 0.0, atol=1e-9)

    def test_radius_matches_fixture(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        assert jnp.isclose(tf.radius, 3.0, atol=1e-6)

    def test_center_at_origin(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        np.testing.assert_allclose(np.asarray(tf.center), np.zeros(3), atol=1e-6)


class TestSignFlip:
    def test_sign_flip_matches_face_orientation(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        expected = 1.0 if sample_sphere_face.Orientation() == TopAbs_FORWARD else -1.0
        assert jnp.isclose(tf.sign_flip, expected)


class TestPolylineOnSphere:
    def test_polyline_vertices_are_on_sphere(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        delta = tf.polyline_3d - tf.center
        dist = jnp.linalg.norm(delta, axis=-1)
        valid = tf.mask > 0.5
        err = jnp.where(valid, jnp.abs(dist - tf.radius), 0.0)
        assert float(jnp.max(err)) < 1e-6

    def test_polyline_matches_frame_parameterisation(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        us = tf.polygon_uv[:, 0]
        vs = tf.polygon_uv[:, 1]
        cos_v = jnp.cos(vs)[:, None]
        sin_v = jnp.sin(vs)[:, None]
        equator_dir = jnp.cos(us)[:, None] * tf.x_dir + jnp.sin(us)[:, None] * tf.y_dir
        expected = tf.center + tf.radius * (cos_v * equator_dir + sin_v * tf.axis)
        np.testing.assert_allclose(
            np.asarray(tf.polyline_3d), np.asarray(expected), atol=1e-9
        )


class TestMaskAndShape:
    def test_default_max_vertices_is_64(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        assert tf.polygon_uv.shape == (64, 2)
        assert tf.polyline_3d.shape == (64, 3)
        assert tf.mask.shape == (64,)

    def test_mask_values_are_bitmask(self, sample_sphere_face) -> None:
        tf = extract_sphere_trim_frame(sample_sphere_face)
        assert jnp.all((tf.mask == 0.0) | (tf.mask == 1.0))
        # Ensure at least one valid slot: all-zero mask would mean the
        # extractor silently produced an empty polygon.
        assert int(tf.mask.sum()) > 0


class TestRejectsNonSphere:
    def test_plane_face_returns_none(self) -> None:
        from brepax._occt.backend import GeomAbs_Plane

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
        assert extract_sphere_trim_frame(plane_face) is None


class TestPropertyBased:
    """Hypothesis-generated OCCT spheres across the float range of radii."""

    @staticmethod
    def _make_sphere_face(radius: float) -> object:
        from brepax._occt.backend import BRepPrimAPI_MakeSphere

        shape = BRepPrimAPI_MakeSphere(radius).Shape()
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            if BRepAdaptor_Surface(face).GetType() == GeomAbs_Sphere:
                return face
            exp.Next()
        raise AssertionError("no sphere face in generated solid")

    @given(radius=st.floats(min_value=0.1, max_value=100.0, allow_nan=False))
    @settings(max_examples=25, deadline=None)
    def test_radius_roundtrips(self, radius: float) -> None:
        face = self._make_sphere_face(radius)
        tf = extract_sphere_trim_frame(face)
        assert tf is not None
        assert jnp.isclose(tf.radius, radius, atol=1e-6)

    @given(radius=st.floats(min_value=0.1, max_value=100.0, allow_nan=False))
    @settings(max_examples=25, deadline=None)
    def test_frame_is_orthonormal(self, radius: float) -> None:
        face = self._make_sphere_face(radius)
        tf = extract_sphere_trim_frame(face)
        assert tf is not None
        assert jnp.isclose(jnp.linalg.norm(tf.axis), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.x_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.y_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.y_dir), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.axis), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.y_dir, tf.axis), 0.0, atol=1e-9)

    @given(radius=st.floats(min_value=0.1, max_value=100.0, allow_nan=False))
    @settings(max_examples=25, deadline=None)
    def test_polyline_on_sphere(self, radius: float) -> None:
        face = self._make_sphere_face(radius)
        tf = extract_sphere_trim_frame(face)
        assert tf is not None
        delta = tf.polyline_3d - tf.center
        dist = jnp.linalg.norm(delta, axis=-1)
        valid = tf.mask > 0.5
        err = jnp.where(valid, jnp.abs(dist - tf.radius), 0.0)
        assert float(jnp.max(err)) < max(1e-6, 1e-8 * radius)
