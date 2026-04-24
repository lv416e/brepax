"""Extract Marschner-composition inputs from OCCT cone faces.

Cones are constructed in-test via ``BRepPrimAPI_MakeCone(R1, R2, H)``:
- ``R1 > R2 = 0`` gives a regular cone that tapers to an apex.
- OCCT's ``SemiAngle`` is signed; this fixture covers the negative
  sign branch that ``extract_cone_trim_frame`` must preserve to keep
  the parametric-surface reconstruction correct.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    BRepPrimAPI_MakeCone,
    GeomAbs_Cone,
    GeomAbs_Plane,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopExp_Explorer,
    TopoDS,
)
from brepax.brep.trim_frame import ConeTrimFrame, extract_cone_trim_frame
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _cone_face(r1: float, r2: float, h: float) -> object:
    shape = BRepPrimAPI_MakeCone(r1, r2, h).Shape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cone:
            return face
        exp.Next()
    raise AssertionError("no cone face in generated solid")


@pytest.fixture()
def sample_cone_face() -> object:
    """Cone: R1=3 at base, R2=0 at top, height 9; semi_angle = -atan(3/9)."""
    return _cone_face(3.0, 0.0, 9.0)


class TestExtractSucceeds:
    def test_returns_cone_trim_frame(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        assert isinstance(tf, ConeTrimFrame)


class TestFrameInvariants:
    def test_axis_is_unit(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        assert jnp.isclose(jnp.linalg.norm(tf.axis), 1.0, atol=1e-9)

    def test_frame_is_orthonormal(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        assert jnp.isclose(jnp.linalg.norm(tf.x_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.y_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.y_dir), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.axis), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.y_dir, tf.axis), 0.0, atol=1e-9)

    def test_ref_radius_matches_fixture(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        assert jnp.isclose(tf.ref_radius, 3.0, atol=1e-6)

    def test_semi_angle_is_signed(self, sample_cone_face) -> None:
        # For R1 > R2, OCCT reports a *negative* semi-angle.  The
        # magnitude should equal atan(R1 / H) = atan(3 / 9).
        tf = extract_cone_trim_frame(sample_cone_face)
        assert float(tf.semi_angle) < 0.0
        assert jnp.isclose(jnp.abs(tf.semi_angle), math.atan(3.0 / 9.0), atol=1e-6)

    def test_apex_lies_on_axis_line_from_location(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        delta = tf.apex - tf.location
        # Component perpendicular to axis must vanish.
        axial = jnp.dot(delta, tf.axis)
        perp = delta - axial * tf.axis
        assert jnp.isclose(jnp.linalg.norm(perp), 0.0, atol=1e-6)


class TestSignFlip:
    def test_sign_flip_matches_face_orientation(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        expected = 1.0 if sample_cone_face.Orientation() == TopAbs_FORWARD else -1.0
        assert jnp.isclose(tf.sign_flip, expected)


class TestPolylineOnCone:
    def test_polyline_vertices_are_on_cone(self, sample_cone_face) -> None:
        # Every valid polyline vertex must satisfy the cone surface
        # equation: at axial distance h from location along +axis,
        # the radius is ref_radius + (h / cos(angle)) * sin(angle)
        # (measured via the parametric slant v = h / cos(angle)).
        tf = extract_cone_trim_frame(sample_cone_face)
        delta = tf.polyline_3d - tf.location
        axial = delta @ tf.axis
        perp = delta - axial[:, None] * tf.axis
        perp_norm = jnp.linalg.norm(perp, axis=-1)

        # v = axial / cos(semi_angle); expected_radius = R + v * sin(angle)
        v = axial / jnp.cos(tf.semi_angle)
        expected_radius = tf.ref_radius + v * jnp.sin(tf.semi_angle)

        valid = tf.mask > 0.5
        err = jnp.where(valid, jnp.abs(perp_norm - expected_radius), 0.0)
        assert float(jnp.max(err)) < 1e-6

    def test_polyline_matches_frame_parameterisation(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        us = tf.polygon_uv[:, 0]
        vs = tf.polygon_uv[:, 1]
        radius_at_v = (tf.ref_radius + vs * jnp.sin(tf.semi_angle))[:, None]
        equator_dir = jnp.cos(us)[:, None] * tf.x_dir + jnp.sin(us)[:, None] * tf.y_dir
        axial_offset = (vs * jnp.cos(tf.semi_angle))[:, None] * tf.axis
        expected = tf.location + radius_at_v * equator_dir + axial_offset
        np.testing.assert_allclose(
            np.asarray(tf.polyline_3d), np.asarray(expected), atol=1e-9
        )


class TestMaskAndShape:
    def test_default_max_vertices_is_64(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        assert tf.polygon_uv.shape == (64, 2)
        assert tf.polyline_3d.shape == (64, 3)
        assert tf.mask.shape == (64,)

    def test_mask_values_are_bitmask(self, sample_cone_face) -> None:
        tf = extract_cone_trim_frame(sample_cone_face)
        assert jnp.all((tf.mask == 0.0) | (tf.mask == 1.0))
        assert int(tf.mask.sum()) > 0


class TestRejectsNonCone:
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
        assert extract_cone_trim_frame(plane_face) is None


class TestCapacityExceeded:
    def test_raises_when_exceeded(self, sample_cone_face) -> None:
        with pytest.raises(ValueError, match="exceeds max_vertices"):
            extract_cone_trim_frame(sample_cone_face, max_vertices=4)


class TestPropertyBased:
    @given(
        r1=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
        h=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
    )
    @settings(max_examples=25, deadline=None)
    def test_ref_radius_roundtrips(self, r1: float, h: float) -> None:
        face = _cone_face(r1, 0.0, h)
        tf = extract_cone_trim_frame(face)
        assert tf is not None
        assert jnp.isclose(tf.ref_radius, r1, atol=1e-5)

    @given(
        r1=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
        h=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
    )
    @settings(max_examples=25, deadline=None)
    def test_frame_is_orthonormal(self, r1: float, h: float) -> None:
        face = _cone_face(r1, 0.0, h)
        tf = extract_cone_trim_frame(face)
        assert tf is not None
        assert jnp.isclose(jnp.linalg.norm(tf.axis), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.x_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.linalg.norm(tf.y_dir), 1.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.y_dir), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.x_dir, tf.axis), 0.0, atol=1e-9)
        assert jnp.isclose(jnp.dot(tf.y_dir, tf.axis), 0.0, atol=1e-9)

    @given(
        r1=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
        h=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
    )
    @settings(max_examples=25, deadline=None)
    def test_semi_angle_matches_geometry(self, r1: float, h: float) -> None:
        # |semi_angle| == atan(r1 / h) for a cone that tapers from r1
        # at the base to 0 at height h.
        face = _cone_face(r1, 0.0, h)
        tf = extract_cone_trim_frame(face)
        assert tf is not None
        assert jnp.isclose(jnp.abs(tf.semi_angle), math.atan(r1 / h), atol=1e-6)

    @given(
        r1=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
        h=st.floats(min_value=0.5, max_value=50.0, allow_nan=False),
    )
    @settings(max_examples=25, deadline=None)
    def test_polyline_on_cone(self, r1: float, h: float) -> None:
        face = _cone_face(r1, 0.0, h)
        tf = extract_cone_trim_frame(face)
        assert tf is not None
        delta = tf.polyline_3d - tf.location
        axial = delta @ tf.axis
        perp = delta - axial[:, None] * tf.axis
        perp_norm = jnp.linalg.norm(perp, axis=-1)
        v = axial / jnp.cos(tf.semi_angle)
        expected_radius = tf.ref_radius + v * jnp.sin(tf.semi_angle)
        valid = tf.mask > 0.5
        err = jnp.where(valid, jnp.abs(perp_norm - expected_radius), 0.0)
        assert float(jnp.max(err)) < max(1e-6, 1e-8 * max(r1, h))
