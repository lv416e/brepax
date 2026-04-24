"""End-to-end trim-aware sphere-face SDF against OCCT.

Two fixtures:

- ``sample_sphere.step`` (full sphere, radius 3 at origin) — the trim
  polygon is the full UV domain, so ``chi ~ 1`` everywhere except at
  the seam.  Off-seam queries exercise the untrimmed collapse where
  ``d_T ~ d_S`` and verify magnitude against OCCT.
- A hemisphere solid built in-test via ``BRepPrimAPI_MakeSphere(r, 0,
  pi/2)``; its sphere face has ``v in [0, pi/2]``, so queries whose
  foot lands below the equator (``v < 0``) are outside trim and
  exercise the phantom-elimination regime.

Tests avoid queries projecting onto the seam (``theta ~ 0 or 2*pi``)
and the poles (``|v| ~ pi/2``) per the limitation noted on full-
revolution trim polygons.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax._occt.backend import (
    BRepAdaptor_Surface,
    BRepBuilderAPI_MakeVertex,
    BRepExtrema_DistShapeShape,
    BRepPrimAPI_MakeSphere,
    GeomAbs_Sphere,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
    gp_Pnt,
)
from brepax.brep.trim_frame import (
    SphereTrimFrame,
    extract_sphere_trim_frame,
    sphere_face_sdf,
    sphere_face_sdf_from_frame,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def full_sphere_face_and_frame() -> tuple[object, SphereTrimFrame]:
    shape = read_step(str(FIXTURES / "sample_sphere.step"))
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Sphere:
            tf = extract_sphere_trim_frame(face)
            assert tf is not None
            return face, tf
        exp.Next()
    raise AssertionError("no sphere face in sample_sphere.step")


@pytest.fixture()
def hemisphere_face_and_frame() -> tuple[object, SphereTrimFrame]:
    """Top hemisphere (v in [0, pi/2]) built via BRepPrimAPI_MakeSphere."""
    shape = BRepPrimAPI_MakeSphere(3.0, 0.0, math.pi / 2).Shape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Sphere:
            tf = extract_sphere_trim_frame(face)
            assert tf is not None
            return face, tf
        exp.Next()
    raise AssertionError("no sphere face in generated hemisphere")


def _occt_face_distance(face: object, query: np.ndarray) -> float:
    vertex = BRepBuilderAPI_MakeVertex(
        gp_Pnt(float(query[0]), float(query[1]), float(query[2]))
    ).Vertex()
    dss = BRepExtrema_DistShapeShape(face, vertex)
    dss.Perform()
    assert dss.IsDone()
    return float(dss.Value())


class TestFullSphereUntrimmedCollapse:
    """For a full sphere, d_T collapses to d_S in the polygon interior."""

    def test_outside_ball_is_positive(self, full_sphere_face_and_frame) -> None:
        # Query on +y axis at distance 5; d_S = 5 - 3 = +2 (FORWARD).
        _, frame = full_sphere_face_and_frame
        d = sphere_face_sdf_from_frame(frame, jnp.array([0.0, 5.0, 0.0]))
        assert float(d) > 0.0
        assert jnp.isclose(d, 2.0, atol=1e-2)

    def test_inside_ball_is_negative(self, full_sphere_face_and_frame) -> None:
        _, frame = full_sphere_face_and_frame
        d = sphere_face_sdf_from_frame(frame, jnp.array([0.0, 1.0, 0.0]))
        assert float(d) < 0.0
        assert jnp.isclose(d, -2.0, atol=1e-2)

    @pytest.mark.parametrize(
        "query",
        [
            np.array([0.0, 5.0, 0.0]),  # equator, off-seam (+y)
            np.array([0.0, 1.0, 0.0]),  # inside ball, off-seam
            np.array([0.0, 4.0, 3.0]),  # generic outside
            np.array([0.0, 2.0, 1.0]),  # generic inside
        ],
    )
    def test_magnitude_matches_occt(
        self, full_sphere_face_and_frame, query: np.ndarray
    ) -> None:
        face, frame = full_sphere_face_and_frame
        d = float(
            sphere_face_sdf_from_frame(frame, jnp.asarray(query, dtype=jnp.float64))
        )
        d_occt = _occt_face_distance(face, query)
        assert jnp.isclose(abs(d), d_occt, atol=1e-2), (
            f"query={query} marschner={d} occt={d_occt}"
        )


class TestHemispherePhantomElimination:
    """v outside [0, pi/2] flips sign via the signed blend."""

    def test_inside_trim_inside_ball_is_negative(
        self, hemisphere_face_and_frame
    ) -> None:
        # (0, 2, 1): dist = sqrt(5) ~= 2.236, d_S = -0.76; foot_v ~=
        # atan2(1, 2) = 0.46 rad, inside [0, pi/2].
        _, frame = hemisphere_face_and_frame
        d = sphere_face_sdf_from_frame(frame, jnp.array([0.0, 2.0, 1.0]))
        assert float(d) < 0.0

    def test_inside_trim_outside_ball_is_positive(
        self, hemisphere_face_and_frame
    ) -> None:
        _, frame = hemisphere_face_and_frame
        d = sphere_face_sdf_from_frame(frame, jnp.array([0.0, 4.0, 3.0]))
        assert float(d) > 0.0

    def test_outside_trim_inside_ball_phantom_eliminated(
        self, hemisphere_face_and_frame
    ) -> None:
        # (0, 1, -2): dist = sqrt(5) ~= 2.236, d_S = -0.76 (inside ball,
        # sign_flip=+1 on FORWARD face → "inside primitive" under the
        # untrimmed classification).  foot_v ~= atan2(-2, 1) ~= -1.107
        # which is outside the hemisphere's trim v-range [0, pi/2], so
        # chi ~ 0 and d_T should flip to +d_partial.
        _, frame = hemisphere_face_and_frame
        d = sphere_face_sdf_from_frame(frame, jnp.array([0.0, 1.0, -2.0]))
        assert float(d) > 0.0, f"phantom not eliminated: d_T={float(d)}"

    def test_phantom_magnitude_matches_occt(self, hemisphere_face_and_frame) -> None:
        face, frame = hemisphere_face_and_frame
        query = np.array([0.0, 1.0, -2.0])
        d = float(
            sphere_face_sdf_from_frame(frame, jnp.asarray(query, dtype=jnp.float64))
        )
        d_occt = _occt_face_distance(face, query)
        # Polyline approximates the equator circle with an inscribed
        # polygon; allow the same ~0.15 slack as the cylinder path.
        assert jnp.isclose(abs(d), d_occt, atol=0.15), (
            f"query={query} marschner={d} occt={d_occt}"
        )


class TestConvenienceWrapper:
    def test_convenience_matches_pure(self, full_sphere_face_and_frame) -> None:
        face, frame = full_sphere_face_and_frame
        query = jnp.array([0.0, 4.0, 3.0])
        d_conv = sphere_face_sdf(face, query)
        d_pure = sphere_face_sdf_from_frame(frame, query)
        assert d_conv is not None
        assert jnp.isclose(d_conv, d_pure, atol=1e-9)

    def test_convenience_returns_none_for_non_sphere(self) -> None:
        shape = read_step(str(FIXTURES / "sample_box.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        d = sphere_face_sdf(face, jnp.array([0.0, 0.0, 0.0]))
        assert d is None


class TestGradient:
    def test_grad_through_query_finite(self, full_sphere_face_and_frame) -> None:
        _, frame = full_sphere_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return sphere_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 4.0, 3.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_at_center(
        self, full_sphere_face_and_frame
    ) -> None:
        # At the sphere's centre the radial direction is undefined;
        # the safe-square pattern must keep the gradient finite.
        _, frame = full_sphere_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return sphere_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_on_axis(
        self, full_sphere_face_and_frame
    ) -> None:
        # On the polar axis above the centre the longitude is
        # undefined; the double-where guard on arctan2 must keep the
        # gradient finite.
        _, frame = full_sphere_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return sphere_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 0.0, 5.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_in_phantom_regime(
        self, hemisphere_face_and_frame
    ) -> None:
        _, frame = hemisphere_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return sphere_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 1.0, -2.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_jit_composition_runs(self, full_sphere_face_and_frame) -> None:
        _, frame = full_sphere_face_and_frame
        jitted = jax.jit(lambda q: sphere_face_sdf_from_frame(frame, q))
        d = jitted(jnp.array([0.0, 4.0, 3.0]))
        assert jnp.isfinite(d)
