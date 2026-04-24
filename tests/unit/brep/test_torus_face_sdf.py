"""End-to-end trim-aware torus-face SDF against OCCT.

Two fixtures:

- ``sample_torus.step`` (full torus, major 5 minor 1 at origin) — the
  trim polygon covers the full UV domain so ``chi ~ 1`` away from the
  four seam edges; off-seam queries verify the untrimmed collapse
  where ``d_T ~ d_S`` and magnitude match against OCCT.
- A quarter torus built in-test via ``BRepPrimAPI_MakeTorus(5, 1,
  pi/2)``; its torus face has ``u in [0, pi/2]``, so queries with
  ``u ~ pi`` (opposite side of the ring) are outside trim and
  exercise the phantom-elimination regime.
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
    BRepPrimAPI_MakeTorus,
    GeomAbs_Torus,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
    gp_Pnt,
)
from brepax.brep.trim_frame import (
    TorusTrimFrame,
    extract_torus_trim_frame,
    torus_face_sdf,
    torus_face_sdf_from_frame,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def full_torus_face_and_frame() -> tuple[object, TorusTrimFrame]:
    shape = read_step(str(FIXTURES / "sample_torus.step"))
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Torus:
            tf = extract_torus_trim_frame(face)
            assert tf is not None
            return face, tf
        exp.Next()
    raise AssertionError("no torus face in sample_torus.step")


@pytest.fixture()
def quarter_torus_face_and_frame() -> tuple[object, TorusTrimFrame]:
    """Quarter revolution (u in [0, pi/2]) built via BRepPrimAPI_MakeTorus."""
    shape = BRepPrimAPI_MakeTorus(5.0, 1.0, math.pi / 2).Shape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Torus:
            tf = extract_torus_trim_frame(face)
            assert tf is not None
            return face, tf
        exp.Next()
    raise AssertionError("no torus face in generated quarter torus")


def _occt_face_distance(face: object, query: np.ndarray) -> float:
    vertex = BRepBuilderAPI_MakeVertex(
        gp_Pnt(float(query[0]), float(query[1]), float(query[2]))
    ).Vertex()
    dss = BRepExtrema_DistShapeShape(face, vertex)
    dss.Perform()
    assert dss.IsDone()
    return float(dss.Value())


class TestFullTorusUntrimmedCollapse:
    """On a full torus the trim indicator is ~1 away from seams.

    Queries in this class are chosen so that both UV coordinates stay
    away from the four seam edges (u in {0, 2*pi}, v in {0, 2*pi}).
    Seam points produce a winding-ambiguous chi that collapses the
    signed blend toward zero — a known limitation of full-revolution
    trim polygons shared with the cylinder and sphere faces.
    """

    def test_outside_tube_is_positive(self, full_torus_face_and_frame) -> None:
        # sample_torus has major=5, minor=1.5.  Query (0, 6, 2) has
        # foot_u = pi/2, foot_v = atan2(2, 1) ~ 1.107 — both off the
        # seams.  tube_dist = sqrt(1 + 4) = sqrt(5), d_S = sqrt(5) -
        # 1.5 ~= 0.736.
        _, frame = full_torus_face_and_frame
        minor = float(frame.minor_radius)
        d = torus_face_sdf_from_frame(frame, jnp.array([0.0, 6.0, 2.0]))
        assert float(d) > 0.0
        assert jnp.isclose(d, math.sqrt(5.0) - minor, atol=0.1)

    def test_inside_tube_is_negative(self, full_torus_face_and_frame) -> None:
        # (0, 5.5, 0.3): r=5.5, axial=0.3, v = atan2(0.3, 0.5) ~ 0.54
        # (off seam).  tube_dist = sqrt(0.25 + 0.09) ~= 0.583, d_S =
        # 0.583 - 1 ~= -0.417.
        _, frame = full_torus_face_and_frame
        d = torus_face_sdf_from_frame(frame, jnp.array([0.0, 5.5, 0.3]))
        assert float(d) < 0.0

    @pytest.mark.parametrize(
        "query",
        [
            np.array([0.0, 6.0, 2.0]),  # outside tube, off-seam
            np.array([0.0, 5.5, 0.3]),  # inside tube, off-seam
            np.array([0.0, 7.0, 1.0]),  # generic far-outside
            np.array([2.0, 5.0, 0.5]),  # u=atan2(5,2)~1.19, v~0.54
        ],
    )
    def test_magnitude_matches_occt(
        self, full_torus_face_and_frame, query: np.ndarray
    ) -> None:
        face, frame = full_torus_face_and_frame
        d = float(
            torus_face_sdf_from_frame(frame, jnp.asarray(query, dtype=jnp.float64))
        )
        d_occt = _occt_face_distance(face, query)
        assert jnp.isclose(abs(d), d_occt, atol=0.15), (
            f"query={query} marschner={d} occt={d_occt}"
        )


class TestQuarterTorusPhantomElimination:
    """u outside [0, pi/2] flips sign via the signed blend."""

    def test_inside_trim_inside_tube_is_negative(
        self, quarter_torus_face_and_frame
    ) -> None:
        # u = pi/4, inside the quarter range.  Tube-centre point:
        # (5*cos(pi/4), 5*sin(pi/4), 0) ~= (3.54, 3.54, 0).  d_S = -1.
        _, frame = quarter_torus_face_and_frame
        d = torus_face_sdf_from_frame(
            frame,
            jnp.array([5 * math.cos(math.pi / 4), 5 * math.sin(math.pi / 4), 0.0]),
        )
        assert float(d) < 0.0

    def test_inside_trim_outside_tube_is_positive(
        self, quarter_torus_face_and_frame
    ) -> None:
        _, frame = quarter_torus_face_and_frame
        # u=pi/4 (inside quarter trim), axial=0.5 so foot_v off the
        # v=0 seam; r=7 well outside the tube of major=5 minor=1.
        tx = 7 * math.cos(math.pi / 4)
        ty = 7 * math.sin(math.pi / 4)
        d = torus_face_sdf_from_frame(frame, jnp.array([tx, ty, 0.5]))
        assert float(d) > 0.0

    def test_outside_trim_inside_tube_phantom_eliminated(
        self, quarter_torus_face_and_frame
    ) -> None:
        # (-5, 0, 0) sits on the tube-centre ring at u=pi, outside the
        # quarter [0, pi/2].  Untrimmed primitive: d_S = -1 (inside
        # tube).  Marschner should flip this via chi -> 0.
        _, frame = quarter_torus_face_and_frame
        d = torus_face_sdf_from_frame(frame, jnp.array([-5.0, 0.0, 0.0]))
        assert float(d) > 0.0, f"phantom not eliminated: d_T={float(d)}"


class TestConvenienceWrapper:
    def test_convenience_matches_pure(self, full_torus_face_and_frame) -> None:
        face, frame = full_torus_face_and_frame
        # Off-seam query.
        query = jnp.array([0.0, 4.5, 0.3])
        d_conv = torus_face_sdf(face, query)
        d_pure = torus_face_sdf_from_frame(frame, query)
        assert d_conv is not None
        assert jnp.isclose(d_conv, d_pure, atol=1e-9)

    def test_convenience_returns_none_for_non_torus(self) -> None:
        shape = read_step(str(FIXTURES / "sample_box.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        d = torus_face_sdf(face, jnp.array([0.0, 0.0, 0.0]))
        assert d is None


class TestGradient:
    def test_grad_through_query_finite(self, full_torus_face_and_frame) -> None:
        _, frame = full_torus_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return torus_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 4.5, 0.3]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_on_axis(self, full_torus_face_and_frame) -> None:
        # Polar axis: r=0, u undefined.  Safe-square + double-where.
        _, frame = full_torus_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return torus_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_on_tube_centre(
        self, full_torus_face_and_frame
    ) -> None:
        # Tube-centre ring point: (0, 5, 0).  r - major = 0, axial = 0,
        # so the tube cross-section angle v is undefined.  The
        # double-where on v must keep grad finite.
        _, frame = full_torus_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return torus_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 5.0, 0.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_in_phantom_regime(
        self, quarter_torus_face_and_frame
    ) -> None:
        _, frame = quarter_torus_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return torus_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([-5.0, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_jit_composition_runs(self, full_torus_face_and_frame) -> None:
        _, frame = full_torus_face_and_frame
        jitted = jax.jit(lambda q: torus_face_sdf_from_frame(frame, q))
        d = jitted(jnp.array([0.0, 4.5, 0.3]))
        assert jnp.isfinite(d)
