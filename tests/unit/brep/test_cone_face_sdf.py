"""End-to-end trim-aware cone-face SDF against OCCT.

Two fixtures:

- A full cone built in-test via ``BRepPrimAPI_MakeCone(3, 0, 9)``.
  The trim polygon covers ``[0, 2*pi]`` in u and the full slant
  range in v; off-seam queries exercise the untrimmed collapse and
  verify magnitude against OCCT.
- A half cone built in-test via ``BRepPrimAPI_MakeCone(3, 0, 9, pi)``
  whose face has u in ``[0, pi]``; queries with u outside that
  range exercise the phantom-elimination regime.
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
    BRepPrimAPI_MakeCone,
    GeomAbs_Cone,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
    gp_Pnt,
)
from brepax._occt.types import TopoDS_Face
from brepax.brep.trim_frame import (
    ConeTrimFrame,
    cone_face_sdf,
    cone_face_sdf_from_frame,
    extract_cone_trim_frame,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _cone_face(
    r1: float, r2: float, h: float, angle: float | None = None
) -> TopoDS_Face:
    if angle is None:
        shape = BRepPrimAPI_MakeCone(r1, r2, h).Shape()
    else:
        shape = BRepPrimAPI_MakeCone(r1, r2, h, angle).Shape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cone:
            return face
        exp.Next()
    raise AssertionError("no cone face in generated solid")


@pytest.fixture()
def full_cone_face_and_frame() -> tuple[TopoDS_Face, ConeTrimFrame]:
    """Full cone: R1=3, R2=0, H=9.  Apex at (0, 0, 9), base at z=0."""
    face = _cone_face(3.0, 0.0, 9.0)
    tf = extract_cone_trim_frame(face)
    assert tf is not None
    return face, tf


@pytest.fixture()
def half_cone_face_and_frame() -> tuple[TopoDS_Face, ConeTrimFrame]:
    """Half revolution (u in [0, pi]) of the same cone."""
    face = _cone_face(3.0, 0.0, 9.0, math.pi)
    tf = extract_cone_trim_frame(face)
    assert tf is not None
    return face, tf


@pytest.fixture()
def frustum_face_and_frame() -> tuple[TopoDS_Face, ConeTrimFrame]:
    """Frustum: R1=3 at z=0, R2=1 at z=9; cone face has v in [0, ~9.22]."""
    face = _cone_face(3.0, 1.0, 9.0)
    tf = extract_cone_trim_frame(face)
    assert tf is not None
    return face, tf


def _occt_face_distance(face: TopoDS_Face, query: np.ndarray) -> float:
    vertex = BRepBuilderAPI_MakeVertex(
        gp_Pnt(float(query[0]), float(query[1]), float(query[2]))
    ).Vertex()
    dss = BRepExtrema_DistShapeShape(face, vertex)
    dss.Perform()
    assert dss.IsDone()
    return float(dss.Value())


class TestFullConeUntrimmedCollapse:
    """Off-seam queries should match the cone primitive's signed distance."""

    def test_radially_outside_is_positive(self, full_cone_face_and_frame) -> None:
        # (0, 5, 3): on +y at cone height 3 (where cone radius = 2),
        # radially 3 units beyond the cone surface.  foot_u = pi/2
        # (off-seam).
        _, frame = full_cone_face_and_frame
        d = cone_face_sdf_from_frame(frame, jnp.array([0.0, 5.0, 3.0]))
        assert float(d) > 0.0

    def test_inside_cone_is_negative(self, full_cone_face_and_frame) -> None:
        # (0, 0.5, 3): on axis side at cone height 3; cone radius 2,
        # query is 1.5 units inside.
        _, frame = full_cone_face_and_frame
        d = cone_face_sdf_from_frame(frame, jnp.array([0.0, 0.5, 3.0]))
        assert float(d) < 0.0

    @pytest.mark.parametrize(
        "query",
        [
            np.array([0.0, 5.0, 3.0]),  # outside, off-seam
            np.array([0.0, 0.5, 3.0]),  # inside, off-seam
            np.array([0.0, 3.5, 5.0]),  # outside near apex
            np.array([1.0, 2.0, 4.0]),  # generic inside-trim
        ],
    )
    def test_magnitude_matches_occt(
        self, full_cone_face_and_frame, query: np.ndarray
    ) -> None:
        face, frame = full_cone_face_and_frame
        d = float(
            cone_face_sdf_from_frame(frame, jnp.asarray(query, dtype=jnp.float64))
        )
        d_occt = _occt_face_distance(face, query)
        assert jnp.isclose(abs(d), d_occt, atol=0.15), (
            f"query={query} marschner={d} occt={d_occt}"
        )


class TestHalfConePhantomElimination:
    """u outside [0, pi] flips sign via the signed blend."""

    def test_inside_trim_inside_cone_is_negative(
        self, half_cone_face_and_frame
    ) -> None:
        # u = pi/2, inside the half range; on axis at cone height 3
        # offset 0.5 in +y so foot_u = pi/2.  Inside cone body.
        _, frame = half_cone_face_and_frame
        d = cone_face_sdf_from_frame(frame, jnp.array([0.0, 0.5, 3.0]))
        assert float(d) < 0.0

    def test_inside_trim_outside_cone_is_positive(
        self, half_cone_face_and_frame
    ) -> None:
        _, frame = half_cone_face_and_frame
        d = cone_face_sdf_from_frame(frame, jnp.array([0.0, 5.0, 3.0]))
        assert float(d) > 0.0

    def test_outside_trim_inside_cone_phantom_eliminated(
        self, half_cone_face_and_frame
    ) -> None:
        # (0, -0.5, 3): foot_u = 3*pi/2 (outside [0, pi]); untrimmed
        # primitive says the query is inside the cone body (1.5 units
        # inside).  Marschner should flip this to positive.
        _, frame = half_cone_face_and_frame
        d = cone_face_sdf_from_frame(frame, jnp.array([0.0, -0.5, 3.0]))
        assert float(d) > 0.0, f"phantom not eliminated: d_T={float(d)}"


class TestFrustumVBoundaryPhantomElimination:
    """A v-trimmed (frustum) cone face exercises the other trim axis.

    The frustum built from ``MakeCone(3, 1, 9)`` has a cone face whose
    slant ``v`` is bounded to ``[0, ~9.22]``.  A query inside the
    untrimmed cone body but axially above the top disk (``foot_v``
    beyond the upper bound) must be flipped to positive by the signed
    blend.
    """

    def test_v_above_top_phantom_eliminated(self, frustum_face_and_frame) -> None:
        # (0.3, 0, 10): axial = 10 (above top disk at z=9), r = 0.3.
        # At v = 10 / cos(semi_angle) ~= 10.24 the extrapolated cone
        # radius ~= 0.78, so the query (r=0.3) is inside the untrimmed
        # cone body: d_s_raw < 0.  foot_v ~= 10.24 sits outside the
        # frustum's trim [0, ~9.22], so chi collapses to 0 and the
        # signed blend flips the sign.
        _, frame = frustum_face_and_frame
        d = cone_face_sdf_from_frame(frame, jnp.array([0.3, 0.0, 10.0]))
        assert float(d) > 0.0, f"phantom not eliminated at v_max: d_T={float(d)}"

    def test_v_below_base_phantom_eliminated(self, frustum_face_and_frame) -> None:
        # (0.0, 2.5, -1): axial = -1 (below base disk at z=0), r = 2.5.
        # At v = -1 / cos(semi_angle) the untrimmed cone extends back
        # to radius ref_radius - small; query is inside the untrimmed
        # body.  foot_v negative sits below the frustum's trim [0, _];
        # chi ~ 0 flips d_T positive.
        _, frame = frustum_face_and_frame
        d = cone_face_sdf_from_frame(frame, jnp.array([0.0, 2.5, -1.0]))
        assert float(d) > 0.0, f"phantom not eliminated at v_min: d_T={float(d)}"


class TestConvenienceWrapper:
    def test_convenience_matches_pure(self, full_cone_face_and_frame) -> None:
        face, frame = full_cone_face_and_frame
        query = jnp.array([0.0, 5.0, 3.0])
        d_conv = cone_face_sdf(face, query)
        d_pure = cone_face_sdf_from_frame(frame, query)
        assert d_conv is not None
        assert jnp.isclose(d_conv, d_pure, atol=1e-9)

    def test_convenience_returns_none_for_non_cone(self) -> None:
        shape = read_step(str(FIXTURES / "sample_box.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        d = cone_face_sdf(face, jnp.array([0.0, 0.0, 0.0]))
        assert d is None


class TestGradient:
    def test_grad_through_query_finite(self, full_cone_face_and_frame) -> None:
        _, frame = full_cone_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return cone_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 5.0, 3.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_on_axis(self, full_cone_face_and_frame) -> None:
        _, frame = full_cone_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return cone_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 0.0, 5.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_at_apex(self, full_cone_face_and_frame) -> None:
        # Query at the cone's apex (0, 0, 9).  radial=0 triggers the
        # double-where on arctan2 for foot_u.
        _, frame = full_cone_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return cone_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, 0.0, 9.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_in_phantom_regime(
        self, half_cone_face_and_frame
    ) -> None:
        _, frame = half_cone_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return cone_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([0.0, -0.5, 3.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_jit_composition_runs(self, full_cone_face_and_frame) -> None:
        _, frame = full_cone_face_and_frame
        jitted = jax.jit(lambda q: cone_face_sdf_from_frame(frame, q))
        d = jitted(jnp.array([0.0, 5.0, 3.0]))
        assert jnp.isfinite(d)
