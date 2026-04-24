"""End-to-end trim-aware cylinder-face SDF against OCCT on box_with_holes.

box_with_holes is a box with two cylindrical through-holes (radii 3 and 4).
Each hole's inner surface is a REVERSED cylinder face in OCCT terms, so
``sign_flip = -1`` applies: a query inside the hole is *outside* the
primitive's half-space (outside the material), while a query in the
surrounding material (perp > radius, axially within the face) is *inside*.

Tests avoid queries with ``theta ~ 0`` or ``theta ~ 2*pi`` (i.e. on
the cylinder face's seam in UV).  The seam is a 2D boundary of the
trim polygon even though it is a single curve in 3D; any query that
projects onto the seam gets a winding-number-ambiguous trim-indicator
value and ``d_T`` collapses towards zero.  Making the composition
seam-invariant for full-revolution faces is a follow-up for a future
PR.
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
    GeomAbs_Cylinder,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
    gp_Pnt,
)
from brepax.brep.trim_frame import (
    CylinderTrimFrame,
    cylinder_face_sdf,
    cylinder_face_sdf_from_frame,
    extract_cylinder_trim_frame,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def hole_face_and_frame() -> tuple[object, CylinderTrimFrame]:
    """The radius-3 through-hole face at (30, 15, z=0..20) in box_with_holes.

    The hole's axis runs in +z; the face covers axial v in [1, 21]
    (box z in [0, 20], with OCCT origin at z=-1).  The face is
    REVERSED, so ``sign_flip = -1``.
    """
    shape = read_step(str(FIXTURES / "box_with_holes.step"))
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    chosen = None
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Cylinder:
            cyl = BRepAdaptor_Surface(face).Cylinder()
            if math.isclose(cyl.Radius(), 3.0, abs_tol=1e-6):
                chosen = face
                break
        exp.Next()
    assert chosen is not None, "radius-3 hole face not found"
    frame = extract_cylinder_trim_frame(chosen)
    assert frame is not None
    # Sanity: REVERSED faces on a through-hole.
    assert float(frame.sign_flip) == -1.0
    return chosen, frame


def _occt_face_distance(face: object, query: np.ndarray) -> float:
    vertex = BRepBuilderAPI_MakeVertex(
        gp_Pnt(float(query[0]), float(query[1]), float(query[2]))
    ).Vertex()
    dss = BRepExtrema_DistShapeShape(face, vertex)
    dss.Perform()
    assert dss.IsDone()
    return float(dss.Value())


class TestFourRegimes:
    """ADR-0018 regime table, reproduced on a real REVERSED cylinder face.

    sign_flip = -1 flips the primitive's own classification: inside the
    hole (perp < radius) is *outside* the primitive's half-space
    (outside the solid material), while the material-filled region
    (perp > radius) is *inside*.
    """

    def test_inside_trim_inside_hole_is_positive(self, hole_face_and_frame) -> None:
        # (30, 15, 10): on axis (perp=0), middle of face (v=11 in [1, 21]).
        # sign_flip * (0 - 3) = +3 => outside material.
        _, frame = hole_face_and_frame
        d = cylinder_face_sdf_from_frame(frame, jnp.array([30.0, 15.0, 10.0]))
        assert float(d) > 0.0
        assert jnp.isclose(d, 3.0, atol=1e-2)

    def test_inside_trim_in_material_is_negative(self, hole_face_and_frame) -> None:
        # (30, 20, 10): perp=5 along +y (theta=pi/2, off-seam),
        # middle of face (v=11).  sign_flip * (5 - 3) = -2 =>
        # inside material.
        _, frame = hole_face_and_frame
        d = cylinder_face_sdf_from_frame(frame, jnp.array([30.0, 20.0, 10.0]))
        assert float(d) < 0.0
        assert jnp.isclose(d, -2.0, atol=1e-2)

    def test_outside_trim_in_material_phantom_eliminated(
        self, hole_face_and_frame
    ) -> None:
        # Key phantom case: perp=5 along +y (theta=pi/2, off-seam),
        # v=26 is above the face's axial range [1, 21] (outside the
        # box entirely).  Untrimmed primitive says "inside material"
        # with sign_flip * (5 - 3) = -2; Marschner must flip this.
        _, frame = hole_face_and_frame
        d = cylinder_face_sdf_from_frame(frame, jnp.array([30.0, 20.0, 25.0]))
        assert float(d) > 0.0, f"phantom not eliminated: d_T={float(d)}"
        # Nearest boundary point: top-rim at theta=pi/2, v=21 i.e.
        # (30, 18, 20); distance = sqrt(0 + 4 + 25) ~= 5.385.
        assert jnp.isclose(d, math.sqrt(29.0), atol=1e-2)

    def test_outside_trim_inside_hole_is_positive(self, hole_face_and_frame) -> None:
        # (30, 15, 25): on axis, v=26 above the face.  Untrimmed
        # primitive already says outside (perp=0 < radius gives
        # sign_flip * (0 - 3) = +3 > 0), but d_T should *grow* to
        # reflect the actual distance to the face boundary, not to
        # the infinite cylinder.
        _, frame = hole_face_and_frame
        d = cylinder_face_sdf_from_frame(frame, jnp.array([30.0, 15.0, 25.0]))
        assert float(d) > 0.0
        # Nearest analytical point on trim boundary: anywhere on the
        # top circle at z=20 radius 3; distance from (30,15,25) =
        # sqrt(9 + 25) = sqrt(34) ~= 5.83.  The polyline is an
        # inscribed octagon so its chord-midpoint distance is slightly
        # smaller (~5.72); we tolerate the 8-samples-per-edge
        # discretisation error with a wide band.
        assert 5.5 < float(d) < 6.0


class TestMagnitudeAgainstOCCT:
    """|d_T| matches OCCT's face distance within the polyline tolerance.

    The 8-samples-per-edge discretisation turns the hole's top and
    bottom circles into inscribed octagons, so when a query projects
    onto the rim (in the phantom regime or the axially-above-axis
    regime), the polyline distance underestimates the analytical
    circle distance by up to about 2% of the radius (~0.1 mm here).
    We allow an absolute 0.15 mm slack to accommodate that systematic
    error; tightening it requires raising the sample-per-edge count.
    """

    @pytest.mark.parametrize(
        "query",
        [
            np.array([30.0, 15.0, 10.0]),  # inside hole, middle of face (axis)
            np.array([30.0, 20.0, 10.0]),  # inside material, middle (theta=pi/2)
            np.array([30.0, 16.0, 10.0]),  # just inside hole, near wall
            np.array([30.0, 20.0, 25.0]),  # phantom regime (theta=pi/2)
            np.array([30.0, 15.0, 25.0]),  # axis-on, axially above
            np.array([30.0, 19.0, 5.0]),  # generic inside-trim (theta=pi/2, perp=4)
        ],
    )
    def test_magnitude_within_polyline_tolerance_of_occt(
        self, hole_face_and_frame, query: np.ndarray
    ) -> None:
        face, frame = hole_face_and_frame
        d_marschner = float(
            cylinder_face_sdf_from_frame(frame, jnp.asarray(query, dtype=jnp.float64))
        )
        d_occt = _occt_face_distance(face, query)
        assert jnp.isclose(abs(d_marschner), d_occt, atol=0.15), (
            f"query={query} marschner={d_marschner} occt={d_occt}"
        )


class TestConvenienceWrapper:
    def test_convenience_matches_pure(self, hole_face_and_frame) -> None:
        face, frame = hole_face_and_frame
        # Off-seam query (theta=pi/2 direction, v=9).
        query = jnp.array([30.0, 17.5, 8.0])
        d_conv = cylinder_face_sdf(face, query)
        d_pure = cylinder_face_sdf_from_frame(frame, query)
        assert d_conv is not None
        assert jnp.isclose(d_conv, d_pure, atol=1e-9)

    def test_convenience_returns_none_for_non_cylinder(self) -> None:
        shape = read_step(str(FIXTURES / "sample_box.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        d = cylinder_face_sdf(face, jnp.array([0.0, 0.0, 0.0]))
        assert d is None


class TestGradient:
    def test_grad_through_query_finite(self, hole_face_and_frame) -> None:
        _, frame = hole_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return cylinder_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([30.0, 17.5, 8.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_in_phantom_regime(
        self, hole_face_and_frame
    ) -> None:
        _, frame = hole_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return cylinder_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([30.0, 20.0, 25.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_on_axis(self, hole_face_and_frame) -> None:
        # Query exactly on the hole's axis triggers the safe-square
        # fallback in cylinder_face_sdf_from_frame; gradient must
        # still be finite.
        _, frame = hole_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return cylinder_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([30.0, 15.0, 10.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_jit_composition_runs(self, hole_face_and_frame) -> None:
        _, frame = hole_face_and_frame
        jitted = jax.jit(lambda q: cylinder_face_sdf_from_frame(frame, q))
        d = jitted(jnp.array([30.0, 17.5, 8.0]))
        assert jnp.isfinite(d)
