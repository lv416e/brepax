"""End-to-end trim-aware plane-face SDF on sample_box's top face."""

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
    GeomAbs_Plane,
    TopAbs_FACE,
    TopExp_Explorer,
    TopoDS,
    gp_Pnt,
)
from brepax.brep.trim_frame import (
    PlaneTrimFrame,
    extract_plane_trim_frame,
    plane_face_sdf,
    plane_face_sdf_from_frame,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def top_face_and_frame() -> tuple[object, PlaneTrimFrame]:
    """Top face of sample_box (z=30, normal most aligned with +z)."""
    shape = read_step(str(FIXTURES / "sample_box.step"))
    best = None
    best_score = -math.inf
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane:
            frame = extract_plane_trim_frame(face)
            assert frame is not None
            score = float(frame.normal[2]) + float(frame.origin[2]) / 30.0
            if score > best_score:
                best_score = score
                best = (face, frame)
        exp.Next()
    assert best is not None
    # Sanity: top face should have origin near z=30 and normal near +z.
    _, frame = best
    assert float(frame.origin[2]) == pytest.approx(30.0, abs=1e-6)
    assert float(frame.normal[2]) == pytest.approx(1.0, abs=1e-6)
    return best


def _occt_face_distance(face: object, query: np.ndarray) -> float:
    vertex = BRepBuilderAPI_MakeVertex(
        gp_Pnt(float(query[0]), float(query[1]), float(query[2]))
    ).Vertex()
    dss = BRepExtrema_DistShapeShape(face, vertex)
    dss.Perform()
    assert dss.IsDone()
    return float(dss.Value())


class TestFourRegimes:
    """ADR-0018 regime table, reproduced on a real OCCT plane face."""

    def test_inside_trim_below_plane_is_negative(self, top_face_and_frame) -> None:
        _, frame = top_face_and_frame
        # (5, 10, 25): inside trim [0,10]x[0,20], below z=30.
        d = plane_face_sdf_from_frame(frame, jnp.array([5.0, 10.0, 25.0]))
        assert float(d) < 0.0
        assert jnp.isclose(d, -5.0, atol=1e-2)

    def test_inside_trim_above_plane_is_positive(self, top_face_and_frame) -> None:
        _, frame = top_face_and_frame
        d = plane_face_sdf_from_frame(frame, jnp.array([5.0, 10.0, 35.0]))
        assert float(d) > 0.0
        assert jnp.isclose(d, 5.0, atol=1e-2)

    def test_outside_trim_below_plane_phantom_eliminated(
        self, top_face_and_frame
    ) -> None:
        # Critical regime: query is below the infinite half-space (d_s < 0)
        # but outside the trim polygon; without Marschner the sign would be
        # negative (phantom material) and the CSG-Stump PMC would classify
        # this point as inside the primitive.
        _, frame = top_face_and_frame
        d = plane_face_sdf_from_frame(frame, jnp.array([-5.0, 10.0, 25.0]))
        assert float(d) > 0.0, f"phantom not eliminated: d_T={float(d)}"

    def test_outside_trim_above_plane_is_positive(self, top_face_and_frame) -> None:
        _, frame = top_face_and_frame
        d = plane_face_sdf_from_frame(frame, jnp.array([-5.0, 10.0, 35.0]))
        assert float(d) > 0.0


class TestMagnitudeAgainstOCCT:
    """|d_T| should match OCCT's face distance well away from the trim boundary."""

    @pytest.mark.parametrize(
        "query",
        [
            np.array([5.0, 10.0, 25.0]),  # inside trim, below plane
            np.array([5.0, 10.0, 35.0]),  # inside trim, above plane
            np.array([-5.0, 10.0, 25.0]),  # phantom regime
            np.array([-5.0, 10.0, 35.0]),  # outside trim above plane
            np.array([15.0, 25.0, 25.0]),  # diagonally outside
        ],
    )
    def test_magnitude_within_1e_2_of_occt(
        self, top_face_and_frame, query: np.ndarray
    ) -> None:
        face, frame = top_face_and_frame
        d_marschner = float(
            plane_face_sdf_from_frame(frame, jnp.asarray(query, dtype=jnp.float64))
        )
        d_occt = _occt_face_distance(face, query)
        assert jnp.isclose(abs(d_marschner), d_occt, atol=1e-2), (
            f"query={query} marschner={d_marschner} occt={d_occt}"
        )


class TestConvenienceWrapper:
    """``plane_face_sdf`` matches ``plane_face_sdf_from_frame`` exactly."""

    def test_convenience_matches_pure(self, top_face_and_frame) -> None:
        face, frame = top_face_and_frame
        query = jnp.array([3.0, 7.0, 31.5])
        d_conv = plane_face_sdf(face, query)
        d_pure = plane_face_sdf_from_frame(frame, query)
        assert d_conv is not None
        assert jnp.isclose(d_conv, d_pure, atol=1e-9)

    def test_convenience_returns_none_for_non_plane(self) -> None:
        shape = read_step(str(FIXTURES / "sample_sphere.step"))
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        d = plane_face_sdf(face, jnp.array([0.0, 0.0, 0.0]))
        assert d is None


class TestOrientationAcrossAllFaces:
    """Each plane face of the box must report positive sdf 1 unit outside.

    Covers both ``TopAbs_FORWARD`` and ``TopAbs_REVERSED`` faces, because
    OCCT stores orientation independently of the underlying Geom_Plane
    normal and the extracted frame must flip the normal when needed.
    """

    def _iter_plane_faces(self, step_path: str) -> list:
        shape = read_step(step_path)
        faces = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            if BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane:
                faces.append(face)
            exp.Next()
        return faces

    def test_one_unit_outside_returns_positive_for_every_face(self) -> None:
        faces = self._iter_plane_faces(str(FIXTURES / "sample_box.step"))
        assert len(faces) == 6
        for face in faces:
            frame = extract_plane_trim_frame(face)
            assert frame is not None
            # Pick a point on the face's centroid then step 1 unit outward.
            n_valid = int(frame.mask.sum())
            centroid_uv = jnp.mean(frame.polygon_uv[:n_valid], axis=0)
            foot_3d = (
                frame.origin
                + centroid_uv[0] * frame.frame_u
                + centroid_uv[1] * frame.frame_v
            )
            outside = foot_3d + frame.normal
            d = plane_face_sdf_from_frame(frame, outside)
            assert float(d) > 0.0, f"face with normal {frame.normal}: d={float(d)}"
            assert jnp.isclose(d, 1.0, atol=1e-2)

    def test_one_unit_inside_returns_negative_for_every_face(self) -> None:
        faces = self._iter_plane_faces(str(FIXTURES / "sample_box.step"))
        for face in faces:
            frame = extract_plane_trim_frame(face)
            assert frame is not None
            n_valid = int(frame.mask.sum())
            centroid_uv = jnp.mean(frame.polygon_uv[:n_valid], axis=0)
            foot_3d = (
                frame.origin
                + centroid_uv[0] * frame.frame_u
                + centroid_uv[1] * frame.frame_v
            )
            inside = foot_3d - frame.normal
            d = plane_face_sdf_from_frame(frame, inside)
            assert float(d) < 0.0, f"face with normal {frame.normal}: d={float(d)}"
            assert jnp.isclose(d, -1.0, atol=1e-2)


class TestGradient:
    """jax.grad through query, plus through jitted composition."""

    def test_grad_through_query_finite(self, top_face_and_frame) -> None:
        _, frame = top_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return plane_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([5.0, 10.0, 25.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_query_finite_in_phantom_regime(
        self, top_face_and_frame
    ) -> None:
        # Phantom-regime query — gradient must still be finite.
        _, frame = top_face_and_frame

        def loss(q: jnp.ndarray) -> jnp.ndarray:
            return plane_face_sdf_from_frame(frame, q)

        g = jax.grad(loss)(jnp.array([-5.0, 10.0, 25.0]))
        assert jnp.all(jnp.isfinite(g))

    def test_jit_composition_runs(self, top_face_and_frame) -> None:
        # Confirm the pure composition is jittable end to end.
        _, frame = top_face_and_frame
        jitted = jax.jit(lambda q: plane_face_sdf_from_frame(frame, q))
        d = jitted(jnp.array([5.0, 10.0, 25.0]))
        assert jnp.isfinite(d)
