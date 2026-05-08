"""TrimmedCSGStump end-to-end on real CAD models.

Per ADR-0019 and ADR-0020, ``TrimmedCSGStump`` is bit-equivalent to
``DifferentiableCSGStump`` on every fixture, analytical or BSpline:
every primitive contributes its raw untrimmed signed distance to the
DNF.  The Marschner trim-aware blend is reserved for the standalone-
face distance-query use case (``brep/trim_frame.py``'s
``*_face_sdf_from_frame`` family) and for future composition
strategies; substituting it into the CSG-Stump DNF removes
legitimate inside regions on multi-face solids (Linkrods volume
collapses by ~99% when BSpline slots route through Marschner — the
empirical evidence behind ADR-0020).

These tests pin the bit-equivalence invariant on three fixtures:

- ``sample_box`` (6 planes): analytical control.
- ``box_with_holes`` (6 planes + 2 cylinders, REVERSED orientation
  on cylinders): non-trivial DNF analytical control.
- ``nurbs_box`` (6 BSpline patches): BSpline regression check.

The trim frames themselves (``BSplineTrimFrame`` and the analytical
``*TrimFrame`` set) are still extracted and stored on the stump, both
because the standalone-face wrappers consume them and because future
composition strategies (e.g. GWN) will consume the same metadata.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from brepax.brep.convert import shape_metadata
from brepax.brep.csg_stump import reconstruct_csg_stump, stump_to_differentiable
from brepax.brep.trimmed_csg_stump import (
    TrimmedCSGStump,
    enrich_with_trim_frames,
)
from brepax.io.step import read_step


def _shape_bounds(shape):
    """Padded bounding box from OCCT metadata, suitable for grid integration."""
    meta = shape_metadata(shape)
    lo = jnp.asarray(meta.bbox_min, dtype=jnp.float64) - 0.5
    hi = jnp.asarray(meta.bbox_max, dtype=jnp.float64) + 0.5
    return lo, hi


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture(scope="module")
def sample_box_stump_and_trimmed() -> tuple:
    shape = read_step(str(FIXTURES / "sample_box.step"))
    stump = reconstruct_csg_stump(shape)
    assert stump is not None
    trimmed = enrich_with_trim_frames(stump, shape)
    return shape, stump, trimmed


@pytest.fixture(scope="module")
def box_with_holes_stump_and_trimmed() -> tuple:
    shape = read_step(str(FIXTURES / "box_with_holes.step"))
    stump = reconstruct_csg_stump(shape)
    assert stump is not None
    trimmed = enrich_with_trim_frames(stump, shape)
    return shape, stump, trimmed


class TestTrimmedSampleBoxMatchesUntrimmed:
    """sample_box: trimmed and untrimmed composites are bit-equivalent."""

    def test_class_instance(self, sample_box_stump_and_trimmed) -> None:
        _, _, trimmed = sample_box_stump_and_trimmed
        assert isinstance(trimmed, TrimmedCSGStump)

    def test_frames_match_primitive_count(self, sample_box_stump_and_trimmed) -> None:
        _, stump, trimmed = sample_box_stump_and_trimmed
        assert len(trimmed.frames) == len(stump.primitives)

    def test_primitives_match_stump(self, sample_box_stump_and_trimmed) -> None:
        _, stump, trimmed = sample_box_stump_and_trimmed
        assert len(trimmed.primitives) == len(stump.primitives)

    def test_volume_uses_stored_bounds_when_unspecified(
        self, sample_box_stump_and_trimmed
    ) -> None:
        # enrich_with_trim_frames stashes bbox on the stump, so
        # volume() without explicit ``lo``/``hi`` must succeed.
        _, _, trimmed = sample_box_stump_and_trimmed
        v = float(trimmed.volume(resolution=32))
        assert v > 0.0

    def test_gradient_flows_through_primitives(
        self, sample_box_stump_and_trimmed
    ) -> None:
        # Analytical primitives carry the differentiable parameters
        # (radius, axis, plane normal/offset).  jax.grad of an SDF
        # objective must produce finite gradients on those fields.
        import jax

        _, _, trimmed = sample_box_stump_and_trimmed

        def loss(t):
            return jnp.sum(t.sdf(jnp.array([5.0, 10.0, 15.0])) ** 2)

        g = jax.grad(loss)(trimmed)
        # Plane primitives expose ``normal`` and ``offset`` as JAX
        # arrays.  Confirm at least one slot received a finite grad.
        assert jnp.all(jnp.isfinite(g.primitives[0].normal))

    def test_sdf_signs_match_untrimmed(self, sample_box_stump_and_trimmed) -> None:
        # Inside the box => both SDFs negative; outside => both
        # positive.  Under ADR-0019 they agree exactly; sign equality
        # is the weakest check that still confirms wiring sanity.
        _, stump, trimmed = sample_box_stump_and_trimmed
        diff = stump_to_differentiable(stump)
        queries = jnp.array(
            [
                [5.0, 10.0, 15.0],  # dead centre
                [-3.0, 10.0, 15.0],  # well outside
                [5.0, 10.0, -3.0],
                [12.0, 22.0, 32.0],
            ]
        )
        d_untrimmed = diff.sdf(queries)
        d_trimmed = trimmed.sdf(queries)
        assert jnp.all(jnp.sign(d_untrimmed) == jnp.sign(d_trimmed))

    def test_volume_matches_untrimmed(self, sample_box_stump_and_trimmed) -> None:
        # Trimmed and untrimmed composites must agree on volume to
        # within floating-point noise: every analytical primitive
        # contributes the same raw SDF in both paths (ADR-0019).
        shape, stump, trimmed = sample_box_stump_and_trimmed
        diff = stump_to_differentiable(stump)
        lo, hi = _shape_bounds(shape)
        v_untrimmed = float(diff.volume(resolution=48, lo=lo, hi=hi))
        v_trimmed = float(trimmed.volume(resolution=48, lo=lo, hi=hi))
        assert abs(v_untrimmed - v_trimmed) < 1e-3


class TestTrimmedBoxWithHolesMatchesUntrimmed:
    """box_with_holes: planes plus cylindrical holes, all analytical.

    Under ADR-0019 every primitive uses its raw untrimmed half-space
    SDF, so the trimmed composite agrees bit-exactly with the
    untrimmed composite on this fixture too.  The fixture is kept in
    the test set because it exercises a non-trivial DNF with
    cylinder slots whose face orientation is REVERSED, ensuring no
    sign or matrix-flip artefacts have crept back in.
    """

    def test_volume_matches_untrimmed(self, box_with_holes_stump_and_trimmed) -> None:
        shape, stump, trimmed = box_with_holes_stump_and_trimmed
        diff = stump_to_differentiable(stump)
        lo, hi = _shape_bounds(shape)
        v_untrimmed = float(diff.volume(resolution=48, lo=lo, hi=hi))
        v_trimmed = float(trimmed.volume(resolution=48, lo=lo, hi=hi))
        assert abs(v_untrimmed - v_trimmed) < 1e-3

    def test_above_box_query_is_outside(self, box_with_holes_stump_and_trimmed) -> None:
        # box_with_holes is the rectangular block [0,40] x [0,30] x
        # [0,20].  A query directly above the top face (z=25) must be
        # classified as outside by the trim-aware composite via the
        # box's top-plane half-space, irrespective of cylinder slots.
        _, _, trimmed = box_with_holes_stump_and_trimmed
        d = float(trimmed.sdf(jnp.array([10.0, 15.0, 25.0])))
        assert d > 0.0, f"above-box query classified as inside: d={d}"


@pytest.fixture(scope="module")
def nurbs_box_stump_and_trimmed() -> tuple:
    shape = read_step(str(FIXTURES / "nurbs_box.step"))
    stump = reconstruct_csg_stump(shape)
    assert stump is not None
    trimmed = enrich_with_trim_frames(stump, shape)
    return shape, stump, trimmed


class TestTrimmedNurbsBoxMatchesUntrimmed:
    """nurbs_box: 6 BSpline patches, all under raw primitive SDF.

    Per ADR-0020, BSpline slots in ``TrimmedCSGStump`` use raw
    ``primitive.sdf(query)`` just like analytical slots.  The trim
    frames are still extracted (for the standalone-face query path
    and future composition strategies) but do not contribute to the
    DNF SDF.  This fixture's role is to lock in the bit-equivalence
    invariant on a BSpline-bearing solid, so any future regression
    of the BSpline DNF path against ``DifferentiableCSGStump``
    surfaces immediately.
    """

    def test_class_instance(self, nurbs_box_stump_and_trimmed) -> None:
        _, _, trimmed = nurbs_box_stump_and_trimmed
        assert isinstance(trimmed, TrimmedCSGStump)

    def test_all_slots_are_bspline_frames(self, nurbs_box_stump_and_trimmed) -> None:
        from brepax.brep.trim_frame import BSplineTrimFrame

        _, _, trimmed = nurbs_box_stump_and_trimmed
        for f in trimmed.frames:
            assert isinstance(f, BSplineTrimFrame)

    def test_volume_matches_untrimmed(self, nurbs_box_stump_and_trimmed) -> None:
        # Trimmed and untrimmed composites must agree on volume to
        # within floating-point noise: every BSpline primitive
        # contributes the same raw SDF in both paths (ADR-0020).
        shape, stump, trimmed = nurbs_box_stump_and_trimmed
        diff = stump_to_differentiable(stump)
        lo, hi = _shape_bounds(shape)
        # res=24 keeps the test under a minute on BSpline projection.
        v_untrimmed = float(diff.volume(resolution=24, lo=lo, hi=hi))
        v_trimmed = float(trimmed.volume(resolution=24, lo=lo, hi=hi))
        assert abs(v_untrimmed - v_trimmed) < 1e-3

    def test_sdf_signs_match_untrimmed(self, nurbs_box_stump_and_trimmed) -> None:
        # nurbs_box is the rectangular block [0,10] x [0,8] x [-8,0];
        # a few interior / exterior queries must agree on sign with
        # the untrimmed composite (and on value, under ADR-0020).
        _, stump, trimmed = nurbs_box_stump_and_trimmed
        diff = stump_to_differentiable(stump)
        queries = jnp.array(
            [
                [5.0, 4.0, -4.0],  # interior
                [-2.0, 4.0, -4.0],  # outside, away from any face
                [12.0, 4.0, -4.0],  # outside on +x
                [5.0, 4.0, 5.0],  # outside on +z
            ]
        )
        d_untrimmed = diff.sdf(queries)
        d_trimmed = trimmed.sdf(queries)
        assert jnp.all(jnp.sign(d_untrimmed) == jnp.sign(d_trimmed))

    def test_gradient_flows_through_control_points(
        self, nurbs_box_stump_and_trimmed
    ) -> None:
        # ``primitive.sdf`` on a BSpline runs the unrolled Newton
        # projection, so jax.grad over a single SDF evaluation must
        # produce finite gradients on at least one slot's
        # ``control_points`` field.
        import jax

        _, _, trimmed = nurbs_box_stump_and_trimmed

        def loss(t):
            return jnp.sum(t.sdf(jnp.array([5.0, 4.0, -4.0])) ** 2)

        g = jax.grad(loss)(trimmed)
        assert jnp.all(jnp.isfinite(g.primitives[0].control_points))
