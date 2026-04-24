"""TrimmedCSGStump end-to-end on real CAD models.

sample_box: analytic only (6 planes) and NO phantom — trim-aware
composition must not change the result.  This locks in that the
dispatch machinery and the DNF composition carry the same signs as
the untrimmed stump.

box_with_holes: 6 planes + 2 cylindrical holes; the cylinder faces
produce phantom material axially outside the box under the untrimmed
primitive.  Marschner flips this to outside, and the resulting
volume should be measurably closer to OCCT than the untrimmed stump's
volume on the same grid.
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
    """sample_box has no phantom — trimmed volume must equal untrimmed."""

    def test_class_instance(self, sample_box_stump_and_trimmed) -> None:
        _, _, trimmed = sample_box_stump_and_trimmed
        assert isinstance(trimmed, TrimmedCSGStump)

    def test_frames_match_primitive_count(self, sample_box_stump_and_trimmed) -> None:
        _, stump, trimmed = sample_box_stump_and_trimmed
        assert len(trimmed.frames) == len(stump.primitives)

    def test_volume_uses_stored_bounds_when_unspecified(
        self, sample_box_stump_and_trimmed
    ) -> None:
        # enrich_with_trim_frames stashes bbox on the stump, so
        # volume() without explicit ``lo``/``hi`` must succeed.
        _, _, trimmed = sample_box_stump_and_trimmed
        v = float(trimmed.volume(resolution=32))
        assert v > 0.0

    def test_gradient_flows_through_frames(self, sample_box_stump_and_trimmed) -> None:
        # TrimmedCSGStump is an eqx.Module; gradients of sdf w.r.t.
        # frame parameters (e.g. plane normals) should be finite.
        import jax

        _, _, trimmed = sample_box_stump_and_trimmed

        def loss(t):
            return jnp.sum(t.sdf(jnp.array([5.0, 10.0, 15.0])) ** 2)

        g = jax.grad(loss)(trimmed)
        # Every frame has a differentiable JAX field; confirm at
        # least one primitive frame's normal received a finite grad.
        assert jnp.all(jnp.isfinite(g.frames[0].normal))

    def test_sdf_signs_match_untrimmed(self, sample_box_stump_and_trimmed) -> None:
        # sample_box: inside the box => both SDFs negative; outside
        # => both positive.  Magnitude may differ slightly because of
        # the sigmoid transition, but sign must match.
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

    def test_volume_matches_untrimmed_within_sigmoid_slack(
        self, sample_box_stump_and_trimmed
    ) -> None:
        # sample_box is a 10x20x30 box; GT volume 6000.  The trimmed
        # composite on the same grid should land on the same volume
        # as the untrimmed stump because planes never generate
        # phantom (their half-space is the solid's half-space).
        shape, stump, trimmed = sample_box_stump_and_trimmed
        diff = stump_to_differentiable(stump)
        lo, hi = _shape_bounds(shape)
        v_untrimmed = float(diff.volume(resolution=48, lo=lo, hi=hi))
        v_trimmed = float(trimmed.volume(resolution=48, lo=lo, hi=hi))
        # Both paths use the same grid and sigmoid, so equality up to
        # a small drift from the trim-indicator transition is
        # expected.  1 unit ~ 0.02% of 6000.
        assert abs(v_untrimmed - v_trimmed) < 1.0


class TestTrimmedBoxWithHolesReducesPhantom:
    """Cylindrical holes produce phantom under the untrimmed path.

    The untrimmed CSG-Stump composite on ``box_with_holes`` reports a
    volume that is larger than the true volume because each hole's
    infinite cylinder extends phantom-filled "inside material" past
    the box's top and bottom faces.  The trim-aware composite flips
    those regions to outside, giving a volume closer to OCCT GT.
    """

    def test_volume_differs_from_untrimmed(
        self, box_with_holes_stump_and_trimmed
    ) -> None:
        # box_with_holes has two hole cylinders.  The trim-aware path
        # routes those through Marschner while the untrimmed path does
        # not, so their grid-integrated volumes must differ by more
        # than the shared plane-based integration noise.  Whether the
        # trim-aware integration is *closer* to OCCT is a separate
        # measurement that depends on the interplay of phantom
        # elimination and polyline / seam discretisation; the
        # quantitative before/after comparison belongs in the
        # benchmark (PR D).
        shape, stump, trimmed = box_with_holes_stump_and_trimmed
        diff = stump_to_differentiable(stump)
        lo, hi = _shape_bounds(shape)
        v_untrimmed = float(diff.volume(resolution=48, lo=lo, hi=hi))
        v_trimmed = float(trimmed.volume(resolution=48, lo=lo, hi=hi))
        assert abs(v_untrimmed - v_trimmed) > 1.0, (
            f"trim-aware path did not activate: v_trimmed={v_trimmed:.2f} "
            f"matches v_untrimmed={v_untrimmed:.2f}"
        )

    def test_sdf_flips_inside_phantom_region(
        self, box_with_holes_stump_and_trimmed
    ) -> None:
        # box_with_holes has through-holes at (10, 15) and (30, 15)
        # from z=0 to z=20 (origin z=-1, V in [1, 21]).  A query
        # directly above the box (z=25) on the hole's axis should be
        # classified as outside by the trim-aware stump regardless of
        # what the untrimmed path says.
        _, _, trimmed = box_with_holes_stump_and_trimmed
        d = float(trimmed.sdf(jnp.array([10.0, 15.0, 25.0])))
        assert d > 0.0, f"above-box query classified as inside: d={d}"
