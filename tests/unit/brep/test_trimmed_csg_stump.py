"""TrimmedCSGStump end-to-end on real CAD models.

Per ADR-0019, ``TrimmedCSGStump`` on analytical-only models is
bit-equivalent to ``DifferentiableCSGStump``: every analytical
primitive contributes its raw untrimmed half-space SDF to the DNF.
The Marschner trim-aware blend is reserved for the standalone-face
distance-query use case (handled by ``brep/trim_frame.py``'s
``*_face_sdf_from_frame`` wrappers and verified separately) and for
the future BSpline-patch path inside the composition.

These tests pin that invariant: ``sample_box`` and ``box_with_holes``
both go through the trim-aware composite *and* the untrimmed composite
on the same grid; their volumes must match within numerical noise.
The trim frames themselves are extracted and stored on the stump so
the BSpline integration has its per-slot frame ready, but they do not
contribute to the analytical SDFs.
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
