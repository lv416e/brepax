"""Unit tests for draft angle violation metric."""

import jax
import jax.numpy as jnp

from brepax.metrics.draft_angle import draft_angle_violation
from brepax.primitives import Box, Sphere


class TestDraftAngleViolation:
    """Tests for the draft_angle_violation function."""

    def test_box_side_walls_violate(self) -> None:
        """Box side walls (0 draft) violate any positive min_angle."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        violation = draft_angle_violation(
            box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=64
        )
        # 4 side walls each 2x2 = total area 16; top+bottom have 90 draft
        total_area = 24.0
        side_area = 16.0
        assert float(violation) > side_area * 0.5, (
            f"violation={float(violation):.2f}, expected >{side_area * 0.5:.1f}"
        )
        assert float(violation) < total_area, (
            f"violation={float(violation):.2f} should be less than total {total_area}"
        )

    def test_box_top_bottom_no_violation(self) -> None:
        """Box top/bottom faces (90 draft) should not violate."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        # min_angle=80: only surfaces with < 80 draft violate
        violation = draft_angle_violation(
            box.sdf, d, jnp.radians(80.0), lo=lo, hi=hi, resolution=64
        )
        total_area = 24.0
        # All 4 side walls (0 draft) + edges near top/bottom violate
        # but top/bottom faces themselves (90 draft) should not
        assert float(violation) < total_area, (
            f"violation={float(violation):.2f} should be less than total"
        )

    def test_zero_min_angle_no_violation(self) -> None:
        """Zero min_angle produces near-zero violation."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        violation = draft_angle_violation(box.sdf, d, 0.0, lo=lo, hi=hi, resolution=64)
        total_area = 24.0
        assert float(violation) < total_area * 0.3, (
            f"violation={float(violation):.2f} should be small for zero min_angle"
        )

    def test_increasing_min_angle_increases_violation(self) -> None:
        """Higher min_angle threshold produces more violation."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.5))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        v_small = draft_angle_violation(
            sphere.sdf, d, jnp.radians(10.0), lo=lo, hi=hi, resolution=48
        )
        v_large = draft_angle_violation(
            sphere.sdf, d, jnp.radians(45.0), lo=lo, hi=hi, resolution=48
        )
        assert float(v_large) > float(v_small), (
            f"45 violation ({float(v_large):.2f}) should exceed "
            f"10 violation ({float(v_small):.2f})"
        )

    def test_differentiable_wrt_min_angle(self) -> None:
        """jax.grad w.r.t. min_angle is non-negative."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])

        def loss(angle: jnp.ndarray) -> jnp.ndarray:
            return draft_angle_violation(box.sdf, d, angle, lo=lo, hi=hi, resolution=48)

        grad_angle = jax.grad(loss)(jnp.radians(5.0))
        assert jnp.isfinite(grad_angle), f"Non-finite gradient: {grad_angle}"
        assert float(grad_angle) >= 0.0, (
            f"Expected non-negative gradient, got {float(grad_angle):.4f}"
        )

    def test_differentiable_wrt_direction(self) -> None:
        """jax.grad w.r.t. mold direction is finite."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def loss(d: jnp.ndarray) -> jnp.ndarray:
            return draft_angle_violation(
                box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=48
            )

        grad_d = jax.grad(loss)(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.all(jnp.isfinite(grad_d)), f"Non-finite gradient: {grad_d}"

    def test_differentiable_wrt_shape(self) -> None:
        """jax.grad w.r.t. shape parameters works."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])

        def loss(he: jnp.ndarray) -> jnp.ndarray:
            box = Box(center=jnp.zeros(3), half_extents=he)
            return draft_angle_violation(
                box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=48
            )

        grad_he = jax.grad(loss)(jnp.array([2.0, 1.5, 1.0]))
        assert jnp.all(jnp.isfinite(grad_he)), f"Non-finite gradient: {grad_he}"

    def test_jit_compatible(self) -> None:
        """draft_angle_violation works under jax.jit."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        @jax.jit
        def compute(d: jnp.ndarray) -> jnp.ndarray:
            return draft_angle_violation(
                box.sdf, d, jnp.radians(5.0), lo=lo, hi=hi, resolution=32
            )

        result = compute(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isfinite(result)
        assert float(result) > 0.0

    def test_sphere_equator_violates(self) -> None:
        """Sphere equator (0 draft) violates; poles (90 draft) do not."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.5))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        d = jnp.array([0.0, 0.0, 1.0])
        violation = draft_angle_violation(
            sphere.sdf, d, jnp.radians(30.0), lo=lo, hi=hi, resolution=64
        )
        total_area = 4.0 * jnp.pi * 1.5**2
        # Equatorial band with draft < 30 should be a fraction of total
        assert float(violation) > 0.0
        assert float(violation) < float(total_area), (
            f"violation={float(violation):.2f} should be partial, not total"
        )


class TestMinDraftAnglePerFace:
    """Per-face minimum draft angle against analytical / OCCT geometry.

    The M6.1 milestone gate clause "Trim-aware face-level metrics"
    requires each face-level metric to be exposed and verified. For
    draft angle the per-face value is the worst-case (minimum) angle
    between the face's surface normal and the mold pull direction's
    perpendicular plane: ``arcsin(|n . mold|)``.

    Trim awareness comes from OCCT BRepMesh; the verification here is
    geometric (the analytical primitives have known per-face draft
    angles for axis-aligned mold directions).
    """

    @staticmethod
    def _read(name: str):
        from pathlib import Path

        from brepax.io.step import read_step

        fixtures_dir = Path(__file__).resolve().parents[2] / "fixtures"
        return read_step(str(fixtures_dir / f"{name}.step"))

    def test_sample_box_axis_aligned(self) -> None:
        """``sample_box`` faces are axis-aligned planes.  For mold = +Z,
        4 side faces give draft 0 and 2 cap faces give draft pi/2."""
        from brepax.metrics.draft_angle import min_draft_angle_per_face

        shape = self._read("sample_box")
        mold = jnp.array([0.0, 0.0, 1.0])
        angles, params = min_draft_angle_per_face(shape, mold)

        assert angles.shape == (len(params),) == (6,)

        # Sort to check the multiset matches: 4 zeros + 2 pi/2.
        sorted_angles = jnp.sort(angles)
        sorted_deg = jnp.rad2deg(sorted_angles)
        # 4 side faces near 0 degrees, 2 cap faces near 90 degrees.
        assert jnp.all(sorted_deg[:4] < 0.5), (
            f"expected 4 side faces ~0 deg, got sorted_deg={sorted_deg}"
        )
        assert jnp.all(sorted_deg[4:] > 89.5), (
            f"expected 2 cap faces ~90 deg, got sorted_deg={sorted_deg}"
        )

    def test_mold_direction_is_normalised(self) -> None:
        """A non-unit mold direction should give the same answer as its
        unit version (the function normalises internally)."""
        from brepax.metrics.draft_angle import min_draft_angle_per_face

        shape = self._read("sample_box")
        mold_unit = jnp.array([0.0, 0.0, 1.0])
        mold_scaled = jnp.array([0.0, 0.0, 7.5])
        a1, _ = min_draft_angle_per_face(shape, mold_unit)
        a2, _ = min_draft_angle_per_face(shape, mold_scaled)
        assert jnp.allclose(a1, a2, atol=1e-6)

    def test_axis_swap_relabels_faces(self) -> None:
        """Switching mold from +Z to +X relabels which faces are caps:
        the +X / -X side faces become caps, the +Z / -Z cap faces
        become sides.  The multiset of per-face draft angles must
        contain four zeros and two pi/2 in either case."""
        from brepax.metrics.draft_angle import min_draft_angle_per_face

        shape = self._read("sample_box")
        for mold in (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])):
            angles, _ = min_draft_angle_per_face(shape, mold)
            sorted_deg = jnp.rad2deg(jnp.sort(angles))
            assert jnp.all(sorted_deg[:4] < 0.5)
            assert jnp.all(sorted_deg[4:] > 89.5)

    def test_finite_and_in_range_on_all_fixtures(self) -> None:
        """Smoke property: every per-face angle is in ``[0, pi/2]`` on
        every fixture in the standard set, regardless of fixture
        geometry."""
        from brepax.metrics.draft_angle import min_draft_angle_per_face

        fixtures = [
            "sample_box",
            "sample_cylinder",
            "sample_sphere",
            "sample_cone",
            "sample_torus",
            "box_with_holes",
            "box_with_pocket",
            "box_with_slot",
            "l_bracket",
            "nurbs_box",
        ]
        mold = jnp.array([0.0, 0.0, 1.0])
        for name in fixtures:
            shape = self._read(name)
            angles, params = min_draft_angle_per_face(shape, mold)
            assert angles.shape == (len(params),)
            assert jnp.all(jnp.isfinite(angles)), f"{name}: NaN angles"
            # arcsin output is in [0, pi/2]; allow tiny clip overshoot.
            assert jnp.all(angles >= 0.0), f"{name}: negative angles"
            assert jnp.all(angles <= jnp.pi / 2 + 1e-5), f"{name}: angles > pi/2"

    def test_degenerate_triangle_masked_from_min(self) -> None:
        """Zero-area triangles must return ``+inf`` from
        ``_per_triangle_draft_angle`` so the per-face ``min`` reduction
        skips them.  If they returned ``0`` (a legal in-range draft
        angle), a single sliver in the OCCT mesh would force the
        face's worst-case draft angle to ``0`` regardless of the rest
        of the face — the CodeRabbit Major issue this test pins."""
        from brepax.metrics.draft_angle import _per_triangle_draft_angle

        # Zero-area triangle (all three vertices identical).
        degenerate = jnp.zeros((3, 3))
        # Non-degenerate triangle with normal +Z.
        good = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        mold = jnp.array([0.0, 0.0, 1.0])

        d_degen = float(_per_triangle_draft_angle(degenerate, mold))
        d_good = float(_per_triangle_draft_angle(good, mold))

        assert d_degen == float("inf"), (
            f"degenerate triangle returned {d_degen}, expected +inf"
        )
        # The good triangle has normal exactly +Z => arcsin(1) ≈ pi/2.
        assert abs(d_good - float(jnp.pi / 2)) < 1e-3, (
            f"good triangle returned {d_good}, expected ~pi/2"
        )
        # min over [degenerate, good] must equal the good triangle's
        # value, NOT zero.  This is the regression behaviour
        # ``+inf``-masking is meant to provide.
        result = float(jnp.min(jnp.array([d_degen, d_good])))
        assert abs(result - d_good) < 1e-6, (
            f"min should pick the good triangle, got {result}"
        )

    def test_gradient_flows_through_triangles(self) -> None:
        """``jax.grad`` of a per-face angle sum must produce finite
        gradients on triangle vertices (the same vertex array
        ``triangulate_shape`` produces and ``divergence_volume`` reduces
        over)."""
        from brepax.brep.triangulate import triangulate_shape
        from brepax.metrics.draft_angle import _per_triangle_draft_angle

        shape = self._read("sample_box")
        triangles, params_list = triangulate_shape(shape)
        n_tris_py = [int(p["n_triangles"]) for p in params_list]
        offsets_py = [0]
        for n in n_tris_py:
            offsets_py.append(offsets_py[-1] + n)

        mold = jnp.array([0.0, 0.0, 1.0])

        def loss(t: jnp.ndarray) -> jnp.ndarray:
            drafts = jax.vmap(lambda tri: _per_triangle_draft_angle(tri, mold))(t)
            per_face = jnp.stack(
                [
                    jnp.min(drafts[offsets_py[i] : offsets_py[i] + n_tris_py[i]])
                    for i in range(len(params_list))
                ]
            )
            return jnp.sum(per_face)

        g = jax.grad(loss)(triangles)
        assert jnp.all(jnp.isfinite(g))
