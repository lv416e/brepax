"""Unit tests for wall thickness metrics."""

from pathlib import Path

import jax
import jax.numpy as jnp

from brepax.brep.csg_eval import make_grid_3d
from brepax.io.step import read_step
from brepax.metrics.wall_thickness import (
    integrate_sdf_thin_wall_volume,
    min_wall_thickness,
    min_wall_thickness_per_face,
    thin_wall_volume,
)
from brepax.primitives import Box, Sphere

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestIntegrateSdfThinWallVolume:
    """Tests for the low-level grid integration function."""

    def test_thin_plate_full_violation(self) -> None:
        """A thin plate with threshold > half-thickness has ~full volume violation."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([5.0, 5.0, 0.25]),
        )
        lo, hi = jnp.array([-7.0] * 3), jnp.array([7.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        total_vol = 10.0 * 10.0 * 0.5  # 50.0
        # threshold=0.5 > half-thickness=0.25: all interior within threshold
        violation = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
        ratio = float(violation) / total_vol
        assert ratio > 0.8, f"Expected most volume violated, got ratio={ratio:.3f}"

    def test_thick_cube_small_violation_ratio(self) -> None:
        """A thick cube with small threshold has low violation ratio."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([5.0, 5.0, 5.0]),
        )
        lo, hi = jnp.array([-7.0] * 3), jnp.array([7.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        total_vol = 10.0**3  # 1000.0
        # threshold=0.5: thin shell near surface ~30% of volume
        violation = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
        ratio = float(violation) / total_vol
        assert ratio < 0.35, f"Expected shell ratio, got ratio={ratio:.3f}"

    def test_threshold_zero_smaller_than_positive(self) -> None:
        """Zero threshold gives less violation than positive threshold."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.ones(3),
        )
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        # Sigmoid smoothing gives nonzero at threshold=0 (surface delta),
        # but it must be less than a positive threshold
        vol_zero = integrate_sdf_thin_wall_volume(sdf, 0.0, lo, hi, 64)
        vol_half = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
        assert float(vol_zero) < float(vol_half), (
            f"zero={float(vol_zero):.4f} should be less than half={float(vol_half):.4f}"
        )


class TestThinWallVolume:
    """Tests for the high-level thin_wall_volume function."""

    def test_sphere_thin_shell(self) -> None:
        """Thin-wall volume of a sphere increases with threshold."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(2.0))
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        vol_small = thin_wall_volume(sphere.sdf, 0.3, lo=lo, hi=hi, resolution=48)
        vol_large = thin_wall_volume(sphere.sdf, 1.0, lo=lo, hi=hi, resolution=48)
        assert float(vol_large) > float(vol_small), (
            f"Larger threshold should give larger violation: "
            f"small={float(vol_small):.3f}, large={float(vol_large):.3f}"
        )

    def test_differentiable_wrt_threshold(self) -> None:
        """jax.grad w.r.t. threshold is positive (more threshold = more volume)."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([2.0, 1.5, 1.0]),
        )
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def vol_of_threshold(t: jnp.ndarray) -> jnp.ndarray:
            return thin_wall_volume(box.sdf, t, lo=lo, hi=hi, resolution=48)

        grad_t = jax.grad(vol_of_threshold)(jnp.array(0.5))
        assert jnp.isfinite(grad_t)
        assert float(grad_t) > 0.0, (
            f"Expected positive gradient, got {float(grad_t):.4f}"
        )

    def test_differentiable_wrt_shape(self) -> None:
        """jax.grad of thin_wall_volume w.r.t. box half_extents works."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def vol_of_extents(he: jnp.ndarray) -> jnp.ndarray:
            box = Box(center=jnp.zeros(3), half_extents=he)
            return thin_wall_volume(box.sdf, 0.5, lo=lo, hi=hi, resolution=48)

        he = jnp.array([2.0, 1.5, 1.0])
        grad_he = jax.grad(vol_of_extents)(he)
        assert jnp.all(jnp.isfinite(grad_he)), f"Non-finite gradient: {grad_he}"

    def test_jit_compatible(self) -> None:
        """thin_wall_volume works under jax.jit."""
        box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        @jax.jit
        def compute(t: jnp.ndarray) -> jnp.ndarray:
            return thin_wall_volume(box.sdf, t, lo=lo, hi=hi, resolution=32)

        result = compute(jnp.array(0.5))
        assert jnp.isfinite(result)
        assert float(result) > 0.0


class TestMinWallThickness:
    """Tests for the min_wall_thickness diagnostic function."""

    def test_thin_plate(self) -> None:
        """Thin plate (2.0 thick) returns ~2.0."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([3.0, 3.0, 1.0]),
        )
        lo, hi = jnp.array([-5.0] * 3), jnp.array([5.0] * 3)
        thickness = min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=64)
        # Thin dimension = 2.0 (2 * 1.0)
        assert jnp.isclose(thickness, 2.0, rtol=0.05), (
            f"thickness={float(thickness):.4f}, expected ~2.0"
        )

    def test_cube_gives_smallest_dimension(self) -> None:
        """Box half_extents=(2, 1.5, 1): min wall thickness ~2 (2*1)."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([2.0, 1.5, 1.0]),
        )
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        thickness = min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=64)
        # Smallest dimension = 2*1.0 = 2.0
        assert jnp.isclose(thickness, 2.0, rtol=0.05), (
            f"thickness={float(thickness):.4f}, expected ~2.0"
        )

    def test_sphere_gives_diameter(self) -> None:
        """Sphere r=1: min wall thickness ~2 (diameter)."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)
        thickness = min_wall_thickness(sphere.sdf, lo=lo, hi=hi, resolution=64)
        assert jnp.isclose(thickness, 2.0, rtol=0.05), (
            f"thickness={float(thickness):.4f}, expected ~2.0"
        )

    def test_differentiable(self) -> None:
        """jax.grad w.r.t. box half_extents works."""
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)

        def thickness_of_extents(he: jnp.ndarray) -> jnp.ndarray:
            box = Box(center=jnp.zeros(3), half_extents=he)
            return min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=48)

        he = jnp.array([2.0, 1.5, 1.0])
        grad_he = jax.grad(thickness_of_extents)(he)
        assert jnp.all(jnp.isfinite(grad_he)), f"Non-finite gradient: {grad_he}"
        # Thinnest direction (z, half_extent=1.0) should have largest gradient
        assert float(grad_he[2]) > float(grad_he[0]), (
            f"z-gradient should dominate: {grad_he}"
        )


class TestMinWallThicknessPerFace:
    """Per-face min wall thickness from each face's centroid.

    The M6.1 milestone gate clause "Trim-aware face-level metrics"
    requires each face-level metric to be exposed and verified.
    The per-face wall thickness is the distance from each face's
    centroid to the nearest triangle on any **other** face — see
    the function docstring for why centroid-based sampling is
    necessary (boundary triangles share edges with neighbours and
    would otherwise force the metric to zero).
    """

    @staticmethod
    def _read(name: str):
        return read_step(str(FIXTURES / f"{name}.step"))

    def test_sample_box_axis_aligned(self) -> None:
        """``sample_box`` is the rectangular block 10x20x30.  Each
        face's centroid is at the centre of that face, and the
        distance to the nearest perpendicular face equals the
        half-extent of the smallest perpendicular dimension.

        - +/- x faces (20x30 area): nearest perpendicular pair is
          +/- y at half-extent 10 or +/- z at 15; min is 10.
        - +/- y faces (10x30 area): nearest is +/- x at 5 or
          +/- z at 15; min is 5.
        - +/- z faces (10x20 area): nearest is +/- x at 5 or
          +/- y at 10; min is 5.
        """
        shape = self._read("sample_box")
        thicknesses, params = min_wall_thickness_per_face(shape)
        assert thicknesses.shape == (len(params),) == (6,)
        sorted_t = jnp.sort(thicknesses)
        # 4 faces give 5 (the +/- y and +/- z pairs), 2 faces give 10
        # (the +/- x pair).  Mesh discretization keeps the result well
        # under 5% of the analytical value.
        assert jnp.allclose(sorted_t[:4], 5.0, rtol=0.05), (
            f"4 small-half-extent faces: expected ~5, got {sorted_t}"
        )
        assert jnp.allclose(sorted_t[4:], 10.0, rtol=0.05), (
            f"2 wide faces: expected ~10, got {sorted_t}"
        )

    def test_sample_cylinder_side_vs_caps(self) -> None:
        """``sample_cylinder`` is radius 5, height 15.  The cylinder
        side face's centroid sits on the axis at z=7.5; its nearest
        other-face triangle is one of the cap rings at z=0 or z=15
        at distance 7.5.  Each cap face's centroid is at the cap's
        centre on the axis; the nearest other-face triangle is on
        the cylinder side at radial distance 5 (modulo discretization).
        """
        shape = self._read("sample_cylinder")
        thicknesses, params = min_wall_thickness_per_face(shape)
        assert len(params) == 3

        for thick, p in zip(thicknesses, params, strict=True):
            t = p["surface_type"]
            if t == "cylinder":
                # side face centroid -> cap at z=0 or z=15 at distance 7.5
                assert jnp.isclose(thick, 7.5, rtol=0.05), (
                    f"cyl side: expected ~7.5, got {float(thick):.4f}"
                )
            elif t == "plane":
                # cap centroid -> side face at radial distance ~5
                # (mesh discretization can underestimate by a few %).
                assert jnp.isclose(thick, 5.0, rtol=0.10), (
                    f"cyl cap: expected ~5, got {float(thick):.4f}"
                )

    def test_single_face_returns_inf(self) -> None:
        """``sample_sphere`` has a single face; there is no other
        face to measure to, so the function returns ``+inf`` rather
        than 0 (an honest "no other surface" signal)."""
        shape = self._read("sample_sphere")
        thicknesses, params = min_wall_thickness_per_face(shape)
        assert len(params) == 1
        assert jnp.isinf(thicknesses[0])

    def test_finite_or_inf_on_all_fixtures(self) -> None:
        """Smoke property: every per-face thickness is either finite
        positive or ``+inf`` on every fixture in the standard set.
        Negative thickness or NaN would indicate a bug in the
        centroid / slice handling."""
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
        for name in fixtures:
            shape = self._read(name)
            thicknesses, params = min_wall_thickness_per_face(shape)
            assert thicknesses.shape == (len(params),)
            for thick in thicknesses:
                v = float(thick)
                assert v > 0.0 and (jnp.isfinite(v) or jnp.isinf(v)), (
                    f"{name}: unexpected thickness {v}"
                )

    def test_gradient_flows_through_triangles(self) -> None:
        """``jax.grad`` of a per-face thickness sum must produce
        finite gradients on triangle vertices (the same vertex array
        ``triangulate_shape`` produces; the metric reduces over
        ``point_triangle_distance`` which is differentiable through
        triangle vertex positions)."""
        from brepax.brep.mesh_sdf import point_triangle_distance
        from brepax.brep.triangulate import triangulate_shape

        shape = self._read("sample_box")
        triangles, params_list = triangulate_shape(shape)
        n_tris_py = [int(p["n_triangles"]) for p in params_list]
        offsets_py = [0]
        for n in n_tris_py:
            offsets_py.append(offsets_py[-1] + n)

        def loss(t: jnp.ndarray) -> jnp.ndarray:
            total = jnp.asarray(0.0)
            for i in range(len(params_list)):
                a = offsets_py[i]
                b = a + n_tris_py[i]
                centroid = jnp.mean(t[a:b].reshape(-1, 3), axis=0)
                if a == 0:
                    others = t[b:]
                elif b == t.shape[0]:
                    others = t[:a]
                else:
                    others = jnp.concatenate([t[:a], t[b:]], axis=0)
                per_tri = jax.vmap(
                    lambda tri, c=centroid: point_triangle_distance(
                        c, tri[0], tri[1], tri[2]
                    )
                )(others)
                total = total + jnp.min(per_tri)
            return total

        grad = jax.grad(loss)(triangles)
        assert jnp.all(jnp.isfinite(grad))
