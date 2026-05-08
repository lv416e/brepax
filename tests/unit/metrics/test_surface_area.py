"""Unit tests for the surface area metric."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from brepax._occt.backend import (
    BRepGProp,
    GProp_GProps,
    TopAbs_FACE,
    TopAbs_SOLID,
    TopExp_Explorer,
    TopoDS,
)
from brepax.brep.csg_eval import make_grid_3d
from brepax.io.step import read_step
from brepax.metrics.surface_area import (
    integrate_sdf_surface_area,
    surface_area,
    surface_area_per_face,
)
from brepax.primitives import Box, Sphere

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestIntegrateSdfSurfaceArea:
    """Tests for the low-level grid integration function."""

    def test_sphere_r1_res64(self) -> None:
        """Sphere r=1 surface area approximates 4*pi."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = sphere.sdf(grid)
        area = integrate_sdf_surface_area(sdf, lo, hi, 64)
        expected = 4.0 * jnp.pi
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )

    def test_box_unit_res64(self) -> None:
        """Unit box surface area approximates 6."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([0.5, 0.5, 0.5]),
        )
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = box.sdf(grid)
        area = integrate_sdf_surface_area(sdf, lo, hi, 64)
        expected = 6.0
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )

    def test_sphere_r2_scales(self) -> None:
        """Sphere r=2 surface area approximates 16*pi."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(2.0))
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        grid, _ = make_grid_3d(lo, hi, 64)
        sdf = sphere.sdf(grid)
        area = integrate_sdf_surface_area(sdf, lo, hi, 64)
        expected = 4.0 * jnp.pi * 4.0
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )


class TestSurfaceArea:
    """Tests for the high-level surface_area function."""

    def test_sphere_matches_analytical(self) -> None:
        """surface_area(sphere.sdf) approximates 4*pi*r^2."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        area = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=64)
        expected = 4.0 * jnp.pi
        assert jnp.isclose(area, expected, rtol=0.05)

    def test_box_matches_analytical(self) -> None:
        """surface_area(box.sdf) approximates 2*(wh+wl+hl)."""
        box = Box(
            center=jnp.zeros(3),
            half_extents=jnp.array([2.0, 1.5, 1.0]),
        )
        lo, hi = jnp.array([-4.0] * 3), jnp.array([4.0] * 3)
        area = surface_area(box.sdf, lo=lo, hi=hi, resolution=64)
        # 2*(4*3 + 4*2 + 3*2) = 2*(12+8+6) = 52
        expected = 52.0
        assert jnp.isclose(area, expected, rtol=0.05), (
            f"area={float(area):.4f}, expected={float(expected):.4f}"
        )

    def test_resolution_convergence(self) -> None:
        """Higher resolution reduces error."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)
        expected = 4.0 * jnp.pi

        area_32 = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=32)
        area_64 = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=64)

        err_32 = jnp.abs(area_32 - expected) / expected
        err_64 = jnp.abs(area_64 - expected) / expected
        assert err_64 < err_32, (
            f"res=64 error ({float(err_64):.4f}) should be less than "
            f"res=32 error ({float(err_32):.4f})"
        )

    def test_differentiable_wrt_radius(self) -> None:
        """jax.grad of surface area w.r.t. sphere radius works."""
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def area_of_radius(r: jnp.ndarray) -> jnp.ndarray:
            sphere = Sphere(center=jnp.zeros(3), radius=r)
            return surface_area(sphere.sdf, lo=lo, hi=hi, resolution=48)

        r = jnp.array(1.0)
        grad_r = jax.grad(area_of_radius)(r)
        # d/dr(4*pi*r^2) = 8*pi*r = 8*pi at r=1
        expected_grad = 8.0 * jnp.pi
        assert jnp.isfinite(grad_r)
        assert jnp.isclose(grad_r, expected_grad, rtol=0.15), (
            f"grad={float(grad_r):.4f}, expected={float(expected_grad):.4f}"
        )

    def test_differentiable_wrt_center(self) -> None:
        """Surface area gradient w.r.t. center is near zero for symmetric domain."""
        lo, hi = jnp.array([-3.0] * 3), jnp.array([3.0] * 3)

        def area_of_center(c: jnp.ndarray) -> jnp.ndarray:
            sphere = Sphere(center=c, radius=jnp.array(1.0))
            return surface_area(sphere.sdf, lo=lo, hi=hi, resolution=48)

        c = jnp.zeros(3)
        grad_c = jax.grad(area_of_center)(c)
        # Centered sphere in symmetric domain: gradient should be ~0
        assert jnp.allclose(grad_c, 0.0, atol=0.5), (
            f"grad_c={grad_c}, expected near zero"
        )

    def test_jit_compatible(self) -> None:
        """surface_area works under jax.jit."""
        sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        lo, hi = jnp.array([-2.0] * 3), jnp.array([2.0] * 3)

        @jax.jit
        def compute(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
            return surface_area(sphere.sdf, lo=lo, hi=hi, resolution=32)

        area = compute(lo, hi)
        assert jnp.isfinite(area)
        assert float(area) > 0.0


def _occt_per_face_areas(shape) -> list[float]:
    """OCCT BRepGProp ground truth per face, in Solid-then-traversal order
    matching ``triangulate_shape`` / ``surface_area_per_face``."""
    out: list[float] = []
    face_sources: list = []
    exp_solid = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp_solid.More():
        face_sources.append(TopoDS.Solid_s(exp_solid.Current()))
        exp_solid.Next()
    if not face_sources:
        face_sources.append(shape)
    for source in face_sources:
        exp = TopExp_Explorer(source, TopAbs_FACE)
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            gp = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, gp)
            out.append(float(gp.Mass()))
            exp.Next()
    return out


class TestSurfaceAreaPerFace:
    """Per-face surface area against OCCT BRepGProp ground truth.

    The M6.1 milestone gate is "all metrics < 5% on all models. Trim-aware
    face-level metrics" (project_roadmap.md).  This class exercises the
    second clause for surface area: each face's area must agree with
    OCCT's analytic per-face surface integral within 5%, on every fixture
    in the standard set.  Trim awareness is delegated to OCCT's BRepMesh
    (the triangulation only covers the trimmed region of each face); the
    test pins that the BRepAX-side reduction reproduces the OCCT value
    face by face, not just in aggregate.
    """

    @pytest.mark.parametrize(
        "fixture",
        [
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
        ],
    )
    def test_each_face_within_5pct_of_occt(self, fixture: str) -> None:
        shape = read_step(str(FIXTURES / f"{fixture}.step"))
        areas, params = surface_area_per_face(shape)
        occt = _occt_per_face_areas(shape)

        assert len(occt) == len(params), (
            f"face count mismatch: BRepAX {len(params)} vs OCCT {len(occt)}"
        )
        assert areas.shape == (len(params),)

        for i, (mesh_a, occt_a) in enumerate(zip(map(float, areas), occt, strict=True)):
            if occt_a < 1e-9:
                # Degenerate face; just require finiteness.
                assert mesh_a == mesh_a  # NaN check
                continue
            err = abs(mesh_a - occt_a) / occt_a
            assert err < 0.05, (
                f"{fixture} face {i}: mesh_area={mesh_a:.4f} "
                f"occt_area={occt_a:.4f} err={err:.4f} > 0.05"
            )

    def test_areas_sum_matches_total(self) -> None:
        """Per-face areas must sum to the same total ``mesh_surface_area``
        produces from the flattened triangle array, modulo float32
        accumulation noise."""
        from brepax.brep.triangulate import mesh_surface_area, triangulate_shape

        shape = read_step(str(FIXTURES / "box_with_holes.step"))
        per_face, _ = surface_area_per_face(shape)
        triangles, _ = triangulate_shape(shape)
        total = float(mesh_surface_area(triangles))
        per_face_sum = float(jnp.sum(per_face))
        assert abs(per_face_sum - total) / total < 1e-3, (
            f"per_face_sum={per_face_sum:.4f} vs total={total:.4f}"
        )

    def test_gradient_flows_through_per_face(self) -> None:
        """``jax.grad`` of a per-face area sum must produce finite
        gradients on triangle vertices via the same path
        ``divergence_volume`` uses."""
        from brepax.brep.triangulate import triangulate_shape

        shape = read_step(str(FIXTURES / "sample_box.step"))
        triangles, params_list = triangulate_shape(shape)
        n_tris_per_face = jnp.asarray([p["n_triangles"] for p in params_list])
        offsets = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.cumsum(n_tris_per_face).astype(jnp.int32),
            ]
        )

        def total_per_face_sum(t: jnp.ndarray) -> jnp.ndarray:
            from brepax.brep.triangulate import mesh_surface_area as _msa

            return jnp.stack(
                [
                    _msa(
                        jax.lax.dynamic_slice_in_dim(
                            t, int(offsets[i]), int(n_tris_per_face[i]), axis=0
                        )
                    )
                    for i in range(len(params_list))
                ]
            ).sum()

        grad = jax.grad(total_per_face_sum)(triangles)
        assert jnp.all(jnp.isfinite(grad))
