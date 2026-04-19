"""Tests for OCCT mesh hybrid triangulation."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.brep.triangulate import (
    _eval_sphere,
    _extract_ax3,
    divergence_volume,
    evaluate_mesh,
    extract_mesh_topology,
    mesh_center_of_mass,
    mesh_inertia_tensor,
    mesh_surface_area,
    triangulate_shape,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _divergence_volume(triangles: jnp.ndarray) -> jnp.ndarray:
    """Divergence theorem volume from a triangle mesh."""
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return jnp.sum(v0 * jnp.cross(v1, v2)) / 6.0


# Reference volumes from OCCT GProp (pre-computed for each fixture)
_REF_VOLUMES = {
    "sample_box": 6000.0,
    "sample_cylinder": 1178.0972,
    "sample_sphere": 113.0973,
    "sample_cone": 54.9779,
    "sample_torus": 222.0661,
}


class TestTriangulateShape:
    """Tests for triangulate_shape()."""

    def test_box_volume_exact(self) -> None:
        """Box triangulation gives exact volume."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, params = triangulate_shape(shape)
        vol = float(_divergence_volume(tris))
        assert vol == pytest.approx(6000.0, rel=1e-3)
        assert len(params) == 6

    @pytest.mark.parametrize(
        "fixture",
        ["sample_cylinder", "sample_sphere", "sample_cone", "sample_torus"],
    )
    def test_curved_surface_volume(self, fixture: str) -> None:
        """Curved surface volume within 1% of OCCT GProp."""
        shape = read_step(FIXTURES / f"{fixture}.step")
        tris, _params = triangulate_shape(shape)
        vol = float(_divergence_volume(tris))
        assert vol == pytest.approx(_REF_VOLUMES[fixture], rel=1e-2)

    def test_bspline_face_triangulated(self) -> None:
        """BSpline face produces triangles with control_points param."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        tris, params = triangulate_shape(shape)
        assert tris.shape[0] > 0
        assert len(params) == 1
        assert "control_points" in params[0]

    def test_deflection_affects_triangle_count(self) -> None:
        """Finer deflection produces more triangles."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        tris_coarse, _ = triangulate_shape(shape, deflection=0.1)
        tris_fine, _ = triangulate_shape(shape, deflection=0.01)
        assert tris_fine.shape[0] > tris_coarse.shape[0]

    def test_empty_shape_returns_empty(self) -> None:
        """Shape with no faces returns empty array."""
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        tris, params = triangulate_shape(compound)
        assert tris.shape == (0, 3, 3)
        assert params == []


class TestMeshMetrics:
    """Tests for mesh-based surface area, center of mass, and inertia tensor."""

    def test_box_surface_area_exact(self) -> None:
        """Box surface area is exact (flat faces)."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        assert float(mesh_surface_area(tris)) == pytest.approx(2200.0, rel=1e-6)

    def test_sphere_surface_area(self) -> None:
        """Sphere surface area within 0.5% of analytical."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        tris, _ = triangulate_shape(shape)
        # sample_sphere radius from GProp: area ≈ 113.10
        assert float(mesh_surface_area(tris)) == pytest.approx(113.10, rel=5e-3)

    def test_box_center_of_mass_exact(self) -> None:
        """Box CoM is exact (10x20x30 at origin → CoM = (5, 10, 15))."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        com = mesh_center_of_mass(tris)
        assert float(com[0]) == pytest.approx(5.0, abs=1e-4)
        assert float(com[1]) == pytest.approx(10.0, abs=1e-4)
        assert float(com[2]) == pytest.approx(15.0, abs=1e-4)

    def test_sphere_center_of_mass_at_origin(self) -> None:
        """Sphere at origin has CoM = (0, 0, 0)."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        tris, _ = triangulate_shape(shape)
        com = mesh_center_of_mass(tris)
        assert jnp.max(jnp.abs(com)) < 0.01

    def test_box_inertia_exact(self) -> None:
        """Box inertia about CoM matches analytical formula."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        inertia = mesh_inertia_tensor(tris)
        # M=6000, I_xx = M/12*(b^2+c^2) = 500*(400+900) = 650000
        assert float(inertia[0, 0]) == pytest.approx(650000.0, rel=1e-4)
        assert float(inertia[1, 1]) == pytest.approx(500000.0, rel=1e-4)
        assert float(inertia[2, 2]) == pytest.approx(250000.0, rel=1e-4)

    def test_inertia_symmetric(self) -> None:
        """Inertia tensor is symmetric."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        inertia = mesh_inertia_tensor(tris)
        assert float(inertia[0, 1]) == pytest.approx(float(inertia[1, 0]), abs=1e-4)
        assert float(inertia[0, 2]) == pytest.approx(float(inertia[2, 0]), abs=1e-4)
        assert float(inertia[1, 2]) == pytest.approx(float(inertia[2, 1]), abs=1e-4)

    def test_surface_area_gradient_finite(self) -> None:
        """Gradient of surface area is finite."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        grad = jax.grad(mesh_surface_area)(tris)
        assert jnp.all(jnp.isfinite(grad))


class TestEvaluateMesh:
    """Tests for extract_mesh_topology + evaluate_mesh."""

    def test_matches_triangulate_shape(self) -> None:
        """evaluate_mesh(topology) gives same volume as triangulate_shape."""
        shape = read_step(FIXTURES / "sample_cylinder.step")
        tris_old, _ = triangulate_shape(shape)
        topology = extract_mesh_topology(shape)
        tris_new = evaluate_mesh(topology)
        v_old = float(divergence_volume(tris_old))
        v_new = float(divergence_volume(tris_new))
        assert v_new == pytest.approx(v_old, abs=1e-4)

    def test_bspline_matches_triangulate_shape(self) -> None:
        """evaluate_mesh matches triangulate_shape for BSpline solid."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        tris_old, _ = triangulate_shape(shape)
        topology = extract_mesh_topology(shape)
        tris_new = evaluate_mesh(topology)
        v_old = float(divergence_volume(tris_old))
        v_new = float(divergence_volume(tris_new))
        assert v_new == pytest.approx(v_old, abs=1e-4)


class TestDivergenceVolumeGradient:
    """Tests for gradient flow through divergence_volume."""

    def test_gradient_finite_and_nonzero(self) -> None:
        """Gradient of volume w.r.t. triangle vertices is finite and nonzero."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        grad = jax.grad(_divergence_volume)(tris)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0)

    def test_gradient_matches_finite_diff(self) -> None:
        """AD gradient matches central finite difference."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        grad = jax.grad(_divergence_volume)(tris)

        eps = 1e-3
        idx = (0, 1, 2)
        tris_p = tris.at[idx].add(eps)
        tris_m = tris.at[idx].add(-eps)
        fd = (_divergence_volume(tris_p) - _divergence_volume(tris_m)) / (2 * eps)
        assert float(grad[idx]) == pytest.approx(float(fd), rel=0.05)


class TestParametricOptimization:
    """Tests for end-to-end parametric optimization via evaluate_mesh."""

    def test_optimize_sphere_radius(self) -> None:
        """Newton optimization converges sphere radius to target volume."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere

        from brepax._occt.backend import (
            BRep_Tool,
            BRepAdaptor_Surface,
            BRepMesh_IncrementalMesh,
            TopAbs_FACE,
            TopAbs_FORWARD,
            TopAbs_SOLID,
            TopExp_Explorer,
            TopLoc_Location,
            TopoDS,
        )

        shape = BRepPrimAPI_MakeSphere(5.0).Shape()
        BRepMesh_IncrementalMesh(shape, 0.01)

        exp_s = TopExp_Explorer(shape, TopAbs_SOLID)
        solid = TopoDS.Solid_s(exp_s.Current())
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        face = TopoDS.Face_s(exp.Current())
        adaptor = BRepAdaptor_Surface(face)
        reverse = face.Orientation() != TopAbs_FORWARD

        center, xdir, ydir, axis = _extract_ax3(adaptor.Sphere().Position())

        loc = TopLoc_Location()
        poly_tri = BRep_Tool.Triangulation_s(face, loc)
        n_nodes = poly_tri.NbNodes()
        us_np = np.empty(n_nodes)
        vs_np = np.empty(n_nodes)
        for i in range(1, n_nodes + 1):
            uv = poly_tri.UVNode(i)
            us_np[i - 1] = uv.X()
            vs_np[i - 1] = uv.Y()

        conn_np = np.empty((poly_tri.NbTriangles(), 3), dtype=np.int32)
        for i in range(1, poly_tri.NbTriangles() + 1):
            n1, n2, n3 = poly_tri.Triangle(i).Get()
            conn_np[i - 1] = [n1 - 1, n2 - 1, n3 - 1]
        if reverse:
            conn_np = conn_np[:, [0, 2, 1]]

        us = jnp.array(us_np)
        vs = jnp.array(vs_np)
        conn = conn_np

        def volume_of_radius(radius: jnp.ndarray) -> jnp.ndarray:
            positions = jax.vmap(
                lambda u, v: _eval_sphere(center, xdir, ydir, axis, radius, u, v)
            )(us, vs)
            return divergence_volume(positions[conn])

        target = (4 / 3) * math.pi * 7.0**3
        radius = jnp.array(5.0)
        vg_fn = jax.value_and_grad(volume_of_radius)

        for _ in range(10):
            vol, dvdr = vg_fn(radius)
            radius = radius - (vol - target) / dvdr

        final_vol = float(volume_of_radius(radius))
        assert final_vol == pytest.approx(target, rel=1e-3)
        assert float(radius) == pytest.approx(7.0, abs=0.05)

    def test_optimize_cylinder_radius(self) -> None:
        """Cylinder radius optimization with multi-face cap tracking."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        shape = BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape()
        topology = extract_mesh_topology(shape)

        def volume_fn(radius: jnp.ndarray) -> jnp.ndarray:
            tris = evaluate_mesh(topology, {"radius": radius}, uv_scale_param="radius")
            return divergence_volume(tris)

        target = 500.0
        radius = jnp.array(5.0)
        vg_fn = jax.value_and_grad(volume_fn)

        for _ in range(10):
            vol, dvdr = vg_fn(radius)
            radius = radius - (vol - target) / dvdr

        r_analytical = math.sqrt(target / (math.pi * 10))
        assert float(volume_fn(radius)) == pytest.approx(target, rel=1e-3)
        assert float(radius) == pytest.approx(r_analytical, rel=0.01)

    def test_optimize_bspline_control_points(self) -> None:
        """BSpline CP optimization converges to target volume."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        topology = extract_mesh_topology(shape)

        face_idx = 0
        original_cp = topology[face_idx]["control_points"]

        # Identify normal axis (constant coordinate across all CPs)
        cp_ranges = original_cp.max(axis=(0, 1)) - original_cp.min(axis=(0, 1))
        normal_axis = int(jnp.argmin(cp_ranges))

        def volume_fn(offset: jnp.ndarray) -> jnp.ndarray:
            cp_new = original_cp.at[:, :, normal_axis].add(offset)
            topo = [{**f} for f in topology]
            topo[face_idx] = {**topo[face_idx], "control_points": cp_new}
            return divergence_volume(evaluate_mesh(topo))

        target = 300.0
        offset = jnp.array(0.0)
        vg_fn = jax.value_and_grad(volume_fn)

        for _ in range(10):
            vol, dvdo = vg_fn(offset)
            offset = offset - (vol - target) / dvdo

        assert float(volume_fn(offset)) == pytest.approx(target, rel=1e-3)

    def test_bspline_cp_gradient_physically_correct(self) -> None:
        """Moving BSpline CPs outward changes volume in expected direction."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        topology = extract_mesh_topology(shape)

        face_idx = 0
        original_cp = topology[face_idx]["control_points"]
        cp_ranges = original_cp.max(axis=(0, 1)) - original_cp.min(axis=(0, 1))
        normal_axis = int(jnp.argmin(cp_ranges))
        face_pos = float(original_cp[0, 0, normal_axis])

        def volume_fn(cp_new: jnp.ndarray) -> jnp.ndarray:
            topo = [{**f} for f in topology]
            topo[face_idx] = {**topo[face_idx], "control_points": cp_new}
            return divergence_volume(evaluate_mesh(topo))

        grad = jax.grad(volume_fn)(original_cp)
        assert jnp.all(jnp.isfinite(grad))

        # Gradient along normal axis should have consistent sign:
        # if face is at x=0, moving CPs in -x shrinks box → dV/d(cp_x) < 0
        grad_normal = grad[:, :, normal_axis]
        if face_pos < 1.0:
            assert float(jnp.mean(grad_normal)) < 0
        else:
            assert float(jnp.mean(grad_normal)) > 0
