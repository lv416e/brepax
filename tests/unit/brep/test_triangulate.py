"""Tests for OCCT mesh hybrid triangulation."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_NurbsConvert,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.Geom import Geom_BezierCurve
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TColgp import TColgp_Array1OfPnt

from brepax._occt.backend import BRepPrimAPI_MakeBox
from brepax.brep.gprop import compute_gprop_ground_truth
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


def _assert_closed_mesh(triangles: jnp.ndarray) -> None:
    """Check opposite edge pairs after 8-decimal coordinate quantization in mm."""
    points = np.round(np.asarray(triangles).reshape(-1, 3), decimals=8)
    _, indices = np.unique(points, axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    directed_edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    _, edge_ids, counts = np.unique(
        np.sort(directed_edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    assert np.all(counts == 2), f"Unpaired edges: {np.count_nonzero(counts != 2)}"
    signs = np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1)
    assert np.all(np.bincount(edge_ids, weights=signs) == 0)


def _scaled_bspline_mesh(topology, scales: jnp.ndarray) -> jnp.ndarray:
    """Apply one shared affine map to the control points of every patch."""
    linked = [
        {**face, "control_points": face["control_points"] * scales} for face in topology
    ]
    return evaluate_mesh(linked)


def _box_cad_metrics(dimensions: np.ndarray) -> np.ndarray:
    """Rebuild the corresponding planar box independently of its mesh."""
    builder = BRepPrimAPI_MakeBox(*map(float, dimensions))
    shape = builder.Shape()
    assert builder.IsDone() and not shape.IsNull()
    reference = compute_gprop_ground_truth(shape)
    return np.array([reference["volume"], reference["surface_area"]])


def _rotation_matrix(
    angle: float, axis: tuple[float, float, float] = (1.0, 2.0, 3.0)
) -> np.ndarray:
    direction = np.asarray(axis) / np.linalg.norm(axis)
    x, y, z = direction
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _curved_roof_shape(unit: float):
    """Extrude a quadratic roof with planar trimmed caps, then convert to NURBS."""
    rotation = _rotation_matrix(0.47)

    def point(x: float, z: float):
        return gp_Pnt(*(unit * (rotation @ np.array([x, 0.0, z]) + [3.0, -2.0, 4.0])))

    corners = [point(0, 0), point(40, 0), point(40, 10), point(0, 10)]
    poles = TColgp_Array1OfPnt(1, 3)
    for i, pole in enumerate([corners[2], point(20, 20), corners[3]], 1):
        poles.SetValue(i, pole)
    wire = BRepBuilderAPI_MakeWire()
    for edge in (
        BRepBuilderAPI_MakeEdge(corners[0], corners[1]),
        BRepBuilderAPI_MakeEdge(corners[1], corners[2]),
        BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)),
        BRepBuilderAPI_MakeEdge(corners[3], corners[0]),
    ):
        wire.Add(edge.Edge())
    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    vector = gp_Vec(*(unit * (rotation @ np.array([0.0, 20.0, 0.0]))))
    shape = BRepBuilderAPI_NurbsConvert(
        BRepPrimAPI_MakePrism(face, vector).Shape(), True
    ).Shape()
    assert BRepCheck_Analyzer(shape).IsValid()
    return shape


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
        vol = float(divergence_volume(tris))
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
        vol = float(divergence_volume(tris))
        assert vol == pytest.approx(_REF_VOLUMES[fixture], rel=1e-2)

    def test_bspline_face_triangulated(self) -> None:
        """BSpline face produces triangles with control_points param."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        tris, params = triangulate_shape(shape)
        assert tris.shape[0] > 0
        assert len(params) == 1
        assert "control_points" in params[0]

    @pytest.mark.parametrize("unit", [1e-3, 1.0, 1e3])
    def test_rotated_bspline_roof_preserves_boundary(self, unit: float) -> None:
        """Mesher UV roundoff must not send boundary vertices to the origin."""
        shape = _curved_roof_shape(unit)
        reference = compute_gprop_ground_truth(shape)
        triangles, params = triangulate_shape(shape, deflection=0.001 * unit)
        topology = extract_mesh_topology(shape, deflection=0.001 * unit)
        assert len(params) == len(topology) == 6
        for mesh in (triangles, evaluate_mesh(topology)):
            _assert_closed_mesh(mesh / unit)
            np.testing.assert_allclose(
                divergence_volume(mesh) / unit**3,
                reference["volume"] / unit**3,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                mesh_surface_area(mesh) / unit**2,
                reference["surface_area"] / unit**2,
                rtol=1e-4,
            )

        def scaled_metrics(scale):
            mesh = _scaled_bspline_mesh(topology, jnp.full(3, scale))
            return jnp.array(
                [divergence_volume(mesh) / unit**3, mesh_surface_area(mesh) / unit**2]
            )

        derivative = jax.jit(jax.jacfwd(scaled_metrics))(jnp.array(1.0))
        np.testing.assert_allclose(
            derivative,
            [
                3 * reference["volume"] / unit**3,
                2 * reference["surface_area"] / unit**2,
            ],
            rtol=1e-4,
        )

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
        expected = np.diag([650000.0, 500000.0, 250000.0])
        np.testing.assert_allclose(inertia, expected, rtol=1e-12, atol=1e-8)

    def test_nonprincipal_inertia_matches_occt(self) -> None:
        """A concave solid has nonzero products of inertia about its CoM."""
        shape = read_step(FIXTURES / "l_bracket.step")
        tris, _ = triangulate_shape(shape)
        expected = compute_gprop_ground_truth(shape)["moment_of_inertia"]
        assert abs(expected[0, 1]) > 1.0
        np.testing.assert_allclose(
            mesh_inertia_tensor(tris), expected, rtol=1e-12, atol=1e-8
        )

    def test_empty_inertia_is_zero(self) -> None:
        np.testing.assert_array_equal(
            mesh_inertia_tensor(jnp.empty((0, 3, 3))), np.zeros((3, 3))
        )

    @settings(max_examples=12, derandomize=True, deadline=None)
    @given(
        angle=st.floats(-math.pi, math.pi),
        axis=st.tuples(*(st.floats(-1.0, 1.0) for _ in range(3))).filter(
            lambda axis: np.linalg.norm(axis) > 0.1
        ),
        translation=st.tuples(*(st.floats(-100.0, 100.0) for _ in range(3))),
        scale=st.floats(1e-9, 1e3),
    )
    @example(angle=0.47, axis=(0.3, -0.4, 0.5), translation=(3.0, -2.0, 4.0), scale=1.0)
    @example(angle=0.0, axis=(0.3, -0.4, 0.5), translation=(1e8, -2e8, 3e8), scale=1.0)
    @example(
        angle=0.47, axis=(0.3, -0.4, 0.5), translation=(3.0, -2.0, 4.0), scale=1e-9
    )
    def test_mesh_metrics_rigid_motion_and_scale(
        self, angle, axis, translation, scale
    ) -> None:
        """Check scalar, vector, and tensor similarity laws against independent OCCT."""
        shape = read_step(FIXTURES / "l_bracket.step")
        tris, _ = triangulate_shape(shape)
        reference = compute_gprop_ground_truth(shape)
        rotation = _rotation_matrix(angle, axis)
        transformed = scale * (tris @ rotation.T + jnp.array(translation))
        np.testing.assert_allclose(
            divergence_volume(transformed) / scale**3, reference["volume"], rtol=2e-9
        )
        np.testing.assert_allclose(
            mesh_surface_area(transformed) / scale**2,
            reference["surface_area"],
            rtol=2e-9,
        )
        # Compare in model units; allow only coordinate-rounding error at large offsets.
        rounding = 8 * np.finfo(float).eps * np.max(np.abs(translation))
        np.testing.assert_allclose(
            mesh_center_of_mass(transformed) / scale - np.array(translation),
            rotation @ reference["center_of_mass"],
            rtol=2e-10,
            atol=2e-10 + rounding,
        )
        expected = rotation @ reference["moment_of_inertia"] @ rotation.T
        np.testing.assert_allclose(
            mesh_inertia_tensor(transformed) / scale**5,
            expected,
            rtol=2e-9,
            atol=2e-9 * np.linalg.norm(expected),
        )

    def test_inertia_transformed_gradients(self) -> None:
        """JIT/vmap/grad match analytic scale and translation derivatives."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        tris = tris - jnp.array([5.0, 10.0, 15.0])
        rotation = jnp.asarray(_rotation_matrix(0.47))
        direction = jnp.array([1.0, 2.0, -3.0]) / jnp.sqrt(14.0)
        reference = jnp.diag(jnp.array([650000.0, 500000.0, 250000.0]))
        coefficient = direction @ rotation @ reference @ rotation.T @ direction

        def objective(scale, shift):
            transformed = scale * (tris @ rotation.T) + shift
            return direction @ mesh_inertia_tensor(transformed) @ direction

        scales = jnp.array([0.25, 1.0, 2.0])
        scale_grad, shift_grad = jax.jit(
            jax.vmap(jax.grad(objective, argnums=(0, 1)), in_axes=(0, None))
        )(scales, jnp.array([3.0, -2.0, 4.0]))
        np.testing.assert_allclose(
            scale_grad, 5.0 * scales**4 * coefficient, rtol=1e-12
        )
        np.testing.assert_allclose(shift_grad, 0.0, atol=1e-7)

    @pytest.mark.parametrize(
        ("metric", "degree", "reference"),
        [(divergence_volume, 3, 6000.0), (mesh_surface_area, 2, 2200.0)],
    )
    def test_scalar_metric_similarity_gradients(
        self, metric, degree, reference
    ) -> None:
        """Scalar gradients co-rotate and scale by s^(degree-1), including under JIT."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        rotation = jnp.asarray(_rotation_matrix(0.47))
        shift = jnp.array([3.0, -2.0, 4.0])

        def objective(scale, translation):
            return metric(scale * (tris @ rotation.T) + translation)

        scales = jnp.array([1e-9, 1e-3, 1.0, 2.0])
        scale_grad, shift_grad = jax.jit(jax.vmap(jax.grad(objective, argnums=(0, 1))))(
            scales, scales[:, None] * shift
        )
        np.testing.assert_allclose(
            scale_grad / scales ** (degree - 1), degree * reference, rtol=1e-12
        )
        np.testing.assert_allclose(
            shift_grad / scales[:, None] ** (degree - 1), 0.0, atol=1e-10
        )

        gradient = jax.jit(jax.grad(metric))
        scale = 0.25
        np.testing.assert_allclose(
            gradient(scale * (tris @ rotation.T) + shift),
            scale ** (degree - 1) * (gradient(tris) @ rotation.T),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_center_of_mass_similarity_derivatives(self) -> None:
        """CoM translation and scale Jacobians match the analytic similarity laws."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        rotation = jnp.asarray(_rotation_matrix(0.47, (0.3, -0.4, 0.5)))

        def center(scale, translation):
            return mesh_center_of_mass(scale * (tris @ rotation.T) + translation)

        scales = jnp.array([1e-9, 1.0, 1e3])
        scale_jac, shift_jac = jax.jit(jax.vmap(jax.jacrev(center, argnums=(0, 1))))(
            scales, scales[:, None] * jnp.array([3.0, -2.0, 4.0])
        )
        np.testing.assert_allclose(
            scale_jac,
            jnp.broadcast_to(rotation @ jnp.array([5.0, 10.0, 15.0]), (3, 3)),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            shift_jac, jnp.tile(jnp.eye(3), (3, 1, 1)), atol=1e-12
        )

    def test_empty_mesh_metrics_are_zero(self) -> None:
        triangles = jnp.empty((0, 3, 3))
        assert float(divergence_volume(triangles)) == 0.0
        assert float(mesh_surface_area(triangles)) == 0.0
        np.testing.assert_array_equal(mesh_center_of_mass(triangles), jnp.zeros(3))

    def test_zero_area_triangles_have_zero_value_and_gradient(self) -> None:
        """At collapsed or collinear triangles, select a zero area subgradient."""
        triangles = jnp.array(
            [[[1.0, 2.0, 3.0]] * 3, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
        )
        value, gradient = jax.jit(jax.value_and_grad(mesh_surface_area))(triangles)
        assert float(value) == 0.0
        np.testing.assert_array_equal(gradient, jnp.zeros_like(triangles))


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

    def test_shared_vertex_gradients_match_analytic_tetrahedron(self) -> None:
        """Compare against a determinant oracle while preserving mesh closure."""
        vertices = jnp.array(
            [[1.2, -0.3, 0.5], [3.4, -0.1, 0.7], [1.5, 2.1, 0.4], [1.0, 0.2, 4.5]]
        )
        faces = jnp.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])

        def volume(points):
            return divergence_volume(points[faces])

        edges = np.asarray(vertices[1:] - vertices[0])
        reference_volume = np.linalg.det(edges) / 6.0
        outer_gradients = reference_volume * np.linalg.inv(edges).T
        reference_gradient = np.vstack((-outer_gradients.sum(axis=0), outer_gradients))
        value, gradient = jax.jit(jax.value_and_grad(volume))(vertices)
        np.testing.assert_allclose(value, reference_volume, rtol=1e-12)
        np.testing.assert_allclose(gradient, reference_gradient, rtol=1e-12, atol=1e-12)

        for step in (1e-3, 1e-4, 1e-5):
            forward = vertices.at[1, 2].add(step)
            backward = vertices.at[1, 2].add(-step)
            fd = (volume(forward) - volume(backward)) / (2 * step)
            np.testing.assert_allclose(gradient[1, 2], fd, rtol=1e-8, atol=1e-10)


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

    def test_optimize_linked_bspline_control_points(self) -> None:
        """A Newton update fits height while keeping the planar patches closed."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        topology = extract_mesh_topology(shape)

        def mesh(height_scale: jnp.ndarray) -> jnp.ndarray:
            return _scaled_bspline_mesh(topology, jnp.array([1.0, 1.0, height_scale]))

        def volume_fn(height_scale: jnp.ndarray) -> jnp.ndarray:
            return divergence_volume(mesh(height_scale))

        target = 300.0  # mm^3; the fixture is a 10 x 8 x 6 mm box.
        height_scale = jnp.array(1.0)
        volume, derivative = jax.jit(jax.value_and_grad(volume_fn))(height_scale)
        np.testing.assert_allclose(derivative, 480.0, rtol=1e-12)
        height_scale = height_scale - (volume - target) / derivative
        assert float(height_scale) == pytest.approx(0.625, abs=1e-12)

        for scale in (1.0, float(height_scale)):
            triangles = mesh(jnp.array(scale))
            _assert_closed_mesh(triangles)
            actual = jnp.stack(
                [divergence_volume(triangles), mesh_surface_area(triangles)]
            )
            reference = _box_cad_metrics(np.array([10.0, 8.0, 6.0 * scale]))
            np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-10)
        assert float(volume_fn(height_scale)) == pytest.approx(target, abs=1e-10)

    def test_linked_bspline_gradients_match_rebuilt_cad(self) -> None:
        """Differentiate a closed fixed-volume map, not independent face motion."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        topology = extract_mesh_topology(shape)

        def scales(parameters: jnp.ndarray) -> jnp.ndarray:
            a, b = parameters
            return jnp.exp(jnp.array([a, b, -a - b]))

        def metrics(parameters: jnp.ndarray) -> jnp.ndarray:
            triangles = _scaled_bspline_mesh(topology, scales(parameters))
            return jnp.stack(
                [divergence_volume(triangles), mesh_surface_area(triangles)]
            )

        def cad_metrics(parameters: np.ndarray) -> np.ndarray:
            a, b = parameters
            dimensions = np.array([10.0, 8.0, 6.0]) * np.exp([a, b, -a - b])
            return _box_cad_metrics(dimensions)

        @jax.jit
        def value_and_jacobian(parameters):
            return metrics(parameters), jax.jacrev(metrics)(parameters)

        points = jnp.array([[0.0, 0.0], [-0.1, 0.1], [0.1, -0.15]])
        values, jacobians = jax.vmap(value_and_jacobian)(points)
        for point, value, jacobian in zip(points, values, jacobians, strict=True):
            _assert_closed_mesh(_scaled_bspline_mesh(topology, scales(point)))
            np.testing.assert_allclose(
                value, cad_metrics(np.asarray(point)), rtol=1e-12
            )
            assert float(value[0]) == pytest.approx(480.0, abs=1e-10)
            a, b = np.asarray(point)
            # A = 2*(80*exp(a+b) + 60*exp(-b) + 48*exp(-a)) mm^2.
            expected = np.array(
                [
                    [0.0, 0.0],
                    [
                        2 * (80 * np.exp(a + b) - 48 * np.exp(-a)),
                        2 * (80 * np.exp(a + b) - 60 * np.exp(-b)),
                    ],
                ]
            )
            np.testing.assert_allclose(jacobian, expected, rtol=1e-11, atol=1e-10)
            for direction in np.eye(2):
                for step in (1e-3, 1e-4, 1e-5):
                    forward = cad_metrics(np.asarray(point) + step * direction)
                    backward = cad_metrics(np.asarray(point) - step * direction)
                    finite_difference = (forward - backward) / (2 * step)
                    np.testing.assert_allclose(
                        jacobian @ direction, finite_difference, rtol=1e-6, atol=1e-7
                    )
