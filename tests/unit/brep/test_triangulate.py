"""Tests for OCCT mesh hybrid triangulation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_NurbsConvert,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakePrism,
    BRepPrimAPI_MakeSphere,
)
from OCP.Geom import Geom_BezierCurve
from OCP.gp import gp_Pln, gp_Pnt, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.TopoDS import TopoDS_Compound

from benchmarks.abc_evaluation import evaluate_step as evaluation
from brepax._occt.backend import BRepPrimAPI_MakeBox
from brepax.brep.gprop import compute_gprop_ground_truth
from brepax.brep.triangulate import (
    _DEFAULT_DEFLECTION,
    _eval_sphere,
    _evaluate_positions_per_face,
    _extract_ax3,
    _extract_face_records,
    _weld_shared_vertices,
    divergence_volume,
    evaluate_mesh,
    extract_mesh_topology,
    mesh_center_of_mass,
    mesh_inertia_tensor,
    mesh_surface_area,
    per_face_geometric_params,
    triangulate_shape,
    triangulate_shape_watertight,
)
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _assert_index_mesh_closed_manifold(
    vertices: jnp.ndarray, faces: jnp.ndarray
) -> None:
    """Independent closure/manifold/orientation check on welded indices.

    Unlike :func:`_assert_closed_mesh`, no coordinate quantization is
    needed: ``faces`` already indexes a deduplicated ``vertices`` array,
    so two triangle corners are the same vertex only if they share an
    index.
    """
    faces_np = np.asarray(faces)
    assert faces_np.shape[0] > 0
    directed_edges = np.concatenate(
        [faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]]
    )
    assert not np.any(directed_edges[:, 0] == directed_edges[:, 1])
    _, edge_ids, counts = np.unique(
        np.sort(directed_edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    assert np.all(counts == 2), f"Unpaired edges: {np.count_nonzero(counts != 2)}"
    signs = np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1)
    assert np.all(np.bincount(edge_ids, weights=signs) == 0)
    assert int(np.max(faces_np)) == np.asarray(vertices).shape[0] - 1


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


def _curved_roof_shape(unit: float, angle: float = 0.47):
    """Extrude a quadratic roof with planar trimmed caps."""
    rotation = _rotation_matrix(angle)

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
    shape = BRepPrimAPI_MakePrism(face, vector).Shape()
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
        shape = BRepBuilderAPI_NurbsConvert(_curved_roof_shape(unit), True).Shape()
        assert BRepCheck_Analyzer(shape).IsValid()
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

    def test_periodic_bspline_sphere_volume_area(self) -> None:
        """A NURBS-converted full sphere (IsUPeriodic) triangulates and matches it.

        ``BRepPrimAPI_MakeSphere`` builds an analytic sphere;
        ``BRepBuilderAPI_NurbsConvert`` rewrites its face as a rational
        B-spline that is periodic in U (``Geom_BSplineSurface.
        IsUPeriodic() == True``). That periodic pole/knot layout used to
        make ``bspline_basis`` raise a broadcasting ``TypeError``
        because its poles and knots do not satisfy the clamped
        ``n_poles == sum(multiplicities) - degree - 1`` convention the
        flat-knot evaluator assumes.
        """
        radius = 5.0
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeSphere(radius).Shape(), True
        ).Shape()
        assert BRepCheck_Analyzer(shape).IsValid()

        exact_volume = 4.0 / 3.0 * math.pi * radius**3
        exact_area = 4.0 * math.pi * radius**2
        reference = compute_gprop_ground_truth(shape)

        tris_coarse, params = triangulate_shape(shape, deflection=0.01)
        assert len(params) == 1
        assert "control_points" in params[0]
        # Mesh closure is not asserted here: OCCT's own mesher places a
        # fan of triangles at each geometric pole, and the general
        # rational evaluator returns positions that agree with each
        # other only to about 1e-10 (not bit-exactly) across the
        # different u values feeding that fan, which is independent of
        # this fix and of the volume/area claim checked below.
        vol_coarse = float(divergence_volume(tris_coarse))
        area_coarse = float(mesh_surface_area(tris_coarse))
        # Same discretization tolerance as the other curved fixtures
        # (test_curved_surface_volume) at the default deflection.
        assert vol_coarse == pytest.approx(exact_volume, rel=1e-2)
        assert area_coarse == pytest.approx(exact_area, rel=1e-2)
        assert vol_coarse == pytest.approx(reference["volume"], rel=1e-2)
        assert area_coarse == pytest.approx(reference["surface_area"], rel=1e-2)

        tris_fine, _ = triangulate_shape(shape, deflection=0.001)
        vol_fine = float(divergence_volume(tris_fine))
        area_fine = float(mesh_surface_area(tris_fine))
        assert abs(vol_fine - exact_volume) < abs(vol_coarse - exact_volume)
        assert abs(area_fine - exact_area) < abs(area_coarse - exact_area)

    def test_periodic_bspline_hemisphere_volume_area(self) -> None:
        """A periodic BSpline cap plus a planar cap triangulates and matches the volume.

        ``BRepPrimAPI_MakeSphere(radius, 0, pi/2)`` trims the sphere to
        the upper hemisphere: the curved cap stays periodic in U (full
        longitude) while the equatorial cut adds a second, planar face
        (also a B-spline after conversion, but non-periodic and
        degree 1), exercising a second surface signature in the same
        triangulation batch.
        """
        radius = 5.0
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeSphere(radius, 0.0, math.pi / 2).Shape(), True
        ).Shape()
        assert BRepCheck_Analyzer(shape).IsValid()

        exact_volume = 2.0 / 3.0 * math.pi * radius**3
        exact_area = 3.0 * math.pi * radius**2
        reference = compute_gprop_ground_truth(shape)

        tris_coarse, params = triangulate_shape(shape, deflection=0.01)
        assert len(params) == 2
        # See test_periodic_bspline_sphere_volume_area: the pole on the
        # curved cap keeps this mesh from being bit-exactly closed under
        # decimal quantization, independent of this fix.
        vol_coarse = float(divergence_volume(tris_coarse))
        area_coarse = float(mesh_surface_area(tris_coarse))
        assert vol_coarse == pytest.approx(exact_volume, rel=1e-2)
        assert area_coarse == pytest.approx(exact_area, rel=1e-2)
        # OCCT's own BRepGProp quadrature over the converted BSpline
        # faces (one of which includes the collapsed pole) carries
        # about 1% residual error on this shape, measured independently
        # of this fix. The looser bound here confirms the mesh path
        # agrees with OCCT's geometric engine to within what GProp
        # itself resolves, not a tighter claim than GProp supports.
        assert vol_coarse == pytest.approx(reference["volume"], rel=2e-2)
        assert area_coarse == pytest.approx(reference["surface_area"], rel=2e-2)

        tris_fine, _ = triangulate_shape(shape, deflection=0.001)
        vol_fine = float(divergence_volume(tris_fine))
        area_fine = float(mesh_surface_area(tris_fine))
        assert abs(vol_fine - exact_volume) < abs(vol_coarse - exact_volume)
        assert abs(area_fine - exact_area) < abs(area_coarse - exact_area)

    def test_deflection_affects_triangle_count(self) -> None:
        """Finer deflection produces more triangles."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        tris_coarse, _ = triangulate_shape(shape, deflection=0.1)
        tris_fine, _ = triangulate_shape(shape, deflection=0.01)
        assert tris_fine.shape[0] > tris_coarse.shape[0]

    @pytest.mark.parametrize(
        "extract", [triangulate_shape, extract_mesh_topology, per_face_geometric_params]
    )
    def test_unsupported_face_rejects_partial_mesh(self, extract) -> None:
        """A valid solid with an unsupported roof must not lose that face."""
        shape = _curved_roof_shape(1.0)
        with pytest.raises(NotImplementedError, match="GeomAbs_SurfaceOfExtrusion"):
            extract(shape)

    def test_unsupported_solid_is_unavailable_in_evaluation(
        self, tmp_path: Path
    ) -> None:
        """Neither mesh nor CSG paths may report a volume after dropping the roof."""
        shape = _curved_roof_shape(1.0, angle=0.0)
        path = tmp_path / "curved_roof.step"
        writer = STEPControl_Writer()
        assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
        assert writer.Write(str(path)) == IFSelect_RetDone
        with pytest.warns(UserWarning, match="Unsupported surface type"):
            report = evaluation.evaluate_step_file(path, resolution=2)
        assert {result.path for result in report.volume_paths} == {
            "divergence",
            "mesh_sdf",
            "gwn",
            "csg_stump",
            "trimmed_csg",
        }
        assert all(
            result.value is None and result.note for result in report.volume_paths
        )
        assert len(report.face_metrics) == 4
        assert all(
            result.n_finite == 0 and result.note for result in report.face_metrics
        )
        assert report.n_faces == 6
        assert report.occt_volume == pytest.approx(40 * 20 * (10 + 10 / 3))

    @pytest.mark.parametrize("extract", [triangulate_shape, extract_mesh_topology])
    def test_unmeshed_face_rejects_partial_mesh(self, extract) -> None:
        """A bounded plane must not conceal a second face that cannot be meshed."""
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        builder.Add(compound, BRepBuilderAPI_MakeFace(gp_Pln(), 0, 1, 0, 1).Face())
        builder.Add(compound, BRepBuilderAPI_MakeFace(gp_Pln()).Face())
        with pytest.raises(ValueError, match="no triangulation"):
            extract(compound)

    def test_empty_shape_returns_empty(self) -> None:
        """Shape with no faces returns empty array."""
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        tris, params = triangulate_shape(compound)
        assert tris.shape == (0, 3, 3)
        assert params == []
        assert extract_mesh_topology(compound) == []

    @pytest.mark.parametrize(
        ("fixture", "expected_hash"),
        [
            (
                "sample_box",
                "efe7fcd638267ba3a70298fecf4a9246ef7309cf64fc3a48f5634e18cdc7a9c4",
            ),
            (
                "sample_sphere",
                "a1e8d505377f48e4b4a0591108cf35e38adac3516fdc00bf4715438bbdc73ae3",
            ),
            (
                "l_bracket",
                "db9c90523ed914ff9701d4616e5f200f5edc0a68343fdc82ca745d88223db2d8",
            ),
            (
                "nurbs_box",
                "12a43173384fa7fd42d48d378d665ac4ce840a334f373c92e73eaff1ae8857d6",
            ),
        ],
    )
    def test_default_output_is_bit_identical(
        self, fixture: str, expected_hash: str
    ) -> None:
        """Adding watertight triangulation must not change the default path.

        Hashes were captured from this exact function and independently
        cross-checked byte-for-byte against the pre-refactor implementation
        (before :func:`triangulate_shape` and :func:`triangulate_shape_watertight`
        shared host-extraction helpers) on every ``.step`` fixture tracked in
        this repository.
        """
        shape = read_step(FIXTURES / f"{fixture}.step")
        tris, _ = triangulate_shape(shape)
        digest = hashlib.sha256(
            np.asarray(tris, dtype=np.float64).tobytes()
        ).hexdigest()
        assert digest == expected_hash


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


class TestTriangulateShapeWatertight:
    """Tests for triangulate_shape_watertight()."""

    def test_box_exact_closure_and_metrics(self) -> None:
        """A box welds its 24 duplicated corners down to the true 8 vertices."""
        shape = read_step(FIXTURES / "sample_box.step")
        mesh = triangulate_shape_watertight(shape)
        assert mesh.vertices.shape == (8, 3)
        assert mesh.faces.shape == (12, 3)
        _assert_index_mesh_closed_manifold(mesh.vertices, mesh.faces)
        assert float(mesh.max_gap) == 0.0
        triangles = mesh.vertices[mesh.faces]
        assert float(divergence_volume(triangles)) == pytest.approx(6000.0, rel=1e-12)
        assert float(mesh_surface_area(triangles)) == pytest.approx(2200.0, rel=1e-12)

    @pytest.mark.parametrize(
        "fixture",
        [
            "sample_cylinder",
            "sample_sphere",
            "sample_cone",
            "sample_torus",
            "l_bracket",
            "nurbs_box",
            "box_with_holes",
            "box_with_pocket",
            "box_with_slot",
        ],
    )
    def test_closed_manifold_matches_occt_gprop(self, fixture: str) -> None:
        """Every tracked solid fixture welds to a closed, manifold mesh.

        Checks the result both against :func:`_assert_index_mesh_closed_manifold`
        (an independent recomputation of edge-count/orientation from the
        returned indices alone) and against OCCT's own ``BRepGProp`` for
        volume and area, the same oracle :func:`triangulate_shape` is
        checked against elsewhere in this module.
        """
        shape = read_step(FIXTURES / f"{fixture}.step")
        reference = compute_gprop_ground_truth(shape)
        mesh = triangulate_shape_watertight(shape)
        _assert_index_mesh_closed_manifold(mesh.vertices, mesh.faces)
        # BRepMesh shares node parameters bit-identically across faces, so
        # independent per-face re-evaluations of the same shared point
        # should agree far tighter than the mesh's own discretization error.
        scale = max(1.0, float(jnp.max(jnp.abs(mesh.vertices))))
        assert float(mesh.max_gap) < 1e-6 * scale
        triangles = mesh.vertices[mesh.faces]
        volume = float(divergence_volume(triangles))
        area = float(mesh_surface_area(triangles))
        assert volume == pytest.approx(reference["volume"], rel=1e-2)
        assert area == pytest.approx(reference["surface_area"], rel=1e-2)

    def test_periodic_bspline_sphere_closes_at_both_poles(self) -> None:
        """A full NURBS-converted sphere welds its two pole fans shut.

        ``triangulate_shape``'s own test for this fixture explicitly does
        not assert closure, because independent per-face evaluations at
        the pole only agree to about 1e-10, not bit-exactly. Averaging
        those evaluations (instead of picking one, as a naive weld would)
        both closes the mesh and keeps ``max_gap`` reporting that residual
        disagreement instead of hiding it.
        """
        radius = 5.0
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeSphere(radius).Shape(), True
        ).Shape()
        assert BRepCheck_Analyzer(shape).IsValid()
        reference = compute_gprop_ground_truth(shape)
        exact_volume = 4.0 / 3.0 * math.pi * radius**3
        exact_area = 4.0 * math.pi * radius**2

        mesh_coarse = triangulate_shape_watertight(shape, deflection=0.01)
        _assert_index_mesh_closed_manifold(mesh_coarse.vertices, mesh_coarse.faces)
        assert float(mesh_coarse.max_gap) < 1e-6
        tris_coarse = mesh_coarse.vertices[mesh_coarse.faces]
        vol_coarse = float(divergence_volume(tris_coarse))
        area_coarse = float(mesh_surface_area(tris_coarse))
        assert vol_coarse == pytest.approx(exact_volume, rel=1e-2)
        assert area_coarse == pytest.approx(exact_area, rel=1e-2)
        assert vol_coarse == pytest.approx(reference["volume"], rel=1e-2)
        assert area_coarse == pytest.approx(reference["surface_area"], rel=1e-2)

        mesh_fine = triangulate_shape_watertight(shape, deflection=0.001)
        _assert_index_mesh_closed_manifold(mesh_fine.vertices, mesh_fine.faces)
        tris_fine = mesh_fine.vertices[mesh_fine.faces]
        vol_fine = float(divergence_volume(tris_fine))
        area_fine = float(mesh_surface_area(tris_fine))
        assert abs(vol_fine - exact_volume) < abs(vol_coarse - exact_volume)
        assert abs(area_fine - exact_area) < abs(area_coarse - exact_area)

    def test_periodic_bspline_hemisphere_closes_pole_and_planar_seam(self) -> None:
        """A curved pole cap and a planar equatorial cap weld at their shared seam."""
        radius = 5.0
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeSphere(radius, 0.0, math.pi / 2).Shape(), True
        ).Shape()
        assert BRepCheck_Analyzer(shape).IsValid()
        reference = compute_gprop_ground_truth(shape)
        exact_volume = 2.0 / 3.0 * math.pi * radius**3
        exact_area = 3.0 * math.pi * radius**2

        mesh = triangulate_shape_watertight(shape, deflection=0.01)
        assert len(mesh.params_list) == 2
        _assert_index_mesh_closed_manifold(mesh.vertices, mesh.faces)
        triangles = mesh.vertices[mesh.faces]
        volume = float(divergence_volume(triangles))
        area = float(mesh_surface_area(triangles))
        assert volume == pytest.approx(exact_volume, rel=1e-2)
        assert area == pytest.approx(exact_area, rel=1e-2)
        assert volume == pytest.approx(reference["volume"], rel=2e-2)
        assert area == pytest.approx(reference["surface_area"], rel=2e-2)

    def test_periodic_bspline_cylinder_seam_without_pole(self) -> None:
        """A periodic (but non-polar) BSpline face welds its own self-seam.

        ``BRepBuilderAPI_NurbsConvert`` rewrites the cylinder's analytic
        lateral surface as a BSpline that is periodic in U but has no
        degenerate edge, exercising the seam-matching path independently
        of degenerate-edge handling.
        """
        radius, height = 2.0, 5.0
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeCylinder(radius, height).Shape(), True
        ).Shape()
        assert BRepCheck_Analyzer(shape).IsValid()
        reference = compute_gprop_ground_truth(shape)
        exact_volume = math.pi * radius**2 * height

        mesh = triangulate_shape_watertight(shape, deflection=0.001)
        _assert_index_mesh_closed_manifold(mesh.vertices, mesh.faces)
        volume = float(divergence_volume(mesh.vertices[mesh.faces]))
        assert volume == pytest.approx(exact_volume, rel=5e-3)
        # OCCT's own BRepGProp quadrature over the NURBS-converted lateral
        # surface (not this mesh) carries its own ~1% residual error here,
        # measured independently of this fix (same effect noted in
        # test_periodic_bspline_hemisphere_closes_pole_and_planar_seam).
        assert volume == pytest.approx(reference["volume"], rel=2e-2)

    def test_cone_apex_closes(self) -> None:
        """A cone's single-point apex (a degenerate edge, not a seam) welds shut."""
        radius, height = 1.0, 2.0
        shape = BRepPrimAPI_MakeCone(radius, 0.0, height).Shape()
        reference = compute_gprop_ground_truth(shape)
        exact_volume = math.pi * radius**2 * height / 3.0

        mesh = triangulate_shape_watertight(shape, deflection=0.001)
        _assert_index_mesh_closed_manifold(mesh.vertices, mesh.faces)
        assert float(mesh.max_gap) < 1e-9
        volume = float(divergence_volume(mesh.vertices[mesh.faces]))
        assert volume == pytest.approx(exact_volume, rel=5e-3)
        assert volume == pytest.approx(reference["volume"], rel=5e-3)

    def test_rejects_open_shell_free_boundary(self) -> None:
        """A single open BSpline patch has no second face at its boundary."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        with pytest.raises(ValueError, match="face occurrences"):
            triangulate_shape_watertight(shape)

    def test_unsupported_face_rejects_partial_mesh(self) -> None:
        """A valid solid with an unsupported roof must not lose that face."""
        shape = _curved_roof_shape(1.0)
        with pytest.raises(NotImplementedError, match="GeomAbs_SurfaceOfExtrusion"):
            triangulate_shape_watertight(shape)

    def test_empty_shape_returns_empty(self) -> None:
        """Shape with no faces returns an empty, zero-gap watertight mesh."""
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        mesh = triangulate_shape_watertight(compound)
        assert mesh.vertices.shape == (0, 3)
        assert mesh.faces.shape == (0, 3)
        assert mesh.params_list == []
        assert float(mesh.max_gap) == 0.0

    def test_tolerance_argument_gates_on_reported_gap(self) -> None:
        """``tolerance`` raises exactly when the reported ``max_gap`` exceeds it."""
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeSphere(5.0).Shape(), True
        ).Shape()
        mesh = triangulate_shape_watertight(shape, deflection=0.01)
        assert float(mesh.max_gap) > 0.0

        # A loose tolerance above the reported gap must not raise.
        triangulate_shape_watertight(shape, deflection=0.01, tolerance=1.0)

        # A tolerance tighter than the reported gap must raise, quoting it.
        with pytest.raises(ValueError, match="exceeds tolerance"):
            triangulate_shape_watertight(
                shape, deflection=0.01, tolerance=float(mesh.max_gap) / 2.0
            )

    def test_gradient_averages_symmetrically_across_incident_faces(self) -> None:
        """A shared vertex's gradient splits evenly between its two faces.

        A cylinder's lateral face and one planar cap share their circular
        boundary two-to-one (unlike a box corner, which is a three-face
        junction). Perturbs one face's own position parameter at a time
        (``center`` for the cylinder, ``origin`` for the cap plane) along
        an arbitrary fixed direction, independent of that face's true
        normal, and differentiates the resulting *welded* vertex position
        with respect to each perturbation. Since the welded position is the
        mean of exactly two incident evaluations here, moving only one
        face's parameter must move the shared vertex by exactly half the
        naive (single-face) displacement -- and moving both faces by the
        same amount must recover the full displacement. Checked both as an
        exact linear identity and against finite differences.
        """
        position_key = {"plane": "origin", "cylinder": "center"}
        shape = BRepPrimAPI_MakeCylinder(2.0, 5.0).Shape()
        face_records = _extract_face_records(shape, _DEFAULT_DEFLECTION)
        group_of, n_vertices = _weld_shared_vertices(shape, face_records)
        offsets = np.cumsum([0] + [r["n_points"] for r in face_records], dtype=np.int64)

        shared_vertex = next(
            v for v in range(n_vertices) if np.count_nonzero(group_of == v) == 2
        )
        member_indices = np.nonzero(group_of == shared_vertex)[0]
        face_a, face_b = (
            int(np.searchsorted(offsets, idx, side="right") - 1)
            for idx in member_indices
        )
        assert face_a != face_b
        key_a = position_key[face_records[face_a]["surface_type"]]
        key_b = position_key[face_records[face_b]["surface_type"]]

        direction = jnp.array([1.0, 0.0, 0.0])
        mask = jnp.asarray(group_of == shared_vertex)

        def welded_vertex(delta_a: jnp.ndarray, delta_b: jnp.ndarray) -> jnp.ndarray:
            patched = []
            for i, rec in enumerate(face_records):
                params = rec["params"]
                if i == face_a:
                    params = {**params, key_a: params[key_a] + delta_a * direction}
                if i == face_b:
                    params = {**params, key_b: params[key_b] + delta_b * direction}
                patched.append({**rec, "params": params})
            positions = jnp.concatenate(_evaluate_positions_per_face(patched), axis=0)
            return jnp.mean(positions[mask], axis=0)

        jac_a, jac_b = jax.jacrev(welded_vertex, argnums=(0, 1))(
            jnp.array(0.0), jnp.array(0.0)
        )
        np.testing.assert_allclose(jac_a, 0.5 * direction, atol=1e-12)
        np.testing.assert_allclose(jac_b, 0.5 * direction, atol=1e-12)

        for step in (1e-2, 1e-3, 1e-4):
            forward = welded_vertex(jnp.array(step), jnp.array(0.0))
            backward = welded_vertex(jnp.array(-step), jnp.array(0.0))
            fd_a = (forward - backward) / (2 * step)
            np.testing.assert_allclose(fd_a, jac_a, rtol=1e-8, atol=1e-10)

            forward = welded_vertex(jnp.array(0.0), jnp.array(step))
            backward = welded_vertex(jnp.array(0.0), jnp.array(-step))
            fd_b = (forward - backward) / (2 * step)
            np.testing.assert_allclose(fd_b, jac_b, rtol=1e-8, atol=1e-10)

        # Moving both faces together recovers the full (unwelded) displacement.
        both = jax.jacrev(lambda d: welded_vertex(d, d))(jnp.array(0.0))
        np.testing.assert_allclose(both, direction, atol=1e-12)
