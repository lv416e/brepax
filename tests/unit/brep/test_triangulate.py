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
from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_NurbsConvert,
    BRepBuilderAPI_Transform,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakePrism,
    BRepPrimAPI_MakeSphere,
)
from OCP.Geom import Geom_BezierCurve, Geom_BSplineSurface
from OCP.gp import gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from OCP.TopoDS import TopoDS_Compound

from benchmarks.abc_evaluation import evaluate_step as evaluation
from brepax._occt.backend import BRepPrimAPI_MakeBox
from brepax.brep.gprop import compute_gprop_ground_truth
from brepax.brep.triangulate import (
    WatertightMesh,
    WatertightTopology,
    _check_manifold_closed,
    _check_vertex_links_manifold,
    _eval_sphere,
    _extract_ax3,
    _extract_face_geometric_params,
    divergence_volume,
    evaluate_mesh,
    evaluate_watertight_mesh,
    extract_mesh_topology,
    extract_watertight_topology,
    mesh_center_of_mass,
    mesh_inertia_tensor,
    mesh_surface_area,
    per_face_geometric_params,
    triangulate_shape,
    triangulate_shape_watertight,
)
from brepax.io.step import read_step
from brepax.nurbs.evaluate import evaluate_surface

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


def _pair_face_of(topology: WatertightTopology) -> np.ndarray:
    """Which face owns each (vertex, face) pair in ``topology``.

    Test-only convenience: recovers the face index for each pair from its
    member nodes' position in ``face_records``, so gradient tests can
    target a specific face by name instead of by a hard-coded index.
    """
    n_faces = len(topology.face_records)
    node_face = np.repeat(
        np.arange(n_faces), [r["n_points"] for r in topology.face_records]
    )
    pair_face = np.zeros(topology.n_pairs, dtype=np.int64)
    pair_face[topology.pair_of] = node_face
    return pair_face


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
        "fixture", ["sample_box", "sample_sphere", "l_bracket", "nurbs_box"]
    )
    def test_default_output_matches_occt_mesh(self, fixture: str) -> None:
        """The default path reproduces OCCT's own mesh, triangle for triangle.

        :func:`triangulate_shape` re-evaluates OCCT's mesh nodes through
        BRepAX's surface formulas, so every output triangle must coincide
        with one OCCT mesh triangle (located node coordinates, face
        orientation applied) up to evaluation round-off. OCCT's mesh itself
        differs between platforms (``sample_sphere`` has 2996 triangles on
        macOS and 3000 on Linux), so a recorded hash or checksum is not a
        portable reference; OCCT's mesh on the same run is.
        """
        from brepax._occt.backend import (
            BRep_Tool,
            TopAbs_FACE,
            TopAbs_FORWARD,
            TopExp_Explorer,
            TopLoc_Location,
            TopoDS,
        )

        shape = read_step(FIXTURES / f"{fixture}.step")
        tris = np.asarray(triangulate_shape(shape)[0], dtype=np.float64)

        reference = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            loc = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(face, loc)
            transform = loc.Transformation()
            nodes = np.array(
                [
                    [c.X(), c.Y(), c.Z()]
                    for c in (
                        poly.Node(i).Transformed(transform)
                        for i in range(1, poly.NbNodes() + 1)
                    )
                ]
            )
            for i in range(1, poly.NbTriangles() + 1):
                a, b, c = poly.Triangle(i).Get()
                if face.Orientation() != TopAbs_FORWARD:
                    b, c = c, b
                reference.append(nodes[[a - 1, b - 1, c - 1]])
            explorer.Next()
        reference_tris = np.asarray(reference)

        assert tris.shape == reference_tris.shape
        scale = float(np.ptp(reference_tris.reshape(-1, 3), axis=0).max())
        centroid_gap = np.linalg.norm(
            tris.mean(axis=1)[:, None, :] - reference_tris.mean(axis=1)[None, :, :],
            axis=-1,
        )
        match = centroid_gap.argmin(axis=1)
        assert np.unique(match).size == match.size
        matched = reference_tris[match]
        # Same orientation: some cyclic rotation of each OCCT triangle
        # coincides with the output triangle vertex by vertex.
        gaps = np.stack(
            [
                np.abs(tris - np.roll(matched, shift, axis=1)).max(axis=(1, 2))
                for shift in range(3)
            ]
        ).min(axis=0)
        assert gaps.max() <= 1e-9 * scale


def _single_span_bspline_surface() -> Geom_BSplineSurface:
    """Build a clamped, non-rational, degree-3 x degree-3 BSpline patch.

    4x4 non-planar control points on the unit knot domain ``[0, 1]`` in
    both directions, matching the clamped convention the flat-knot
    evaluator in ``nurbs/evaluate.py`` assumes.
    """
    poles = TColgp_Array2OfPnt(1, 4, 1, 4)
    for i in range(1, 5):
        for j in range(1, 5):
            poles.SetValue(i, j, gp_Pnt(float(i), float(j), float(i * j)))
    knots = TColStd_Array1OfReal(1, 2)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)
    mults = TColStd_Array1OfInteger(1, 2)
    mults.SetValue(1, 4)
    mults.SetValue(2, 4)
    return Geom_BSplineSurface(poles, knots, knots, mults, mults, 3, 3)


class TestBsplineMeshParameterExtension:
    """Tests for evaluating out-of-domain B-spline mesh UV parameters.

    OCCT's mesher can place a trimmed-face mesh node outside the
    surface's knot domain (observed up to ~3.3e-4 on a real NIST part,
    not just a roundoff-sized overshoot); ``Geom_BSplineSurface::Value``
    still evaluates it there by continuing the end span's polynomial
    piece. ``triangulate_shape`` passes mesh UV parameters through
    unmodified (see ``bspline_basis`` in ``nurbs/evaluate.py``), so an
    out-of-domain parameter must evaluate to the same point as OCCT,
    not the origin a vanishing basis used to produce.
    """

    def test_out_of_domain_mesh_parameter_matches_occt_not_origin(self) -> None:
        """Reproduces the defect directly and checks the fix against OCCT."""
        bspl = _single_span_bspline_surface()
        face = BRepBuilderAPI_MakeFace(bspl, 1e-6).Face()
        adaptor = BRepAdaptor_Surface(face)
        result = _extract_face_geometric_params(adaptor)
        assert result is not None
        surface_type, params = result
        assert surface_type == "bspline"

        knots_u = params["knots_u"]
        degree_u = params["deg_u"]
        lower_u = float(np.asarray(knots_u)[degree_u])
        # A few ulps, like the roundoff case, and a much larger
        # undershoot, like the NIST STC-10 face-139 anomaly: both must
        # match OCCT, not just the roundoff-sized one.
        v_mid = 0.5
        for undershoot in (lower_u - 9 * np.finfo(np.float64).eps, lower_u - 3.3e-4):
            point = evaluate_surface(
                params["control_points"],
                knots_u,
                params["knots_v"],
                degree_u,
                params["deg_v"],
                jnp.array(undershoot),
                jnp.array(v_mid),
            )
            oracle = bspl.Value(undershoot, v_mid)
            occt = np.array([oracle.X(), oracle.Y(), oracle.Z()])
            np.testing.assert_allclose(np.asarray(point), occt, atol=1e-9)
            # The historical bug: a vanishing basis collapsed this to
            # the origin, which disagrees with OCCT's own node here.
            assert not np.allclose(occt, 0.0)

    @pytest.mark.parametrize("fixture", ["nurbs_box", "nurbs_saddle"])
    def test_no_origin_collapsed_corners_on_tracked_fixtures(
        self, fixture: str
    ) -> None:
        """No BSpline mesh corner collapses to the origin on tracked fixtures.

        Both tracked fixtures happen to have a legitimate mesh vertex
        at the world origin, so this translates the shape by a large,
        arbitrary offset first: UV knot parametrization is unaffected
        by a rigid transform, so any post-translation corner still at
        the origin can only be a vanishing-basis artifact, not a
        coincidentally-placed real point.
        """
        offset = gp_Vec(1234.5, -6789.25, 42.75)
        transform = gp_Trsf()
        transform.SetTranslation(offset)

        shape = read_step(FIXTURES / f"{fixture}.step")
        moved = BRepBuilderAPI_Transform(shape, transform, True).Shape()

        triangles, params = triangulate_shape(moved)
        assert any(p["surface_type"] == "bspline" for p in params)
        corners = np.asarray(triangles).reshape(-1, 3)
        n_origin = int(np.sum(np.all(corners == 0.0, axis=-1)))
        assert n_origin == 0


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
        assert float(divergence_volume(triangles)) == pytest.approx(6000.0, rel=1e-10)
        assert float(mesh_surface_area(triangles)) == pytest.approx(2200.0, rel=1e-10)

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

        topology = extract_watertight_topology(shape, deflection=0.01)
        assert len(topology.face_records) == 2
        mesh = evaluate_watertight_mesh(topology, topology.params_list)
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
        assert float(mesh.max_gap) == 0.0

        topology = extract_watertight_topology(compound)
        assert topology.face_records == []
        assert topology.params_list == []

    def test_tolerance_argument_gates_on_reported_gap(self) -> None:
        """``tolerance`` raises exactly when the reported ``max_gap`` exceeds it.

        Uses the hemisphere fixture (a curved pole cap and a planar cap
        meeting at the equator): its ``max_gap`` is a reliably nonzero
        residual (the raw per-node evaluation noise described in
        :class:`WatertightMesh`), large enough to exercise both the
        loose-tolerance and tight-tolerance branches below without being
        exactly zero to machine precision.
        """
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeSphere(5.0, 0.0, math.pi / 2).Shape(), True
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

    def test_gradient_through_production_evaluator_is_symmetric(self) -> None:
        """``evaluate_watertight_mesh`` (the production averaging) splits a
        shared vertex's gradient evenly between its two incident faces.

        A cylinder's lateral face and one planar cap share their circular
        boundary two-to-one (unlike a box corner, which is a three-face
        junction) and each contribute exactly one node there, so this is
        the simple case of the two-level mean. Perturbs one face's own
        ``control_points`` at a time (the shape is NURBS-converted so both
        faces expose that parameter) through :func:`extract_watertight_topology`
        and :func:`evaluate_watertight_mesh` -- the actual production
        welding, not a hand-rolled mean -- and checks that moving only one
        face's parameter moves the shared vertex by exactly half the
        naive (single-face) displacement, under both ``jax.grad`` and
        ``jax.jit``, and against finite differences.
        """
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeCylinder(2.0, 5.0).Shape(), True
        ).Shape()
        topology = extract_watertight_topology(shape, deflection=0.01)

        pair_counts = np.bincount(topology.pair_of, minlength=topology.n_pairs)
        pair_face = _pair_face_of(topology)

        # A vertex with exactly two incident faces, each contributing one node.
        vertex_ids, vertex_pair_counts = np.unique(
            topology.pair_vertex, return_counts=True
        )
        ordinary_vertex = next(
            int(v)
            for v, n_incident in zip(vertex_ids, vertex_pair_counts, strict=True)
            if n_incident == 2
            and all(
                pair_counts[p] == 1 for p in np.nonzero(topology.pair_vertex == v)[0]
            )
        )
        pairs_at_vertex = np.nonzero(topology.pair_vertex == ordinary_vertex)[0]
        face_a, face_b = (int(pair_face[p]) for p in pairs_at_vertex)
        assert face_a != face_b

        direction = jnp.array([1.0, 0.0, 0.0])

        def vertex_of(delta_a: jnp.ndarray, delta_b: jnp.ndarray) -> jnp.ndarray:
            params = list(topology.params_list)
            params[face_a] = {
                **params[face_a],
                "control_points": params[face_a]["control_points"]
                + delta_a * direction,
            }
            params[face_b] = {
                **params[face_b],
                "control_points": params[face_b]["control_points"]
                + delta_b * direction,
            }
            mesh = evaluate_watertight_mesh(topology, params)
            return mesh.vertices[ordinary_vertex]

        jac_a, jac_b = jax.jit(jax.jacrev(vertex_of, argnums=(0, 1)))(
            jnp.array(0.0), jnp.array(0.0)
        )
        np.testing.assert_allclose(jac_a, 0.5 * direction, atol=1e-12)
        np.testing.assert_allclose(jac_b, 0.5 * direction, atol=1e-12)

        for step in (1e-2, 1e-3, 1e-4):
            forward = vertex_of(jnp.array(step), jnp.array(0.0))
            backward = vertex_of(jnp.array(-step), jnp.array(0.0))
            fd_a = (forward - backward) / (2 * step)
            np.testing.assert_allclose(fd_a, jac_a, rtol=1e-6, atol=1e-9)

        # Moving both faces together recovers the full (unwelded) displacement.
        both = jax.jacrev(lambda d: vertex_of(d, d))(jnp.array(0.0))
        np.testing.assert_allclose(both, direction, atol=1e-12)

    def test_seam_endpoint_weighted_by_face_not_by_node(self) -> None:
        """A face that touches a shared vertex through two of its own nodes
        (both sides of its own periodic seam) still counts as one face.

        ``BRepBuilderAPI_NurbsConvert`` makes the cylinder's lateral face
        periodic in U with no degenerate edge, so its own self-seam meets
        the planar cap's boundary at one welded vertex where the lateral
        face contributes *two* nodes (one per side of its seam) and the
        cap contributes *one*. A flat mean over nodes would weight that
        vertex 2/3 lateral, 1/3 cap; :func:`evaluate_watertight_mesh`'s
        two-level mean (per face, then across faces) must instead split
        it exactly half and half, checked through production code, not a
        hand-picked two-node group.
        """
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeCylinder(2.0, 5.0).Shape(), True
        ).Shape()
        topology = extract_watertight_topology(shape, deflection=0.01)

        pair_counts = np.bincount(topology.pair_of, minlength=topology.n_pairs)
        pair_face = _pair_face_of(topology)

        # A seam-endpoint vertex: exactly two incident faces, one of which
        # (the periodic lateral face) supplies more than one of its own
        # nodes there.
        vertex_ids, vertex_pair_counts = np.unique(
            topology.pair_vertex, return_counts=True
        )
        seam_vertex = next(
            int(v)
            for v, n_incident in zip(vertex_ids, vertex_pair_counts, strict=True)
            if n_incident == 2
            and any(
                pair_counts[p] > 1 for p in np.nonzero(topology.pair_vertex == v)[0]
            )
        )
        pairs_at_vertex = np.nonzero(topology.pair_vertex == seam_vertex)[0]
        multi_node_pair = next(p for p in pairs_at_vertex if pair_counts[p] > 1)
        lateral_face = int(pair_face[multi_node_pair])
        cap_face = int(next(p for p in pairs_at_vertex if p != multi_node_pair))
        cap_face = int(pair_face[cap_face])
        assert lateral_face != cap_face

        direction = jnp.array([1.0, 0.0, 0.0])

        def vertex_of(
            delta_lateral: jnp.ndarray, delta_cap: jnp.ndarray
        ) -> jnp.ndarray:
            params = list(topology.params_list)
            params[lateral_face] = {
                **params[lateral_face],
                "control_points": params[lateral_face]["control_points"]
                + delta_lateral * direction,
            }
            params[cap_face] = {
                **params[cap_face],
                "control_points": params[cap_face]["control_points"]
                + delta_cap * direction,
            }
            mesh = evaluate_watertight_mesh(topology, params)
            return mesh.vertices[seam_vertex]

        jac_lateral, jac_cap = jax.jacrev(vertex_of, argnums=(0, 1))(
            jnp.array(0.0), jnp.array(0.0)
        )
        np.testing.assert_allclose(jac_lateral, 0.5 * direction, atol=1e-12)
        np.testing.assert_allclose(jac_cap, 0.5 * direction, atol=1e-12)

        for step in (1e-2, 1e-3, 1e-4):
            forward = vertex_of(jnp.array(step), jnp.array(0.0))
            backward = vertex_of(jnp.array(-step), jnp.array(0.0))
            fd_lateral = (forward - backward) / (2 * step)
            np.testing.assert_allclose(fd_lateral, jac_lateral, rtol=1e-6, atol=1e-9)

    def test_two_tetrahedra_sharing_one_vertex_is_rejected(self) -> None:
        """Two fans pinched at one shared vertex pass edge-pairing alone.

        Each tetrahedron's own six edges are already paired exactly twice
        with consistent orientation within itself, so
        ``_check_manifold_closed`` (edge count and orientation only) finds
        nothing wrong even though the combined mesh is not a manifold at
        the shared vertex: its link is two disjoint triangles, not one.
        ``_check_vertex_links_manifold`` must reject it. This constructs
        the index arrays directly (not via OCCT) and checks the two
        private functions as black boxes, rather than reimplementing
        their link-cycle criterion in the test.
        """
        tetra_a = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
        tetra_b = np.array([[4, 5, 6], [0, 6, 5], [0, 4, 6], [0, 5, 4]])
        pinched = np.concatenate([tetra_a, tetra_b], axis=0)

        _check_manifold_closed(pinched)  # does not raise: every edge is fine
        with pytest.raises(ValueError, match="non-manifold vertex"):
            _check_vertex_links_manifold(pinched)

        # An ordinary single tetrahedron is rejected by neither check.
        _check_manifold_closed(tetra_a)
        _check_vertex_links_manifold(tetra_a)

    def test_evaluate_watertight_mesh_jit_with_bspline_params(self) -> None:
        """``evaluate_watertight_mesh`` is ``jax.jit``-compatible over a full
        ``params_list`` that includes a B-spline face.

        B-spline degrees are shape-determining dispatch/cache keys, not
        differentiable data (see ``WatertightTopology.face_records``): if
        they leaked into the traced ``params_list`` pytree, ``jax.jit``
        would turn them into unhashable tracers used as dict keys inside
        ``_evaluate_positions_per_face``'s batched-group dispatch.
        """
        shape = read_step(FIXTURES / "nurbs_box.step")
        topology = extract_watertight_topology(shape)
        assert any(rec["surface_type"] == "bspline" for rec in topology.face_records)
        assert all("deg_u" not in p for p in topology.params_list)

        jitted = jax.jit(lambda params: evaluate_watertight_mesh(topology, params))
        mesh = jitted(topology.params_list)
        eager = evaluate_watertight_mesh(topology, topology.params_list)
        np.testing.assert_allclose(mesh.vertices, eager.vertices)
        np.testing.assert_array_equal(mesh.faces, eager.faces)
        # Both values are at machine-epsilon scale for this exact box, so
        # only an absolute tolerance is meaningful: XLA's fused jit
        # evaluation order and eager NumPy/JAX dispatch round differently
        # at that scale without indicating a real discrepancy.
        np.testing.assert_allclose(mesh.max_gap, eager.max_gap, atol=1e-9)

    def test_topology_is_a_valid_jit_static_argument(self) -> None:
        """A topology can be passed to ``jit`` as a static argument.

        JAX hashes static arguments; the topology holds lists and arrays,
        so it must hash and compare by identity.
        """
        topology = extract_watertight_topology(read_step(FIXTURES / "nurbs_box.step"))
        jitted = jax.jit(evaluate_watertight_mesh, static_argnums=0)
        mesh = jitted(topology, topology.params_list)
        eager = evaluate_watertight_mesh(topology, topology.params_list)
        np.testing.assert_allclose(mesh.vertices, eager.vertices, atol=1e-12)
        assert topology == topology
        assert topology != extract_watertight_topology(
            read_step(FIXTURES / "nurbs_box.step")
        )

    def test_evaluate_watertight_mesh_returns_from_jit_and_vmap(self) -> None:
        """A ``WatertightMesh`` is a registered pytree (an ``equinox.Module``
        of arrays only), so ``jax.jit`` and ``jax.vmap`` can return it
        directly, not just take it as an input.
        """
        shape = read_step(FIXTURES / "sample_box.step")
        topology = extract_watertight_topology(shape)

        jitted = jax.jit(lambda params: evaluate_watertight_mesh(topology, params))
        mesh = jitted(topology.params_list)
        assert isinstance(mesh, WatertightMesh)
        assert mesh.vertices.shape == (8, 3)
        assert mesh.faces.shape == (12, 3)

        def scaled(scale: jnp.ndarray) -> WatertightMesh:
            params = [
                {**p, "origin": p["origin"] * scale} for p in topology.params_list
            ]
            return evaluate_watertight_mesh(topology, params)

        batched = jax.vmap(scaled)(jnp.array([0.5, 1.0, 2.0]))
        assert isinstance(batched, WatertightMesh)
        assert batched.vertices.shape == (3, 8, 3)
        assert batched.faces.shape == (3, 12, 3)
        assert batched.max_gap.shape == (3,)

    def test_max_gap_reports_raw_node_disagreement_within_one_face(self) -> None:
        """``max_gap`` exposes two of the *same* face's own nodes
        disagreeing, which the face-weighted mean used for ``vertices``
        does not.

        Edits one control point of a full periodic B-spline sphere, near
        its own pole: the pole is touched by only one face, so the
        per-face mean used for ``vertices`` has nothing to average it
        against and just absorbs the edit into that face's own mean.
        ``max_gap`` instead looks at the raw per-node evaluations before
        either level of averaging, so it must report the edit's own
        scale, not an unrelated near-zero baseline residual.
        """
        shape = BRepBuilderAPI_NurbsConvert(
            BRepPrimAPI_MakeSphere(5.0).Shape(), True
        ).Shape()
        topology = extract_watertight_topology(shape, deflection=0.01)
        baseline = evaluate_watertight_mesh(topology, topology.params_list)
        assert float(baseline.max_gap) < 1e-6

        params = [dict(p) for p in topology.params_list]
        control_points = params[0]["control_points"]
        edited = control_points.at[0, 0].add(jnp.array([10.0, 0.0, 0.0]))
        params[0] = {**params[0], "control_points": edited}
        edited_mesh = evaluate_watertight_mesh(topology, params)

        assert float(edited_mesh.max_gap) == pytest.approx(10.0, rel=1e-6)
