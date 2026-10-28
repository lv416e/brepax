"""Actual CAD acceptance, rejection, and covariance of contact preparation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax._occt.backend import (
    BRep_Tool,
    BRepAdaptor_Surface,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepBuilderAPI_Transform,
    BRepGProp,
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    GeomAbs_Plane,
    GProp_GProps,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopExp_Explorer,
    TopoDS,
    gp_Ax2,
    gp_Dir,
    gp_Pnt,
    gp_Trsf,
)
from brepax.experimental.planar_contact import planar_contact_volume_direction
from brepax.experimental.planar_contact_cad import prepare_planar_contact_reference


def _box(origin, size):
    return BRepPrimAPI_MakeBox(gp_Pnt(*origin), *size).Shape()


def _pair(gap=0.0):
    return _box((-2, -2, -1), (4, 4, 1)), _box((-1, -1, gap), (2, 2, 1))


def _faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    result = []
    while explorer.More():
        result.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    return result


def _support_id(shape, expected, offset=0.0):
    for index, face in enumerate(_faces(shape)):
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() != GeomAbs_Plane:
            continue
        plane = adaptor.Plane()
        direction, location = plane.Axis().Direction(), plane.Location()
        normal = np.array([direction.X(), direction.Y(), direction.Z()])
        if face.Orientation() == TopAbs_REVERSED:
            normal = -normal
        point = np.array([location.X(), location.Y(), location.Z()])
        if np.array_equal(normal, expected) and np.dot(point, normal) == offset:
            return index
    raise AssertionError("Fixture support face not found")


def _prepare(fixed, moving, mode="zero_residual", fixed_id=5, moving_id=4):
    return prepare_planar_contact_reference(
        fixed,
        moving,
        fixed_face_id=fixed_id,
        moving_face_id=moving_id,
        mode=mode,
    )


def _rate(prepared, velocity, angular_velocity=(0, 0, 0), pivot=(0, 0, 0)):
    return float(
        planar_contact_volume_direction(
            prepared.patch,
            jnp.asarray(velocity, dtype=jnp.float64),
            jnp.asarray(angular_velocity, dtype=jnp.float64),
            jnp.asarray(pivot, dtype=jnp.float64),
        )
    )


def test_selected_cad_faces_supply_oriented_patch_to_numerical_kernel():
    prepared = _prepare(*_pair())
    assert _rate(prepared, (0, 0, -1)) == pytest.approx(4.0, abs=1e-13)
    assert _rate(prepared, (0, 0, 1)) == pytest.approx(0.0, abs=1e-13)
    assert _rate(prepared, (0, 0, 0), (1, 0, 0)) == pytest.approx(1.0, abs=1e-13)
    assert _rate(prepared, (0, 0, 0), (-1, 0, 0)) == pytest.approx(1.0, abs=1e-13)
    diagnostics = prepared.diagnostics
    assert diagnostics.plane_distance_range == (0.0, 0.0)
    assert diagnostics.opposed_normal_residual == 0.0
    assert diagnostics.reference_kind == "projected_support_polygons"
    assert diagnostics.original_cad_contact_certified is False


def test_partial_overlap_matches_independent_occt_face_area():
    fixed = _box((-0.5, -2, -1), (2.5, 4, 1))
    moving = _pair()[1]
    prepared = _prepare(fixed, moving)
    intersection = BRepAlgoAPI_Common(_faces(fixed)[5], _faces(moving)[4])
    assert intersection.IsDone()
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(intersection.Shape(), properties)
    assert properties.Mass() == pytest.approx(3.0, abs=1e-13)
    assert _rate(prepared, (0, 0, -1)) == pytest.approx(properties.Mass(), abs=1e-13)


@pytest.mark.parametrize("gap", [1e-3, 1e-8, 1e-16, -1e-8])
def test_nonzero_source_gap_is_not_snapped_by_occt_tolerance(gap):
    fixed, moving = _pair(gap)
    assert BRep_Tool.Tolerance_s(_faces(moving)[4]) >= 1e-7
    with pytest.raises(ValueError, match="Nonzero contact residual"):
        _prepare(fixed, moving)


def test_nominal_reference_retains_positive_source_gap_and_different_target():
    prepared = _prepare(*_pair(1e-16), mode="projected_nominal")
    diagnostics = prepared.diagnostics
    assert diagnostics.signed_plane_gap == pytest.approx(1e-16, rel=1e-14, abs=0)
    assert diagnostics.plane_distance_range == (1e-16, 1e-16)
    assert diagnostics.original_cad_contact_certified is False
    # The nominal contact model has rate 4. The source solid has a fixed
    # positive gap, so its true t->0+ intersection derivative is zero.
    assert _rate(prepared, (0, 0, -1)) == pytest.approx(4.0, abs=1e-13)
    with pytest.raises(ValueError, match="nominal roundoff bound"):
        _prepare(*_pair(1e-8), mode="projected_nominal")


def _transform(shape, scale, rotation, translation):
    transform = gp_Trsf()
    coefficients = np.column_stack((scale * rotation, translation)).reshape(-1)
    transform.SetValues(*map(float, coefficients))
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_actual_cad_rigid_placement_and_units_preserve_nominal_rate(scale):
    cosine, sine = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    translation = scale * np.array([3, -2, 5])
    fixed, moving = [
        _transform(shape, scale, rotation, translation) for shape in _pair()
    ]
    prepared = _prepare(fixed, moving, mode="projected_nominal")
    velocity = scale * rotation @ np.array([0, 0, -0.3])
    angular_velocity = rotation @ np.array([0.7, -0.2, 0.4])
    baseline = _rate(_prepare(*_pair()), (0, 0, -0.3), (0.7, -0.2, 0.4))
    actual = _rate(prepared, velocity, angular_velocity, translation)
    assert actual / scale**3 == pytest.approx(baseline, rel=3e-13)


@pytest.mark.parametrize("origin", [(3, -1, 0), (2, -1, 0), (2, 2, 0)])
def test_disjoint_edge_only_and_vertex_only_face_contacts_reject(origin):
    with pytest.raises(ValueError, match="no positive-area"):
        _prepare(_pair()[0], _box(origin, (2, 2, 1)))


def test_same_oriented_selected_faces_reject():
    with pytest.raises(ValueError, match="normals must oppose"):
        _prepare(*_pair(), moving_id=5)


def test_curved_solid_rejects_even_if_selected_cap_is_planar():
    cylinder = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.5, 1
    ).Shape()
    with pytest.raises(ValueError, match=r"planar faces|straight edges"):
        _prepare(_pair()[0], cylinder, moving_id=_support_id(cylinder, [0, 0, -1]))


@pytest.mark.parametrize("kind", ["hole", "notch"])
def test_holes_and_nonconvex_faces_reject(kind):
    fixed, moving = _pair()
    if kind == "hole":
        cutter = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 0.4, 3
        ).Shape()
        message = "one wire|planar faces"
    else:
        cutter = _box((0, 0, -1), (2, 2, 3))
        message = "convex"
    moving = BRepAlgoAPI_Cut(moving, cutter).Shape()
    with pytest.raises(ValueError, match=message):
        _prepare(fixed, moving, moving_id=_support_id(moving, [0, 0, -1]))


@pytest.mark.parametrize("kind", ["bottom_and_side", "two_top_patches"])
def test_other_contact_patches_fail_whole_solid_support_validation(kind):
    fixed, moving = _pair()
    if kind == "bottom_and_side":
        wall = _box((1, -2, -1), (1, 4, 3))
        fixed = BRepAlgoAPI_Fuse(fixed, wall).Shape()
    else:
        slot = _box((-0.5, -3, -0.5), (1, 6, 1))
        fixed = BRepAlgoAPI_Cut(fixed, slot).Shape()
    with pytest.raises(ValueError, match=r"convex|multiple faces"):
        _prepare(fixed, moving, fixed_id=_support_id(fixed, [0, 0, 1]))


@pytest.mark.parametrize("face_id", [-1, 6, True])
def test_invalid_face_selection_rejects(face_id):
    with pytest.raises(ValueError, match=r"Face ID|face ID"):
        _prepare(*_pair(), fixed_id=face_id)


def test_mode_and_precision_are_explicit():
    with pytest.raises(TypeError, match="mode"):
        prepare_planar_contact_reference(*_pair(), fixed_face_id=5, moving_face_id=4)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="x64"):
            _prepare(*_pair())
    finally:
        jax.config.update("jax_enable_x64", True)
