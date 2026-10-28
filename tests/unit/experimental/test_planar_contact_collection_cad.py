"""Exhaustive CAD-cell contact inventory and source/reference distinctions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax._occt.backend import (
    BRep_Builder,
    BRep_Tool,
    BRepAlgoAPI_Common,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_Transform,
    BRepCheck_Analyzer,
    BRepGProp,
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakePrism,
    GProp_GProps,
    TopAbs_VERTEX,
    TopExp_Explorer,
    TopoDS,
    gp_Pnt,
    gp_Trsf,
    gp_Vec,
)
from brepax.experimental._polyhedral_contact import hull_relation, vertex_hull
from brepax.experimental.planar_contact import planar_contact_set_volume_direction
from brepax.experimental.planar_contact_collection_cad import (
    ContactPreparationError,
    prepare_planar_contact_set_reference,
)


def _box(lower, upper):
    return BRepPrimAPI_MakeBox(gp_Pnt(*lower), gp_Pnt(*upper)).Shape()


def _corner():
    return (
        _box((-1, 0, -1), (1, 1, 0)),
        _box((-1, 0, 0), (0, 1, 1)),
    ), (_box((0, 0, 0), (1, 1, 1)),)


def _rate(prepared, velocity):
    return float(
        planar_contact_set_volume_direction(
            prepared.contact_set,
            jnp.asarray(velocity, dtype=jnp.float64),
            jnp.zeros(3),
            jnp.zeros(3),
        )
    )


def _volume(shape):
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return properties.Mass()


def _transform(shape, rotation, translation):
    transform = gp_Trsf()
    transform.SetValues(*map(float, np.column_stack((rotation, translation)).ravel()))
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _prism(vertices, vector):
    polygon = BRepBuilderAPI_MakePolygon()
    for point in vertices:
        polygon.Add(gp_Pnt(*point))
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(*vector)).Shape()


def _move_topological_vertices(shape, update):
    builder = BRep_Builder()
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    visited = []
    while explorer.More():
        vertex = TopoDS.Vertex_s(explorer.Current())
        if not any(vertex.IsSame(previous) for previous in visited):
            point = BRep_Tool.Pnt_s(vertex)
            updated = update(np.array([point.X(), point.Y(), point.Z()]))
            builder.UpdateVertex(
                vertex, gp_Pnt(*map(float, updated)), BRep_Tool.Tolerance_s(vertex)
            )
            visited.append(vertex)
        explorer.Next()
    assert BRepCheck_Analyzer(shape, True, False, True).IsValid()


def test_all_contact_patches_recover_corner_volume_and_cell_provenance():
    fixed, moving = _corner()
    prepared = prepare_planar_contact_set_reference(fixed, moving, mode="zero_residual")
    assert len(prepared.patches) == 2
    assert [
        (pair.first_cell, pair.second_cell, pair.relation)
        for pair in prepared.cell_pairs
    ] == [
        (0, 0, "face_contact"),
        (1, 0, "face_contact"),
    ]
    assert [
        (source.patch_index, source.fixed_cell_id, source.moving_cell_id)
        for source in prepared.patch_sources
    ] == [
        (0, 0, 0),
        (1, 1, 0),
    ]
    assert prepared.partition_pairs[0].relation == "face_contact"
    assert prepared.original_cad_contact_certified is False
    assert _rate(prepared, (-1, 0, -1)) == pytest.approx(2.0, abs=1e-13)
    for step in (1e-2, 1e-3):
        tool = _transform(moving[0], np.eye(3), (-step, 0, -step))
        volume = sum(_volume(BRepAlgoAPI_Common(cell, tool).Shape()) for cell in fixed)
        assert volume == pytest.approx(2 * step - step**2, rel=2e-10)


def test_opposing_slot_walls_have_nonzero_opposite_direction_responses():
    fixed = (_box((-1, 0, 0), (0, 1, 1)), _box((1, 0, 0), (2, 1, 1)))
    moving = (_box((0, 0, 0), (1, 1, 1)),)
    prepared = prepare_planar_contact_set_reference(fixed, moving, mode="zero_residual")
    assert len(prepared.patches) == 2
    assert prepared.partition_pairs[0].relation == "separated"
    assert _rate(prepared, (1, 0, 0)) == pytest.approx(1.0, abs=1e-13)
    assert _rate(prepared, (-1, 0, 0)) == pytest.approx(1.0, abs=1e-13)


def test_inventory_retains_face_edge_point_and_separated_pairs():
    fixed = (_box((0, 0, 0), (1, 1, 1)),)
    moving = (
        _box((1, 0, 0), (2, 1, 1)),
        _box((1, 1, 0), (2, 2, 1)),
        _box((1, 1, 1), (2, 2, 2)),
        _box((3, 0, 0), (4, 1, 1)),
    )
    prepared = prepare_planar_contact_set_reference(fixed, moving, mode="zero_residual")
    assert [pair.relation for pair in prepared.cell_pairs] == [
        "face_contact",
        "edge_contact",
        "point_contact",
        "separated",
    ]
    assert len(prepared.partition_pairs) == 6
    assert len(prepared.patches) == 1


@pytest.mark.parametrize("gap", [1e-3, 1e-8, 1e-16])
@pytest.mark.parametrize("mode", ["zero_residual", "projected_nominal"])
def test_every_represented_positive_gap_stays_separated(gap, mode):
    fixed = (_box((-1, -1, -1), (1, 1, 0)),)
    moving = (_box((-1, -1, gap), (1, 1, 1)),)
    prepared = prepare_planar_contact_set_reference(fixed, moving, mode=mode)
    assert prepared.cell_pairs[0].relation == "separated"
    assert prepared.cell_pairs[0].axis_gap == pytest.approx(gap, rel=1e-14, abs=0)
    assert not prepared.patches
    assert _rate(prepared, (0, 0, -1)) == 0.0


@pytest.mark.parametrize("depth", [0.2, 1e-8, 1e-16])
def test_initial_overlap_rejects_with_pair_evidence_even_below_cad_tolerance(depth):
    fixed = (_box((-1, -1, -1), (1, 1, 0)),)
    moving = (_box((-1, -1, -depth), (1, 1, 1)),)
    with pytest.raises(
        ContactPreparationError, match="existing represented overlap"
    ) as error:
        prepare_planar_contact_set_reference(fixed, moving, mode="projected_nominal")
    assert error.value.pair.scope == "cross"
    assert error.value.pair.relation == "overlap"
    assert error.value.pair.axis_gap == pytest.approx(-depth, rel=1e-14, abs=0)


@pytest.mark.parametrize("body", ["fixed", "moving"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_same_body_overlap_or_duplicate_cells_cannot_double_count(body, duplicate):
    cell = _box((0, 0, 0), (1, 1, 1))
    other = cell if duplicate else _box((0.5, 0, 0), (1.5, 1, 1))
    args = ((cell, other), ()) if body == "fixed" else ((), (cell, other))
    with pytest.raises(
        ContactPreparationError, match="overlapping represented interiors"
    ) as error:
        prepare_planar_contact_set_reference(*args, mode="zero_residual")
    assert error.value.pair.scope == body
    assert error.value.pair.relation == "overlap"


def test_face_against_prism_edge_is_not_misclassified_as_face_contact():
    wedge = _prism(((-1, -0.5, 1), (0, -0.5, 0), (1, -0.5, 1)), (0, 1, 0))
    fixed = (_box((-2, -2, -1), (2, 2, 0)),)
    prepared = prepare_planar_contact_set_reference(
        fixed, (wedge,), mode="zero_residual"
    )
    assert prepared.cell_pairs[0].relation == "edge_contact"
    assert not prepared.patches
    # A zero rate does not mean a finite downward edit is collision free.
    moved = _transform(wedge, np.eye(3), (0, 0, -0.01))
    assert _volume(BRepAlgoAPI_Common(fixed[0], moved).Shape()) == pytest.approx(
        1e-4, rel=1e-9
    )


def test_crossing_support_edges_have_point_contact_and_cubic_finite_overlap():
    fixed = _prism(((-0.5, -1, -1), (-0.5, 0, 0), (-0.5, 1, -1)), (1, 0, 0))
    moving = _prism(((-1, -0.5, 1), (0, -0.5, 0), (1, -0.5, 1)), (0, 1, 0))
    prepared = prepare_planar_contact_set_reference(
        (fixed,), (moving,), mode="zero_residual"
    )
    assert prepared.cell_pairs[0].relation == "point_contact"
    assert not prepared.patches
    step = 0.01
    moved = _transform(moving, np.eye(3), (0, 0, -step))
    # The overlap-layer height is t-|x|-|y| on the diamond |x|+|y|<=t.
    assert _volume(BRepAlgoAPI_Common(fixed, moved).Shape()) == pytest.approx(
        2 * step**3 / 3, rel=1e-8
    )


def test_nominal_mode_only_projects_faces_after_represented_touching():
    fixed, moving = _corner()
    cosine, sine = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    placed_fixed = tuple(_transform(cell, rotation, (3, -2, 5)) for cell in fixed)
    placed_moving = tuple(_transform(cell, rotation, (3, -2, 5)) for cell in moving)
    with pytest.raises(
        ContactPreparationError, match="source-face reference is unresolved"
    ) as error:
        prepare_planar_contact_set_reference(
            placed_fixed, placed_moving, mode="zero_residual"
        )
    assert error.value.pair.relation == "uncertain"
    assert error.value.pair.axis_gap == 0
    prepared = prepare_planar_contact_set_reference(
        placed_fixed, placed_moving, mode="projected_nominal"
    )
    assert [pair.relation for pair in prepared.cell_pairs] == ["face_contact"] * 2
    assert any(
        source.diagnostics.plane_distance_range != (0.0, 0.0)
        for source in prepared.patch_sources
    )
    assert _rate(prepared, rotation @ np.array([-1.0, 0, -1.0])) == pytest.approx(
        2.0, abs=2e-13
    )
    assert prepared.original_cad_contact_certified is False


def test_unresolved_narrow_contact_cannot_select_an_unrelated_source_plane():
    epsilon = 1e-13
    fixed = _box((-100, 0, -1), (0, 100, 0))
    moving = _box((-epsilon, 0, -epsilon), (100, 100, 1))
    _move_topological_vertices(
        moving, lambda point: np.array([point[0], point[1], max(point[2], 0.0)])
    )
    # Represented contact is a narrow horizontal patch. CAD still has nearby
    # vertical source faces, which must not be selected when that patch is
    # below the planar preparation's area-resolution bound.
    with pytest.raises(ContactPreparationError, match="no positive-area") as error:
        prepare_planar_contact_set_reference(
            (fixed,), (moving,), mode="projected_nominal"
        )
    assert error.value.pair.relation == "uncertain"
    assert error.value.pair.axis == (0.0, 0.0, 1.0)


def test_parallel_source_planes_must_resolve_the_represented_support_plane():
    fixed = _box((-100, 0, -100), (0, 1, 0))
    moving = _box((-100, 0, 0), (0, 1, 100))
    # Both source planes coincide at z=0, but the represented support lies at
    # z=1e-12. This fits each long wire's consistency allowance while exceeding
    # the contact face's area-based length bound. Matching normals is not enough.
    for shape in (fixed, moving):
        _move_topological_vertices(shape, lambda point: point + np.array([0, 0, 1e-12]))
    with pytest.raises(
        ContactPreparationError, match="exact represented contact support"
    ) as error:
        prepare_planar_contact_set_reference(
            (fixed,), (moving,), mode="projected_nominal"
        )
    assert error.value.pair.relation == "uncertain"


def test_exact_contact_support_does_not_use_an_underflowed_gap_diagnostic():
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, -np.ldexp(1.0, -500)],
            [0.0, np.nextafter(0.0, 1.0), 0.0],
        ]
    )
    first = vertex_hull(np.vstack((base, [-1.0, 0, 0])))
    second = vertex_hull(np.vstack((base, [1.0, 0, 0])))
    relation = hull_relation(first, second)
    assert relation.relation == "face_contact"
    assert relation.support is not None
    assert relation.support.axis == (1, 0, 0)
    assert relation.support.offset == 0


def test_empty_collection_and_explicit_mode_precision():
    empty = prepare_planar_contact_set_reference((), (), mode="zero_residual")
    assert not empty.cell_pairs and not empty.partition_pairs and not empty.patches
    assert _rate(empty, (1, 2, 3)) == 0
    with pytest.raises(TypeError, match="mode"):
        prepare_planar_contact_set_reference((), ())
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="x64"):
            prepare_planar_contact_set_reference((), (), mode="zero_residual")
    finally:
        jax.config.update("jax_enable_x64", True)
