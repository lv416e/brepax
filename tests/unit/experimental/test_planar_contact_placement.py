"""Explicit placement preserves local contact meaning, units and motion fields."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax._occt.backend import (
    BRep_Builder,
    BRepAlgoAPI_Common,
    BRepBuilderAPI_Transform,
    BRepGProp,
    BRepPrimAPI_MakeBox,
    GProp_GProps,
    TopLoc_Location,
    TopoDS_Compound,
    gp_Pnt,
    gp_Trsf,
)
from brepax._occt.types import TopoDS_Shape
from brepax.experimental.planar_contact import (
    prepare_planar_contact_patch,
    prepare_planar_contact_set,
)
from brepax.experimental.planar_contact_collection_cad import ContactPreparationError
from brepax.experimental.planar_contact_placement import (
    place_planar_contact_set,
    placed_planar_contact_set_volume_direction,
    prepare_located_planar_contact_set_reference,
    prepare_placed_planar_contact_set_reference,
)

ROTATION = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
ORIGIN = np.array([3.0, -2.0, 4.0])


def _contacts():
    floor = prepare_planar_contact_patch(
        [[0, 0], [2, 0], [2, 1], [0, 1]],
        origin=[0, 0, 0],
        basis=[[1, 0], [0, 1], [0, 0]],
    )
    wall = prepare_planar_contact_patch(
        [[0, 0], [1, 0], [1, 3], [0, 3]],
        origin=[0, 0, 0],
        basis=[[0, 0], [1, 0], [0, 1]],
    )
    return prepare_planar_contact_set((floor, wall), capacity=8)


def _box(lower, upper):
    return BRepPrimAPI_MakeBox(gp_Pnt(*lower), gp_Pnt(*upper)).Shape()


def _corner():
    return (
        _box((-1, 0, -1), (1, 1, 0)),
        _box((-1, 0, 0), (0, 1, 1)),
    ), (_box((0, 0, 0), (1, 1, 1)),)


def _location(linear=ROTATION, origin=ORIGIN):
    transform = gp_Trsf()
    transform.SetValues(*map(float, np.column_stack((linear, origin)).ravel()))
    return TopLoc_Location(transform)


def _rate(model, velocity, angular=(0, 0, 0), pivot=(0, 0, 0)):
    return placed_planar_contact_set_volume_direction(
        model,
        jnp.asarray(velocity, dtype=jnp.float64),
        jnp.asarray(angular, dtype=jnp.float64),
        jnp.asarray(pivot, dtype=jnp.float64),
    )


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_world_motion_covariance_masking_and_compensated_instantaneous_pivot(scale):
    model = place_planar_contact_set(
        _contacts(), rotation=ROTATION, origin=ORIGIN, length_scale=scale
    )
    velocity = np.array([[-2, 3, -4], [0, 0, 0], [0, 0, 0], [1, 2, 3]])
    angular = np.array([[0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    # Independent integrals: floor area 2, wall area 3; the two tilts
    # integrate y over a 2x1 rectangle and x over that same rectangle.
    expected = scale**3 * np.array([14.0, 1.0, 2.0, 0.0])
    world_v = scale * velocity @ ROTATION.T
    world_w = angular @ ROTATION.T
    world_p = np.broadcast_to(ORIGIN, (4, 3))
    evaluate = jax.jit(
        jax.vmap(placed_planar_contact_set_volume_direction, in_axes=(None, 0, 0, 0))
    )
    actual = evaluate(
        model, jnp.asarray(world_v), jnp.asarray(world_w), jnp.asarray(world_p)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=scale**3 * 1e-14)
    delta = scale * np.array([[1, 2, 3], [-2, 1, 1], [3, 1, -1], [1, 0, 2]])
    compensated = evaluate(
        model,
        jnp.asarray(world_v + np.cross(world_w, delta)),
        jnp.asarray(world_w),
        jnp.asarray(world_p + delta),
    )
    np.testing.assert_allclose(compensated, expected, rtol=2e-13, atol=scale**3 * 1e-14)


@pytest.mark.parametrize(
    "scale,speed,expected", [(1e120, 1e-220, 2e20), (1e-120, 1e220, 2e-20)]
)
def test_area_scaling_avoids_unnecessary_cubic_overflow(scale, speed, expected):
    model = place_planar_contact_set(
        _contacts(), rotation=np.eye(3), origin=np.zeros(3), length_scale=scale
    )
    result = jax.jit(_rate)(model, jnp.array([0.0, 0.0, -speed]))
    assert result == pytest.approx(expected, rel=2e-14, abs=0)


def test_empty_model_keeps_zero_rate_distinct_from_invalid_motion():
    model = place_planar_contact_set(
        prepare_planar_contact_set(()), rotation=ROTATION, origin=ORIGIN
    )
    assert jax.jit(_rate)(model, jnp.array([1.0, 2.0, 3.0])) == 0
    assert np.isnan(jax.jit(_rate)(model, jnp.array([1.0, 2.0, jnp.nan])))


@pytest.mark.parametrize(
    "rotation",
    [np.diag([-1, 1, 1]), np.diag([1, 1, 1.001]), np.full((3, 3), np.nan), np.eye(2)],
)
def test_invalid_rotation_is_rejected(rotation):
    with pytest.raises(ValueError, match=r"rotation|Rotation"):
        place_planar_contact_set(_contacts(), rotation=rotation, origin=np.zeros(3))


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, float("nan"), float("inf"), 1e200, 1e-200]
)
def test_unsupported_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        place_planar_contact_set(
            _contacts(), rotation=ROTATION, origin=ORIGIN, length_scale=scale
        )


def test_host_validation_and_array_ownership():
    local = _contacts()
    rotation, origin = ROTATION.copy(), ORIGIN.copy()
    placed = place_planar_contact_set(local, rotation=rotation, origin=origin)
    rotation[:] = 0
    origin[:] = 100
    np.testing.assert_array_equal(placed.rotation, ROTATION)
    np.testing.assert_array_equal(placed.origin, ORIGIN)
    with pytest.raises(ValueError, match="origin"):
        place_planar_contact_set(local, rotation=ROTATION, origin=[0, 0, np.inf])
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="x64"):
            place_planar_contact_set(local, rotation=ROTATION, origin=ORIGIN)
    finally:
        jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("gap", [2.0**-20, 2.0**-40, -(2.0**-40)])
def test_explicit_local_frame_preserves_gap_and_overlap(gap):
    fixed = (_box((0, 0, -1), (1, 1, 0)),)
    moving = (_box((0, 0, gap), (1, 1, 1)),)
    if gap < 0:
        with pytest.raises(ContactPreparationError) as caught:
            prepare_placed_planar_contact_set_reference(
                fixed,
                moving,
                rotation=ROTATION,
                origin=ORIGIN,
                length_scale=0.1,
                mode="projected_nominal",
            )
        assert caught.value.pair.relation == "overlap"
        assert caught.value.pair.axis_gap == pytest.approx(gap, rel=1e-13, abs=0)
    else:
        prepared = prepare_placed_planar_contact_set_reference(
            fixed,
            moving,
            rotation=ROTATION,
            origin=ORIGIN,
            length_scale=0.1,
            mode="projected_nominal",
        )
        pair = prepared.local_reference.cell_pairs[0]
        assert pair.relation == "separated"
        assert pair.axis_gap == pytest.approx(gap, rel=1e-13, abs=0)
        assert prepared.provenance == "supplied_local_frame"
        assert not prepared.original_cad_contact_certified
        assert _rate(prepared.model, (0, 0, -1)) == 0


def test_shared_cad_location_recovers_frame_and_agrees_with_finite_cad():
    fixed, moving = _corner()
    location = _location()
    fixed_world = tuple(shape.Located(location, True) for shape in fixed)
    moving_world = tuple(shape.Located(location, True) for shape in moving)
    prepared = prepare_located_planar_contact_set_reference(
        fixed_world, moving_world, mode="zero_residual"
    )
    assert prepared.provenance == "shared_cad_location"
    assert not prepared.original_cad_contact_certified
    assert len(prepared.local_reference.patches) == 2
    actual = location.Transformation()
    matrix = np.array([[actual.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    offset = np.array([actual.Value(i, 4) for i in (1, 2, 3)])
    velocity = matrix @ np.array([-1.0, 0, -1])
    assert _rate(prepared.model, velocity, pivot=offset) == pytest.approx(
        2.0, rel=1e-13
    )
    step = 0.02
    translated = _box((-step, 0, -step), (1 - step, 1, 1 - step)).Located(
        location, True
    )
    measured = 0.0
    for cell in fixed_world:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(
            BRepAlgoAPI_Common(cell, translated).Shape(), props
        )
        measured += props.Mass()
    assert measured == pytest.approx(2 * step - step**2, rel=1e-10, abs=1e-13)
    assert all(
        shape.Location().IsEqual(location) for shape in fixed_world + moving_world
    )


def test_removing_shared_top_level_location_preserves_nested_cell_placement():
    fixed, moving = _corner()
    child_location = _location(np.eye(3), np.array([2.0, 0.0, 3.0]))
    common_location = _location()
    builder = BRep_Builder()
    world_cells = []
    children = []
    for cell in fixed + moving:
        child = cell.Located(child_location, True)
        children.append(child)
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        builder.Add(compound, child)
        world_cells.append(compound.Located(common_location, True))
    prepared = prepare_located_planar_contact_set_reference(
        world_cells[:2], world_cells[2:], mode="zero_residual"
    )
    # y-axis rotation about local zero integrates x over the floor's [2,3]
    # interval. Losing the nested translation would give 0.5 instead of 2.5.
    transform = common_location.Transformation()
    world_axis = np.array([transform.Value(i, 2) for i in (1, 2, 3)])
    world_pivot = np.array([transform.Value(i, 4) for i in (1, 2, 3)])
    assert _rate(prepared.model, (0, 0, 0), world_axis, world_pivot) == pytest.approx(
        2.5, rel=2e-13
    )
    assert all(child.Location().IsEqual(child_location) for child in children)
    assert all(cell.Location().IsEqual(common_location) for cell in world_cells)


@pytest.mark.parametrize("linear", [0.1 * ROTATION, 10 * ROTATION, np.diag([-1, 1, 1])])
def test_scaled_or_reflected_source_locations_are_rejected(linear):
    fixed, moving = _corner()
    location = _location(linear)
    with pytest.raises(ValueError, match="proper rigid"):
        prepare_located_planar_contact_set_reference(
            tuple(shape.Located(location) for shape in fixed),
            tuple(shape.Located(location) for shape in moving),
            mode="projected_nominal",
        )


def test_shared_location_is_not_inferred_from_equal_matrices_or_baked_coordinates():
    fixed, moving = _corner()
    first, separately_created = _location(), _location()
    with pytest.raises(ValueError, match="same explicit"):
        prepare_located_planar_contact_set_reference(
            tuple(shape.Located(first) for shape in fixed),
            tuple(shape.Located(separately_created) for shape in moving),
            mode="projected_nominal",
        )
    transform = first.Transformation()
    baked_fixed = tuple(
        BRepBuilderAPI_Transform(shape, transform, True).Shape() for shape in fixed
    )
    baked_moving = tuple(
        BRepBuilderAPI_Transform(shape, transform, True).Shape() for shape in moving
    )
    assert all(shape.Location().IsIdentity() for shape in baked_fixed + baked_moving)
    with pytest.raises(ContactPreparationError):
        prepare_located_planar_contact_set_reference(
            baked_fixed, baked_moving, mode="projected_nominal"
        )


@pytest.mark.parametrize("fixed,moving", [((), ()), ((TopoDS_Shape(),), ())])
def test_location_frame_requires_a_nonnull_source(fixed, moving):
    with pytest.raises(ValueError, match="non-null"):
        prepare_located_planar_contact_set_reference(
            fixed, moving, mode="zero_residual"
        )
