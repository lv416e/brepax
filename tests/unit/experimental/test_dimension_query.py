"""Certified minimal volume changes against brute force and rebuilt CAD."""

import itertools
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.GProp import GProp_GProps

from brepax.experimental.applications.cad_frame import CadFrame
from brepax.experimental.applications.dimension_query import (
    DimensionLaw,
    TravelLimit,
    face_offset_laws,
    hole_ligament_constraints,
    minimal_volume_change,
)
from brepax.experimental.applications.face_offset import (
    prepare_face_offset,
    recognize_face_offsets,
)
from brepax.experimental.applications.hole_edit import prepare_hole_edit
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
ZERO = Fraction(0)


def face_law(area, lower=-2.0, upper=2.0, weight=1.0):
    return DimensionLaw(
        "face", 0.0, lower, upper, weight, (ZERO, Fraction(area), ZERO), (ZERO,) * 3
    )


def hole_law(radius, height, weight=1.0):
    return DimensionLaw(
        "hole",
        radius,
        radius / 2,
        radius * 1.2,
        weight,
        (ZERO,) * 3,
        (ZERO, ZERO, -Fraction(height)),
    )


def contribution(law, x, pi):
    def value(c, t):
        return c[0] + t * (c[1] + t * c[2])

    x, x0 = Fraction(x), Fraction(law.reference)
    return (
        value(law.rational, x)
        - value(law.rational, x0)
        + pi * (value(law.pi_coefficients, x) - value(law.pi_coefficients, x0))
    )


def brute_force(laws, band, steps):
    """Grid over all but the last handle; solve the last one densely too."""
    pi = Fraction(math.pi)
    best = None
    grids = [
        [law.lower + (law.upper - law.lower) * k / steps for k in range(steps + 1)]
        + [law.reference]
        for law in laws
    ]
    for values in itertools.product(*grids):
        total = sum(
            contribution(law, x, pi) for law, x in zip(laws, values, strict=True)
        )
        if band[0] <= total <= band[1]:
            cost = sum(
                law.weight * abs(x - law.reference)
                for law, x in zip(laws, values, strict=True)
            )
            best = cost if best is None else min(best, cost)
    return best


@pytest.mark.parametrize(
    ("laws", "band"),
    [
        ((face_law(12.0), face_law(30.0, weight=2.0)), (-25.0, -24.0)),
        ((hole_law(1.5, 4.0), face_law(8.0)), (-12.0, -11.0)),
        ((hole_law(1.0, 6.0), hole_law(2.0, 3.0)), (5.0, 6.0)),
        ((hole_law(2.0, 5.0, weight=3.0), hole_law(1.2, 2.0)), (-9.0, -8.5)),
    ],
)
def test_certified_optimum_is_no_worse_than_brute_force(laws, band):
    result = minimal_volume_change(laws, band)
    assert result.status == "optimal_within_tolerance"
    pi = Fraction(math.pi)
    total = sum(
        contribution(law, x, pi) for law, x in zip(laws, result.values, strict=True)
    )
    assert band[0] - 1e-12 <= total <= band[1] + 1e-12
    assert band[0] <= result.volume_change[0] <= result.volume_change[1] <= band[1]
    found = brute_force(laws, band, steps=400)
    assert found is not None
    # The certified cost cannot beat the true optimum by more than the gap,
    # and a dense grid cannot beat the certified lower bound.
    assert float(result.lower_bound) <= found + 1e-9
    assert float(result.change) <= found + 1e-9
    assert float(result.change - result.lower_bound) <= 1e-6


def test_unreachable_band_is_infeasible_with_the_exact_range():
    laws = (face_law(10.0), hole_law(1.0, 2.0))
    result = minimal_volume_change(laws, (100.0, 101.0))
    assert result.status == "infeasible"
    assert result.values is None
    # Face: [-20, 20]; hole radius 0.5..1.2 removes up to pi*2*(1.44-1)
    # and adds up to pi*2*(1-0.25).
    low = -20 - math.pi * 2 * 0.44
    high = 20 + math.pi * 2 * 0.75
    assert float(result.reachable[0]) == pytest.approx(low, rel=1e-12)
    assert float(result.reachable[1]) == pytest.approx(high, rel=1e-12)


def test_weights_choose_the_cheaper_handle():
    laws = (face_law(10.0, weight=1.0), face_law(10.0, weight=5.0))
    result = minimal_volume_change(laws, (-10.0, -10.0))
    assert result.values == (-1.0, 0.0)
    laws = (face_law(10.0, weight=5.0), face_law(10.0, weight=1.0))
    assert minimal_volume_change(laws, (-10.0, -10.0)).values == (0.0, -1.0)


def test_node_budget_leaves_a_valid_lower_bound():
    laws = tuple(hole_law(0.5 + 0.1 * k, 2.0 + k) for k in range(6))
    full = minimal_volume_change(laws, (-20.0, -19.0))
    assert full.status == "optimal_within_tolerance"
    capped = minimal_volume_change(laws, (-20.0, -19.0), max_nodes=0)
    assert capped.status == "unresolved"
    assert capped.lower_bound <= full.change


def test_lower_bound_holds_when_the_gap_tolerance_prunes_boxes():
    laws = (hole_law(1.07, 3.33, weight=2.94), hole_law(0.71, 2.17, weight=1.18))
    band = (3.25, 3.55)
    exact = minimal_volume_change(laws, band)
    assert exact.status == "optimal_within_tolerance"
    loose = minimal_volume_change(laws, band, tolerance=0.01)
    assert loose.status == "optimal_within_tolerance"
    # Boxes discarded as within the gap of the incumbent still count.
    assert loose.lower_bound <= exact.change
    assert loose.change - loose.lower_bound <= Fraction(0.01)


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError, match="low <= high"):
        minimal_volume_change((face_law(1.0),), (1.0, 0.0))
    turning = DimensionLaw(
        "turning", 0.0, -1.0, 1.0, 1.0, (ZERO, ZERO, Fraction(1)), (ZERO,) * 3
    )
    with pytest.raises(ValueError, match="not monotone"):
        minimal_volume_change((turning,), (0.1, 0.2))


def mass(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, 1e-9)
    return props.Mass()


def test_face_offset_witness_rebuilds_into_the_band():
    shape = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 40, 24, 6).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(5, 7, 6), 10, 10, 4).Shape(),
    ).Shape()
    frame = CadFrame(1.0, (0.0, 0.0, 0.0))
    faces, _ = recognize_face_offsets(shape, frame=frame)
    top = next(f for f in faces if np.isclose(f.centroid[2], 10.0))
    bottom = next(f for f in faces if np.isclose(f.centroid[2], 0.0))
    edit = prepare_face_offset(shape, [top.face_id, bottom.face_id], frame=frame)
    before = mass(shape)
    band = (-0.02 * before, -0.015 * before)
    result = minimal_volume_change(face_offset_laws(edit), band)
    assert result.status == "optimal_within_tolerance"
    # The bottom face (area 960) removes volume more cheaply than the boss top.
    assert result.values[0] == 0.0
    change = mass(edit.rebuild(jnp.array(result.values))) - before
    assert band[0] - 1e-9 * before <= change <= band[1] + 1e-9 * before


def test_hole_radius_witness_rebuilds_into_the_band():
    jax.config.update("jax_enable_x64", True)
    shape = read_step(FIXTURES / "box_with_holes.step")
    edit = prepare_hole_edit(shape)
    before = mass(shape)
    band = (-0.004 * before, -0.003 * before)
    laws, limits = hole_ligament_constraints(edit)
    result = minimal_volume_change(laws, band, travel_limits=limits)
    assert result.status == "optimal_within_tolerance"
    radii = jnp.array(result.values)
    # Query and JAX model are the same law, and the witness is editable.
    assert bool(edit.model.is_valid(radii))
    assert float(edit.model.volume(radii)) - before == pytest.approx(
        float(result.volume_change[0]), rel=1e-9
    )
    change = mass(edit.rebuild(radii)) - before
    assert band[0] - 1e-9 * before <= change <= band[1] + 1e-9 * before


def named(law, label):
    return replace(law, label=label)


def limit_travel(change, limit):
    return max(change, 0.0) if limit.increases_only else abs(change)


def limited_brute_force(laws, band, limits, steps=400):
    best = None
    grids = [np.linspace(law.lower, law.upper, steps) for law in laws]
    for values in itertools.product(*grids):
        change = {
            law.label: x - law.reference for law, x in zip(laws, values, strict=True)
        }
        if any(
            limit_travel(change[t.first], t) + limit_travel(change[t.second], t)
            > t.limit
            for t in limits
        ):
            continue
        total = sum(
            float(contribution(law, x, Fraction(math.pi)))
            for law, x in zip(laws, values, strict=True)
        )
        if band[0] <= total <= band[1]:
            cost = sum(
                law.weight * abs(x - law.reference)
                for law, x in zip(laws, values, strict=True)
            )
            best = cost if best is None else min(best, cost)
    return best


@pytest.mark.parametrize(
    "limit, band, weights",
    [
        (0.25, (-6.0, -5.0), (1.0, 1.0)),
        (0.15, (-4.5, -4.0), (1.0, 3.0)),
        (0.4, (-3.0, -2.5), (2.0, 1.0)),
    ],
)
def test_travel_limits_hold_and_lower_bound_stays_valid(limit, band, weights):
    laws = (
        named(hole_law(1.5, 4.0, weight=weights[0]), "a"),
        named(hole_law(1.0, 6.0, weight=weights[1]), "b"),
    )
    limits = (TravelLimit("a", "b", limit),)
    result = minimal_volume_change(laws, band, travel_limits=limits, max_nodes=50_000)
    assert result.status == "optimal_within_tolerance"
    travels = [
        abs(Fraction(x) - Fraction(law.reference))
        for law, x in zip(laws, result.values, strict=True)
    ]
    assert sum(travels) <= Fraction(limit)
    assert band[0] <= result.volume_change[0] <= result.volume_change[1] <= band[1]
    found = limited_brute_force(laws, band, limits)
    assert found is not None
    assert float(result.lower_bound) <= found + 1e-9
    assert float(result.change) <= found + 1e-9


@pytest.mark.parametrize(
    "limit, band, weights",
    [
        (0.25, (-6.0, -5.0), (1.0, 1.0)),
        (0.15, (-4.5, -4.0), (1.0, 3.0)),
        (0.2, (-1.0, 1.0), (1.0, 1.0)),
        # Reaching this band takes more decrease than the limit would allow.
        (0.1, (4.0, 5.0), (1.0, 2.0)),
    ],
)
def test_travel_limits_on_increases_leave_decreases_free(limit, band, weights):
    laws = (
        named(hole_law(1.5, 4.0, weight=weights[0]), "a"),
        named(hole_law(1.0, 6.0, weight=weights[1]), "b"),
    )
    limits = (TravelLimit("a", "b", limit, increases_only=True),)
    result = minimal_volume_change(laws, band, travel_limits=limits, max_nodes=50_000)
    assert result.status == "optimal_within_tolerance"
    growth = [
        max(Fraction(x) - Fraction(law.reference), Fraction(0))
        for law, x in zip(laws, result.values, strict=True)
    ]
    assert sum(growth) <= Fraction(limit)
    assert band[0] <= result.volume_change[0] <= result.volume_change[1] <= band[1]
    found = limited_brute_force(laws, band, limits)
    assert found is not None
    assert float(result.lower_bound) <= found + 1e-9
    assert float(result.change) <= found + 1e-9


def test_a_binding_travel_limit_raises_the_cost():
    laws = (named(hole_law(1.5, 4.0), "a"), named(hole_law(1.0, 6.0), "b"))
    band = (-4.0, -3.5)
    free = minimal_volume_change(laws, band)
    travel = [abs(x - law.reference) for law, x in zip(laws, free.values, strict=True)]
    limit = TravelLimit("a", "b", sum(travel) * 0.9)
    limited = minimal_volume_change(laws, band, travel_limits=(limit,))
    # Travel is the cost here, so a tighter limit on it cannot reach the band.
    assert limited.status == "infeasible"


def test_infeasible_when_a_travel_limit_excludes_the_band():
    laws = (named(hole_law(1.5, 4.0), "a"), named(hole_law(1.0, 6.0), "b"))
    result = minimal_volume_change(
        laws, (-8.0, -7.0), travel_limits=(TravelLimit("a", "b", 0.01),)
    )
    assert result.status == "infeasible"
    assert result.values is None


@pytest.mark.parametrize(
    "limit",
    [
        TravelLimit("a", "missing", 1.0),
        TravelLimit("a", "a", 1.0),
        TravelLimit("a", "b", -1.0),
        TravelLimit("a", "b", math.nan),
    ],
)
def test_unsupported_travel_limits_are_refused(limit):
    laws = (named(hole_law(1.5, 4.0), "a"), named(hole_law(1.0, 6.0), "b"))
    with pytest.raises(ValueError, match="travel limit"):
        minimal_volume_change(laws, (-1.0, 0.0), travel_limits=(limit,))


def test_a_travel_limit_cannot_name_an_ambiguous_label():
    laws = (hole_law(1.5, 4.0), hole_law(1.0, 6.0), named(hole_law(1.0, 6.0), "b"))
    with pytest.raises(ValueError, match="travel limit"):
        minimal_volume_change(
            laws, (-1.0, 0.0), travel_limits=(TravelLimit("hole", "b", 1.0),)
        )


def ligament_plate():
    shape = BRepPrimAPI_MakeBox(40, 24, 6).Shape()
    for base, radius in (((10, 12, 0), 2), ((22, 12, 0), 3)):
        tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(*base), gp_Dir(0, 0, 1)), radius, 6
        )
        shape = BRepAlgoAPI_Cut(shape, tool.Shape()).Shape()
    return shape


def test_ligament_constraints_change_the_answer_and_the_rebuild_keeps_them():
    edit = prepare_hole_edit(ligament_plate(), frame=CadFrame(1.0, (0.0, 0.0, 0.0)))
    band = (-40.0, -35.0)
    # Growing the small hole is cheap but travels further than the large one.
    weights = [1.0 if hole.radius == 2 else 5.0 for hole in edit.holes]
    laws, limits = hole_ligament_constraints(edit, weights=weights)
    free = minimal_volume_change(laws, band, travel_limits=limits)
    assert free.status == "optimal_within_tolerance"
    assert float(jnp.min(edit.model.ligament_bounds(jnp.asarray(free.values)))) < 6.65
    minimum = 6.65
    laws, limits = hole_ligament_constraints(edit, minimum, weights)
    assert limits
    result = minimal_volume_change(laws, band, travel_limits=limits, max_nodes=50_000)
    assert result.status == "optimal_within_tolerance"
    radii = jnp.asarray(result.values)
    assert result.change > free.change
    # Exact arithmetic on the measured distance, which is what is promised.
    travel = sum(
        abs(Fraction(x) - Fraction(law.reference))
        for x, law in zip(result.values, laws, strict=True)
    )
    between = float(edit.model.hole_clearance[0, 1])
    assert travel <= Fraction(between) - Fraction(minimum)
    assert band[0] - 1e-9 <= cad_volume_change(edit, radii) <= band[1] + 1e-9


def test_default_constraints_reach_past_half_the_distance_for_each_hole():
    edit = prepare_hole_edit(ligament_plate(), frame=CadFrame(1.0, (0.0, 0.0, 0.0)))
    model = edit.model
    # What both holes reach at half their distance, the most that equal shares
    # of it could allow.
    half = (model.reference_radii + 3.5) ** 2 - model.reference_radii**2
    band_edge = -float(jnp.sum(jnp.pi * model.heights * half))
    laws, limits = hole_ligament_constraints(edit)
    assert limits
    band = (2 * band_edge, 1.1 * band_edge)
    result = minimal_volume_change(laws, band, travel_limits=limits)
    assert result.status == "optimal_within_tolerance"
    radii = jnp.asarray(result.values)
    assert bool(model.is_valid(radii))
    assert float(jnp.min(model.ligament_bounds(radii))) >= float(model.contact_margin)
    assert band[0] - 1e-9 <= cad_volume_change(edit, radii) <= band[1] + 1e-9


def test_shrinking_is_free_unless_a_minimum_is_asked_for():
    # 1.5 apart, with radii that could shrink by 5 each.
    edit = plate_with(((20, 10), (41.5, 10)))
    gain = float(jnp.pi * edit.model.heights[0] * (100 - 49))
    band = (gain, 1.2 * gain)
    laws, limits = hole_ligament_constraints(edit)
    assert limits
    result = minimal_volume_change(laws, band, travel_limits=limits)
    assert result.status == "optimal_within_tolerance"
    radii = jnp.asarray(result.values)
    assert bool(edit.model.is_valid(radii))
    assert band[0] - 1e-9 <= cad_volume_change(edit, radii) <= band[1] + 1e-9
    # The ligament bound counts shrinking, so asking for one narrows this.
    laws, limits = hole_ligament_constraints(edit, 0.0)
    held = minimal_volume_change(laws, band, travel_limits=limits)
    assert held.status == "infeasible"


def test_a_minimum_the_part_already_misses_is_refused():
    edit = prepare_hole_edit(ligament_plate(), frame=CadFrame(1.0, (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="below the minimum"):
        hole_ligament_constraints(edit, 7.5)
    with pytest.raises(ValueError, match="not be negative"):
        hole_ligament_constraints(edit, -1.0)


def test_ligament_constraints_narrow_intervals_for_fixed_faces():
    shape = ligament_plate()
    frame = CadFrame(1.0, (0.0, 0.0, 0.0))
    first = prepare_hole_edit(shape, frame=frame).holes[0]
    edit = prepare_hole_edit(shape, [first.face_id], frame=frame)
    laws, limits = hole_ligament_constraints(edit, 6.9)
    assert limits == ()
    # 7 away from the unselected hole: 0.1 of travel remains either way.
    assert laws[0].upper == pytest.approx(first.radius + 0.1)
    assert laws[0].lower == pytest.approx(first.radius - 0.1)


@pytest.mark.slow
def test_an_imported_part_reaches_a_band_beyond_a_fifth_of_its_clearance():
    path = FIXTURES / "nist" / "nist_ftc_10_asme1_rb.stp"
    if not path.exists():
        pytest.skip("NIST fixture not available")
    edit = prepare_hole_edit(read_step(path), merge_same_domain=True)
    model = edit.model
    before = float(model.reference_volume * model.scale**3)
    band = (-0.005 * before, -0.003 * before)
    laws, limits = hole_ligament_constraints(edit)
    result = minimal_volume_change(laws, band, travel_limits=limits)
    assert result.status == "optimal_within_tolerance"
    radii = jnp.asarray(result.values)
    assert bool(model.is_valid(radii))
    clearance = jnp.array([hole.clearance for hole in edit.holes])
    assert float(jnp.max((radii - model.reference_radii) / clearance)) > 0.2
    rebuilt = edit.rebuild(radii)
    assert mass(rebuilt) - before == pytest.approx(
        float(result.volume_change[0]), rel=1e-5
    )


def cad_volume_change(edit, radii):
    def measured(shape):
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props, 1e-9)
        return props.Mass()

    reference = edit.frame.restore_shape(edit.working_shape)
    return measured(edit.rebuild(radii)) - measured(reference)


def test_a_zero_slope_at_the_far_end_keeps_the_lower_bound_valid():
    # Cost against volume is convex here and its slope diverges at x = 0, where
    # the law is flat; a chord would lie above the curve.
    laws = (
        DimensionLaw("a", 1.0, 0.0, 1.0, 1.0, (ZERO, ZERO, Fraction(1)), (ZERO,) * 3),
        DimensionLaw("b", 0.0, 0.0, 1.0, 0.8, (ZERO, Fraction(-1), ZERO), (ZERO,) * 3),
    )
    band = (-0.76, -0.75)
    result = minimal_volume_change(laws, band, max_nodes=50_000)
    # Moving a to 0.5 reaches -0.75 at cost 0.5.
    assert float(result.lower_bound) <= 0.5
    assert result.status == "optimal_within_tolerance"
    assert float(result.change) <= 0.5 + 1e-9


def plate_with(holes, size=(80, 50, 6)):
    shape = BRepPrimAPI_MakeBox(*size).Shape()
    for x, radius in holes:
        tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(x, size[1] / 2, 0), gp_Dir(0, 0, 1)), radius, size[2]
        )
        shape = BRepAlgoAPI_Cut(shape, tool.Shape()).Shape()
    frame = CadFrame(1.0, (0.0, 0.0, 0.0))
    first = prepare_hole_edit(shape, frame=frame)
    ids = [hole.face_id for hole in sorted(first.holes, key=lambda h: h.base[0])]
    return prepare_hole_edit(shape, ids, frame=frame)


@pytest.mark.parametrize(
    "holes, minimum, values",
    [
        # 7 - 0.1 rounds up as a float, so a limit computed in floats exceeds
        # the exact room.
        (((20, 10), (47, 10)), 0.1, (5.0, 8.1)),
        # The largest travels only just exceed the room, and rounding their sum
        # can hide that.
        (((20, 2), (25.3, 1.3)), 0.3500000000000001, (1.0, 0.65)),
    ],
)
def test_ligament_constraints_exclude_values_below_the_exact_minimum(
    holes, minimum, values
):
    edit = plate_with(holes)
    laws, limits = hole_ligament_constraints(edit, minimum)
    between = np.asarray(edit.model.hole_clearance)
    travel = [
        abs(Fraction(x) - Fraction(law.reference))
        for law, x in zip(laws, values, strict=True)
    ]
    pair = min(Fraction(float(between[0, 1])), Fraction(float(between[1, 0])))
    allowed = all(
        law.lower <= x <= law.upper for law, x in zip(laws, values, strict=True)
    ) and all(sum(travel) <= Fraction(limit.limit) for limit in limits)
    # These values travel past the exact room, so nothing may admit them.
    assert sum(travel) > pair - Fraction(minimum)
    assert not allowed


def test_a_minimum_below_the_contact_margin_still_keeps_the_margin():
    edit = plate_with(((20, 2), (30, 2)))
    margin = float(edit.model.contact_margin)
    _, limits = hole_ligament_constraints(edit, 0.0)
    assert limits[0].limit == pytest.approx(6 - margin)


def test_ligament_constraints_keep_an_exactly_feasible_endpoint():
    edit = plate_with(((10, 2),), (40, 24, 6))
    laws, limits = hole_ligament_constraints(edit, 7.0)
    # The clearance is exactly 8, so a radius of exactly 3 leaves exactly 7.
    assert laws[0].upper == 3.0
    edge = -30 * math.pi
    result = minimal_volume_change(
        laws, (math.nextafter(edge, -math.inf), edge), travel_limits=limits
    )
    assert result.status != "infeasible"
