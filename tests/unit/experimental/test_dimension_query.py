"""Certified minimal volume changes against brute force and rebuilt CAD."""

import itertools
import math
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps

from brepax.experimental.applications.cad_frame import CadFrame
from brepax.experimental.applications.dimension_query import (
    DimensionLaw,
    face_offset_laws,
    hole_radius_laws,
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
    result = minimal_volume_change(hole_radius_laws(edit), band)
    assert result.status == "optimal_within_tolerance"
    change = mass(edit.rebuild(jnp.array(result.values))) - before
    assert band[0] - 1e-9 * before <= change <= band[1] + 1e-9 * before
