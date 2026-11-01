"""Design search against independent interval products and section integrals."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from brepax.experimental.polyhedral_boundary import (
    PreparedPolyhedralEventBoundary,
    prepare_polyhedral_event_boundary,
    specialize_polyhedral_event_boundary,
)
from brepax.experimental.polyhedral_design import polyhedral_design_target
from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell
from brepax.experimental.polyhedral_targets import polyhedral_translation_targets

_BOUNDS = (Fraction(-3, 8), Fraction(3, 8))
_PARAMETER_TOLERANCE = Fraction(1, 2**12)
_STEP_TOLERANCE = Fraction(1, 2**30)
_VOLUME_TOLERANCE = Fraction(1, 2**20)


def _box(lower, upper):
    return np.array(list(product(*zip(lower, upper, strict=True))), dtype=float)


def _prepared_box(
    slope: Fraction,
    *,
    scale: Fraction = Fraction(1),
    transformed: bool = False,
) -> PreparedPolyhedralEventBoundary:
    fixed = _box((0, 0, 0), (1, 1, 1))
    moving = _box((-1.25, 0.25, -0.25), (-0.25, 0.75, 1.25))
    displacement = np.zeros((1, 8, 3))
    displacement[0, moving[:, 0] == -1.25, 0] = -1
    displacement[0, moving[:, 1] == 0.25, 1] = float(slope) / 2
    displacement[0, moving[:, 1] == 0.75, 1] = -float(slope) / 2
    rotation = (
        np.array(((0, -1, 0), (0, 0, 1), (-1, 0, 0))) if transformed else np.eye(3)
    )
    offset = np.array((8, -4, 16)) if transformed else np.zeros(3)
    return prepare_polyhedral_event_boundary(
        [
            AffinePolyhedralCell(
                float(scale) * fixed @ rotation.T + offset, np.zeros((1, 8, 3))
            )
        ],
        [
            AffinePolyhedralCell(
                float(scale) * moving @ rotation.T + offset,
                float(scale) * displacement @ rotation.T,
            )
        ],
        float(scale) * rotation @ np.array((1, 0, 0)),
        max_step=3,
        parameter_radius=Fraction(1, 8),
    )


@pytest.fixture(scope="module")
def smooth_boundary():
    return _prepared_box(Fraction(1))


def _box_volume(step, parameter, slope=Fraction(1)):
    overlap = max(
        Fraction(0),
        min(Fraction(1), step - Fraction(1, 4))
        - max(Fraction(0), step - Fraction(5, 4) - parameter),
    )
    return overlap * (Fraction(1, 2) - slope * parameter)


def _assert_box_witness(
    result, target, initial_parameter, volume_tolerance, *, slope=Fraction(1), scale=1
):
    witness = result.witness
    assert witness is not None
    assert isinstance(witness.parameter, float) and isinstance(witness.step, float)
    parameter, step = Fraction(witness.parameter), Fraction(witness.step)
    assert _BOUNDS[0] <= parameter <= _BOUNDS[1]
    assert 0 <= step <= 3
    expected = scale**3 * _box_volume(step, parameter, slope)
    assert isinstance(witness.volume, Fraction) and witness.volume == expected
    assert witness.residual == expected - target
    assert abs(witness.residual) <= volume_tolerance
    assert witness.objective == abs(parameter - Fraction(initial_parameter))
    assert result.objective_upper == witness.objective


def _assert_radical_objective_bounds(result, shift, square):
    # The analytic continuous optimum is shift-sqrt(square); squaring keeps
    # both enclosure checks exact without rounding an irrational reference.
    lower, upper = result.objective_lower, result.objective_upper
    assert lower is not None and upper is not None
    assert 0 <= lower <= upper
    assert lower <= shift and (shift - lower) ** 2 >= square
    assert upper >= shift or (shift - upper) ** 2 <= square
    assert result.optimality_gap == upper - lower
    assert result.optimality_gap <= _PARAMETER_TOLERANCE


def test_search_crosses_infeasible_boundary_to_nearest_feasible_island(
    smooth_boundary,
):
    target = Fraction(35, 64)
    result = polyhedral_design_target(
        smooth_boundary,
        target,
        parameter_bounds=_BOUNDS,
        initial_parameter=Fraction(1, 8),
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=_VOLUME_TOLERANCE,
    )
    assert result.status == "optimal_within_tolerance"
    _assert_box_witness(result, target, Fraction(1, 8), _VOLUME_TOLERANCE)
    assert result.witness.parameter < 0
    # M(theta)=9/16-(theta+1/4)^2 on the left chart. Even theta=0
    # misses this target band, so stopping at the exchange is insufficient.
    assert Fraction(1, 2) < target - _VOLUME_TOLERANCE
    _assert_radical_objective_bounds(
        result, Fraction(3, 8), Fraction(1, 64) + _VOLUME_TOLERANCE
    )


def _prepared_tent():
    fixed = np.array(
        [(x, y, z) for x, y in ((0, 0), (1, 1), (3, 0)) for z in (0, 1)],
        dtype=float,
    )
    moving = _box((-3.25, -1, -1), (-0.25, 2, 2))
    displacement = np.zeros((1, 8, 3))
    displacement[0, moving[:, 0] == -3.25, 0] = -1
    return prepare_polyhedral_event_boundary(
        [AffinePolyhedralCell(fixed, np.zeros((1, 6, 3)))],
        [AffinePolyhedralCell(moving, displacement)],
        (1, 0, 0),
        max_step=7,
        parameter_radius=Fraction(1, 8),
    )


def _tent_volume(step, parameter):
    def section_integral(x):
        if x <= 0:
            return Fraction(0)
        if x <= 1:
            return x**2 / 2
        if x <= 3:
            return Fraction(3, 2) - (3 - x) ** 2 / 4
        return Fraction(3, 2)

    return section_integral(step - Fraction(1, 4)) - section_integral(
        step - Fraction(13, 4) - parameter
    )


def test_interior_translation_maximum_controls_the_optimal_design():
    prepared = _prepared_tent()
    target = Fraction(143, 96)
    allowance = Fraction(1, 2**22)
    result = polyhedral_design_target(
        prepared,
        target,
        parameter_bounds=(Fraction(-1, 2), Fraction(1, 2)),
        initial_parameter=Fraction(-1, 2),
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=allowance,
    )
    assert result.status == "optimal_within_tolerance"
    witness = result.witness
    assert witness is not None
    theta, step = Fraction(witness.parameter), Fraction(witness.step)
    assert Fraction(-1, 2) <= theta <= Fraction(1, 2) and 0 <= step <= 7

    volume = _tent_volume(step, theta)
    assert witness.volume == volume
    assert witness.residual == volume - target
    assert abs(witness.residual) <= allowance
    assert witness.objective == abs(theta + Fraction(1, 2))
    # Maximizing the two triangular missing caps gives M=3/2-theta^2/6.
    # Its step=13/4+2theta/3 is not an event when theta<0.
    _assert_radical_objective_bounds(
        result, Fraction(1, 2), Fraction(1, 16) + 6 * allowance
    )


def test_unresolved_root_query_does_not_hide_an_acceptable_band_action():
    prepared = _prepared_tent()
    target, allowance, initial = Fraction(7, 5), Fraction(1, 10), Fraction(-1, 4)
    roots = polyhedral_translation_targets(
        specialize_polyhedral_event_boundary(prepared, initial),
        target,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=allowance,
        max_refinements=1,
    )
    assert roots.status == "unresolved" and roots.exact_reachable
    result = polyhedral_design_target(
        prepared,
        target,
        parameter_bounds=(Fraction(-1, 2), Fraction(1, 2)),
        initial_parameter=initial,
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=allowance,
        max_boxes=1,
        max_refinements=1,
    )
    assert result.status == "optimal_within_tolerance"
    assert result.witness is not None and result.witness.parameter == float(initial)
    step = Fraction(result.witness.step)
    assert 0 <= step <= 7
    volume = _tent_volume(step, Fraction(result.witness.parameter))
    assert (
        result.witness.volume == volume and result.witness.residual == volume - target
    )
    assert abs(result.witness.residual) <= allowance
    assert result.objective_lower == result.objective_upper == 0


def test_global_volume_bound_proves_unreachable_target(smooth_boundary):
    result = polyhedral_design_target(
        smooth_boundary,
        Fraction(37, 64),
        parameter_bounds=_BOUNDS,
        initial_parameter=Fraction(1, 8),
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=_VOLUME_TOLERANCE,
    )
    # The full family reaches at most 9/16, not the stock's full unit volume.
    assert result.status == "infeasible"
    assert result.witness is None
    assert result.remaining_intervals == ()
    assert result.objective_upper is None and result.optimality_gap is None


@pytest.mark.parametrize(
    "target", [Fraction(1, 4), Fraction(1, 2) + Fraction(1, 2**21)]
)
def test_initial_design_already_satisfies_the_volume_band(smooth_boundary, target):
    result = polyhedral_design_target(
        smooth_boundary,
        target,
        parameter_bounds=_BOUNDS,
        initial_parameter=0,
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=_VOLUME_TOLERANCE,
        max_boxes=1,
    )
    assert result.status == "optimal_within_tolerance"
    _assert_box_witness(result, target, 0, _VOLUME_TOLERANCE)
    assert result.witness.parameter == 0
    assert (
        result.objective_lower == result.objective_upper == result.optimality_gap == 0
    )


def test_initial_zero_volume_is_usable_when_target_band_contains_zero(smooth_boundary):
    result = polyhedral_design_target(
        smooth_boundary,
        _VOLUME_TOLERANCE,
        parameter_bounds=_BOUNDS,
        initial_parameter=Fraction(1, 8),
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=_VOLUME_TOLERANCE,
        max_boxes=1,
    )
    assert result.status == "optimal_within_tolerance"
    _assert_box_witness(result, _VOLUME_TOLERANCE, Fraction(1, 8), _VOLUME_TOLERANCE)
    assert result.witness.volume == 0
    assert result.objective_lower == result.objective_upper == 0


def test_exact_boundary_singleton_has_a_representable_action():
    prepared = _prepared_box(Fraction(1, 4))
    result = polyhedral_design_target(
        prepared,
        Fraction(1, 2),
        parameter_bounds=_BOUNDS,
        initial_parameter=Fraction(1, 8),
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=0,
    )
    assert result.status == "optimal_within_tolerance"
    _assert_box_witness(result, Fraction(1, 2), Fraction(1, 8), 0, slope=Fraction(1, 4))
    assert result.witness.parameter == 0 and result.witness.step == 1.25
    assert result.objective_lower <= Fraction(1, 8) == result.objective_upper
    assert result.optimality_gap <= _PARAMETER_TOLERANCE


def test_nonbinary_singleton_is_not_falsely_reported_infeasible():
    prepared = _prepared_box(Fraction(3, 4))
    result = polyhedral_design_target(
        prepared,
        Fraction(25, 48),
        parameter_bounds=_BOUNDS,
        initial_parameter=0,
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=0,
        max_boxes=8,
    )
    # M=25/48-3(theta+1/6)^2/4. The only real feasible theta is -1/6,
    # and no binary64 design can attain it with zero volume allowance.
    assert result.status == "unresolved"
    assert result.witness is None and result.objective_upper is None
    assert any(lo <= Fraction(-1, 6) <= hi for lo, hi in result.remaining_intervals)


def test_positive_allowance_accepts_exactly_measured_float_action():
    prepared = _prepared_box(Fraction(3, 4))
    initial = float(Fraction(-1, 6))
    target, allowance = Fraction(25, 48), Fraction(1, 10**30)
    result = polyhedral_design_target(
        prepared,
        target,
        parameter_bounds=_BOUNDS,
        initial_parameter=initial,
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=allowance,
        max_boxes=8,
    )
    assert result.status == "optimal_within_tolerance"
    _assert_box_witness(result, target, initial, allowance, slope=Fraction(3, 4))
    deficit = Fraction(3, 4) * (Fraction(initial) + Fraction(1, 6)) ** 2
    assert 0 < deficit <= allowance
    assert result.witness.residual == -deficit
    assert result.objective_lower == result.objective_upper == 0


def test_rational_search_window_without_binary64_parameter_stays_unresolved(
    smooth_boundary,
):
    left = Fraction(0.125)
    right = Fraction(float(np.nextafter(0.125, np.inf)))
    bounds = ((2 * left + right) / 3, (left + 2 * right) / 3)
    result = polyhedral_design_target(
        smooth_boundary,
        0,
        parameter_bounds=bounds,
        initial_parameter=sum(bounds) / 2,
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=0,
        max_boxes=8,
    )
    # Every real parameter is feasible at step zero, but neither adjacent
    # binary64 value belongs to the requested closed window.
    assert result.status == "unresolved"
    assert result.witness is None and result.objective_upper is None
    assert result.objective_lower == 0
    assert result.remaining_intervals
    assert all(
        bounds[0] <= lo <= hi <= bounds[1] for lo, hi in result.remaining_intervals
    )


def test_exhausted_search_budget_retains_uncertainty(smooth_boundary):
    result = polyhedral_design_target(
        smooth_boundary,
        Fraction(35, 64),
        parameter_bounds=_BOUNDS,
        initial_parameter=Fraction(1, 8),
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=_VOLUME_TOLERANCE,
        max_boxes=1,
    )
    assert result.status == "unresolved"
    assert result.remaining_intervals
    assert 0 <= result.boxes_examined <= 1 and result.point_queries >= 0
    if result.witness is not None:
        _assert_box_witness(result, Fraction(35, 64), Fraction(1, 8), _VOLUME_TOLERANCE)
        assert result.optimality_gap > _PARAMETER_TOLERANCE


@pytest.mark.parametrize("scale", [Fraction(1, 8), Fraction(8)])
def test_exact_frame_and_unit_changes_preserve_design_objective(scale):
    prepared = _prepared_box(Fraction(1), scale=scale, transformed=True)
    target = scale**3 * Fraction(35, 64)
    allowance = scale**3 * _VOLUME_TOLERANCE
    result = polyhedral_design_target(
        prepared,
        target,
        parameter_bounds=_BOUNDS,
        initial_parameter=Fraction(1, 8),
        parameter_tolerance=_PARAMETER_TOLERANCE,
        step_tolerance=_STEP_TOLERANCE,
        volume_tolerance=allowance,
    )
    assert result.status == "optimal_within_tolerance"
    _assert_box_witness(result, target, Fraction(1, 8), allowance, scale=scale)
    _assert_radical_objective_bounds(
        result, Fraction(3, 8), Fraction(1, 64) + _VOLUME_TOLERANCE
    )


@pytest.mark.parametrize(
    "override",
    [
        {"target": -1},
        {"target": np.nan},
        {"parameter_bounds": (Fraction(1, 4), Fraction(-1, 4))},
        {"parameter_bounds": (Fraction(-1, 2), Fraction(1, 2))},
        {"initial_parameter": np.inf},
        {"parameter_tolerance": 0},
        {"step_tolerance": 0},
        {"volume_tolerance": -1},
        {"max_boxes": 0},
        {"max_refinements": 0},
    ],
)
def test_invalid_search_requests_are_rejected(smooth_boundary, override):
    arguments = {
        "target": Fraction(35, 64),
        "parameter_bounds": _BOUNDS,
        "initial_parameter": 0,
        "parameter_tolerance": _PARAMETER_TOLERANCE,
        "step_tolerance": _STEP_TOLERANCE,
        "volume_tolerance": _VOLUME_TOLERANCE,
    }
    arguments.update(override)
    with pytest.raises(ValueError):
        polyhedral_design_target(smooth_boundary, **arguments)
