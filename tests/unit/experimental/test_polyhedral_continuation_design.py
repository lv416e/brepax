"""Global design guarantees with discovered boundaries and incomplete coverage."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from brepax.experimental.polyhedral_continuation import prepare_polyhedral_continuation
from brepax.experimental.polyhedral_design import polyhedral_design_target
from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell

_BOUNDS = (Fraction(-1, 16), Fraction(5, 8))
_TARGET = Fraction(25, 32)
_OPTIMUM = Fraction(319, 600)


def _box(lower, upper):
    return np.array(list(product(*zip(lower, upper, strict=True))), dtype=float)


def _stepped_cells():
    fixed = (
        _box((0, 0, 0), (0.75, 0.5, 1)),
        _box((0.75, 0, 0), (1.875, 1, 1)),
    )
    moving = _box((-0.75, 0, -0.25), (-0.25, 0.75, 1.25))
    displacement = np.zeros((1, 8, 3))
    displacement[0, moving[:, 0] == -0.75, 0] = -1.5
    displacement[0, moving[:, 1] == 0.75, 1] = -0.25
    return (
        tuple(AffinePolyhedralCell(v, np.zeros((1, 8, 3))) for v in fixed),
        (AffinePolyhedralCell(moving, displacement),),
    )


def _volume(parameter, step):
    width = Fraction(1, 2) + Fraction(3, 2) * parameter
    height = Fraction(3, 4) - parameter / 4
    right = step - Fraction(1, 4)

    def overlap(lower, upper):
        return max(Fraction(0), min(upper, right) - max(lower, right - width))

    return (
        overlap(Fraction(0), Fraction(3, 4)) / 2
        + overlap(Fraction(3, 4), Fraction(15, 8)) * height
    )


def _prepare(*, max_charts=8, seed=0, bounds=_BOUNDS):
    return prepare_polyhedral_continuation(
        *_stepped_cells(),
        (1, 0, 0),
        max_step=4,
        parameter_bounds=bounds,
        seed_parameter=seed,
        max_charts=max_charts,
    )


def _query(prepared, *, target=_TARGET, initial=0, max_boxes=256):
    return polyhedral_design_target(
        prepared,
        target,
        parameter_bounds=prepared.parameter_bounds,
        initial_parameter=initial,
        parameter_tolerance=Fraction(1, 1000),
        step_tolerance=Fraction(1, 10**10),
        volume_tolerance=target / 1000,
        max_boxes=max_boxes,
    )


@pytest.mark.parametrize(
    ("scale", "factor", "offset", "rotated"),
    [
        (Fraction(1), Fraction(1), Fraction(0), False),
        (Fraction(1, 8), Fraction(1), Fraction(0), True),
        (Fraction(8), Fraction(-2), Fraction(1, 8), True),
    ],
)
def test_search_discovers_two_boundaries_and_preserves_coordinate_laws(
    scale, factor, offset, rotated
):
    rotation = np.array(((0, -1, 0), (0, 0, 1), (-1, 0, 0))) if rotated else np.eye(3)
    translation = np.array((8, -4, 16)) if rotated else np.zeros(3)
    cells = tuple(
        tuple(
            AffinePolyhedralCell(
                float(scale)
                * (
                    np.asarray(cell.vertices)
                    - float(offset / factor)
                    * np.asarray(cell.parameter_displacements)[0]
                )
                @ rotation.T
                + translation,
                float(scale / factor)
                * np.asarray(cell.parameter_displacements)
                @ rotation.T,
            )
            for cell in body
        )
        for body in _stepped_cells()
    )
    bounds = tuple(sorted(factor * t + offset for t in _BOUNDS))
    prepared = prepare_polyhedral_continuation(
        *cells,
        float(scale) * rotation @ np.array((1, 0, 0)),
        max_step=4,
        parameter_bounds=bounds,
        seed_parameter=offset,
    )
    assert set(prepared.boundary_parameters) == {
        factor * Fraction(1, 6) + offset,
        factor * Fraction(5, 12) + offset,
    }
    assert not prepared.uncovered
    target, allowance = scale**3 * _TARGET, scale**3 * _TARGET / 1000
    result = polyhedral_design_target(
        prepared,
        target,
        parameter_bounds=bounds,
        initial_parameter=offset,
        parameter_tolerance=abs(factor) / 1000,
        step_tolerance=Fraction(1, 10**10),
        volume_tolerance=allowance,
    )
    assert result.status == "optimal_within_tolerance"
    witness = result.witness
    assert witness is not None
    parameter = (Fraction(witness.parameter) - offset) / factor
    step = Fraction(witness.step)
    assert parameter > Fraction(5, 12)
    assert witness.volume == scale**3 * _volume(parameter, step)
    assert witness.residual == witness.volume - target
    assert abs(witness.residual) <= allowance
    assert witness.objective == abs(Fraction(witness.parameter) - offset)
    assert result.objective_lower <= abs(factor) * _OPTIMUM <= result.objective_upper
    assert 0 <= result.optimality_gap <= abs(factor) / 1000


@pytest.mark.parametrize(
    ("cap", "frontier"), [(1, Fraction(1, 6)), (2, Fraction(5, 12))]
)
def test_chart_budget_retains_global_unknown_region(cap, frontier):
    prepared = _prepare(max_charts=cap)
    result = _query(prepared)
    assert prepared.uncovered
    assert result.status == "unresolved"
    assert result.witness is None
    assert result.objective_lower == frontier
    assert any(lo <= _OPTIMUM <= hi for lo, hi in result.remaining_intervals)


def test_missing_endpoint_singleton_cannot_be_reported_infeasible():
    endpoint = Fraction(1, 6)
    prepared = _prepare(bounds=(_BOUNDS[0], endpoint))
    assert len(prepared.uncovered) == 1
    assert prepared.uncovered[0].lower == prepared.uncovered[0].upper == endpoint
    result = _query(prepared, target=Fraction(1))
    assert result.status == "unresolved"
    assert result.witness is None
    assert result.remaining_intervals == ((endpoint, endpoint),)
    assert result.objective_lower == endpoint


def test_distant_unknown_region_does_not_prevent_zero_gap_certificate():
    prepared = _prepare(max_charts=1)
    assert prepared.uncovered
    result = _query(prepared, target=Fraction(1, 4))
    assert result.status == "optimal_within_tolerance"
    assert result.witness is not None and result.witness.parameter == 0
    assert (
        result.objective_lower == result.objective_upper == result.optimality_gap == 0
    )
    assert result.remaining_intervals


def test_failed_seed_does_not_make_zero_target_infeasible():
    prepared = _prepare(seed=Fraction(1, 6))
    assert prepared.chart_count == 0
    result = _query(prepared, target=Fraction(0))
    assert result.status == "unresolved"
    assert result.witness is None
    assert result.objective_lower == 0
    assert result.boxes_examined == result.point_queries == 0
    assert any(lo <= 0 <= hi for lo, hi in result.remaining_intervals)


def test_search_box_budget_is_shared_across_charts():
    prepared = _prepare()
    assert prepared.chart_count == 3
    result = _query(prepared, max_boxes=1)
    assert result.status == "unresolved"
    assert result.boxes_examined == 1
    assert any(lo <= _OPTIMUM <= hi for lo, hi in result.remaining_intervals)


def test_complete_coverage_can_exclude_the_entire_domain():
    prepared = _prepare()
    result = _query(prepared, target=Fraction(1))
    assert not prepared.uncovered
    assert result.status == "infeasible"
    assert result.witness is None
    assert result.remaining_intervals == ()


def test_subquery_uses_its_own_closed_domain():
    prepared = _prepare(max_charts=1)
    assert prepared.uncovered
    arguments = {
        "initial_parameter": 0,
        "parameter_tolerance": Fraction(1, 1000),
        "step_tolerance": Fraction(1, 10**10),
        "volume_tolerance": 0,
    }
    result = polyhedral_design_target(
        prepared,
        1,
        parameter_bounds=(_BOUNDS[0], Fraction(1, 8)),
        **arguments,
    )
    assert result.status == "infeasible" and not result.remaining_intervals
    with pytest.raises(ValueError, match="inside the continuation domain"):
        polyhedral_design_target(
            prepared, 1, parameter_bounds=(_BOUNDS[0], Fraction(1)), **arguments
        )
