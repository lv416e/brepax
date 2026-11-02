"""Automatic coverage checked against interval products and section integrals."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from brepax.experimental.polyhedral_continuation import (
    prepare_polyhedral_continuation,
    specialize_polyhedral_continuation,
)
from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell
from brepax.experimental.polyhedral_targets import (
    polyhedral_translation_range,
    polyhedral_translation_targets,
)
from brepax.experimental.polyhedral_translation import (
    polyhedral_translation_volume,
    prepare_polyhedral_translation,
)


def _box(lower, upper):
    return np.array(list(product(*zip(lower, upper, strict=True))), dtype=float)


def _family(kind):
    if kind == "growing_box":
        fixed = _box((0, 0, 0), (1, 1, 1))
        moving = _box((-0.75, 0.375, -0.25), (-0.25, 0.625, 1.25))
        displacement = np.zeros((1, 8, 3))
        displacement[0, moving[:, 0] == -0.75, 0] = -3
        displacement[0, moving[:, 1] == 0.375, 1] = -1 / 8
        displacement[0, moving[:, 1] == 0.625, 1] = 1 / 8
        maximum = 4
    elif kind == "transverse_box":
        fixed = _box((0, 0, 0), (1, 1, 1))
        moving = _box((-0.75, 0.25, -0.25), (-0.25, 0.75, 1.25))
        displacement = np.zeros((1, 8, 3))
        displacement[0, :, 1] = 1
        maximum = 3
    else:
        length = 3 if kind == "asymmetric_tent" else 2
        fixed = np.array(
            [(x, y, z) for x, y in ((0, 0), (1, 1), (length, 0)) for z in (0, 1)],
            dtype=float,
        )
        moving = _box((-0.75, -1, -1), (-0.25, 2, 2))
        displacement = np.zeros((1, 8, 3))
        displacement[0, moving[:, 0] == -0.75, 0] = -3
        maximum = 7 if length == 3 else 5
    return (
        [AffinePolyhedralCell(fixed, np.zeros((1, len(fixed), 3)))],
        [AffinePolyhedralCell(moving, displacement)],
        maximum,
    )


def _volume(kind, step, theta):
    width = Fraction(1, 2) if kind == "transverse_box" else Fraction(1, 2) + 3 * theta
    right, left = step - Fraction(1, 4), step - Fraction(1, 4) - width
    if kind in ("growing_box", "transverse_box"):
        x = max(Fraction(0), min(Fraction(1), right) - max(Fraction(0), left))
        height = (
            Fraction(1, 4) + theta / 4
            if kind == "growing_box"
            else max(
                Fraction(0),
                min(Fraction(1), Fraction(3, 4) + theta)
                - max(Fraction(0), Fraction(1, 4) + theta),
            )
        )
        return x * height
    length = Fraction(3 if kind == "asymmetric_tent" else 2)

    def primitive(x):
        if x <= 0:
            return Fraction(0)
        if x <= 1:
            return x**2 / 2
        if x <= length:
            return length / 2 - (length - x) ** 2 / (2 * (length - 1))
        return length / 2

    return primitive(right) - primitive(left)


def _assert_profile(prepared, kind, theta, maximum):
    path = specialize_polyhedral_continuation(prepared, theta)
    for step in (Fraction(maximum * i, 8) for i in range(9)):
        actual = polyhedral_translation_volume(path.profile, float(step))
        assert float(actual) == pytest.approx(
            float(_volume(kind, step, theta)), abs=2e-14
        )
    return path


@pytest.mark.parametrize(
    "kind,bounds,events,parameters,target,optimal_parameter,optimal_step",
    [
        (
            "growing_box",
            (Fraction(-1, 16), Fraction(5, 8)),
            (Fraction(1, 6),),
            (Fraction(0), Fraction(1, 6), Fraction(1, 2)),
            Fraction(3, 8),
            Fraction(1, 2),
            Fraction(5, 4),
        ),
        (
            "asymmetric_tent",
            (Fraction(-1, 16), Fraction(3, 4)),
            (Fraction(1, 6), Fraction(1, 2)),
            (
                Fraction(0),
                Fraction(1, 6),
                Fraction(1, 3),
                Fraction(1, 2),
                Fraction(2, 3),
            ),
            Fraction(35, 24),
            Fraction(2, 3),
            Fraction(35, 12),
        ),
    ],
)
def test_discovers_exchanges_and_reaches_designs_beyond_seed_chart(
    kind, bounds, events, parameters, target, optimal_parameter, optimal_step
):
    fixed, moving, maximum = _family(kind)
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=bounds,
        seed_parameter=0,
    )
    assert prepared.parameter_bounds == bounds and prepared.seed_parameter == 0
    assert prepared.boundary_parameters == events
    assert prepared.chart_count == len(events) + 1
    assert prepared.uncovered == ()
    for theta in (*bounds, *parameters):
        _assert_profile(prepared, kind, theta, maximum)
    # Both analytic maxima grow on the domain. These target designs require
    # crossing every discovered exchange; the original seed chart cannot suffice.
    assert _volume(kind, optimal_step, optimal_parameter) == target
    path = specialize_polyhedral_continuation(prepared, optimal_parameter)
    targets = polyhedral_translation_targets(
        path, target, step_tolerance=1e-10, volume_tolerance=1e-12
    )
    assert targets.exact_reachable and targets.status == "reachable"
    assert any(
        root.lower <= optimal_step <= root.upper for root in targets.roots
    ) or any(
        interval.lower <= optimal_step <= interval.upper
        for interval in targets.intervals
    )


def test_rational_seed_and_exchange_preserve_the_original_affine_family():
    fixed, moving, maximum = _family("growing_box")
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=(Fraction(-1, 16), Fraction(5, 8)),
        seed_parameter=Fraction(1, 10),
    )
    assert prepared.seed_parameter == Fraction(1, 10)
    assert prepared.boundary_parameters == (Fraction(1, 6),)
    assert prepared.uncovered == ()
    for theta, expected_maximum in (
        (Fraction(1, 10), Fraction(11, 50)),
        (Fraction(1, 6), Fraction(7, 24)),
    ):
        path = specialize_polyhedral_continuation(prepared, theta)
        value_range = polyhedral_translation_range(
            path, step_tolerance=1e-12, volume_tolerance=0
        )
        # Float volume comparisons would hide a wrongly rounded rebase: at
        # theta=1/6 the exact y endpoints are 17/48 and 31/48.
        assert value_range.maximum == (expected_maximum, expected_maximum)


def test_regular_seed_on_right_discovers_exchanges_in_reverse():
    fixed, moving, maximum = _family("asymmetric_tent")
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=(Fraction(-1, 16), Fraction(3, 4)),
        seed_parameter=Fraction(5, 8),
    )
    assert prepared.boundary_parameters == (Fraction(1, 6), Fraction(1, 2))
    assert prepared.chart_count == 3 and not prepared.uncovered
    for theta in (Fraction(0), Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)):
        _assert_profile(prepared, "asymmetric_tent", theta, maximum)


@pytest.mark.parametrize(
    "kind,bounds,frontier,reason,far_parameter,target,step",
    [
        (
            "transverse_box",
            (Fraction(-1, 8), Fraction(1, 2)),
            Fraction(1, 4),
            "non_event",
            Fraction(3, 8),
            Fraction(3, 16),
            Fraction(3, 4),
        ),
        (
            "simultaneous_tent",
            (Fraction(-1, 16), Fraction(1, 3)),
            Fraction(1, 6),
            "multiple",
            Fraction(1, 4),
            Fraction(55, 64),
            Fraction(15, 8),
        ),
    ],
)
def test_unsupported_frontier_retains_valid_far_side_geometry_as_uncovered(
    kind, bounds, frontier, reason, far_parameter, target, step
):
    fixed, moving, maximum = _family(kind)
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=bounds,
        seed_parameter=0,
    )
    assert prepared.boundary_parameters == () and prepared.chart_count == 1
    assert len(prepared.uncovered) == 1
    gap = prepared.uncovered[0]
    assert (gap.lower, gap.upper) == (frontier, bounds[1])
    assert gap.lower_closed and gap.upper_closed and reason in gap.reason
    _assert_profile(prepared, kind, Fraction(0), maximum)
    for theta in (frontier, far_parameter):
        with pytest.raises(ValueError, match=r"covered|uncovered"):
            specialize_polyhedral_continuation(prepared, theta)

    # The far-side inputs are valid dyadic coordinates; a complete fixed
    # rebuild independently confirms a feasible action in the uncovered region.
    bodies = [
        [
            np.asarray(cell.vertices)
            + float(far_parameter) * np.asarray(cell.parameter_displacements)[0]
            for cell in body
        ]
        for body in (fixed, moving)
    ]
    rebuilt = prepare_polyhedral_translation(
        *bodies, (1, 0, 0), max_step=float(maximum)
    )
    targets = polyhedral_translation_targets(
        rebuilt, target, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert targets.exact_reachable
    assert _volume(kind, step, far_parameter) == target
    assert float(polyhedral_translation_volume(rebuilt.profile, float(step))) == float(
        target
    )


@pytest.mark.parametrize(
    "max_charts,events,frontier",
    [(1, (), Fraction(1, 6)), (2, (Fraction(1, 6),), Fraction(1, 2))],
)
def test_chart_budget_preserves_unfinished_region(max_charts, events, frontier):
    fixed, moving, maximum = _family("asymmetric_tent")
    upper = Fraction(3, 4)
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=(Fraction(-1, 16), upper),
        seed_parameter=0,
        max_charts=max_charts,
    )
    assert prepared.chart_count == max_charts and prepared.boundary_parameters == events
    assert len(prepared.uncovered) == 1
    gap = prepared.uncovered[0]
    assert (gap.lower, gap.upper) == (frontier, upper)
    assert gap.lower_closed and gap.upper_closed and "budget" in gap.reason
    with pytest.raises(ValueError, match=r"covered|uncovered"):
        specialize_polyhedral_continuation(prepared, Fraction(2, 3))
    _assert_profile(prepared, "asymmetric_tent", Fraction(0), maximum)


def test_nonregular_event_seed_leaves_the_requested_domain_uncovered():
    fixed, moving, maximum = _family("growing_box")
    bounds = (Fraction(-1, 16), Fraction(5, 8))
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=bounds,
        seed_parameter=Fraction(1, 6),
    )
    assert prepared.chart_count == 0 and prepared.boundary_parameters == ()
    assert len(prepared.uncovered) == 1
    gap = prepared.uncovered[0]
    assert (gap.lower, gap.upper) == bounds
    assert gap.lower_closed and gap.upper_closed and "seed" in gap.reason
    assert prepared.attempts and any(attempt.message for attempt in prepared.attempts)
    with pytest.raises(ValueError, match=r"covered|uncovered"):
        specialize_polyhedral_continuation(prepared, Fraction(0))


@pytest.mark.parametrize(
    "bounds,seed",
    [
        ((Fraction(-1, 16), Fraction(1, 6)), Fraction(0)),
        ((Fraction(1, 6), Fraction(5, 8)), Fraction(1, 2)),
    ],
)
def test_event_at_window_endpoint_retains_single_uncovered_point(bounds, seed):
    fixed, moving, maximum = _family("growing_box")
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=bounds,
        seed_parameter=seed,
    )
    assert prepared.boundary_parameters == ()
    assert len(prepared.uncovered) == 1
    gap = prepared.uncovered[0]
    assert gap.lower == gap.upper == Fraction(1, 6)
    assert gap.lower_closed and gap.upper_closed and "endpoint" in gap.reason
    _assert_profile(prepared, "growing_box", seed, maximum)
    with pytest.raises(ValueError, match=r"covered|uncovered"):
        specialize_polyhedral_continuation(prepared, Fraction(1, 6))


def test_closed_single_parameter_window_is_evaluable():
    fixed, moving, maximum = _family("growing_box")
    theta = Fraction(1, 10)
    prepared = prepare_polyhedral_continuation(
        fixed,
        moving,
        (1, 0, 0),
        max_step=maximum,
        parameter_bounds=(theta, theta),
        seed_parameter=theta,
    )
    assert prepared.chart_count == 1 and not prepared.uncovered
    _assert_profile(prepared, "growing_box", theta, maximum)
    for outside in (Fraction(0), Fraction(1, 5)):
        with pytest.raises(ValueError, match=r"outside|domain|bounds"):
            specialize_polyhedral_continuation(prepared, outside)


@pytest.mark.parametrize("scale", [Fraction(1, 8), Fraction(8)])
def test_exact_frame_and_units_preserve_values_on_common_coverage(scale):
    fixed, moving, maximum = _family("growing_box")
    rotation = np.array(((0, -1, 0), (0, 0, 1), (-1, 0, 0)))
    bodies = [
        [
            AffinePolyhedralCell(
                float(scale) * np.asarray(cell.vertices) @ rotation.T + (8, -4, 16),
                float(scale) * np.asarray(cell.parameter_displacements) @ rotation.T,
            )
            for cell in body
        ]
        for body in (fixed, moving)
    ]
    theta = Fraction(1, 10)
    prepared = prepare_polyhedral_continuation(
        *bodies,
        float(scale) * rotation @ np.array((1, 0, 0)),
        max_step=maximum,
        parameter_bounds=(Fraction(-1, 16), Fraction(5, 8)),
        seed_parameter=theta,
    )
    # The sufficient separating-certificate choice can change overall coverage
    # with frame. Only values at this common regular seed are asserted here.
    path = specialize_polyhedral_continuation(prepared, theta)
    value_range = polyhedral_translation_range(
        path, step_tolerance=1e-12, volume_tolerance=0
    )
    expected_maximum = scale**3 * Fraction(11, 50)
    assert value_range.maximum == (expected_maximum, expected_maximum)
    for step in (Fraction(1, 2), Fraction(1), Fraction(5, 4), Fraction(2)):
        expected = scale**3 * _volume("growing_box", step, theta)
        actual = float(polyhedral_translation_volume(path.profile, float(step)))
        assert actual == pytest.approx(float(expected), abs=float(scale**3) * 2e-14)


@pytest.mark.parametrize(
    "override",
    [
        {"parameter_bounds": (1, -1)},
        {"parameter_bounds": (0, np.inf)},
        {"seed_parameter": 1},
        {"seed_parameter": np.nan},
        {"max_step": 0},
        {"max_charts": 0},
        {"max_center_trials": 0},
    ],
)
def test_invalid_continuation_requests_are_rejected(override):
    fixed, moving, maximum = _family("growing_box")
    arguments = {
        "max_step": maximum,
        "parameter_bounds": (Fraction(-1, 16), Fraction(5, 8)),
        "seed_parameter": 0,
    }
    arguments.update(override)
    with pytest.raises(ValueError):
        prepare_polyhedral_continuation(fixed, moving, (1, 0, 0), **arguments)
