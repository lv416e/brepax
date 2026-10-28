"""Fresh finite-CAD checks of directional limits and their validity boundaries."""

import json

import numpy as np
import pytest

from benchmarks.face_contact.cad import (
    Fixture,
    Motion,
    Placement,
    build_pair,
    cad_volume,
    difference_sweep,
    placement_from_matrix,
    volume_sample,
)
from benchmarks.face_contact.stress import collect_cad_controls
from brepax._occt.backend import BRepGProp, GProp_GProps

SQUARE = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
TRAPEZOID = ((-1.1, -0.7), (1.2, -0.7), (0.8, 1.0), (-0.4, 1.0))


def _slopes(samples):
    for sample in samples:
        assert sample.error is None
        assert sample.valid and sample.fixed_valid and sample.moving_valid
        assert sample.working_volume is not None
    return np.array([sample.working_difference for sample in samples[1:]])


def test_opposite_tilts_have_same_positive_overlap_directional_limit():
    slopes = []
    for sign in (-1.0, 1.0):
        samples = difference_sweep(
            Fixture(SQUARE), Motion(angular_velocity=(sign, 0.0, 0.0)), (1e-3, 1e-4)
        )
        assert samples[0].working_volume == 0
        observed = _slopes(samples)
        # The square section has overlap volume tan(t), hence slope tan(t)/t.
        np.testing.assert_allclose(
            observed, np.tan([1e-3, 1e-4]) / [1e-3, 1e-4], atol=2e-12
        )
        assert abs(observed[1] - 1) < abs(observed[0] - 1) / 50
        slopes.append(observed[1])
    np.testing.assert_allclose(slopes, [1, 1], atol=4e-9)
    # A linear gradient would require D(-omega) = -D(omega).
    assert sum(slopes) > 1.999999


def test_mixed_trapezoid_motion_converges_to_independent_patch_integral():
    samples = difference_sweep(
        Fixture(TRAPEZOID),
        Motion((0.2, -0.1, -0.25), (0.5, 0.7, 0.3)),
        (1e-3, 1e-4),
    )
    # Independent 80-digit vertical-slice integration of max(.25+.7*x-.5*y,0).
    expected = 1.0588976190476189458156746312153086148
    errors = np.abs(_slopes(samples) - expected)
    assert errors[1] < errors[0] / 8
    assert errors[1] < 5e-6


@pytest.mark.parametrize(
    "placement",
    [
        Placement(0.1, (0.3, -0.2, 0.4), (0.2, 0.4, -0.3)),
        Placement(10.0, (30.0, -20.0, 40.0), (0.2, 0.4, -0.3)),
    ],
)
def test_actual_transformed_cad_has_volume_similarity_law(placement):
    fixture = Fixture(TRAPEZOID)
    motion = Motion((0.1, 0.2, -0.1), (-0.4, 0.6, 0.2), (0.3, -0.4, 0.5))
    reference = volume_sample(fixture, motion, 0.01)
    transformed = volume_sample(fixture, motion, 0.01, placement)
    for sample in (reference, transformed):
        assert sample.error is None
        assert sample.valid and sample.fixed_valid and sample.moving_valid
    np.testing.assert_allclose(
        transformed.working_volume, reference.working_volume, rtol=2e-10, atol=1e-13
    )


def test_matrix_placement_preserves_actual_solid_centroid_at_half_turn():
    # This exact matrix exercises the axis-angle branch at a half turn.
    placement = placement_from_matrix(0.1, (3.0, -2.0, 4.0), np.diag([-1.0, 1.0, -1.0]))
    _, moving = build_pair(Fixture(SQUARE), placement=placement)
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(moving, properties, 1e-12)
    center = properties.CentreOfMass()
    np.testing.assert_allclose(
        [center.X(), center.Y(), center.Z()], [3.0, -2.0, 3.95], atol=2e-14
    )
    np.testing.assert_allclose(cad_volume(moving), 0.004, atol=1e-16)
    with pytest.raises(ValueError, match="proper rotation"):
        placement_from_matrix(1.0, (0.0, 0.0, 0.0), np.diag([1.0, 1.0, -1.0]))


def test_gap_closure_is_a_finite_action_not_a_contact_directional_derivative():
    samples = difference_sweep(
        Fixture(SQUARE, gap=1e-3), Motion((0.0, 0.0, -1.0)), (0.01, 1e-4)
    )
    np.testing.assert_allclose(_slopes(samples), [3.6, 0], atol=2e-12)
    assert samples[0].working_volume == 0


def test_initial_overlap_can_have_negative_volume_direction():
    samples = difference_sweep(
        Fixture(SQUARE, gap=-1e-3), Motion((0.0, 0.0, 1.0)), (1e-4, 1e-5)
    )
    np.testing.assert_allclose(samples[0].working_volume, 0.004, atol=2e-15)
    np.testing.assert_allclose(_slopes(samples), [-4, -4], atol=2e-10)


def test_second_contact_patch_contributes_to_actual_cad_volume():
    fixture = Fixture(SQUARE, right_wall=True)
    side = difference_sweep(fixture, Motion((1.0, 0.0, 0.0)), (1e-3, 1e-4))
    combined = difference_sweep(fixture, Motion((1.0, 0.0, -1.0)), (1e-3, 1e-4))
    np.testing.assert_allclose(_slopes(side), [2, 2], atol=2e-11)
    # Two strips overlap by 2*t**2: the union volume is 6*t - 2*t**2.
    np.testing.assert_allclose(
        _slopes(combined), 6 - 2 * np.array([1e-3, 1e-4]), atol=2e-11
    )
    # Using only the bottom face would predict zero and four, respectively.


def test_intersection_patch_area_may_be_smaller_than_the_tool_footprint():
    samples = difference_sweep(
        Fixture(SQUARE, stock_bounds=(-0.5, 2.0, -2.0, 2.0)),
        Motion((0.0, 0.0, -1.0)),
        (1e-3, 1e-4),
    )
    np.testing.assert_allclose(_slopes(samples), [3, 3], atol=2e-11)


def test_stress_report_retains_unresolved_limits_and_inapplicable_predictions():
    report = collect_cad_controls()
    json.dumps(report, allow_nan=False)
    assert report["passed"]
    assert report["unresolved_cad_limit_cases"] == [
        "gap_below_occt_tolerance",
        "milliscale_mixed",
    ]
    assert report["finite_difference_reference_departure_count"] > 0
    rows = {row["name"]: row for row in report["rows"]}
    gap = rows["gap_below_occt_tolerance"]
    assert not gap["contact_law_applicable"]
    assert gap["kernel_reference_agreement"] is None
    assert gap["cad_assessed_step_agreement"] is None
    assert gap["expected_direction"] == 0
    assert gap["selected_patch_prediction"] == 4
    assert any(sample["working_difference"] == 0 for sample in gap["samples"])
    overlap = rows["initial_overlap"]
    assert overlap["expected_direction"] == -4
    assert overlap["selected_patch_prediction"] == 0
    assert overlap["cad_assessed_step_agreement"]
