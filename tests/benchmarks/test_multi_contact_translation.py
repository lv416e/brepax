"""Finite controls compare independent geometry with the maintained workflow."""

import numpy as np
import pytest

from benchmarks.multi_contact.finite_references import (
    clipped_rotation_volume,
    rotation_volume,
    translation_cases,
)
from benchmarks.multi_contact.translation import measure_translation, target_demo


@pytest.mark.parametrize("name", ["positive_gap", "oblique_sliding_face"])
def test_finite_profile_crosses_events_and_matches_rebuilt_cad(name):
    case = next(case for case in translation_cases() if case.name == name)
    result = measure_translation(case)
    assert result["passed"], result
    assert len(result["values"]) == len(case.steps)
    if name == "positive_gap":
        assert result["event_slopes"][0]["actual"] == pytest.approx([0.0, 1.0])
        assert result["values"][-1]["cad"]["volume"] == pytest.approx(0.0)


def test_identical_initial_velocity_has_path_dependent_quadratic_term():
    steps = (0.04, 0.02, 0.01, 0.005)
    for pivot_x in (0.0, 1.0):
        residuals = []
        for step in steps:
            value = rotation_volume(step, 1.0, 0.5, 1.0, pivot_x)
            polygon = clipped_rotation_volume(step, 1.0, 0.5, 1.0, pivot_x)
            assert value == pytest.approx(polygon, abs=1e-14)
            residuals.append((value - 0.125 * step) / step**2)
        coefficient = -0.25 * pivot_x
        assert abs(residuals[-1] - coefficient) < abs(residuals[0] - coefficient)
        assert residuals[-1] == pytest.approx(coefficient, abs=1e-3)
    # Crossing the far-side event invalidates the earlier triangular formula.
    step = 0.4
    actual = rotation_volume(step, 0.1, 0.5, 1.0, 0.0)
    assert actual == pytest.approx(
        clipped_rotation_volume(step, 0.1, 0.5, 1.0, 0.0), abs=1e-14
    )
    assert 0.125 * np.tan(step) - actual > 0.01


def test_profile_target_selection_produces_valid_cut_and_step_roundtrip(tmp_path):
    result = target_demo(tmp_path / "finite.step")
    assert result["passed"], result
    assert result["step"] == pytest.approx(result["analytic_step"], abs=1e-14)
    assert result["relative_target_error"] < 1e-10
    assert result["cad"]["common_calls"] == 1
    assert result["cad"]["cut_calls"] == 1
    assert result["step_artifact"]["passed"]
    assert result["step_artifact"]["roundtrip_volume"] == pytest.approx(2.8, abs=1e-10)
