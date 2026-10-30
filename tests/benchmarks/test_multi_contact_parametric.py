"""Finite-edit claims use independent sections, rebuilt CAD and retained failures."""

from dataclasses import asdict

import numpy as np
import pytest

from benchmarks.multi_contact.parametric_evaluation import evaluate_case, summarize
from benchmarks.multi_contact.parametric_fixtures import parametric_fixtures
from benchmarks.multi_contact.parametric_references import (
    clipped_volume,
    reference_volume,
)
from benchmarks.multi_contact.parametric_sensitivity import (
    sensitivity_case,
    symmetry_and_degeneracy,
)


def test_frozen_section_formulas_match_independent_polygon_clipping():
    for fixture in parametric_fixtures():
        for step in np.linspace(0.0, fixture.max_step, 23):
            assert reference_volume(fixture, step) == pytest.approx(
                clipped_volume(fixture.bodies(), step), abs=1e-13
            )


@pytest.mark.parametrize("name", ["tent_0", "split_0"])
def test_finite_roots_include_exit_plateaus_and_path_infeasibility(name):
    fixture = next(f for f in parametric_fixtures() if f.name == name)
    result = evaluate_case(fixture, (("finite", 4),))
    assert result["passed"], result
    policy = result["policies"][0]
    assert all(row["within_one_percent"] for row in policy["selected"][:3])
    unreachable = policy["selected"][3]
    assert unreachable["correct_unreachable"]
    assert unreachable["target"] < unreachable["reference_stock_volume"]
    assert unreachable["probe_attempts"] == 0
    assert unreachable["validation"] is None
    maximum = next(
        row for row in result["inventory"]["targets"] if row["fraction"] == 1
    )
    if name == "tent_0":
        assert len(maximum["actual"]["roots"]) == 1
        assert maximum["actual"]["intervals"] == ()
    else:
        assert len(maximum["actual"]["intervals"]) == 2
        partial = result["inventory"]["targets"][1]
        assert len(partial["actual"]["roots"]) == 4


@pytest.mark.parametrize("dimension", ["stock_height", "slab_width"])
def test_design_sensitivity_preserves_constant_volume_in_rebuilt_cad(dimension):
    fixture = parametric_fixtures()[0]
    result = sensitivity_case(fixture, second=dimension, target=15 / 128)
    assert result["passed"], result
    entry, exit_root = result["roots"]
    assert entry["step"] == pytest.approx(23 / 32, abs=2e-12)
    assert exit_root["step"] == pytest.approx(5 / 4, abs=2e-12)
    expected = [1, -15 / 32] if dimension == "stock_height" else [1, -11 / 8]
    np.testing.assert_allclose(entry["position_gradient"], expected, atol=2e-12)
    assert result["physical_calls"]["cut"] > 0
    assert all(row["passed"] for row in result["roots"])


def test_parameter_covariance_and_distinct_degenerate_inverse_cases():
    result = symmetry_and_degeneracy()
    assert result["passed"], result
    assert result["degeneracy"]["smooth_maximum_matches_reference"]
    assert result["degeneracy"]["numerical_maximum_derivative_unresolved"]
    assert result["degeneracy"]["zero_slope_position_rejected"]
    assert result["degeneracy"]["event_order_tie_rejected"]


def test_whole_case_failure_stays_in_requested_policy_denominators():
    first, failed = parametric_fixtures()[:2]
    policies = (("finite", 4),)
    good = evaluate_case(first, policies)
    assert good["passed"]
    missing = {
        "fixture": asdict(failed),
        "passed": False,
        "error": "Source construction failed before a policy record was returned",
    }
    result = summarize([good, missing], policies)["finite_K4"]
    assert result["shape_groups"] == 2
    assert result["targets"] == 8
    assert result["reachable_targets"] == 6
    assert result["unconditional_success_fraction"] == 3 / 8
    assert result["reachable_success_fraction"] == 3 / 6
    assert result["unsupported_or_failed_count"] == 4
    assert result["call_counts_are_lower_bounds"]
    assert not result["call_accounting_complete"]
    all_failed = summarize([missing], policies)["finite_K4"]
    assert all_failed["mean_error_among_returned_edits"] is None
    assert all_failed["median_first_target_seconds"] is None
