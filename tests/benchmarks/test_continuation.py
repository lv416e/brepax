"""Automatic CAD edits, retained unsupported coverage, and honest milestones."""

from fractions import Fraction

import pytest

from benchmarks.multi_contact.continuation import audit_continuation
from benchmarks.multi_contact.continuation_fixtures import TASKS
from benchmarks.multi_contact.design_search_selection import (
    DesignSearchResult,
    Observation,
    evaluate_design_search,
)
from benchmarks.multi_contact.parametric_cad import CadMeasurement


def test_automatic_edit_crosses_two_discovered_boundaries_with_fresh_cut() -> None:
    task = TASKS[1]
    result = evaluate_design_search(task, "automatic_continuation")
    audit = audit_continuation(task, result)
    assert result.operational_success
    assert result.model_status == "optimal_within_tolerance"
    assert result.continuation is not None
    assert result.continuation.boundary_parameters == (Fraction(1, 6), Fraction(5, 12))
    assert result.continuation.uncovered == ()
    assert audit["crossed_discovered_boundaries"] == [Fraction(1, 6), Fraction(5, 12)]
    assert audit["objective_excess_bounds"][1] <= Fraction(1, 1000)
    assert not audit["invalid_optimality_certificate"]
    assert "crosses_supplied_boundary" not in audit
    assert all(observation.source_preserved for observation in result.observations)

    common = audit["first_design_qualified_common"]
    final = audit["final_cut_validation"]
    assert common is not None and final is not None
    assert result.model_query_elapsed_seconds is not None
    assert result.model_query_elapsed_seconds <= common["elapsed_seconds"]
    assert common["elapsed_seconds"] < final["elapsed_seconds"] <= result.total_seconds
    assert common["cumulative_calls"] == {"common": 1, "cut": 0, "fuse": 1}
    assert final["cumulative_calls"] == {"common": 2, "cut": 1, "fuse": 2}
    assert result.calls.common == 2 and result.calls.cut == 1 and result.calls.fuse == 2
    assert final["operational_success"]
    assert abs(final["actual_removed_volume"] - float(task.target)) <= float(
        task.target / 1000
    )


@pytest.mark.parametrize("index", [4, 5])
def test_unsupported_frontiers_remain_unresolved_in_benchmark(index: int) -> None:
    task = TASKS[index]
    result = evaluate_design_search(task, "automatic_continuation")
    audit = audit_continuation(task, result)
    assert result.model_status == "unresolved"
    assert result.continuation is not None and result.continuation.uncovered
    assert result.errors == []
    assert not audit["false_infeasible_certificate"]
    assert not audit["false_feasible_certificate"]
    assert audit["model_certificate_completed_elapsed_seconds"] is None
    assert audit["first_design_qualified_common"] is None


def test_cad_band_discrepancy_cannot_claim_impossible_design_quality() -> None:
    task = TASKS[1]
    allowance = task.target / 1000
    theta = float(Fraction(319, 600)) - 1e-6
    step = 17 / 8
    assert task.reference_volume(theta, step) < task.target - allowance
    observed = float(task.target - allowance) + 1e-8
    record = CadMeasurement(
        step,
        volume=observed,
        working_volume=observed,
        source_valid=True,
        common_valid=True,
    )
    result = DesignSearchResult(task.name, "rebuild_profile", task.target, allowance)
    result.observations.append(Observation(theta, record, "inside", True))
    audit = audit_continuation(task, result)
    assert audit["first_volume_qualified_common"] is not None
    assert audit["first_design_qualified_common"] is None
