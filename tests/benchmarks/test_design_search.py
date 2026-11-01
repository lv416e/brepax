"""Physical witnesses, whole-task resources, and independent report semantics."""

from dataclasses import replace
from fractions import Fraction

import pytest

from benchmarks.multi_contact.design_search import audit_result
from benchmarks.multi_contact.design_search_fixtures import TASKS
from benchmarks.multi_contact.design_search_selection import (
    DesignSearchResult,
    SearchBudget,
    classify_band,
    evaluate_design_search,
)
from brepax.experimental.polyhedral_design import (
    DesignTargetResult,
    DesignTargetWitness,
)


def test_budget_counts_source_fusions_and_reserved_fresh_cut() -> None:
    task = replace(TASKS[1], target=Fraction(1, 2), theta0=Fraction(0))
    result = evaluate_design_search(
        task, "rebuild_profile", SearchBudget(boolean_operations=7)
    )
    assert result.operational_success
    assert result.objective == 0
    assert result.calls.common == 2
    assert result.calls.cut == 1
    assert result.calls.fuse == 4
    assert result.final is not None
    assert result.final.actual_removed_volume == pytest.approx(0.5, abs=1e-12)
    assert all(item.source_preserved for item in result.observations)

    capped = evaluate_design_search(
        task, "adaptive_cad", SearchBudget(boolean_operations=2)
    )
    assert not capped.operational_success
    assert capped.resource_limit is not None
    assert capped.model_status != "infeasible"
    assert capped.calls.common + capped.calls.cut + capped.calls.fuse <= 2


def test_rebuild_accepts_band_only_maximum_without_exact_target_root() -> None:
    task = replace(TASKS[-1], theta0=Fraction(-1, 32))
    assert task.reference_maximum(task.theta0) < task.target
    result = evaluate_design_search(task, "rebuild_profile")
    assert result.operational_success and result.objective == 0
    assert result.final is not None
    assert result.final.actual_removed_volume == pytest.approx(1023 / 2048, abs=1e-12)
    assert audit_result(task, result)["selected_action_reference_feasible"]


def test_certified_search_crosses_boundary_to_oblique_feasible_island() -> None:
    task = TASKS[0]
    result = evaluate_design_search(task, "certified_boundary")
    audit = audit_result(task, result)
    assert result.model_status == "optimal_within_tolerance"
    assert result.operational_success
    assert audit["crosses_supplied_boundary"]
    assert audit["model_witness_reference_feasible"]
    assert not audit["certificate_excludes_reference"]
    assert not audit["invalid_optimality_certificate"]
    assert audit["objective_excess_bounds"][1] <= Fraction(1, 1000)


def test_report_audits_rejected_model_witness_and_false_feasible_claim() -> None:
    task = TASKS[3]
    witness = DesignTargetWitness(0.0, 0.0, Fraction(0), -task.target, Fraction(0))
    query = DesignTargetResult(
        task.target,
        witness,
        Fraction(0),
        Fraction(0),
        "optimal_within_tolerance",
        (),
        1,
        1,
        (),
    )
    result = DesignSearchResult(
        task.name,
        "certified_boundary",
        task.target,
        task.target / 1000,
        model_status=query.status,
        model_query=query,
        objective_lower=Fraction(0),
        objective_upper=Fraction(0),
    )
    audit = audit_result(task, result)
    assert result.selected_parameter is None
    assert audit["model_witness_reference_feasible"] is False
    assert audit["false_feasible_certificate"]
    assert audit["invalid_optimality_certificate"]


def test_tolerance_endpoint_is_not_silently_widened() -> None:
    target, allowance = Fraction(1, 2), Fraction(1, 2000)
    assert classify_band(0.4995, target, allowance) == "boundary_indeterminate"
    assert classify_band(0.4996, target, allowance) == "inside"
    assert classify_band(0.4994, target, allowance) == "outside"


@pytest.mark.parametrize("wall_seconds", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_wall_budget_rejected_before_work(wall_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        SearchBudget(wall_seconds=wall_seconds)
