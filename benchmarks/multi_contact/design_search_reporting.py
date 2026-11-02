"""Independent post-run audits of model certificates and physical CAD actions."""

from dataclasses import asdict
from fractions import Fraction
from typing import Any

from .design_search_selection import BoundedDesignTask, DesignSearchResult


def audit_result(task: BoundedDesignTask, result: DesignSearchResult) -> dict[str, Any]:
    """Evaluate independent formulas after online policy and final CAD finish."""
    reference = task.reference_optimality(result.volume_tolerance)
    audit: dict[str, Any] = {
        "reference": asdict(reference),
        "reference_exact_target": asdict(task.reference_optimality(Fraction(0))),
        "false_infeasible_certificate": result.model_status == "infeasible"
        and reference.status != "infeasible",
        "objective_excess_bounds": None,
        "certificate_excludes_reference": False,
        "model_witness_reference_feasible": None,
        "model_witness_reference_error": None,
        "model_witness_objective_consistent": None,
        "selected_action_reference_feasible": None,
        "false_feasible_certificate": False,
        "invalid_optimality_certificate": False,
        "common_reference_error": None,
        "removed_reference_error": None,
        "crosses_supplied_boundary": False,
    }
    query = result.model_query
    if reference.status == "infeasible":
        audit["false_feasible_certificate"] = (
            result.model_status == "optimal_within_tolerance"
            or result.objective_upper is not None
            or (query is not None and query.witness is not None)
        )
    if query is not None and query.witness is not None:
        witness = query.witness
        theta, step = Fraction(witness.parameter), Fraction(witness.step)
        volume = task.reference_volume(theta, step)
        audit["model_witness_reference_feasible"] = (
            task.bounds[0] <= theta <= task.bounds[1]
            and 0 <= step <= Fraction(task.extent)
            and abs(volume - task.target) <= result.volume_tolerance
        )
        audit["model_witness_reference_error"] = abs(volume - witness.volume)
        audit["model_witness_objective_consistent"] = witness.objective == abs(
            theta - task.theta0
        )
    if result.model_status == "optimal_within_tolerance":
        audit["invalid_optimality_certificate"] = (
            query is None
            or query.witness is None
            or query.optimality_gap is None
            or not 0 <= query.optimality_gap <= result.parameter_tolerance
            or audit["model_witness_reference_feasible"] is not True
            or audit["model_witness_reference_error"] != 0
            or audit["model_witness_objective_consistent"] is not True
            or query.objective_upper != query.witness.objective
        )
    probe_audits: list[dict[str, Any]] = []
    for observation in result.observations:
        value = task.reference_volume(
            Fraction(observation.parameter), Fraction(observation.measurement.step)
        )
        observed = observation.measurement.working_volume
        probe_audits.append(
            {
                "parameter": observation.parameter,
                "step": observation.measurement.step,
                "final": observation.final,
                "reference_volume": value,
                "reference_band_feasible": abs(value - task.target)
                <= result.volume_tolerance,
                "absolute_error": None
                if observed is None
                else abs(observed - float(value)),
            }
        )
    audit["probe_reference_audits"] = probe_audits
    audit["max_all_probe_reference_error"] = max(
        (
            item["absolute_error"]
            for item in probe_audits
            if item["absolute_error"] is not None
        ),
        default=None,
    )
    if reference.objective is not None:
        if result.objective is not None:
            audit["objective_excess_bounds"] = (
                result.objective - reference.objective[1],
                result.objective - reference.objective[0],
            )
        if result.objective_lower is not None:
            audit["certificate_excludes_reference"] = (
                result.objective_lower > reference.objective[1]
            )
        if result.objective_upper is not None:
            audit["certificate_excludes_reference"] |= (
                result.objective_upper < reference.objective[0]
            )
    if result.selected_parameter is not None and result.selected_step is not None:
        theta, step = (
            Fraction(result.selected_parameter),
            Fraction(result.selected_step),
        )
        value = task.reference_volume(theta, step)
        audit["selected_action_reference_feasible"] = (
            abs(value - task.target) <= result.volume_tolerance
        )
        audit["crosses_supplied_boundary"] = theta * task.theta0 < 0
        if result.final is not None:
            common = result.final.measurement.working_volume
            removed = result.final.actual_removed_volume
            if common is not None:
                audit["common_reference_error"] = abs(common - float(value))
            if removed is not None:
                audit["removed_reference_error"] = abs(removed - float(value))
    return audit
