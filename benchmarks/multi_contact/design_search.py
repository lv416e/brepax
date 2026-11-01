"""Run a compact mechanism study of bounded dimension and translation search."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import jax

from .design_search_fixtures import TASKS, DesignTask
from .design_search_selection import (
    DEFAULT_BUDGET,
    DesignSearchResult,
    Method,
    SearchBudget,
    evaluate_design_search,
)

METHODS: tuple[Method, ...] = ("certified_boundary", "rebuild_profile", "adaptive_cad")


def audit_result(task: DesignTask, result: DesignSearchResult) -> dict[str, Any]:
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


def run_study(
    tasks: tuple[DesignTask, ...] = TASKS,
    methods: tuple[Method, ...] = METHODS,
    budget: SearchBudget = DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Retain every attempted task, including unsupported and capped results."""
    rows = []
    for task in tasks:
        for method in methods:
            result = evaluate_design_search(task, method, budget)
            rows.append({"result": asdict(result), "audit": audit_result(task, result)})
    return {
        "scope": (
            "Deterministic mechanism tasks; "
            "no generalization or timing superiority claim"
        ),
        "mode": "cold task setup included; interpreter startup excluded equally",
        "budget": asdict(budget),
        "task_inputs": [asdict(task) for task in tasks],
        "rows": rows,
    }


def _encode(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=[task.name for task in TASKS], action="append"
    )
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--boolean-budget", type=int, default=128)
    parser.add_argument("--wall-seconds", type=float, default=60.0)
    parser.add_argument("--model-work-limit", type=int, default=256)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    selected = tuple(
        task for task in TASKS if args.task is None or task.name in args.task
    )
    methods = METHODS if args.method is None else tuple(args.method)
    budget = SearchBudget(args.boolean_budget, args.wall_seconds, args.model_work_limit)
    payload = run_study(selected, methods, budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_encode(payload), indent=2) + "\n")
    for row in payload["rows"]:
        result = row["result"]
        print(
            result["task"],
            result["method"],
            result["decision"],
            f"CAD={result['operational_success']}",
            f"seconds={result['total_seconds']:.3f}",
        )


if __name__ == "__main__":
    main()
