"""Compare automatic chart continuation with complete rebuild and CAD search."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import jax

from .continuation_fixtures import TASKS, ContinuationTask
from .design_search_reporting import audit_result
from .design_search_selection import (
    DEFAULT_BUDGET,
    DesignSearchResult,
    Method,
    Observation,
    SearchBudget,
    evaluate_design_search,
)
from .parametric_cad import valid_observation

METHODS: tuple[Method, ...] = (
    "automatic_continuation",
    "rebuild_profile",
    "adaptive_cad",
)


def _milestone(observation: Observation) -> dict[str, Any]:
    return {
        "parameter": observation.parameter,
        "step": observation.measurement.step,
        "elapsed_seconds": observation.elapsed_seconds,
        "cumulative_calls": asdict(observation.cumulative_calls),
        "volume": observation.measurement.working_volume,
    }


def audit_continuation(
    task: ContinuationTask, result: DesignSearchResult
) -> dict[str, Any]:
    """Label observed actions after the run; reference optima never guide search."""
    audit = audit_result(task, result)
    audit.pop("crosses_supplied_boundary")
    reference = task.reference_optimality(result.volume_tolerance)
    first_volume = None
    first_objective = None
    for observation in result.observations:
        if (
            observation.final
            or observation.band_status != "inside"
            or not observation.source_preserved
            or not valid_observation(observation.measurement)
        ):
            continue
        if first_volume is None:
            first_volume = _milestone(observation)
        theta, step = (
            Fraction(observation.parameter),
            Fraction(observation.measurement.step),
        )
        objective = abs(theta - task.theta0)
        if (
            first_objective is None
            and reference.objective is not None
            and task.bounds[0] <= theta <= task.bounds[1]
            and 0 <= step <= Fraction(task.extent)
            and abs(task.reference_volume(theta, step) - task.target)
            <= result.volume_tolerance
            and objective - reference.objective[0] <= result.parameter_tolerance
        ):
            first_objective = _milestone(observation)
            first_objective["objective_excess_upper"] = (
                objective - reference.objective[0]
            )
    audit["first_volume_qualified_common"] = first_volume
    audit["first_design_qualified_common"] = first_objective
    audit["final_cut_validation"] = (
        None
        if result.final is None
        else {
            **_milestone(result.final),
            "actual_removed_volume": result.final.actual_removed_volume,
            "operational_success": result.operational_success,
        }
    )
    audit["model_certificate_completed_elapsed_seconds"] = (
        result.model_query_elapsed_seconds
        if result.model_status in ("optimal_within_tolerance", "infeasible")
        else None
    )
    audit["crossed_discovered_boundaries"] = (
        None
        if result.continuation is None or result.selected_parameter is None
        else [
            boundary
            for boundary in result.continuation.boundary_parameters
            if min(task.theta0, Fraction(result.selected_parameter))
            < boundary
            < max(task.theta0, Fraction(result.selected_parameter))
        ]
    )
    return audit


def run_comparison(
    tasks: tuple[ContinuationTask, ...] = TASKS,
    methods: tuple[Method, ...] = METHODS,
    budget: SearchBudget = DEFAULT_BUDGET,
) -> dict[str, Any]:
    """Run cold tasks with all failed/unsupported attempts retained."""
    rows = []
    for task in tasks:
        for method in methods:
            result = evaluate_design_search(task, method, budget)
            rows.append(
                {"result": asdict(result), "audit": audit_continuation(task, result)}
            )
    return {
        "scope": (
            "Two explicit geometry mechanisms; deterministic validation, "
            "no generalization claim"
        ),
        "timing_contract": (
            "Common milestones are actual completed observations, not completed Cuts. "
            "Objective qualification is retrospective only. The automatic monolithic "
            "query completes before its first CAD observation. Cold setup is included; "
            "full-budget tail costs are not a runtime superiority claim."
        ),
        "budget": asdict(budget),
        "task_inputs": [asdict(task) for task in tasks],
        "rows": rows,
    }


def _json_fraction(value: object) -> str:
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=[task.name for task in TASKS], action="append"
    )
    parser.add_argument("--method", choices=METHODS, action="append")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    tasks = tuple(task for task in TASKS if args.task is None or task.name in args.task)
    methods = METHODS if args.method is None else tuple(args.method)
    payload = run_comparison(tasks, methods)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, default=_json_fraction, indent=2) + "\n")
    for row in payload["rows"]:
        result = row["result"]
        print(
            result["task"],
            result["method"],
            result["decision"],
            f"CAD={result['operational_success']}",
        )


if __name__ == "__main__":
    main()
