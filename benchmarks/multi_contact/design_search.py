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
from .design_search_reporting import audit_result
from .design_search_selection import (
    DEFAULT_BUDGET,
    Method,
    SearchBudget,
    evaluate_design_search,
)

METHODS: tuple[Method, ...] = ("certified_boundary", "rebuild_profile", "adaptive_cad")


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
