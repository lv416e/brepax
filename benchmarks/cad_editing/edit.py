"""Validate bounded inverse CAD edits, independently of their frozen mesh loss.

Run from the repository root:
    uv run --group benchmark python -m benchmarks.cad_editing.edit --output RESULT.json

Writes JSON and STEP artifacts beside it. Three target fits and three expected
failure controls share one working mesh. Every converged fit is independently
rebuilt, transformed, remeshed and STEP-roundtripped. Failures retain their
trajectories and exit nonzero when unexpected. No DFM, arbitrary STEP editing,
topology-change, remeshing-gradient, or performance claim is made.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import reference_measures
from .mesh import prepare_mesh
from .optimization import EditSolver
from .verification import RELATIVE_TOLERANCE, verify_edit

# Fixed before running: changing both coordinates together and in opposite directions.
EDIT_CASES = (
    ("increase", (4.0, 2.55), (6.0, 3.45)),
    ("decrease", (6.0, 3.45), (4.0, 2.55)),
    ("opposite", (4.0, 3.5), (6.0, 2.5)),
)


def run(deflection: float, artifact_dir: Path) -> dict[str, Any]:
    """Retain every case; a successful control means the edit was rejected."""
    report: dict[str, Any] = {
        "settings": {
            "working_deflection": deflection
            if np.isfinite(deflection)
            else str(deflection),
            "required_dtype": "float64",
            "relative_tolerance": RELATIVE_TOLERANCE,
            "solver_residual_tolerance": 1e-10,
            "max_updates": 8,
            "scope": "Supplied roof/hole family with fixed topology",
        },
        "cases": [],
        "controls": [],
        "passed": False,
    }
    try:
        if jnp.asarray(0.0).dtype != jnp.float64:
            raise ValueError("The CAD comparison requires jax_enable_x64=True")
        model, report["mesh"] = prepare_mesh(deflection)
        solver = EditSolver(model)
    except Exception as error:
        report["setup_error"] = f"{type(error).__name__}: {error}"
        return report

    for name, initial, goal in EDIT_CASES:
        row: dict[str, Any] = {
            "name": name,
            "initial": initial,
            "target_dimensions": goal,
            "passed": False,
        }
        report["cases"].append(row)
        try:
            target = np.asarray(reference_measures(np.array(goal)), dtype=np.float64)
            row["target_measures"] = target.tolist()
            result = solver.solve(np.array(initial), target)
            row["optimization"] = asdict(result)
            if result.status != "converged":
                continue
            dimensions = np.array(result.steps[-1].dimensions)
            row["parameter_relative_error"] = float(np.max(abs(dimensions / goal - 1)))
            checks = verify_edit(dimensions, target, deflection, artifact_dir / name)
            row["geometry"] = [asdict(check) for check in checks]
            row["passed"] = bool(
                checks
                and all(check.passed for check in checks)
                and row["parameter_relative_error"] <= RELATIVE_TOLERANCE
            )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"

    target = np.asarray(reference_measures(np.array([6.0, 3.45])), dtype=np.float64)
    controls = (
        ("outside_edit_box", np.array([3.0, 3.0]), target, 8, "invalid_input"),
        (
            "unreachable_target",
            np.array([5.0, 3.0]),
            np.array([1e6, 1e6]),
            8,
            "no_descent",
        ),
        ("zero_budget", np.array([4.0, 2.55]), target, 0, "iteration_limit"),
    )
    for name, initial_array, measures, budget, expected in controls:
        result = solver.solve(initial_array, measures, max_updates=budget)
        report["controls"].append(
            {
                "name": name,
                "initial": initial_array.tolist(),
                "target_measures": measures.tolist(),
                "max_updates": budget,
                "expected_status": expected,
                "optimization": asdict(result),
                "passed": result.status == expected,
            }
        )
    report["passed"] = all(
        row["passed"] for row in report["cases"] + report["controls"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deflection", type=float, default=1e-4)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A fresh directory prevents an earlier successful STEP from masking a failed run.
    artifacts = Path(
        tempfile.mkdtemp(prefix=f"{args.output.stem}-", dir=args.output.parent)
    )
    report = run(args.deflection, artifacts)
    report["artifact_dir"] = str(artifacts)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
