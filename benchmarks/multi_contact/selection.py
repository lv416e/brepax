"""Compare finite edit selection under a fixed CAD observation budget.

Run ``uv run python -m benchmarks.multi_contact.selection
--output /tmp/selection.json``.
The default protocol uses eight pilot and 32 fresh synthetic designs, five
grouped partitions, three target fractions, and 1/2/4/8 candidate queries.
Off-grid amount refinement is a separate bounded heuristic comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

from .selection_evaluation import evaluate
from .selection_experiment import (
    SelectionOracle,
    collect_case,
    placement_check,
    valid_measurement,
)
from .selection_fixtures import fixtures
from .selection_policies import BUDGETS, STEPS, TARGETS


def _snapshot(root: Path) -> dict[str, str]:
    # Include transitive library/benchmark imports and the locked environment.
    paths = sorted((root / "src/brepax").rglob("*.py"))
    paths.extend(sorted((root / "benchmarks").rglob("*.py")))
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def run(variants: int = 8) -> dict[str, Any]:
    """Measure the frozen policies without requiring any performance winner."""
    if not 2 <= variants <= 8:
        raise ValueError("Use two to eight variants; eight is the full protocol")
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    root = Path(__file__).parents[2]
    before = _snapshot(root)
    started = perf_counter()
    pilots = tuple(collect_case(f) for f in fixtures(2, cohort="pilot"))
    cases = tuple(collect_case(f) for f in fixtures(variants))
    oracle = SelectionOracle()
    evaluation = evaluate(cases, oracle)
    placed = placement_check(cases[0].fixture)
    physical = [c.initial for c in (*pilots, *cases)]
    physical.extend(e.measurement for c in (*pilots, *cases) for e in c.endpoints)
    physical.extend(oracle.offgrid.values())
    physical.extend(oracle.edits.values())
    unchanged = before == _snapshot(root)
    all_valid = all(
        c.initial_valid
        and c.preparation_error is None
        and all(valid_measurement(e.measurement, c.cutter_volume) for e in c.endpoints)
        for c in (*pilots, *cases)
    )
    all_edits_valid = all(row["valid_edit"] for row in evaluation["selections"])
    return {
        "passed": bool(
            all_valid and all_edits_valid and unchanged and placed["passed"]
        ),
        "passed_scope": (
            "Complete bounded synthetic comparison and valid selected cuts. "
            "Passing does not mean targets are reachable, a method is superior, "
            "or arbitrary source CAD is supported."
        ),
        "protocol": {
            "variants_per_family": variants,
            "pilot_variants_per_family": 2,
            "target_fractions": TARGETS,
            "steps": list(map(float, STEPS)),
            "budgets": BUDGETS,
            "primary_budget": 4,
            "input_scope": "Axis-aligned box assemblies at identity placement",
            "evaluation_scope": "Five shape-grouped partitions of a synthetic cohort",
            "offgrid_scope": (
                "Observed-endpoint heuristic; no monotonicity or reachability guarantee"
            ),
            "source_intake_scope": (
                "Nominal contact model, not original-CAD certification"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("jax", "numpy", "cadquery-ocp-novtk")
            },
        },
        "source_sha256": before,
        "sources_unchanged": unchanged,
        "pilot": [asdict(case) for case in pilots],
        "cases": [asdict(case) for case in cases],
        "evaluation": evaluation,
        "observations": oracle.records(),
        "physical_comparison_calls": {
            name: sum(getattr(m, name, 0) for m in physical)
            for name in ("common_calls", "cut_calls", "fusion_calls")
        },
        "call_accounting": (
            "Physical deterministic observations and final cuts are cached across "
            "policies. Each selection records its own attempted query budget, actual "
            "Common counts, initial validation and final Common/Cut costs. Offline "
            "training/scoring observations are reported separately. Placement checks "
            "are separate validation work. No end-to-end speed advantage is claimed."
        ),
        "placement_check": placed,
        "seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", default=8, type=int)
    args = parser.parse_args()
    if args.output.exists() and not args.output.is_file():
        parser.error("Output must name a regular file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.variants)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "seconds": result["seconds"],
                "summary": result["evaluation"]["summary"],
            }
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
