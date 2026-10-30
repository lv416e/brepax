"""Reproduce finite target selection and fixed-normal design sensitivities.

Run ``uv run --group benchmark python -m benchmarks.multi_contact.parametric
--output /tmp/parametric.json``. Inputs are twelve predeclared synthetic groups,
not an external CAD population. Every online observation executes actual OCCT.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

from .parametric_evaluation import POLICIES, TARGET_FRACTIONS, evaluate_case, summarize
from .parametric_fixtures import parametric_fixtures
from .parametric_sensitivity import sensitivity_case, symmetry_and_degeneracy


def _snapshot(root: Path) -> dict[str, str]:
    paths = sorted((root / "src/brepax").rglob("*.py"))
    paths.extend(sorted((root / "benchmarks").rglob("*.py")))
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def run(
    *,
    names: tuple[str, ...] = (),
    method: str = "all",
    include_sensitivity: bool = True,
) -> dict[str, Any]:
    """Execute selected frozen cases, retaining each failed/unsupported outcome."""
    fixtures = parametric_fixtures()
    if any(name not in {fixture.name for fixture in fixtures} for name in names):
        raise ValueError("Unknown fixture name")
    if method not in ("all", "finite", "first_order", "cad"):
        raise ValueError("Unknown method")
    selected = [fixture for fixture in fixtures if not names or fixture.name in names]
    policies = tuple(
        policy for policy in POLICIES if method == "all" or policy[0] == method
    )
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    root = Path(__file__).parents[2]
    before = _snapshot(root)
    started = perf_counter()
    cases = []
    for fixture in selected:
        try:
            cases.append(evaluate_case(fixture, policies))
        except Exception as error:
            cases.append(
                {
                    "fixture": asdict(fixture),
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    sensitivity = []
    invariants = None
    if include_sensitivity:
        mapping = {fixture.name: fixture for fixture in fixtures}
        for name in ("ramp_0", "ramp_1", "tent_0"):
            sensitivity.append(sensitivity_case(mapping[name]))
        sensitivity.append(
            sensitivity_case(mapping["ramp_0"], second="slab_width", target=15 / 128)
        )
        try:
            invariants = symmetry_and_degeneracy()
        except Exception as error:
            invariants = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    summary = summarize(cases, policies)
    unchanged = before == _snapshot(root)
    return {
        "passed": bool(
            unchanged
            and all(case["passed"] for case in cases)
            and all(case["passed"] for case in sensitivity)
            and (invariants is None or invariants["passed"])
        ),
        "passed_scope": (
            "Independent finite-profile root/range and parameter-derivative evidence, "
            "completed bounded comparisons and valid selected Cuts; "
            "no performance winner required."
        ),
        "protocol": {
            "geometry": (
                "Explicit convex XY prisms and pure translation, "
                "twelve frozen synthetic groups"
            ),
            "methods": policies,
            "target_fractions_of_independent_path_maximum": TARGET_FRACTIONS,
            "inverse_controls": (
                "Zero and maximum targets checked separately from "
                "positive-target comparisons"
            ),
            "units": (
                "Working length units, dimensionless step and "
                "normalized design parameters"
            ),
            "design_objective": (
                "Constant physical target volume under parameter perturbation"
            ),
            "data_use": (
                "Prior 32 groups excluded; no training or fitted ordering "
                "in this deterministic panel"
            ),
            "observations": (
                "No cached CAD observation; initial and independently rebuilt "
                "final checks counted separately"
            ),
            "timing": (
                "Single-process actual timings; imports excluded, JIT cache may "
                "be reused across cases. Not isolated cold-process evidence."
            ),
            "interpretation": (
                "Per-method runtime and target-error panels do not establish "
                "arbitrary-CAD generalization"
            ),
        },
        "cases": cases,
        "summary": summary,
        "sensitivities": sensitivity,
        "symmetry_and_degeneracy": invariants,
        "sources": before,
        "sources_unchanged": unchanged,
        "seconds": perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": importlib.metadata.version("jax"),
            "ocp": importlib.metadata.version("cadquery-ocp-novtk"),
        },
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--method", choices=("all", "finite", "first_order", "cad"), default="all"
    )
    parser.add_argument("--skip-sensitivity", action="store_true")
    arguments = parser.parse_args()
    result = run(
        names=tuple(arguments.case),
        method=arguments.method,
        include_sensitivity=not arguments.skip_sensitivity,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(_json_value(result), indent=2, allow_nan=False) + "\n"
    )
    print(
        f"passed={result['passed']} cases={len(result['cases'])} "
        f"seconds={result['seconds']:.3f}"
    )
    print(f"output={arguments.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
