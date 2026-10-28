"""Reproduce grouped exact-contact response and finite CAD-action comparisons.

Run with ``uv run --group benchmark python -m benchmarks.face_contact.run
--output /path/outside/repository/face-contact.json``. The output includes all
measurements, split-specific calibration choices, failures and setup costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

from .controls import collect_preparation_controls
from .evaluation import evaluate
from .experiment import (
    collect_analytic,
    collect_cad,
    collect_grid_controls,
    collect_sampled,
)
from .fixtures import MAIN_RESOLUTION, SPLIT_SEEDS
from .preparation import collect_prepared_contact
from .stress import collect_cad_controls


def run(resolution: int = MAIN_RESOLUTION) -> dict[str, Any]:
    """Compute once per physical case; reuse observations across five splits."""
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    if resolution < 2:
        raise ValueError("Resolution must be at least two")
    files = sorted(Path(__file__).parent.glob("*.py"))
    root = Path(__file__).parents[2]
    files.extend(
        root / path
        for path in (
            "src/brepax/experimental/planar_contact.py",
            "src/brepax/experimental/planar_contact_cad.py",
            "src/brepax/boolean/stratum.py",
            "src/brepax/primitives/_base.py",
            "src/brepax/primitives/box.py",
            "pyproject.toml",
            "uv.lock",
            "src/brepax/_occt/backend.py",
            "benchmarks/__init__.py",
            "benchmarks/contact_transition/baselines.py",
            "benchmarks/contact_transition/measure.py",
        )
    )
    source_before = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    started = perf_counter()
    stages: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("sampled", lambda: collect_sampled(resolution)),
        ("analytic", collect_analytic),
        ("prepared", collect_prepared_contact),
        ("cad", collect_cad),
        ("preparation_controls", collect_preparation_controls),
        ("cad_controls", collect_cad_controls),
        ("grid_controls", collect_grid_controls),
    )
    measurements: dict[str, Any] = {}
    stage_errors = {}
    for name, collect in stages:
        try:
            measurements[name] = collect()
        except Exception as error:
            stage_errors[name] = f"{type(error).__name__}: {error}"
    evaluation: dict[str, Any]
    if stage_errors:
        evaluation = {"assessed": False, "passed": False, "stage_errors": stage_errors}
    else:
        try:
            evaluation = evaluate(
                measurements["sampled"],
                measurements["analytic"],
                measurements["cad"],
                measurements["prepared"],
            )
            evaluation["passed"] &= all(
                measurements[name]["passed"]
                for name in ("preparation_controls", "cad_controls", "grid_controls")
            )
        except Exception as error:
            evaluation = {
                "assessed": False,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
    source_after = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    sources_unchanged = source_before == source_after
    evaluation["passed"] = bool(evaluation["passed"] and sources_unchanged)
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=root
        ).strip(),
        "source_sha256": source_after,
        "source_sha256_before": source_before,
        "sources_unchanged": sources_unchanged,
        "split_seeds": SPLIT_SEEDS,
        **measurements,
        "evaluation": evaluation,
        "total_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=MAIN_RESOLUTION)
    arguments = parser.parse_args()
    result = run(arguments.resolution)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "passed": result["evaluation"]["passed"],
                "seconds": result["total_seconds"],
            }
        )
    )
    raise SystemExit(0 if result["evaluation"]["passed"] else 1)


if __name__ == "__main__":
    main()
