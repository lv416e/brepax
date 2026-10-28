"""Reproduce multiple-contact selection, actual CAD cuts, and a STEP artifact.

Run ``uv run --group benchmark python -m benchmarks.multi_contact.run
--output /tmp/multi-contact.json --step-output /tmp/cutter-edit.step``.
The optional STEP artifact assigns millimetres to the demo's working lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import numpy as np

from benchmarks.face_contact.cad import cad_volume
from brepax._occt.backend import (
    BRepCheck_Analyzer,
    IFSelect_RetDone,
    STEPControl_AsIs,
    STEPControl_Writer,
)
from brepax.io.step import read_step

from .cad import apply_cut
from .evaluation import evaluate
from .experiment import collect_geometry, validate_selections
from .fixtures import candidate_motions, fixtures
from .sampling import collect_sampled


def _snapshot(root: Path) -> dict[str, str]:
    files = sorted((root / "src/brepax").rglob("*.py"))
    files.extend(sorted((root / "benchmarks").rglob("*.py")))
    files.extend(root / name for name in ("pyproject.toml", "uv.lock"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise ValueError(f"Output must name a regular file: {path}")


def _export_demo(
    path: Path, evaluation: dict[str, Any], variants: int
) -> dict[str, Any]:
    selected = next(
        record
        for record in evaluation["selections"]
        if record["method"] == "full_contact"
        and record["objective"] == "target"
        and record["placement_name"] == "identity"
    )
    fixture = next(
        item for item in fixtures(variants) if item.group_id == selected["group_id"]
    )
    motion = candidate_motions(fixture)[selected["direction"]].motion
    edited = apply_cut(fixture, motion, selected["step"])
    measurement = edited.measurement
    if (
        measurement.error is not None
        or not measurement.source_valid
        or not measurement.valid
        or not measurement.cut_valid
        or edited.shape is None
        or measurement.cut_volume is None
        or measurement.working_balance_error is None
        or abs(measurement.working_balance_error) > selected["target_volume"] * 1e-7
    ):
        raise RuntimeError(f"Demonstration Cut failed verification: {measurement}")
    writer = STEPControl_Writer()
    # Assign an explicit unit to this generated example, without changing
    # process-global OCCT settings or pretending to infer imported STEP units.
    writer.Model().SetLocalLengthUnit(1.0)
    writer.Model().SetWriteLengthUnit(1.0)
    if writer.Transfer(edited.shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError("STEP transfer failed")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError("STEP write failed")
    restored = read_step(path)
    restored_volume = cad_volume(restored)
    tolerance = selected["target_volume"] * 1e-7
    if not BRepCheck_Analyzer(restored).IsValid() or not np.isclose(
        restored_volume, measurement.cut_volume, rtol=0, atol=tolerance
    ):
        raise RuntimeError("STEP roundtrip changed the verified cut volume or validity")
    return {
        "passed": True,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "working_length_unit": "millimetre",
        "selection": selected,
        "measurement": asdict(measurement),
        "roundtrip_volume": restored_volume,
        "roundtrip_absolute_tolerance": tolerance,
        "scope": "A valid finite cut; target error remains as measured",
    }


def run(
    variants: int = 5, resolution: int = 48, *, step_output: Path | None = None
) -> dict[str, Any]:
    """Measure, calibrate on training groups, and validate selected real edits."""
    if variants < 2 or resolution < 2:
        raise ValueError("Use at least two variants per family and grid resolution two")
    if step_output is not None:
        _output_path(step_output)
    root = Path(__file__).parents[2]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "equinox", "cadquery-ocp-novtk")
        },
    }
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    before = _snapshot(root)
    started = perf_counter()
    stages: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("geometry", lambda: collect_geometry(variants)),
        ("sampled", lambda: collect_sampled(variants, resolution)),
    )
    measurements: dict[str, Any] = {}
    errors = {}
    for name, function in stages:
        try:
            measurements[name] = function()
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    evaluation: dict[str, Any] = {"passed": False, "assessed": False}
    edits: dict[str, Any] = {"passed": False, "assessed": False}
    artifact: dict[str, Any] | None = None
    if not errors:
        try:
            evaluation = evaluate(measurements["geometry"], measurements["sampled"])
            if evaluation["passed"]:
                edits = validate_selections(evaluation["selections"], variants=variants)
                if edits["passed"] and step_output is not None:
                    artifact = _export_demo(step_output, evaluation, variants)
        except Exception as error:
            errors["evaluation_or_edit"] = f"{type(error).__name__}: {error}"
    try:
        after = _snapshot(root)
    except Exception as error:
        errors["source_snapshot"] = f"{type(error).__name__}: {error}"
        after = {}
    unchanged = before == after
    return {
        "passed": bool(
            not errors and evaluation["passed"] and edits["passed"] and unchanged
        ),
        "passed_scope": (
            "Complete specified design-family comparison and verified finite edits. "
            "Source-intake rejections and target-volume errors remain explicit."
        ),
        "head": head,
        "environment": environment,
        "variants_per_family": variants,
        "resolution": resolution,
        "source_sha256_before": before,
        "source_sha256": after,
        "sources_unchanged": unchanged,
        "stage_errors": errors,
        **measurements,
        "evaluation": evaluation,
        "edits": edits,
        "step_artifact": artifact,
        "seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-output", type=Path)
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=48)
    args = parser.parse_args()
    _output_path(args.output)
    if args.step_output is not None:
        _output_path(args.step_output)
        if args.step_output.resolve() == args.output.resolve():
            parser.error("JSON and STEP outputs must have different paths")
    result = run(args.variants, args.resolution, step_output=args.step_output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "seconds": result["seconds"],
                "summary": result["evaluation"].get("summary"),
            }
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
