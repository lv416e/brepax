"""Reproduce finite translation profiles, rotation limits, and a CAD edit.

Run ``uv run python -m benchmarks.multi_contact.translation --output /tmp/finite.json
--step-output /tmp/finite.step``. The STEP demonstration uses millimetres.
The deterministic controls establish bounded geometric identities, not method
superiority, generic CAD input coverage, or a runtime improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.face_contact.cad import cad_volume, rotation_matrix, transform_shape
from brepax._occt.backend import (
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepCheck_Analyzer,
    BRepPrimAPI_MakePrism,
    IFSelect_RetDone,
    STEPControl_AsIs,
    STEPControl_Writer,
    gp_Pnt,
    gp_Vec,
)
from brepax._occt.types import TopoDS_Shape
from brepax.experimental.polyhedral_translation import (
    polyhedral_translation_slope,
    polyhedral_translation_volume,
    prepare_polyhedral_translation,
)
from brepax.io.step import read_step

from .finite_references import (
    IDENTITY,
    Matrix3,
    TranslationCase,
    box_vertices,
    clipped_rotation_volume,
    interval_slope,
    interval_volume,
    rotation_volume,
    translation_cases,
)
from .fixtures import Box


@dataclass
class CadObservation:
    """Attempt counts and independent shape/balance checks for one endpoint."""

    volume: float | None = None
    source_valid: bool = False
    common_valid: bool = False
    cut_valid: bool | None = None
    cut_volume: float | None = None
    balance_error: float | None = None
    common_calls: int = 0
    cut_calls: int = 0
    error: str | None = None


def _prism(box: Box, matrix: Matrix3 = IDENTITY) -> TopoDS_Shape:
    linear = np.asarray(matrix)
    lo, hi = np.asarray(box.lower), np.asarray(box.upper)
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])):
        polygon.Add(gp_Pnt(*map(float, linear @ np.array([x, y, lo[2]]))))
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    extrusion = linear @ np.array([0.0, 0.0, hi[2] - lo[2]])
    return BRepPrimAPI_MakePrism(face, gp_Vec(*map(float, extrusion))).Shape()


def _union(shapes: tuple[TopoDS_Shape, ...]) -> TopoDS_Shape:
    result = shapes[0]
    for shape in shapes[1:]:
        operation = BRepAlgoAPI_Fuse(result, shape)
        if not operation.IsDone():
            raise RuntimeError("Source component fusion failed")
        result = operation.Shape()
    return result


def _observe(
    fixed: TopoDS_Shape, moving: TopoDS_Shape, *, cut: bool = False
) -> tuple[CadObservation, TopoDS_Shape | None]:
    observation = CadObservation()
    cut_shape = None
    try:
        observation.source_valid = bool(
            BRepCheck_Analyzer(fixed).IsValid() and BRepCheck_Analyzer(moving).IsValid()
        )
        observation.common_calls += 1
        common = BRepAlgoAPI_Common(fixed, moving)
        if not common.IsDone() or common.Shape().IsNull():
            raise RuntimeError("Common did not produce a shape")
        observation.common_valid = bool(BRepCheck_Analyzer(common.Shape()).IsValid())
        observation.volume = cad_volume(common.Shape())
        if cut:
            observation.cut_calls += 1
            operation = BRepAlgoAPI_Cut(fixed, moving)
            if not operation.IsDone() or operation.Shape().IsNull():
                raise RuntimeError("Cut did not produce a shape")
            cut_shape = operation.Shape()
            observation.cut_valid = bool(BRepCheck_Analyzer(cut_shape).IsValid())
            observation.cut_volume = cad_volume(cut_shape)
            observation.balance_error = (
                cad_volume(fixed) - observation.cut_volume - observation.volume
            )
    except Exception as error:
        observation.error = f"{type(error).__name__}: {error}"
    return observation, cut_shape


def _translation_observation(
    case: TranslationCase, step: float, *, cut: bool = False
) -> tuple[CadObservation, TopoDS_Shape | None]:
    fixed = _union(tuple(_prism(box, case.matrix) for box in case.fixed))
    moving = _union(tuple(_prism(box, case.matrix) for box in case.moving))
    shift = step * np.asarray(case.matrix) @ np.asarray(case.velocity)
    moving = transform_shape(moving, np.eye(3), shift)
    return _observe(fixed, moving, cut=cut)


def measure_translation(case: TranslationCase) -> dict[str, Any]:
    """Compare one finite profile with independent intervals and fresh CAD."""
    prepared = prepare_polyhedral_translation(
        [box_vertices(box, case.matrix) for box in case.fixed],
        [box_vertices(box, case.matrix) for box in case.moving],
        np.asarray(case.matrix) @ np.asarray(case.velocity),
        max_step=case.max_step,
    )
    profile = prepared.profile
    values = np.asarray(
        jax.jit(jax.vmap(polyhedral_translation_volume, in_axes=(None, 0)))(
            profile,
            jnp.asarray(case.steps),
        )
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Finite in-domain steps produced a nonfinite profile value")
    rows = []
    for step, predicted in zip(case.steps, values, strict=True):
        expected = interval_volume(case, step)
        observed, _ = _translation_observation(case, step)
        rows.append(
            {
                "step": step,
                "prediction": float(predicted),
                "reference": expected,
                "cad": asdict(observed),
                "passed": bool(
                    np.isclose(predicted, expected, rtol=0, atol=1e-11)
                    and observed.error is None
                    and observed.source_valid
                    and observed.common_valid
                    and observed.volume is not None
                    and np.isclose(observed.volume, expected, rtol=0, atol=1e-10)
                ),
            }
        )
    slopes = []
    determinant = abs(float(np.linalg.det(np.asarray(case.matrix))))
    sides: tuple[Literal["left", "right"], ...] = ("left", "right")
    for step, left, right in case.slope_events:
        actual = [
            float(polyhedral_translation_slope(profile, step, side=side))
            for side in sides
        ]
        if not np.isfinite(actual).all():
            raise RuntimeError("An internal event produced a nonfinite one-sided slope")
        expected_slopes = [left * determinant, right * determinant]
        slopes.append(
            {
                "step": step,
                "actual": actual,
                "reference": expected_slopes,
                "passed": bool(
                    np.allclose(actual, expected_slopes, rtol=0, atol=1e-11)
                ),
            }
        )
    interior = []
    gradient = jax.jit(jax.grad(polyhedral_translation_volume, argnums=1))
    for lo, hi in zip(prepared.exact_knots[:-1], prepared.exact_knots[1:], strict=True):
        midpoint = float((lo + hi) / 2)
        actual_slope = float(gradient(profile, midpoint))
        if not np.isfinite(actual_slope):
            raise RuntimeError("An open chamber produced a nonfinite AD derivative")
        expected = interval_slope(case, midpoint)
        interior.append(
            {
                "step": midpoint,
                "actual": actual_slope,
                "reference": expected,
                "passed": bool(np.isclose(actual_slope, expected, rtol=0, atol=1e-11)),
            }
        )
    return {
        "case": case.name,
        "knots": [str(value) for value in prepared.exact_knots],
        "pair_count": prepared.pair_count,
        "values": rows,
        "event_slopes": slopes,
        "interior_ad": interior,
        "passed": all(row["passed"] for row in rows + slopes + interior),
    }


def measure_rotation() -> dict[str, Any]:
    """Retain a path-dependent second term and a finite chamber transition."""
    event = float(np.arctan(0.1 / 0.5))
    configurations = [
        (1.0, q, t) for q in (0.0, 0.5, 1.0) for t in (0.2, 0.1, 0.05, 0.025, 0.0125)
    ]
    configurations += [(0.1, 0.0, t) for t in (event * 0.8, event, event * 1.2, 0.4)]
    rows = []
    for width, pivot_x, step in configurations:
        fixed = _prism(Box((-3.0, -3.0, 0.0), (0.0, 3.0, 1.0)))
        moving = _prism(Box((0.0, -0.5, 0.0), (width, 0.5, 1.0)))
        rotation = rotation_matrix(np.array([0.0, 0.0, step]))
        pivot = np.array([pivot_x, 0.0, 0.0])
        shift = pivot - rotation @ pivot + step * np.array([0.0, pivot_x, 0.0])
        observed, _ = _observe(fixed, transform_shape(moving, rotation, shift))
        expected = rotation_volume(step, width, 0.5, 1.0, pivot_x)
        polygon = clipped_rotation_volume(step, width, 0.5, 1.0, pivot_x)
        rows.append(
            {
                "width": width,
                "pivot_x": pivot_x,
                "step": step,
                "reference": expected,
                "polygon_reference": polygon,
                "cad": asdict(observed),
                "second_coefficient": -0.25 * pivot_x,
                "normalized_linear_residual": (expected - 0.125 * step) / step**2,
                "uncapped_zero_pivot_extension": float(0.125 * np.tan(step))
                if pivot_x == 0
                else None,
                "passed": bool(
                    observed.error is None
                    and observed.source_valid
                    and observed.common_valid
                    and observed.volume is not None
                    and np.isclose(observed.volume, expected, rtol=0, atol=1e-10)
                    and np.isclose(polygon, expected, rtol=0, atol=1e-12)
                ),
            }
        )
    return {
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "thickness_event": event,
        "path": "p + R_z(t)(x-p) + t v; p=(q,0,0), v=(0,q,0)",
        "scope": (
            "Same initial velocity does not imply the same finite path "
            "or quadratic term"
        ),
    }


def target_demo(step_output: Path | None = None) -> dict[str, Any]:
    """Invert the proved monotone corner branch, then verify a real finite Cut.

    On [0,.5], the independent corner formula is 2t-t^2 with positive slope.
    Bisection here is a bounded demonstration, not a generic monotonicity claim.
    Profile queries use no CAD; the final validation uses one Common and one Cut.
    """
    case = next(case for case in translation_cases() if case.name == "corner")
    prepared = prepare_polyhedral_translation(
        [box_vertices(box) for box in case.fixed],
        [box_vertices(box) for box in case.moving],
        np.asarray(case.velocity),
        max_step=0.5,
    )
    value = jax.jit(polyhedral_translation_volume)
    target, lower, upper = 0.2, 0.0, 0.5
    for _ in range(48):
        midpoint = (lower + upper) / 2
        if float(value(prepared.profile, midpoint)) < target:
            lower = midpoint
        else:
            upper = midpoint
    step = (lower + upper) / 2
    observed, shape = _translation_observation(case, step, cut=True)
    passed = bool(
        observed.error is None
        and observed.source_valid
        and observed.common_valid
        and observed.cut_valid
        and observed.volume is not None
        and shape is not None
        and observed.balance_error is not None
        and abs(observed.balance_error) < 1e-10
        and abs(observed.volume - target) < 1e-10
        and abs(interval_volume(case, step) - target) < 1e-12
    )
    result: dict[str, Any] = {
        "passed": passed,
        "target": target,
        "step": step,
        "analytic_step": float(1.0 - np.sqrt(1.0 - target)),
        "profile_selection_queries": 48,
        "cad": asdict(observed),
        "relative_target_error": abs(observed.volume - target) / target
        if observed.volume is not None
        else None,
    }
    if (
        step_output is not None
        and passed
        and shape is not None
        and observed.cut_volume is not None
    ):
        step_output.parent.mkdir(parents=True, exist_ok=True)
        writer = STEPControl_Writer()
        writer.Model().SetLocalLengthUnit(1.0)
        writer.Model().SetWriteLengthUnit(1.0)
        if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
            raise RuntimeError("STEP transfer failed")
        if writer.Write(str(step_output)) != IFSelect_RetDone:
            raise RuntimeError("STEP write failed")
        restored = read_step(step_output)
        restored_volume = cad_volume(restored)
        roundtrip = bool(BRepCheck_Analyzer(restored).IsValid()) and bool(
            np.isclose(
                restored_volume,
                observed.cut_volume,
                rtol=0,
                atol=1e-10,
            )
        )
        result["step_artifact"] = {
            "path": str(step_output),
            "working_length_unit": "millimetre",
            "sha256": hashlib.sha256(step_output.read_bytes()).hexdigest(),
            "roundtrip_volume": restored_volume,
            "passed": roundtrip,
        }
        result["passed"] = passed and roundtrip
    return result


def _snapshot() -> dict[str, str]:
    root = Path(__file__).parents[2]
    names = (
        "benchmarks/multi_contact/translation.py",
        "benchmarks/multi_contact/finite_references.py",
        "benchmarks/face_contact/cad.py",
        "src/brepax/experimental/polyhedral_translation.py",
        "src/brepax/experimental/_polyhedral_translation.py",
        "src/brepax/experimental/_polyhedral_contact.py",
        "src/brepax/_occt/backend.py",
        "src/brepax/_occt/types.py",
        "src/brepax/io/step.py",
        "uv.lock",
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }


def run(*, step_output: Path | None = None) -> dict[str, Any]:
    """Keep each attempted control and every stage failure in the run artifact."""
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    before = _snapshot()
    started = perf_counter()
    rows = []
    for case in translation_cases():
        try:
            rows.append(measure_translation(case))
        except Exception as error:
            rows.append(
                {
                    "case": case.name,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    stages = {}
    for name, operation in (
        ("rotation", measure_rotation),
        ("demo", lambda: target_demo(step_output)),
    ):
        try:
            stages[name] = operation()
        except Exception as error:
            stages[name] = {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
    after = _snapshot()
    return {
        "passed": all(row["passed"] for row in rows)
        and all(stage["passed"] for stage in stages.values())
        and before == after,
        "scope": (
            "Specified finite translation controls and bounded rotation identities; "
            "no generic CAD or speed guarantee"
        ),
        "translation": rows,
        **stages,
        "sources_unchanged": before == after,
        "source_sha256": after,
        "source_sha256_before": before,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("jax", "numpy", "cadquery-ocp-novtk")
            },
        },
        "seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-output", type=Path)
    args = parser.parse_args()
    if (
        args.step_output is not None
        and args.step_output.resolve() == args.output.resolve()
    ):
        parser.error("JSON and STEP outputs must differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(step_output=args.step_output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(args.output),
                "seconds": result["seconds"],
            }
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
