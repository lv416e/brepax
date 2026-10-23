"""Validate fixed-frame dimensional CAD editing against independent references.

Run from the repository root:
    uv run --group benchmark python -m benchmarks.cad_editing.run --output RESULT.json

The default deflection is 1e-4 working units. Use --deflection to investigate
convergence; a run outside the declared error budget exits nonzero and retains
its failed cases. Setup and mesh accuracy are not a performance claim. The
three unit conventions reuse one fixed working mesh, not independent remeshes.
Existing STEP operands, topology changes, and generic CAD repair are out of scope.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from brepax.experimental.applications.cad_frame import CadFrame

from .geometry import (
    BASE_DIMENSIONS,
    EDIT_POINTS,
    LENGTH,
    WIDTH,
    bottom_landmark_error,
    cad_measures,
    make_shape,
    reference_jacobian,
    reference_measures,
)
from .mesh import check_mesh, prepare_mesh

FD_STEPS = (1e-3, 1e-4, 1e-5)


def relative_error(actual: NDArray[Any], expected: NDArray[Any]) -> float:
    """Reject nonfinite evidence so failed cases remain JSON-serializable."""
    error = float(np.max(abs(actual / expected - 1)))
    if not np.isfinite(error):
        raise FloatingPointError("Nonfinite reference comparison")
    return error


def restored_measures(
    dimensions: NDArray[np.float64], frame: CadFrame
) -> NDArray[np.float64]:
    """Rebuild CAD from normalized dimensions, then measure the restored solid."""
    working = np.asarray(frame.working_lengths(jnp.asarray(dimensions)))
    return cad_measures(frame.restore_shape(make_shape(working)))


def finite_differences(
    dimensions: NDArray[np.float64], frame: CadFrame
) -> list[dict[str, float]]:
    """Independent rebuilt-CAD derivatives, retaining every relative step."""
    scale = frame.length_scale
    expected = reference_jacobian(dimensions / scale)
    rows = []
    for relative_step in FD_STEPS:
        columns = []
        for i, direction in enumerate(np.eye(2)):
            h = relative_step * scale * BASE_DIMENSIONS[i]
            plus = restored_measures(dimensions + h * direction, frame)
            minus = restored_measures(dimensions - h * direction, frame)
            columns.append((plus - minus) / (2 * h))
        derivative = np.stack(columns, axis=1) / np.array([scale**2, scale])[:, None]
        rows.append(
            {
                "relative_step": relative_step,
                "jacobian_relative_error": relative_error(derivative, expected),
            }
        )
    return rows


def run(deflection: float) -> dict[str, Any]:
    """Evaluate nine unit/shape cases and one large-origin placement control."""
    report: dict[str, Any] = {
        "settings": {
            "required_dtype": "float64",
            "units": [0.001, 1.0, 1000.0],
            "dimensions": EDIT_POINTS.tolist(),
            "fd_steps": FD_STEPS,
            "acceptance": (
                "mesh value/Jacobian <=1e-4; restored CAD value <=1e-7; "
                "best rebuilt-CAD FD Jacobian <=1e-6; "
                "landmark error <=1e-6 working units; "
                "closed nondegenerate unflipped mesh"
            ),
            "scope": (
                "One supplied fixed-topology family; shared working mesh across units; "
                "no imported-STEP healing or performance claim"
            ),
        },
        "cases": [],
    }
    try:
        if jnp.asarray(0.0).dtype != jnp.float64:
            raise ValueError("The CAD comparison requires jax_enable_x64=True")
        model, metadata = prepare_mesh(deflection)
        report["mesh"] = metadata
    except Exception as error:
        report.update(passed=False, setup_error=f"{type(error).__name__}: {error}")
        return report
    # Reuse each closure across all edits in its fixed frame, not per evaluation.
    frames = [CadFrame(s, (3 * s, -2 * s, 4 * s)) for s in (0.001, 1.0, 1000.0)]
    frames.append(CadFrame(0.001, (1000.0, -2000.0, 3000.0)))
    for index, frame in enumerate(frames):
        measures = partial(model.external_measures, frame=frame)
        evaluate = jax.jit(measures)
        derivative = jax.jit(jax.jacrev(measures))
        points = EDIT_POINTS if index < 3 else EDIT_POINTS[-1:]
        for dimensions in points:
            row: dict[str, Any] = {
                "length_scale": frame.length_scale,
                "origin": frame.origin,
                "working_dimensions": dimensions.tolist(),
            }
            try:
                scale = frame.length_scale
                external = scale * dimensions
                expected, expected_jac = (
                    reference_measures(dimensions),
                    reference_jacobian(dimensions),
                )
                values = np.asarray(evaluate(external)) / np.array([scale**3, scale**2])
                jacobian = (
                    np.asarray(derivative(external))
                    / np.array([scale**2, scale])[:, None]
                )
                working = np.asarray(frame.working_lengths(jnp.asarray(external)))
                shape = frame.restore_shape(make_shape(working))
                cad_values = cad_measures(shape) / np.array([scale**3, scale**2])
                corners = np.array(
                    [[0, 0, 0], [LENGTH, 0, 0], [0, WIDTH, 0], [LENGTH, WIDTH, 0]]
                )
                landmarks = np.array(frame.origin) + scale * corners
                fd = finite_differences(external, frame)
                row.update(
                    mesh_value_error=relative_error(values, expected),
                    mesh_jacobian_error=relative_error(jacobian, expected_jac),
                    restored_cad_value_error=relative_error(cad_values, expected),
                    landmark_error=bottom_landmark_error(shape, landmarks, scale),
                    mesh_valid=check_mesh(model, working),
                    finite_difference=fd,
                )
                row["passed"] = bool(
                    row["mesh_valid"]
                    and row["mesh_value_error"] <= 1e-4
                    and row["mesh_jacobian_error"] <= 1e-4
                    and row["restored_cad_value_error"] <= 1e-7
                    and row["landmark_error"] <= 1e-6
                    and min(item["jacobian_relative_error"] for item in fd) <= 1e-6
                )
            except Exception as error:
                # An experiment case must remain visible when CAD or extraction fails.
                row.update(passed=False, error=f"{type(error).__name__}: {error}")
            report["cases"].append(row)
    report["passed"] = all(row["passed"] for row in report["cases"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deflection", type=float, default=1e-4)
    args = parser.parse_args()
    # JAX exposes config.update without a return annotation.
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    report = run(args.deflection)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "output": str(args.output),
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
