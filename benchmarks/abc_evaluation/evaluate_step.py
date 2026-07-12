"""Evaluate a single STEP file through the BRepAX benchmark pipeline.

Reads one STEP file, runs the same five volume paths and four
face-level metrics that ``benchmarks/integration_report/run_benchmark.py``
uses, and prints a compact result table.  Designed as the reusable
building block for large-scale evaluation (M7a: 20+ external models).

Run:
    uv run python -m benchmarks.abc_evaluation.evaluate_step path/to/file.step

Scope:
    No new algorithms.  Orchestrates the existing public APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from brepax._occt.backend import TopAbs_SOLID
from brepax._occt.types import TopoDS_Shape
from brepax.brep.convert import shape_metadata
from brepax.brep.csg_eval import integrate_sdf_volume, make_grid_3d
from brepax.brep.csg_stump import (
    reconstruct_csg_stump,
    stump_to_differentiable,
)
from brepax.brep.gprop import compute_gprop_ground_truth
from brepax.brep.mesh_sdf import mesh_sdf
from brepax.brep.triangulate import divergence_volume, triangulate_shape
from brepax.brep.trimmed_csg_stump import enrich_with_trim_frames
from brepax.brep.winding import winding_number
from brepax.io.step import read_step
from brepax.metrics import (
    mean_curvature_per_face,
    min_draft_angle_per_face,
    min_wall_thickness_per_face,
    surface_area_per_face,
)

VOLUME_RESOLUTION = 32
GWN_SHARPNESS = 32.0
GWN_CHUNK_SIZE = 1024
MOLD_DIRECTION = jnp.array([0.0, 0.0, 1.0])


@dataclass
class PathResult:
    path: str
    value: float | None
    err_pct: float | None
    elapsed_s: float
    note: str


@dataclass
class MetricResult:
    metric: str
    n_faces: int
    n_finite: int
    n_nan: int
    n_inf: int
    note: str = ""


@dataclass
class ModelResult:
    file: str
    shape_type: str
    occt_volume: float
    occt_surface_area: float
    n_faces: int
    volume_paths: list[PathResult]
    face_metrics: list[MetricResult]


def _safe(fn: Callable[[], Any]) -> tuple[float | None, float, str]:
    t0 = time.perf_counter()
    try:
        v = float(fn())
        return v, time.perf_counter() - t0, ""
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
        return None, time.perf_counter() - t0, msg


def _shape_grid_bounds(shape: TopoDS_Shape) -> tuple[jnp.ndarray, jnp.ndarray]:
    meta = shape_metadata(shape)
    margin = 0.5
    return jnp.array(meta.bbox_min) - margin, jnp.array(meta.bbox_max) + margin


def _evaluate_volume_paths(shape: TopoDS_Shape, *, resolution: int) -> list[PathResult]:
    results: list[PathResult] = []

    # Triangulate once (shared by divergence, mesh_sdf, gwn)
    t0 = time.perf_counter()
    try:
        triangles, _ = triangulate_shape(shape)
    except Exception as exc:
        note = f"triangulate: {type(exc).__name__}"
        for name in ("divergence", "mesh_sdf", "gwn"):
            results.append(PathResult(name, None, None, time.perf_counter() - t0, note))
        # CSG paths don't need triangles
        triangles = None
    else:
        if triangles.shape[0] == 0:
            triangles = None

    lo, hi = _shape_grid_bounds(shape)

    # divergence
    if triangles is not None:
        v, elapsed, note = _safe(lambda: divergence_volume(triangles))
        results.append(PathResult("divergence", v, None, elapsed, note))
    elif not any(r.path == "divergence" for r in results):
        results.append(PathResult("divergence", None, None, 0.0, "no triangles"))

    # mesh_sdf
    if triangles is not None:

        def _mesh_sdf_vol() -> jnp.ndarray:
            grid, _ = make_grid_3d(lo, hi, resolution)
            flat = grid.reshape(-1, 3)
            sdf_vals = mesh_sdf(flat, triangles)
            return integrate_sdf_volume(sdf_vals, lo, hi, resolution)

        v, elapsed, note = _safe(_mesh_sdf_vol)
        results.append(PathResult("mesh_sdf", v, None, elapsed, note))
    elif not any(r.path == "mesh_sdf" for r in results):
        results.append(PathResult("mesh_sdf", None, None, 0.0, "no triangles"))

    # gwn
    if triangles is not None:

        def _gwn_vol() -> jnp.ndarray:
            grid, cell_vol = make_grid_3d(lo, hi, resolution)
            flat = grid.reshape(-1, 3)
            n = flat.shape[0]
            rem = n % GWN_CHUNK_SIZE
            pad = 0 if rem == 0 else GWN_CHUNK_SIZE - rem
            padded = jnp.pad(flat, ((0, pad), (0, 0)))
            chunks = padded.reshape(-1, GWN_CHUNK_SIZE, 3)

            def _chunk(c: jnp.ndarray) -> jnp.ndarray:
                return jax.vmap(lambda p: winding_number(p, triangles))(c)

            wn = jax.lax.map(_chunk, chunks).reshape(-1)[:n]
            indicator = jax.nn.sigmoid((wn - 0.5) * GWN_SHARPNESS)
            return jnp.sum(indicator) * cell_vol

        v, elapsed, note = _safe(_gwn_vol)
        results.append(PathResult("gwn", v, None, elapsed, note))
    elif not any(r.path == "gwn" for r in results):
        results.append(PathResult("gwn", None, None, 0.0, "no triangles"))

    # CSG-Stump
    t0_csg = time.perf_counter()
    try:
        stump = reconstruct_csg_stump(shape)
    except Exception as exc:
        stump = None
        csg_note = f"reconstruct: {type(exc).__name__}"
    else:
        csg_note = "" if stump is not None else "reconstruct: None"

    if stump is not None:
        diff = stump_to_differentiable(stump)
        v, _, note = _safe(lambda: diff.volume(resolution=resolution, lo=lo, hi=hi))
        results.append(
            PathResult("csg_stump", v, None, time.perf_counter() - t0_csg, note)
        )
    else:
        results.append(
            PathResult("csg_stump", None, None, time.perf_counter() - t0_csg, csg_note)
        )

    # TrimmedCSG
    if stump is not None:
        t0_trim = time.perf_counter()
        try:
            trimmed = enrich_with_trim_frames(stump, shape)
            v, _, note = _safe(
                lambda: trimmed.volume(resolution=resolution, lo=lo, hi=hi)
            )
            results.append(
                PathResult("trimmed_csg", v, None, time.perf_counter() - t0_trim, note)
            )
        except Exception as exc:
            results.append(
                PathResult(
                    "trimmed_csg",
                    None,
                    None,
                    time.perf_counter() - t0_trim,
                    f"enrich: {type(exc).__name__}",
                )
            )
    else:
        results.append(PathResult("trimmed_csg", None, None, 0.0, csg_note))

    return results


def _evaluate_face_metrics(shape: TopoDS_Shape) -> list[MetricResult]:
    results: list[MetricResult] = []
    metrics: dict[str, Callable] = {
        "surface_area": lambda s: surface_area_per_face(s),
        "min_draft_angle": lambda s: min_draft_angle_per_face(
            s, mold_direction=MOLD_DIRECTION
        ),
        "mean_curvature": lambda s: mean_curvature_per_face(s),
        "min_wall_thickness": lambda s: min_wall_thickness_per_face(s),
    }
    for name, fn in metrics.items():
        try:
            values, _ = fn(shape)
            n = int(values.shape[0])
            n_nan = int(jnp.sum(jnp.isnan(values)))
            n_inf = int(jnp.sum(jnp.isinf(values)))
            n_finite = n - n_nan - n_inf
            results.append(MetricResult(name, n, n_finite, n_nan, n_inf))
        except Exception as exc:
            results.append(
                MetricResult(name, 0, 0, 0, 0, f"ERROR: {type(exc).__name__}")
            )
    return results


def evaluate_step_file(
    path: str | Path, *, resolution: int = VOLUME_RESOLUTION
) -> ModelResult:
    path = Path(path)
    shape = read_step(str(path))
    shape_type = shape.ShapeType().name.split("_")[-1].lower()

    gt = compute_gprop_ground_truth(shape)
    occt_volume = float(gt["volume"])
    occt_area = float(gt["surface_area"])

    is_solid = shape.ShapeType() == TopAbs_SOLID
    if is_solid:
        vol_results = _evaluate_volume_paths(shape, resolution=resolution)
    else:
        note = f"{shape_type} (volume paths require closed solid)"
        vol_results = [
            PathResult(p, None, None, 0.0, note)
            for p in ("divergence", "mesh_sdf", "gwn", "csg_stump", "trimmed_csg")
        ]

    # Compute err_pct for each volume path
    for r in vol_results:
        if r.value is not None and abs(occt_volume) > 1e-6:
            r.err_pct = abs(r.value - occt_volume) / abs(occt_volume) * 100

    face_results = _evaluate_face_metrics(shape)
    n_faces = max((m.n_faces for m in face_results), default=0)

    return ModelResult(
        file=str(path),
        shape_type=shape_type,
        occt_volume=occt_volume,
        occt_surface_area=occt_area,
        n_faces=n_faces,
        volume_paths=vol_results,
        face_metrics=face_results,
    )


def render_text(result: ModelResult) -> str:
    lines: list[str] = []
    lines.append(f"File: {result.file}")
    lines.append(f"Shape type: {result.shape_type}")
    lines.append(f"OCCT volume: {result.occt_volume:.4f}")
    lines.append(f"OCCT surface area: {result.occt_surface_area:.4f}")
    lines.append(f"Faces: {result.n_faces}")
    lines.append("")
    lines.append("Volume paths:")
    lines.append(f"  {'path':<14} {'value':>12} {'err%':>8} {'time':>8}  notes")
    lines.append(f"  {'-' * 14} {'-' * 12} {'-' * 8} {'-' * 8}  -----")
    for r in result.volume_paths:
        val = f"{r.value:.4f}" if r.value is not None else "—"
        err = f"{r.err_pct:.2f}%" if r.err_pct is not None else "—"
        lines.append(
            f"  {r.path:<14} {val:>12} {err:>8} {r.elapsed_s:>7.2f}s  {r.note}"
        )
    lines.append("")
    lines.append("Face-level metrics:")
    lines.append(
        f"  {'metric':<20} {'n_faces':>7} {'finite':>6} {'nan':>5} {'inf':>5}  notes"
    )
    lines.append(f"  {'-' * 20} {'-' * 7} {'-' * 6} {'-' * 5} {'-' * 5}  -----")
    for m in result.face_metrics:
        row = (
            f"  {m.metric:<20} {m.n_faces:>7}"
            f" {m.n_finite:>6} {m.n_nan:>5} {m.n_inf:>5}"
            f"  {m.note}"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single STEP file through the BRepAX pipeline."
    )
    parser.add_argument("step_file", type=Path, help="Path to a STEP file")
    parser.add_argument(
        "--resolution",
        type=int,
        default=VOLUME_RESOLUTION,
        help="Per-axis grid resolution for sigmoid integrators (default 32)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of text table",
    )
    args = parser.parse_args()

    if not args.step_file.exists():
        print(f"File not found: {args.step_file}", file=sys.stderr)
        sys.exit(1)

    print(f"[eval] {args.step_file}", file=sys.stderr)
    result = evaluate_step_file(args.step_file, resolution=args.resolution)

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        print(render_text(result))


if __name__ == "__main__":
    main()
