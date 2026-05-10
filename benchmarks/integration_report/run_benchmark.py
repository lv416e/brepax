"""Reproducible integration benchmark report.

Measures the BRepAX volume paths (mesh divergence theorem, CSG-Stump
DNF grid integration, trimmed CSG-Stump grid integration) against
OCCT BRepGProp on the project's standard STEP fixture set, and reports
the per-fixture coverage of the four trim-aware face-level metrics
shipped in PR #81-#84.

Run:
    uv run python -m benchmarks.integration_report.run_benchmark

Output:
    benchmarks/integration_report/REPORT.md (overwritten in place)

Scope:
    No new algorithms.  No CSG/GWN/Marschner changes.  Only orchestrates
    existing public APIs and writes a markdown table.
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from brepax._occt.backend import TopAbs_SOLID
from brepax._occt.types import TopoDS_Shape
from brepax.brep.convert import shape_metadata
from brepax.brep.csg_stump import (
    CSGStump,
    reconstruct_csg_stump,
    stump_to_differentiable,
)
from brepax.brep.gprop import compute_gprop_ground_truth
from brepax.brep.triangulate import divergence_volume, triangulate_shape
from brepax.brep.trimmed_csg_stump import enrich_with_trim_frames
from brepax.io.step import read_step
from brepax.metrics import (
    mean_curvature_per_face,
    min_draft_angle_per_face,
    min_wall_thickness_per_face,
    surface_area_per_face,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"
COMPETITOR_PATH = Path(__file__).resolve().parent / "competitor_landscape.md"

FIXTURES: tuple[str, ...] = (
    "sample_box",
    "sample_cylinder",
    "sample_sphere",
    "sample_cone",
    "sample_torus",
    "box_with_holes",
    "box_with_pocket",
    "box_with_slot",
    "l_bracket",
    "nurbs_box",
    "nurbs_revol",
    "nurbs_saddle",
)

VOLUME_RESOLUTION = 32
MOLD_DIRECTION = jnp.array([0.0, 0.0, 1.0])


@dataclass
class VolumeResult:
    value: float | None
    err_pct: float | None
    elapsed_s: float
    note: str


@dataclass
class CoverageCell:
    n_faces: int
    n_finite: int
    n_nan: int
    n_inf: int
    note: str = ""


def _measure(fn: Callable[[], Any]) -> tuple[float | None, float, str]:
    t0 = time.perf_counter()
    try:
        v = float(fn())
        return v, time.perf_counter() - t0, ""
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
        return None, time.perf_counter() - t0, msg


def measure_divergence(shape: TopoDS_Shape) -> VolumeResult:
    """Time the full STEP-to-volume path: triangulate + integrate.

    Wrapping the triangulation step in the same timer as the
    integration keeps the per-row time comparable to the CSG paths,
    which include their reconstruct cost.
    """
    t0 = time.perf_counter()
    try:
        triangles, _ = triangulate_shape(shape)
    except Exception as exc:
        return VolumeResult(
            None,
            None,
            time.perf_counter() - t0,
            f"triangulate: {type(exc).__name__}",
        )
    v, _, note = _measure(lambda: divergence_volume(triangles))
    return VolumeResult(v, None, time.perf_counter() - t0, note)


def _shape_grid_bounds(shape: TopoDS_Shape) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the OCCT shape bbox padded by 0.5 on every side.

    Both ``DifferentiableCSGStump.volume`` and ``TrimmedCSGStump.volume``
    accept explicit ``lo`` / ``hi``; passing the same bounds to both
    eliminates the auto-bbox divergence (``DifferentiableCSGStump``
    derives bounds from primitives, which is unreliable for unbounded
    primitives like ``Plane``; ``TrimmedCSGStump`` uses the stump's
    stored bbox).
    """
    meta = shape_metadata(shape)
    margin = 0.5
    lo = jnp.array(meta.bbox_min) - margin
    hi = jnp.array(meta.bbox_max) + margin
    return lo, hi


@dataclass
class _StumpHandle:
    """Cached output of ``reconstruct_csg_stump``.

    ``stump`` is ``None`` when reconstruction returned ``None`` or
    raised; in that case ``note`` carries the reason.  ``elapsed_s`` is
    the cost paid once for the shape and is added to *both* the CSG
    and Trimmed-CSG measurement times so each row reflects the full
    STEP-to-volume cost.
    """

    stump: CSGStump | None
    elapsed_s: float
    note: str


def reconstruct_stump_cached(shape: TopoDS_Shape) -> _StumpHandle:
    t0 = time.perf_counter()
    try:
        stump = reconstruct_csg_stump(shape)
    except Exception as exc:
        return _StumpHandle(
            None,
            time.perf_counter() - t0,
            f"reconstruct: {type(exc).__name__}",
        )
    elapsed = time.perf_counter() - t0
    if stump is None:
        return _StumpHandle(None, elapsed, "reconstruct: None")
    return _StumpHandle(stump, elapsed, "")


def measure_csg_stump(
    shape: TopoDS_Shape,
    *,
    resolution: int,
    handle: _StumpHandle,
) -> VolumeResult:
    if handle.stump is None:
        return VolumeResult(None, None, handle.elapsed_s, handle.note)
    t0 = time.perf_counter()
    diff = stump_to_differentiable(handle.stump)
    lo, hi = _shape_grid_bounds(shape)
    v, _, note = _measure(lambda: diff.volume(resolution=resolution, lo=lo, hi=hi))
    return VolumeResult(v, None, handle.elapsed_s + (time.perf_counter() - t0), note)


def measure_trimmed_csg_stump(
    shape: TopoDS_Shape,
    *,
    resolution: int,
    handle: _StumpHandle,
) -> VolumeResult:
    if handle.stump is None:
        return VolumeResult(None, None, handle.elapsed_s, handle.note)
    t0 = time.perf_counter()
    try:
        trimmed = enrich_with_trim_frames(handle.stump, shape)
    except Exception as exc:
        return VolumeResult(
            None,
            None,
            handle.elapsed_s + (time.perf_counter() - t0),
            f"enrich: {type(exc).__name__}",
        )
    lo, hi = _shape_grid_bounds(shape)
    v, _, note = _measure(lambda: trimmed.volume(resolution=resolution, lo=lo, hi=hi))
    return VolumeResult(v, None, handle.elapsed_s + (time.perf_counter() - t0), note)


def _coverage_cell(values: jnp.ndarray) -> CoverageCell:
    n = int(values.shape[0])
    n_nan = int(jnp.sum(jnp.isnan(values)))
    n_inf = int(jnp.sum(jnp.isinf(values)))
    n_finite = n - n_nan - n_inf
    return CoverageCell(n, n_finite, n_nan, n_inf)


def measure_face_coverage(shape: TopoDS_Shape) -> dict[str, CoverageCell]:
    out: dict[str, CoverageCell] = {}
    metrics = {
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
            out[name] = _coverage_cell(values)
        except Exception as exc:
            out[name] = CoverageCell(0, 0, 0, 0, f"ERROR: {type(exc).__name__}")
    return out


def _format_pct(v: float | None, ref: float) -> str:
    if v is None:
        return "—"
    if abs(ref) < 1e-6:
        return "n/a"
    return f"{abs(v - ref) / abs(ref) * 100:.2f}%"


def _format_value(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _format_time(s: float) -> str:
    return f"{s:.2f}s"


def render_report(
    volume_rows: list[dict],
    coverage_rows: list[dict],
    *,
    resolution: int,
) -> str:
    lines: list[str] = []
    lines.append("# BRepAX Integration Benchmark Report")
    lines.append("")
    lines.append(
        "Reproducible measurement of the BRepAX volume paths and "
        "face-level metric coverage on the project's standard STEP "
        "fixture set."
    )
    lines.append("")
    lines.append(
        "**This report is the output of one benchmark command** — "
        "`uv run python -m benchmarks.integration_report.run_benchmark`. "
        "No claim of novelty is made beyond what the tables below "
        "directly show.  Comparisons against external systems "
        "(Manifold, PyTorch3D, JAX-FEM) are qualitative; see "
        "`competitor_landscape.md` for the framing and its "
        "limitations."
    )
    lines.append("")
    lines.append(
        f"Volume paths use a sigmoid grid integration at "
        f"`resolution={resolution}` (per-axis cell count); the mesh "
        "divergence path uses the OCCT BRepMesh tessellation at the "
        "default deflection.  OCCT BRepGProp is the reference and is "
        "computed analytically on the exact B-Rep, not on the mesh."
    )
    lines.append("")

    # Volume table
    lines.append("## Volume accuracy")
    lines.append("")
    lines.append(
        "| Fixture | OCCT (ref) | divergence | div err | div t | "
        "CSG-Stump | csg err | csg t | TrimmedCSG | trim err | trim t | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in volume_rows:
        notes: list[str] = []
        for label, key in (
            ("div", "divergence"),
            ("csg", "csg_stump"),
            ("trim", "trimmed_csg"),
        ):
            n = row[key].note
            if n:
                notes.append(f"{label}: {n}")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['fixture']}`",
                    f"{row['occt_volume']:.4f}",
                    _format_value(row["divergence"].value),
                    _format_pct(row["divergence"].value, row["occt_volume"]),
                    _format_time(row["divergence"].elapsed_s),
                    _format_value(row["csg_stump"].value),
                    _format_pct(row["csg_stump"].value, row["occt_volume"]),
                    _format_time(row["csg_stump"].elapsed_s),
                    _format_value(row["trimmed_csg"].value),
                    _format_pct(row["trimmed_csg"].value, row["occt_volume"]),
                    _format_time(row["trimmed_csg"].elapsed_s),
                    "; ".join(notes) if notes else "",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("**How to read this table.**")
    lines.append("")
    lines.append(
        "- `divergence` is the mesh divergence-theorem volume (Stokes' "
        "theorem on the BRepMesh tessellation).  Differentiable through "
        "triangle vertex positions.  Strongest production path."
    )
    lines.append(
        "- `CSG-Stump` is the analytical primitive DNF, integrated with "
        "a sigmoid indicator.  Differentiable through primitive "
        "parameters.  Bounded by the BSpline half-space limitation "
        "(see `project_bspline_halfspace.md` in memory; concretely the "
        "CSG-Stump DNF cannot consume a finite trimmed BSpline patch "
        "as a half-space ingredient — ADR-0019, ADR-0020)."
    )
    lines.append(
        "- `TrimmedCSGStump` carries per-face trim metadata for "
        "standalone trimmed-face SDF queries; on the DNF path it is "
        "**bit-equivalent** to `DifferentiableCSGStump` per "
        "ADR-0019 / ADR-0020.  Equality of the `csg` and `trim` "
        "columns is the expected outcome."
    )
    lines.append("")

    # Coverage table
    lines.append("## Face-level metric coverage")
    lines.append("")
    lines.append(
        "Each cell shows `(finite / nan / inf)` counts out of the "
        "fixture's total face count.  Single-face shapes return `+inf` "
        "for `min_wall_thickness_per_face` (no other surface to measure "
        "against).  `mean_curvature_per_face` returns NaN on cone, "
        "torus, and BSpline faces (analytical handler not yet added)."
    )
    lines.append("")
    metric_order = [
        "surface_area",
        "min_draft_angle",
        "mean_curvature",
        "min_wall_thickness",
    ]
    header = "| Fixture | n_faces | " + " | ".join(m for m in metric_order) + " |"
    sep = "|---" * (len(metric_order) + 2) + "|"
    lines.append(header)
    lines.append(sep)
    for row in coverage_rows:
        cells = [f"`{row['fixture']}`"]
        # Pull n_faces from the first metric that succeeded
        n_faces_seen = next(
            (row[m].n_faces for m in metric_order if row[m].n_faces),
            0,
        )
        cells.append(str(n_faces_seen))
        for m in metric_order:
            c = row[m]
            if c.note:
                cells.append(c.note)
            else:
                cells.append(f"{c.n_finite}/{c.n_nan}/{c.n_inf}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Competitor landscape (loaded from sibling file)
    if COMPETITOR_PATH.exists():
        lines.append("## Qualitative competitor landscape")
        lines.append("")
        lines.append(COMPETITOR_PATH.read_text().strip())
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Generated by `benchmarks/integration_report/run_benchmark.py`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        default=VOLUME_RESOLUTION,
        help="Per-axis grid resolution for sigmoid volume integration",
    )
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=None,
        help="Restrict to a subset of fixture names (omit for all)",
    )
    parser.add_argument(
        "--skip-csg",
        action="store_true",
        help="Skip CSG-Stump and TrimmedCSGStump paths (useful when "
        "iterating on the divergence/coverage tables only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORT_PATH,
        help="Output markdown report path",
    )
    args = parser.parse_args()

    fixtures = args.fixtures if args.fixtures else FIXTURES

    volume_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for name in fixtures:
        path = FIXTURES_DIR / f"{name}.step"
        if not path.exists():
            print(f"[skip] {name}: fixture not found at {path}")
            continue
        print(f"[run]  {name}")
        try:
            shape = read_step(str(path))
        except Exception:
            print(f"[fail] {name}: read_step")
            traceback.print_exc()
            continue

        try:
            gt = compute_gprop_ground_truth(shape)
            occt_volume = float(gt["volume"])
        except Exception as exc:
            print(f"[fail] {name}: gprop {exc}")
            continue

        is_solid = shape.ShapeType() == TopAbs_SOLID
        if not is_solid:
            shape_label = shape.ShapeType().name.split("_")[-1].lower()
            shell_note = f"{shape_label} (volume paths require closed solid)"
            volume_rows.append(
                {
                    "fixture": name,
                    "occt_volume": occt_volume,
                    "divergence": VolumeResult(None, None, 0.0, shell_note),
                    "csg_stump": VolumeResult(None, None, 0.0, ""),
                    "trimmed_csg": VolumeResult(None, None, 0.0, ""),
                }
            )
        else:
            div = measure_divergence(shape)
            if args.skip_csg:
                csg = VolumeResult(None, None, 0.0, "skipped")
                trim = VolumeResult(None, None, 0.0, "skipped")
            else:
                handle = reconstruct_stump_cached(shape)
                csg = measure_csg_stump(
                    shape, resolution=args.resolution, handle=handle
                )
                trim = measure_trimmed_csg_stump(
                    shape, resolution=args.resolution, handle=handle
                )

            volume_rows.append(
                {
                    "fixture": name,
                    "occt_volume": occt_volume,
                    "divergence": div,
                    "csg_stump": csg,
                    "trimmed_csg": trim,
                }
            )

        cov = measure_face_coverage(shape)
        coverage_rows.append({"fixture": name, **cov})

    text = render_report(volume_rows, coverage_rows, resolution=args.resolution)
    args.out.write_text(text)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
