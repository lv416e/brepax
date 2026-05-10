"""Reproducible integration benchmark report.

Measures the BRepAX volume paths against OCCT BRepGProp on the
project's standard STEP fixture set:

- ``divergence_volume`` (mesh divergence theorem on BRepMesh)
- ``mesh_sdf`` (mesh signed distance + sigmoid grid integration)
- ``gwn`` (generalized winding number indicator + grid integration)
- ``DifferentiableCSGStump.volume`` (analytical primitive DNF + sigmoid)
- ``TrimmedCSGStump.volume`` (same DNF, ADR-0019/0020 bit-equivalent)

and reports the per-fixture coverage of the four trim-aware face-level
metrics shipped in PR #81-#84.

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

import jax
import jax.numpy as jnp

from brepax._occt.backend import TopAbs_SOLID
from brepax._occt.types import TopoDS_Shape
from brepax.brep.convert import shape_metadata
from brepax.brep.csg_eval import integrate_sdf_volume, make_grid_3d
from brepax.brep.csg_stump import (
    CSGStump,
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
GWN_SHARPNESS = 32.0  # sigmoid sharpness around the wn=0.5 threshold
GWN_CHUNK_SIZE = 1024  # winding-number evaluation chunk on the grid


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


def _triangulate_or_none(
    shape: TopoDS_Shape,
) -> tuple[jnp.ndarray | None, str]:
    try:
        triangles, _ = triangulate_shape(shape)
    except Exception as exc:
        return None, f"triangulate: {type(exc).__name__}"
    if triangles.shape[0] == 0:
        return None, "no triangles"
    return triangles, ""


def measure_divergence(shape: TopoDS_Shape) -> VolumeResult:
    """Time the full STEP-to-volume path: triangulate + integrate.

    Wrapping the triangulation step in the same timer as the
    integration keeps the per-row time comparable to the CSG paths,
    which include their reconstruct cost.
    """
    t0 = time.perf_counter()
    triangles, note = _triangulate_or_none(shape)
    if triangles is None:
        return VolumeResult(None, None, time.perf_counter() - t0, note)
    v, _, note = _measure(lambda: divergence_volume(triangles))
    return VolumeResult(v, None, time.perf_counter() - t0, note)


def measure_mesh_sdf(shape: TopoDS_Shape, *, resolution: int) -> VolumeResult:
    """Mesh SDF volume: signed distance on a grid + sigmoid integration.

    Differentiable through triangle vertex positions (the SDF chain) and
    through the sigmoid integrator, identical in shape to the CSG-Stump
    grid path so the bias source can be compared directly.
    """
    t0 = time.perf_counter()
    triangles, note = _triangulate_or_none(shape)
    if triangles is None:
        return VolumeResult(None, None, time.perf_counter() - t0, note)
    lo, hi = _shape_grid_bounds(shape)

    def _eval() -> jnp.ndarray:
        grid, _ = make_grid_3d(lo, hi, resolution)
        flat = grid.reshape(-1, 3)
        sdf_vals = mesh_sdf(flat, triangles)
        return integrate_sdf_volume(sdf_vals, lo, hi, resolution)

    v, _, note = _measure(_eval)
    return VolumeResult(v, None, time.perf_counter() - t0, note)


def measure_gwn(shape: TopoDS_Shape, *, resolution: int) -> VolumeResult:
    """GWN volume: winding-number indicator on a grid + integration.

    GWN ~= 1 inside a closed mesh, ~= 0 outside; integrating
    ``sigmoid((wn - 0.5) * GWN_SHARPNESS)`` over the grid recovers the
    interior volume.  Sharpness is fixed at module level so the run
    is reproducible across fixtures (the field is dimensionless 0..1,
    so the cell-width-based sharpness used by ``integrate_sdf_volume``
    is not appropriate here).
    """
    t0 = time.perf_counter()
    triangles, note = _triangulate_or_none(shape)
    if triangles is None:
        return VolumeResult(None, None, time.perf_counter() - t0, note)
    lo, hi = _shape_grid_bounds(shape)

    def _eval() -> jnp.ndarray:
        grid, cell_vol = make_grid_3d(lo, hi, resolution)
        flat = grid.reshape(-1, 3)
        n = flat.shape[0]
        # Pad to a multiple of GWN_CHUNK_SIZE so lax.map sees static
        # shapes while we cap the per-step memory footprint.
        rem = n % GWN_CHUNK_SIZE
        pad = 0 if rem == 0 else GWN_CHUNK_SIZE - rem
        padded = jnp.pad(flat, ((0, pad), (0, 0)))
        chunks = padded.reshape(-1, GWN_CHUNK_SIZE, 3)

        def _chunk(c: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(lambda p: winding_number(p, triangles))(c)

        wn_padded = jax.lax.map(_chunk, chunks).reshape(-1)
        wn = wn_padded[:n]
        indicator = jax.nn.sigmoid((wn - 0.5) * GWN_SHARPNESS)
        return jnp.sum(indicator) * cell_vol

    v, _, note = _measure(_eval)
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

    # Volume table — one row per fixture, one column per path showing
    # "value (err%)".  A separate timings column carries per-path
    # wall-clock seconds; notes carry any per-path failure/skip note.
    lines.append("## Volume accuracy")
    lines.append("")
    path_keys = (
        ("divergence", "divergence"),
        ("mesh_sdf", "mesh_sdf"),
        ("gwn", "gwn"),
        ("csg_stump", "CSG-Stump"),
        ("trimmed_csg", "TrimmedCSG"),
    )
    header_paths = " | ".join(label for _, label in path_keys)
    lines.append(f"| Fixture | OCCT (ref) | {header_paths} | timings (s) | Notes |")
    lines.append("|---|---|" + "|".join("---" for _ in path_keys) + "|---|---|")

    def _path_cell(result: VolumeResult, ref: float) -> str:
        if result.value is None:
            return "—"
        return f"{_format_value(result.value)} ({_format_pct(result.value, ref)})"

    for row in volume_rows:
        notes: list[str] = []
        # Row-level note (e.g. shell fixture) renders once, without
        # prefixing it with any single path key — the constraint
        # applies to every volume path on this row, not just the
        # first.
        if row.get("shape_note"):
            notes.append(row["shape_note"])
        for key, _ in path_keys:
            n = row[key].note
            if n:
                notes.append(f"{key}: {n}")
        timing_str = " ".join(f"{key}={row[key].elapsed_s:.2f}" for key, _ in path_keys)
        cells = [
            f"`{row['fixture']}`",
            f"{row['occt_volume']:.4f}",
        ]
        for key, _ in path_keys:
            cells.append(_path_cell(row[key], row["occt_volume"]))
        cells.append(timing_str)
        cells.append("; ".join(notes) if notes else "")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**How to read this table.**")
    lines.append("")
    lines.append(
        "- `divergence` is the mesh divergence-theorem volume (Stokes' "
        "theorem on the BRepMesh tessellation).  Differentiable through "
        "triangle vertex positions.  Strongest production path."
    )
    lines.append(
        "- `mesh_sdf` is the mesh-based signed distance field "
        "(unsigned distance to the nearest triangle, signed by "
        "generalized winding number) integrated with a sigmoid "
        "indicator on the same grid as the CSG paths.  Differentiable "
        "through triangle vertex positions and the sigmoid integrator."
    )
    lines.append(
        "- `gwn` is the generalized-winding-number indicator integrated "
        "directly: ``sigmoid((wn - 0.5) * GWN_SHARPNESS)`` summed over "
        "the grid.  Sharpness is fixed at module level (the field is "
        "dimensionless 0..1, so the cell-width-based sharpness used by "
        "the SDF integrators is not appropriate here).  This single "
        "fixed configuration is one operating point; sweeping sharpness "
        "/ resolution / mesh deflection is out of scope for this PR."
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
                    "shape_note": shell_note,
                    "divergence": VolumeResult(None, None, 0.0, ""),
                    "mesh_sdf": VolumeResult(None, None, 0.0, ""),
                    "gwn": VolumeResult(None, None, 0.0, ""),
                    "csg_stump": VolumeResult(None, None, 0.0, ""),
                    "trimmed_csg": VolumeResult(None, None, 0.0, ""),
                }
            )
        else:
            div = measure_divergence(shape)
            mesh = measure_mesh_sdf(shape, resolution=args.resolution)
            gwn = measure_gwn(shape, resolution=args.resolution)
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
                    "mesh_sdf": mesh,
                    "gwn": gwn,
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
