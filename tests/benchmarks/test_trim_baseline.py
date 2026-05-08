"""Volume trim baseline against OCCT reference.

Measures the volume on each analytical fixture via two CSG-Stump SDF
paths and compares against OCCT's analytic reference:

- ``sdf_direct``: untrimmed CSG-Stump composite via
  :class:`DifferentiableCSGStump`. Each primitive contributes its raw
  half-space SDF.
- ``sdf_trim``: trim-aware composite via :class:`TrimmedCSGStump`.
  Per ADR-0019, every analytical primitive (plane, cylinder, sphere,
  cone, torus) also contributes its raw half-space SDF; the
  Marschner blend from ADR-0018 is reserved for the standalone-face
  distance-query use case and for the future BSpline-patch path
  inside the composition.

On analytical-only fixtures the two columns must agree to within
floating-point noise. The purpose of this benchmark is to lock that
invariant in: any drift between ``sdf_direct`` and ``sdf_trim``
indicates the analytical dispatch has regressed from raw-half-space
semantics. Phantom reduction over OCCT ground truth, the original
motivation for ADR-0018, surfaces only when BSpline primitives are
wired into the trim-aware path; that is a separate measurement on
fixtures that contain BSpline faces (Linkrods being the worst
measured case at +219%, ADR-0016).

Parameters are adaptive per model:

- ``lo``, ``hi``: the shape's axis-aligned bounding box padded by 5%.
- ``resolution``: 64 per axis (cell size scales with bbox).

The model set is limited to fixtures whose CSG-Stump composite SDF can
be evaluated within current memory limits at 64-cube resolution. Highly
freeform parts whose CSG-Stump expansion contains many BSpline primitives
(occt_linkrods etc.) are excluded here and tracked separately; their
inclusion will require either a chunked composite-SDF evaluator or the
mesh-bypassing trim-aware path that replaces per-primitive Newton
iteration with the analytical surface SDF + trim composition.

Assertions are liveness only; the deviations from OCCT are the data
this benchmark exists to surface.

Run explicitly (slow, not part of default suite):
    uv run pytest tests/benchmarks/test_trim_baseline.py -m slow -s
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brepax
from brepax._occt.backend import Bnd_Box, BRepBndLib
from brepax._occt.types import TopoDS_Shape
from brepax.brep.csg_eval import integrate_sdf_volume, make_grid_3d
from brepax.brep.csg_stump import reconstruct_csg_stump, stump_to_differentiable
from brepax.brep.gprop import compute_gprop_ground_truth
from brepax.brep.trimmed_csg_stump import enrich_with_trim_frames
from brepax.io.step import read_step

FIXTURES = Path(__file__).parents[1] / "fixtures"

MODELS: list[tuple[str, str]] = [
    ("sample_box", "sample_box.step"),
    ("sample_cylinder", "sample_cylinder.step"),
    ("box_with_holes", "box_with_holes.step"),
    ("l_bracket", "l_bracket.step"),
]

RESOLUTION = 64
BBOX_PADDING_FRAC = 0.05


class VolumeRow(NamedTuple):
    """One model's volume record for the printed table."""

    model: str
    sdf_direct: float
    sdf_trim: float
    occt_gt: float
    err_direct_pct: float
    err_trim_pct: float
    t_direct_sec: float
    t_trim_sec: float


def _bbox(shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box corners for a TopoDS_Shape."""
    bnd = Bnd_Box()
    BRepBndLib.Add_s(shape, bnd)
    xmin, ymin, zmin, xmax, ymax, zmax = bnd.Get()
    lo = np.array([xmin, ymin, zmin], dtype=np.float64)
    hi = np.array([xmax, ymax, zmax], dtype=np.float64)
    return lo, hi


def _adaptive_bounds(
    lo_bbox: np.ndarray, hi_bbox: np.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pad the bbox by 5% on each axis."""
    pad = BBOX_PADDING_FRAC * (hi_bbox - lo_bbox)
    lo = jnp.array(lo_bbox - pad, dtype=jnp.float32)
    hi = jnp.array(hi_bbox + pad, dtype=jnp.float32)
    return lo, hi


def _relerr(value: float, reference: float) -> float:
    """Signed relative error in percent; NaN when the reference is tiny."""
    if abs(reference) < 1e-12:
        return float("nan")
    return 100.0 * (value - reference) / reference


def _volume_from_sdf(
    sdf_fn: Any, lo: jnp.ndarray, hi: jnp.ndarray, resolution: int
) -> float:
    """Integrate ``sdf_fn`` on the grid and return the scalar volume."""
    grid = make_grid_3d(lo, hi, resolution)[0]
    sdf_vals = sdf_fn(grid)
    vol = integrate_sdf_volume(sdf_vals, lo, hi, resolution)
    return float(vol)


@pytest.mark.slow
@pytest.mark.benchmark
def test_trim_volume_baseline() -> None:
    """Print per-model untrimmed and trim-aware volumes against OCCT GT."""
    brepax.enable_compilation_cache()

    rows: list[VolumeRow] = []

    for label, rel_path in MODELS:
        print(f"\n[{label}] load {rel_path}", flush=True)
        path = FIXTURES / rel_path
        assert path.exists(), f"missing fixture: {path}"
        shape = read_step(str(path))

        lo_bbox, hi_bbox = _bbox(shape)
        lo, hi = _adaptive_bounds(lo_bbox, hi_bbox)

        gt = compute_gprop_ground_truth(shape)

        print(f"[{label}] csg-stump reconstruct", flush=True)
        stump = reconstruct_csg_stump(shape)
        assert stump is not None, f"CSG reconstruction failed for {label}"
        diff = stump_to_differentiable(stump)

        print(f"[{label}] volume integral (sdf_direct)", flush=True)
        t0 = time.perf_counter()
        v_direct = _volume_from_sdf(diff.sdf, lo, hi, RESOLUTION)
        t_direct = time.perf_counter() - t0

        print(f"[{label}] volume integral (sdf_trim)", flush=True)
        trimmed = enrich_with_trim_frames(stump, shape)
        t0 = time.perf_counter()
        v_trim = _volume_from_sdf(trimmed.sdf, lo, hi, RESOLUTION)
        t_trim = time.perf_counter() - t0

        gt_v = float(gt["volume"])
        rows.append(
            VolumeRow(
                model=label,
                sdf_direct=v_direct,
                sdf_trim=v_trim,
                occt_gt=gt_v,
                err_direct_pct=_relerr(v_direct, gt_v),
                err_trim_pct=_relerr(v_trim, gt_v),
                t_direct_sec=t_direct,
                t_trim_sec=t_trim,
            )
        )

        jax.clear_caches()  # type: ignore[no-untyped-call]

    _print_table(rows)

    for r in rows:
        assert math.isfinite(r.sdf_direct), f"non-finite sdf_direct: {r}"
        assert math.isfinite(r.sdf_trim), f"non-finite sdf_trim: {r}"
        assert r.occt_gt > 0.0, f"non-positive OCCT volume: {r}"


def _print_table(rows: list[VolumeRow]) -> None:
    print()
    header = (
        f"{'model':18s} "
        f"{'V_direct':>14s} {'err_d%':>8s} "
        f"{'V_trim':>14s} {'err_t%':>8s} "
        f"{'V_gt':>14s} "
        f"{'t_d':>6s} {'t_t':>6s}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model:18s} "
            f"{r.sdf_direct:14.4g} {r.err_direct_pct:8.2f} "
            f"{r.sdf_trim:14.4g} {r.err_trim_pct:8.2f} "
            f"{r.occt_gt:14.4g} "
            f"{r.t_direct_sec:6.2f} {r.t_trim_sec:6.2f}"
        )
    print("=" * len(header))
