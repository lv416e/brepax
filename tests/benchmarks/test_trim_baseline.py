"""Volume trim baseline against OCCT reference.

Measures the volume on each fixture via the CSG-Stump SDF path and
compares against OCCT's analytic reference. The purpose is to document,
with a hard number, how far the CSG-Stump SDF path drifts from the true
volume on trimmed geometry before any trim-aware integration is added.

Scope is deliberately narrow: a single metric (volume), a single SDF
path (CSG-Stump composite built from primitive parameters), and OCCT
ground truth via analytic surface-integral quadrature. Extensions to
surface area, wall thickness, curvature, and to a second SDF path
(mesh-based) are deferred to their own follow-ups because each adds
non-trivial compute and compilation cost.

The ``sdf_direct`` path ignores any trim boundary and therefore produces
phantom regions outside the true solid whenever a BSpline face is
trimmed. Models with analytic-only primitives serve as controls where
both paths should agree within the sigmoid-indicator first-order
smearing bias.

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
    occt_gt: float
    err_pct: float
    t_sec: float


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
    """Print per-model CSG-Stump volume against OCCT ground truth."""
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

        print(f"[{label}] volume integral", flush=True)
        t0 = time.perf_counter()
        v_direct = _volume_from_sdf(diff.sdf, lo, hi, RESOLUTION)
        t_elapsed = time.perf_counter() - t0

        err = _relerr(v_direct, gt["volume"])

        rows.append(
            VolumeRow(
                model=label,
                sdf_direct=v_direct,
                occt_gt=float(gt["volume"]),
                err_pct=err,
                t_sec=t_elapsed,
            )
        )

        jax.clear_caches()  # type: ignore[no-untyped-call]

    _print_table(rows)

    for r in rows:
        assert math.isfinite(r.sdf_direct), f"non-finite sdf_direct: {r}"
        assert r.occt_gt > 0.0, f"non-positive OCCT volume: {r}"


def _print_table(rows: list[VolumeRow]) -> None:
    print()
    header = (
        f"{'model':18s} {'V_direct':>14s} {'V_gt':>14s} {'err%':>10s} {'t_sec':>7s}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model:18s} "
            f"{r.sdf_direct:14.4g} {r.occt_gt:14.4g} "
            f"{r.err_pct:10.2f} "
            f"{r.t_sec:7.2f}"
        )
    print("=" * len(header))
