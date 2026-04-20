"""Multi-model triangulation baseline across complexity spectrum.

Reports cold/warm triangulation, volume, and gradient timings on a
small set of fixture models that span face count (1-37) and BSpline
ratio (0-100%). CTC-02 lives in its own benchmark; this file covers
the complement so the M5 optimizations can be sanity-checked beyond
the single large part.

Single-shot timings; treat as order-of-magnitude reference, not
paper-grade statistics. The models run sequentially within one test
process, so compiled JIT artifacts for a surface type or BSpline
signature seen in an earlier model are reused by later ones.  Cold
timings here therefore reflect "first time this signature is seen in
this process", not "single-model fresh process cold".  Isolated
per-model cold measurement needs subprocess runs.

Run explicitly (slow, not part of default suite):
    uv run pytest tests/benchmarks/test_multi_model_baseline.py -m slow -s
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.triangulate import divergence_volume, triangulate_shape
from brepax.io.step import read_step

FIXTURES = Path(__file__).parents[1] / "fixtures"

MODELS: list[tuple[str, str]] = [
    ("sample_box", "sample_box.step"),
    ("sample_sphere", "sample_sphere.step"),
    ("sample_cylinder", "sample_cylinder.step"),
    ("box_with_holes", "box_with_holes.step"),
    ("nurbs_saddle", "nurbs_saddle.step"),
    ("nurbs_box", "nurbs_box.step"),
    ("hex_nut_m6", "freecad/ISO4032_Hex_Nut_M6.step"),
    ("occt_linkrods", "misc/occt_linkrods.step"),
]


def _time(fn: object, *args: object) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn(*args)  # type: ignore[operator]
    jax.block_until_ready(out)
    return time.perf_counter() - t0, out


@pytest.mark.slow
@pytest.mark.benchmark
def test_multi_model_baseline() -> None:
    """Print a per-model timing table; asserts basic liveness."""
    rows: list[tuple[str, int, int, int, float, float, float, float, float, float]] = []
    grad_volume = jax.grad(divergence_volume)

    for label, rel_path in MODELS:
        path = FIXTURES / rel_path
        assert path.exists(), f"missing fixture: {path}"
        shape = read_step(str(path))

        t_tri_cold, pair_cold = _time(triangulate_shape, shape)
        triangles_cold, params_list = pair_cold  # type: ignore[misc]
        t_tri_warm, _ = _time(triangulate_shape, shape)

        n_tris = int(triangles_cold.shape[0])
        n_faces = len(params_list)
        n_bspline = sum(1 for p in params_list if "control_points" in p)

        t_vol_cold, vol = _time(divergence_volume, triangles_cold)
        t_vol_warm, _ = _time(divergence_volume, triangles_cold)

        t_grad_cold, grad = _time(grad_volume, triangles_cold)
        t_grad_warm, _ = _time(grad_volume, triangles_cold)

        assert jnp.isfinite(vol)
        assert jnp.all(jnp.isfinite(grad))

        rows.append(
            (
                label,
                n_faces,
                n_bspline,
                n_tris,
                t_tri_cold,
                t_tri_warm,
                t_vol_cold,
                t_vol_warm,
                t_grad_cold,
                t_grad_warm,
            )
        )

    print()
    header = (
        f"{'model':18s}  {'faces':>5s} {'bspl':>4s} {'tris':>7s}  "
        f"{'tri_c':>7s} {'tri_w':>7s}  {'vol_c':>7s} {'vol_w':>7s}  "
        f"{'grd_c':>7s} {'grd_w':>7s}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r[0]:18s}  {r[1]:5d} {r[2]:4d} {r[3]:7d}  "
            f"{r[4]:7.3f} {r[5]:7.3f}  {r[6]:7.3f} {r[7]:7.3f}  "
            f"{r[8]:7.3f} {r[9]:7.3f}"
        )
    print("=" * len(header))
