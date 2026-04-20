"""CTC-02 triangulation baseline benchmark (M5 entry point).

Measures the STEP -> triangulate_shape -> divergence_volume -> gradient
pipeline on NIST CTC-02 (664 faces, 34 BSpline, ~247K triangles).

ADR-0016 recorded ~95s for BSpline face re-evaluation in this pipeline.
This benchmark reproduces and decomposes that number so M5 improvements
can be A/B compared.

Run explicitly (slow, not part of default suite):
    uv run pytest tests/benchmarks/bench_ctc02_triangulate.py -m slow -s
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.triangulate import divergence_volume, triangulate_shape
from brepax.io.step import read_step

CTC02 = Path(__file__).parents[1] / "fixtures" / "nist" / "nist_ctc_02_asme1_rc.stp"


def _fmt(t: float) -> str:
    return f"{t:7.3f}s"


@pytest.mark.slow
@pytest.mark.benchmark
def test_ctc02_baseline() -> None:
    """Phase-decomposed baseline. Prints to stdout; no pass/fail."""
    assert CTC02.exists(), f"Missing fixture: {CTC02}"

    t0 = time.perf_counter()
    shape = read_step(str(CTC02))
    t_step = time.perf_counter() - t0

    t0 = time.perf_counter()
    triangles_cold, params_list = triangulate_shape(shape)
    jax.block_until_ready(triangles_cold)
    t_tri_cold = time.perf_counter() - t0

    n_tris = int(triangles_cold.shape[0])
    n_faces = len(params_list)
    n_bspline = sum(1 for p in params_list if "control_points" in p)

    # Warm re-call: triangulation re-runs OCCT + rebuilds closures,
    # so this is NOT a JAX-cache-hit scenario. Measured to show that
    # the bottleneck is per-call, not amortizable by simple reuse.
    t0 = time.perf_counter()
    triangles_warm, _ = triangulate_shape(shape)
    jax.block_until_ready(triangles_warm)
    t_tri_warm = time.perf_counter() - t0

    t0 = time.perf_counter()
    vol = divergence_volume(triangles_cold)
    jax.block_until_ready(vol)
    t_vol_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    vol2 = divergence_volume(triangles_cold)
    jax.block_until_ready(vol2)
    t_vol_warm = time.perf_counter() - t0

    grad_vol = jax.grad(divergence_volume)
    t0 = time.perf_counter()
    g_cold = grad_vol(triangles_cold)
    jax.block_until_ready(g_cold)
    t_grad_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    g_warm = grad_vol(triangles_cold)
    jax.block_until_ready(g_warm)
    t_grad_warm = time.perf_counter() - t0

    total_cold = t_step + t_tri_cold + t_vol_cold + t_grad_cold

    print()
    print("=" * 62)
    print(
        f"CTC-02 baseline  |  {n_faces} faces ({n_bspline} BSpline)  |  {n_tris} tris"
    )
    print("=" * 62)
    print(f"  STEP load                  {_fmt(t_step)}")
    print(f"  triangulate_shape (cold)   {_fmt(t_tri_cold)}")
    print(f"  triangulate_shape (warm)   {_fmt(t_tri_warm)}")
    print(f"  divergence_volume (cold)   {_fmt(t_vol_cold)}")
    print(f"  divergence_volume (warm)   {_fmt(t_vol_warm)}")
    print(f"  grad(divergence)  (cold)   {_fmt(t_grad_cold)}")
    print(f"  grad(divergence)  (warm)   {_fmt(t_grad_warm)}")
    print("-" * 62)
    print(f"  total (cold forward+grad)  {_fmt(total_cold)}")
    print(f"  volume                     {float(vol):.3f} mm^3")
    print("=" * 62)

    assert jnp.isfinite(vol), "volume non-finite"
    assert g_cold.shape == triangles_cold.shape
    assert jnp.all(jnp.isfinite(g_cold))
