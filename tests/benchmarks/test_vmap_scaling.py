"""Vmap scaling benchmark.

Measures throughput of jit-compiled gradient computation as batch size
increases from 1 to 1024. Both Method (A) and Method (C) are tested.

Pass condition: T(1024) / T(1) >= 0.7 * 1024 = 716.8 (70% efficiency).
"""

import time

import jax
import jax.numpy as jnp
import pytest

from brepax.boolean import union_area
from brepax.primitives import Disk

BATCH_SIZES = [1, 4, 16, 64, 256, 1024]
WARMUP_ROUNDS = 3
MEASURE_ROUNDS = 10


def _make_batch(batch_size, *, seed=42):
    """Generate random disk configurations avoiding boundary-adjacent cases."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    c1 = jax.random.normal(k1, (batch_size, 2)) * 0.5
    r1 = jax.random.uniform(k2, (batch_size,), minval=0.5, maxval=1.5)
    c2 = jax.random.normal(k3, (batch_size, 2)) * 0.5 + jnp.array([1.0, 0.0])
    r2 = jax.random.uniform(k4, (batch_size,), minval=0.5, maxval=1.5)
    return c1, r1, c2, r2


def _throughput(fn, batch_size, c1, r1, c2, r2):
    """Measure throughput in samples/sec."""
    # Warmup
    for _ in range(WARMUP_ROUNDS):
        fn(c1, r1, c2, r2).block_until_ready()

    # Measure
    times = []
    for _ in range(MEASURE_ROUNDS):
        t0 = time.perf_counter()
        fn(c1, r1, c2, r2).block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_time = sorted(times)[len(times) // 2]
    return batch_size / median_time


def _stratum_grad_batch(c1, r1, c2, r2):
    """Batched Method (C) gradient via vmap."""

    def single_grad(c1_i, r1_i, c2_i, r2_i):
        def f(r):
            a = Disk(center=c1_i, radius=r)
            b = Disk(center=c2_i, radius=r2_i)
            return union_area(a, b, method="stratum")

        return jax.grad(f)(r1_i)

    return jax.vmap(single_grad)(c1, r1, c2, r2)


def _smoothing_grad_batch(c1, r1, c2, r2):
    """Batched Method (A) gradient via vmap."""

    def single_grad(c1_i, r1_i, c2_i, r2_i):
        def f(r):
            a = Disk(center=c1_i, radius=r)
            b = Disk(center=c2_i, radius=r2_i)
            return union_area(a, b, method="smoothing", k=0.1, beta=0.1, resolution=64)

        return jax.grad(f)(r1_i)

    return jax.vmap(single_grad)(c1, r1, c2, r2)


class TestVmapScalingMethodC:
    """Method (C) vmap throughput scaling."""

    def test_vmap_produces_correct_shape(self) -> None:
        c1, r1, c2, r2 = _make_batch(4)
        jitted = jax.jit(_stratum_grad_batch)
        result = jitted(c1, r1, c2, r2)
        assert result.shape == (4,)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.slow
    def test_scaling_efficiency(self) -> None:
        """GPU scaling benchmark. Skipped in CI (CPU only). See ADR-0010."""
        jitted = jax.jit(_stratum_grad_batch)

        throughputs = {}
        for bs in BATCH_SIZES:
            c1, r1, c2, r2 = _make_batch(bs)
            tp = _throughput(jitted, bs, c1, r1, c2, r2)
            throughputs[bs] = tp
            print(f"\n  Method (C) batch_size={bs:>4d}: {tp:>10.0f} samples/sec")

        ratio = throughputs[1024] / throughputs[1]
        efficiency = ratio / 1024
        print(f"\n  T(1024)/T(1) = {ratio:.1f} ({efficiency:.1%} efficiency)")

        assert ratio >= 716.8, (
            f"vmap scaling below 70% efficiency: "
            f"T(1024)/T(1) = {ratio:.1f}, need >= 716.8"
        )


class TestVmapScalingMethodA:
    """Method (A) vmap throughput scaling."""

    def test_vmap_produces_correct_shape(self) -> None:
        c1, r1, c2, r2 = _make_batch(4)
        jitted = jax.jit(_smoothing_grad_batch)
        result = jitted(c1, r1, c2, r2)
        assert result.shape == (4,)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.slow
    def test_scaling_efficiency(self) -> None:
        jitted = jax.jit(_smoothing_grad_batch)

        throughputs = {}
        for bs in BATCH_SIZES:
            c1, r1, c2, r2 = _make_batch(bs)
            tp = _throughput(jitted, bs, c1, r1, c2, r2)
            throughputs[bs] = tp
            print(f"\n  Method (A) batch_size={bs:>4d}: {tp:>10.0f} samples/sec")

        ratio = throughputs[1024] / throughputs[1]
        efficiency = ratio / 1024
        print(f"\n  T(1024)/T(1) = {ratio:.1f} ({efficiency:.1%} efficiency)")

        assert ratio >= 716.8, (
            f"vmap scaling below 70% efficiency: "
            f"T(1024)/T(1) = {ratio:.1f}, need >= 716.8"
        )
