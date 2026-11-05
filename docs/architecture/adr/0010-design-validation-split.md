# ADR-0010: Gate 3 Split into Design Verification and Performance Verification

## Status

Accepted

## Context

Gate criterion 3 specified "vmap x jit with batch size 1 to 1000 scaling nearly linearly on GPU." This implicitly combined two independent concerns:

1. **Design verification**: Does `jax.custom_vjp` (Method C) compose correctly with `vmap`?
2. **Performance verification**: Does throughput scale linearly with batch size on GPU?

The concept proof was conducted on CPU (Apple Silicon). The custom_vjp + vmap integration works correctly (shape tests pass, gradients are accurate across batches), but throughput scaling shows typical CPU memory bandwidth saturation starting at batch size 64.

CPU scaling results:

| batch | throughput | incremental scaling |
|-------|-----------|-------------------|
| 1 | 119k/s | baseline |
| 4 | 508k/s | 4.3x |
| 16 | 1.9M/s | 3.8x |
| 64 | 4.4M/s | 2.3x |
| 256 | 6.9M/s | 1.5x |
| 1024 | 12.0M/s | 1.7x |

Total T(1024)/T(1) = 100x (9.8% efficiency), well below the 70% target. This is a hardware limitation, not a design limitation.

## Decision

Split Gate 3 into two sub-criteria:

- **Gate 3a (design verification)**: custom_vjp + vmap integration works correctly. **PASS.**
- **Gate 3b (performance verification)**: GPU throughput scaling meets 70% efficiency. **DEFERRED to the 3D extension.**

## Consequences

- Concept proof gate evaluation proceeds with Gate 3a PASS
- The 3D extension will include GPU benchmark as part of the primitive test suite
- CPU flattening data serves as a baseline for GPU comparison (replaced; see
  Later measurements)
- The 70% efficiency threshold remains unchanged for the GPU evaluation

## Later measurements

The CPU table above measured the closed-form disk-disk union that Method (C)
was at the time. Commit `ae86dd0`, the generalization of ADR-0012, replaced it
with a grid SDF evaluation at `resolution=128`, and per-stratum dispatch added
more grid passes. The same benchmark at each commit, on the same class of Apple
Silicon CPU under load, gave about 9.8M samples/s at batch 1024 before that
change, about 15K/s after it, and between 5K/s and 9K/s on 2026-11-05. The table
therefore does not describe current Method (C) throughput, and it cannot serve
as the baseline for a GPU comparison: that comparison needs a fresh CPU run of
the same code. Throughput was traded for generality; the code changed on
purpose, and no gradual regression was found across the commits measured.
