# ADR-0001: JAX-Native Rationale

## Status

Accepted

## Context

BRepAX needs a differentiable computation backend for geometric operations. The primary candidates are JAX, PyTorch, and custom C++ with autodiff. The library targets researchers in computational geometry and CAD optimization who need composable transformations (jit, vmap, grad) over geometric primitives.

## Decision

Implement BRepAX as a pure JAX library, without C++ extensions or PyTorch interop.

Key reasons:

- **Composable transformations**: JAX's functional transformation model (jit/vmap/grad) maps naturally to geometric operations on SDF fields
- **custom_vjp/custom_jvp**: Essential for implementing stratum-aware gradient corrections at topological boundaries
- **Ecosystem alignment**: Equinox, Diffrax, Optimistix provide battle-tested patterns for scientific JAX libraries
- **Static shape model**: Forces explicit handling of topology changes via padding/masking, which aligns with stratum tracking design

## Consequences

- Users must work within the JAX ecosystem (no PyTorch interop)
- Dynamic topology changes require padding strategies (see ADR-0005)
- Performance-critical inner loops stay in Python/JAX until profiling shows otherwise
- C++ extensions are deferred unless benchmarks reveal specific bottlenecks
