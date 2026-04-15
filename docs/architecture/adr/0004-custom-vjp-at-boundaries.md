# ADR-0004: Custom VJP at Boundaries

## Status

Proposed

## Context

At stratum boundaries (e.g., two disks becoming tangent), the standard autodiff gradient is undefined or discontinuous. The core thesis of BRepAX is that these boundaries can be handled analogously to contact events in differentiable physics simulation.

## Decision

Use `jax.custom_vjp` to provide corrected gradients near stratum boundaries.

Three strategies are implemented (corresponding to the three Boolean methods):

- **Smoothing**: No custom_vjp needed; smooth approximation handles boundaries implicitly
- **TOI correction**: custom_vjp applies implicit-function-theorem correction at detected boundary crossings
- **Stratum-aware**: custom_vjp switches gradient formulas based on active stratum label

## Consequences

- Requires careful testing of VJP correctness (analytical gradient comparison)
- The custom_vjp implementations are the most mathematically sensitive code in the library
- Each method's VJP must be independently validated against analytical ground truth
