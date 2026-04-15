# ADR-0002: Equinox PyTree Pattern

## Status

Accepted

## Context

Geometric primitives need to carry differentiable parameters (center, radius, etc.) and be compatible with JAX transformations. The standard approaches are: (1) plain JAX arrays with functional APIs, (2) flax.struct, (3) equinox.Module.

## Decision

All geometric primitives and composite shapes inherit from `equinox.Module`.

- Parameters are declared as dataclass-style fields with jaxtyping annotations
- The Module automatically registers as a PyTree, enabling seamless jit/vmap/grad
- Follows the pattern established by Diffrax, Optimistix, and other scientific JAX libraries

## Consequences

- Clean, readable API: `Disk(center=..., radius=...)` instead of dictionaries
- Automatic PyTree flattening/unflattening for all JAX transformations
- Equinox becomes a core dependency (acceptable given ecosystem alignment)
- Users familiar with Equinox/Diffrax will find the API immediately familiar
