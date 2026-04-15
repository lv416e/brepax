# ADR-0005: Half-Edge Padding Strategy

## Status

Proposed

## Context

JAX requires static shapes for jit compilation. Topological data structures (half-edge meshes) have dynamic sizes that change with Boolean operations. A strategy is needed to represent variable-topology meshes within JAX's static shape constraint.

## Decision

Use padding with masking:

- Allocate fixed-capacity arrays for vertices, edges, and faces
- Use boolean mask arrays to indicate valid (non-padding) elements
- Capacity is set at construction time based on expected maximum complexity
- Operations that exceed capacity raise a clear error rather than silently truncating

## Consequences

- Memory overhead from padding (acceptable for typical CAD complexity)
- All topology operations must respect masks (adds implementation complexity)
- No need for `pure_callback` or escaping jit for topology changes
