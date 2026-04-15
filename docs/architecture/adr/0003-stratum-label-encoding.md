# ADR-0003: Stratum Label Encoding

## Status

Proposed

## Context

The stratification module needs to classify and track which topological stratum a configuration belongs to. For two disks, there are 3 strata (disjoint, intersecting, contained). For N primitives with Boolean operations, the number of strata grows combinatorially. Labels must be JAX-traceable for use inside jit-compiled functions.

## Decision

To be determined based on empirical results. Candidates:

1. **Integer encoding**: Simple enum-like integers. Fast but opaque.
2. **Tuple-based encoding**: Hashable tuples describing the combinatorial type. Readable but not directly JAX-traceable.
3. **Binary vector encoding**: Fixed-length binary vector where each bit represents a pairwise relationship. JAX-traceable and extensible.

## Consequences

The choice affects stratum caching, transition detection performance, and scalability to many primitives. Will be resolved based on empirical results with multi-primitive configurations.
