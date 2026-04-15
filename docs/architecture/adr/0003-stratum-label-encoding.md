# ADR-0003: Stratum Label Encoding

## Status

Accepted

## Context

The stratification module needs to classify and track which topological stratum a configuration belongs to. For two disks, there are 3 strata (disjoint, intersecting, contained). For N primitives with Boolean operations, the number of strata grows combinatorially. Labels must be JAX-traceable for use inside jit-compiled functions, vmap-compatible for batched configurations, and usable as `jax.custom_vjp` residuals.

Stage 1 confirmed that `from __future__ import annotations` is incompatible with jaxtyping runtime checks, and that all traced values must be plain JAX arrays (not Python objects) to work with `eqx.filter_jit` and `eqx.filter_vmap`.

## Candidates Evaluated

### (a) Fixed-length `Integer[Array, "n_pairs"]`

Each element independently encodes one pairwise relationship sign. Equality check via `jnp.array_equal`. Works with jit, vmap, and as custom_vjp residual. For N primitives, `n_pairs = N*(N-1)//2` with static shape. O(n_pairs) equality check.

### (b) Single `Integer[Array, ""]` (bit-packed)

O(1) scalar equality comparison. Minimal memory. The current `disk_disk_stratum_label` already returns this shape (values 0/1/2). With 64-bit int, caps at ~40 pairs (~9 primitives) before overflow. Bit-packing logic adds opacity for debugging.

### (c) Static tuple via `eqx.field(static=True)`

Each unique label triggers jit recompilation. Under vmap, all batch elements must share the same static value. Fundamentally incompatible with batched stratum detection.

### (d) IntEnum + jax_dataclasses

Python ints are not traced arrays. Same recompilation issue as (c) unless converted to plain arrays, at which point it reduces to (a) or (b). Experimental dependency.

## Decision

Start with **(b) scalar `Integer[Array, ""]`** for two-disk configurations. Migrate to **(a) `Integer[Array, "n_pairs"]`** when N-primitive support is implemented.

Rationale:

- The current `disk_disk_stratum_label` is already (b) without explicit bit-packing (values 0/1/2 encode the stratum implicitly)
- Zero migration cost for the concept proof
- When N-primitive support arrives, (a) scales cleanly with static shapes, works with vmap, and serves as custom_vjp residuals without bit manipulation
- Define `StratumLabel = Integer[Array, "n_pairs"]` type alias early so call sites are prepared for the shape change

**(c) and (d) are rejected** -- both break under vmap and cause recompilation pathologies.

## Consequences

- Concept proof code continues using scalar int labels as-is
- Type alias `StratumLabel` will be introduced in `stratification/label.py`
- Migration trigger: when the first N > 2 primitive composition is implemented
- Migration scope: change the type alias shape and update equality checks
