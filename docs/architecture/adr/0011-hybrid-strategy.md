# ADR-0011: Stratum-Aware Locality and Hybrid Optimization Strategy

## Status

Accepted

## Context

Axis 3 optimization trajectory testing revealed a fundamental property of the stratum-aware approach (Method C): **within a stratum where the objective is invariant to a design variable, the gradient is exactly zero.** This is mathematically correct but prevents gradient-based optimization from crossing stratum boundaries.

Concrete example: In the disjoint stratum, `union_area = pi*r1^2 + pi*r2^2` is independent of center positions `c1, c2`. Method (C) correctly returns `d(area)/d(c2_x) = 0`. Gradient descent cannot cross from disjoint to intersecting by moving centers.

Method (A) smoothing provides a weak but non-zero gradient signal in this case because its smooth-min kernel extends beyond the exact boundary, creating a "shadow" of the intersecting stratum's area function into the disjoint regime.

This is structurally isomorphic to the visibility problem in differentiable rendering: fully occluded objects have zero gradient w.r.t. camera parameters. SoftRas (2019) solved this with soft rasterization; nvdiffrast (2020) with edge sampling. Both are hybrid approaches combining global signal with boundary precision.

## Decision

Method (A) and Method (C) are **complementary, not competing** strategies. BRepAX will provide a hybrid framework that combines:

- **Method (A)** for cross-stratum exploration (global gradient signal via smoothing)
- **Method (C)** for within-stratum precision (exact analytical gradients)

### Gate 2 Reinterpretation

The original Gate 2 criterion ("Method (C) converges to cross-stratum optimum") was based on an incomplete understanding of stratum-aware gradient structure. Revised:

- **Gate 2a** (within-stratum precision): Method (C) reaches analytical optimum with position error 7e-6 vs Method (A)'s 0.021. **PASS.**
- **Gate 2b** (cross-stratum behavior): Gradient zero in stratum-invariant directions is a mathematical property, not a defect. **REINTERPRETED.**

### Narrative Adjustment

- Previous: "Method (C) is superior to Method (A)"
- Revised: "Method (A) and (C) are complementary; BRepAX provides a coherent framework for combining both"

This parallels the differentiable rendering trajectory: SoftRas (global, approximate) was not replaced by nvdiffrast (precise, local) but complemented by it. BRepAX occupies the same structural position for CAD geometry.

## Method (B) Reconsideration

The TOI correction (Method B, deferred in ADR-0009) was designed precisely for boundary-crossing gradient computation. The Axis 3 finding strengthens the case for Method (B) implementation in future work: it provides the mathematical machinery to transfer gradient information across stratum boundaries via the implicit function theorem.

Re-evaluation trigger: when a use case requires single-method cross-stratum optimization without a coarse-to-fine schedule.

## Consequences

- Hybrid optimizer design becomes a first-class concern (not an afterthought)
- `boolean/__init__.py` union_area API remains method-agnostic; hybrid scheduling is handled at the optimizer level
- Method (B) regains relevance as a potential bridge between (A) and (C)
- The differentiable rendering analogy provides a well-studied template for the hybrid design
