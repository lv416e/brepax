"""Time-of-impact corrected Boolean operations.

Implementation deferred to hybrid optimizer context.
Mathematical derivation complete: see docs/explanation/toi_derivation.md.
Design rationale: see ADR-0009 (deferral) and ADR-0011 (hybrid strategy).

The TOI correction provides cross-stratum gradient transfer via the
implicit function theorem, complementing Method (C) stratum-aware
gradients which are exact within each stratum but zero for
stratum-invariant design variables.
"""
