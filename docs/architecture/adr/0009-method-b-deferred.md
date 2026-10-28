# ADR-0009: Method (B) TOI Correction Deferred from Concept Proof

## Status

Accepted: Method B remains deferred.

Mathematical rationale corrected on 2026-10-28. This preserves the deferral;
it does not approve implementing the former formula.

## Historical Context and Decision

The concept proof proposed comparing smoothing (A), TOI correction (B),
and relation-dispatched gradients (C) on two-disk union area. Method B was
deferred to reduce scope while comparing A and C.

The original rationale incorrectly asserted a first-order jump at internal
disk tangency, exact Method-C gradients everywhere, and a complete static
TOI correction. These assertions are withdrawn. For positive unequal
radii, ordinary internal and external disk contacts both have continuous
first area derivatives. The current intersecting backward pass is a grid
STE; see [Stratification Theory](../../explanation/stratification_theory.md).

## Current Rationale

The [TOI derivation](../../explanation/toi_derivation.md) establishes
event-time sensitivity for smooth, transverse, locally isolated events and
gives a counterexample to the former static correction. A future event
cannot change the derivative of a locally constant static volume while
preserving the same objective.

Historical forced-label experiments showed intersecting STE error even
with correct labels. They do not rule out additional detection error,
establish universal bounds, or validate an unimplemented TOI method.

## Re-evaluation Conditions

Reconsider event-time differentiation for a concrete event time, look-ahead
objective, or trajectory map. Specify grazing, event-order changes, and
degeneracies before choosing an AD rule.

Positive-area face contact warrants study of one-sided static derivatives,
but a first-order kink alone does not validate the discarded TOI formula.
Compare any proposal with independent derivatives and matched baselines.

## Consequences

- The historical concept proof compares A and C only.
- `boolean/toi.py` remains a placeholder; `method="toi"` raises
  `NotImplementedError`.
- The explanation retains valid event-time sensitivity and the reason not
  to implement the former static correction as derived.
