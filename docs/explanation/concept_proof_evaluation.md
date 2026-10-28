# Concept Proof Evaluation: Historical Scope

This records the interpretation of the original two-disk concept proof,
not a current acceptance report. The former overall **PASS** depended on
claims that the current implementation does not satisfy.

## Corrections to the Original Conclusions

| Original conclusion | Current assessment |
|---|---|
| Method C is analytical to floating-point precision in every relation region | Partial overlap uses a grid sigmoid STE; sampled relation detection can also fail near contact |
| Internal disk tangency has a first-order area-gradient jump | For positive unequal radii, ordinary internal and external tangency are both $C^1$ |
| Method C reaches radius error $7\times10^{-6}$ and is 3000 times more precise | Superseded by the maintained grid-based demo; not current evidence of accuracy or general superiority |
| Smoothing plus Method C supplies a validated hybrid optimizer | Manual switching is possible; the experimental optimizer is unimplemented |
| TOI supplies exact cross-boundary gradient transfer | Event-time sensitivity does not establish the proposed static correction |
| CPU scaling identifies a hardware limit and establishes GPU expectations | GPU throughput remains unassessed by those CPU measurements |

The canonical mathematical and implementation scope is in
[Stratification Theory](stratification_theory.md), with the rejected static
correction in the [TOI discussion](toi_derivation.md).

## What the Maintained Demo Establishes

The [Method-C demo](../reference/method_c_demo.md) reports radius errors
$1.31\times10^{-3}$ for Method C and $2.12\times10^{-2}$ for Method A after
200 steps on one disk-union objective. Method A uses fixed
$k=\beta=0.1$; Method C has a hard sampled forward value.

These are reported results for that configuration, not newly measured here,
universal error bounds, or a search over both methods' best settings.
A smoothed objective can have a small residual while parameters remain
biased relative to the analytical target. Grid error also limits Method C.
Compare parameter error and an independent geometric objective, not only
each method's own loss.

## Findings That Remain Useful

For disjoint disks, union area is independent of centers, so a zero center
gradient is correct. Escaping the plateau requires additional exploration
information. Smoothing is one choice, without a guarantee that its signal
is large enough or points toward a desired solution.

Array-based custom-VJP dispatch can compose with batching in tested cases.
That proves neither all higher-order/forward-mode AD contracts nor accuracy
of the backward formula.

Safe intermediate values help avoid invalid derivatives from unselected
branches. Their clipping thresholds must themselves enter near-contact
accuracy analysis; see the [JAX discussion](jax_where_gradient_pitfall.md).

Do not reuse the original gates as current proof of exact intersecting
gradients, novelty, or optimization superiority.
