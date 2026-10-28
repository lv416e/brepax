# ADR-0004: Custom VJP at Boundaries

## Status

Accepted: retain custom-VJP relation dispatch.

Mathematical rationale corrected on 2026-10-28. The historical choice of
a backward-rule interface is not proof of exact boundary derivatives.

## Historical Context and Decision

The concept proof selected `jax.custom_vjp` with a relation label saved
as a residual. Array dispatch selects primitive-volume derivatives or an
intersecting estimate and supports tested batching cases.

The original design preferred an intersecting formula at exact tangency,
assuming every contact had an undefined gradient and partial overlap was
uniquely highest-dimensional. Both premises were incorrect. The current
generic implementation uses sampled labels, not an exact-contact classifier.

## Corrected Mathematical Scope

Ordinary external and unequal-radius internal disk contacts have continuous
first area derivatives. Other geometries or objectives can have different
one-sided limits; equal-radius coincidence is a distinct degeneracy.
Four relation labels do not establish a Whitney stratification.
[Stratification Theory](../../explanation/stratification_theory.md)
defines these distinctions and the current estimator.

Neither a universal tangency jump nor an asymmetric error signature follows
from this ADR. A measure-zero contact set does not make its convention
irrelevant: an optimizer or deterministic fixture can land on it exactly.
A nonsmooth derivative convention needs its own contract.

The current intersecting custom VJP differentiates a smoothed indicator but
returns a hard sampled forward value. It eliminates neither smoothing nor
quadrature error and is generally not the derivative of that discrete
forward computation. Historical boundary-proximity targets were hypotheses,
not implementation guarantees.

## JAX Rationale

`lax.cond` is not incompatible with `vmap`: with batched predicates JAX
converts it to selection. Both branches are traced; domain safety still
matters. See [JAX documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html).

A custom VJP supplies a backward rule with a stated mathematical target.
It does not itself make the rule exact, avoid all unused expressions, or
ensure every AD transform works. Nonlinear dependence on primal values
does not exclude a custom JVP; linearity concerns tangent inputs.

## Consequences

- Retain the custom-VJP interface and tested array dispatch.
- Validate values and backward rules against their separate stated targets.
- Include one-sided limits and near-contact detection failures.
- Use safe intermediates where needed, with accuracy limits described in
  the [JAX discussion](../../explanation/jax_where_gradient_pitfall.md).
