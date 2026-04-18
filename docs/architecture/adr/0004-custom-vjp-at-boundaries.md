# ADR-0004: Custom VJP at Boundaries

## Status

Accepted

## Context

At stratum boundaries (e.g., two disks at tangency, d = r1 + r2), the union area formula has a gradient discontinuity. The `_intersection_area` function uses `jnp.where` for branchless evaluation, but `jnp.where` evaluates both branches and the gradient of the "wrong" branch leaks through at the boundary. The core thesis (contact dynamics analogy) requires correct gradients at these topology changes.

Early prototyping confirmed that `eqx.filter_jit`, `eqx.filter_vmap`, and `eqx.filter_grad` are required for Equinox modules, and that all traced control flow must use array operations rather than Python conditionals.

## Candidates Evaluated

### (a) `jax.lax.cond` with pure autodiff per branch

Simplest implementation. Each branch has well-defined gradient. However, `lax.cond` cannot vectorize different branches for different batch elements under vmap -- a known JAX limitation. Also, the boolean predicate itself depends on differentiable parameters, and its own gradient is lost.

### (b) Full `jax.custom_vjp` with stratum label as residual

Complete control over backward pass. Forward evaluates normally (branchless `jnp.where`). Backward saves stratum label as residual and dispatches to the correct gradient formula per stratum using `jnp.where`. Fully vmap-compatible because both residuals and gradient dispatch use array operations. Standard pattern in differentiable physics (Brax, DiffTaichi contact handling).

### (c) `optimistix.bisection` for boundary distance + smooth blending

Sidesteps discontinuity by smooth-blending across a boundary band. Tunable epsilon for accuracy/smoothness tradeoff. However, this is fundamentally a smoothing approach -- it does not provide exact stratum-aware gradients. The epsilon parameter introduces a bias that does not vanish. For the two-disk case, boundary distance is already computed in closed form, making bisection unnecessary overhead.

### (d) Hybrid (b) + (c)

Clean gradients in stratum interiors via custom_vjp, smooth transition near boundaries via blending. The transition between modes itself introduces another discontinuity at distance = epsilon. Significantly more complex than either alone. Over-engineered for the concept proof.

## Decision

Use **(b): full `jax.custom_vjp`** with stratum label saved as residual.

The forward pass keeps the existing branchless `jnp.where` evaluation. The backward pass uses `fwd` to save `(primal_out, stratum_label)` as residuals, and `bwd` dispatches to the correct gradient formula per stratum using `jnp.where`. The stratum label residual uses the scalar int encoding from ADR-0003.

**(a) is rejected** due to vmap incompatibility. **(c) and (d) are rejected** because they contradict Method (C)'s intent of exact stratum-aware gradients.

## Boundary Exact Case Convention

When the configuration lies exactly on a stratum boundary (boundary distance = 0), the gradient is mathematically undefined (discontinuous). A convention is required.

### Candidates for boundary convention

1. **Intersecting stratum gradient** (chosen): Use the gradient formula for the intersecting regime. Simplest implementation -- no additional state needed.
2. **Path-dependent (previous stratum)**: Use the gradient from whichever stratum the optimization trajectory came from. Requires storing `prev_label` in custom_vjp residuals, adding state complexity.
3. **Subgradient (convex combination)**: Average gradients from adjacent strata. Correct from convex analysis but significantly more complex, and the boundary is measure-zero in parameter space.

**Chosen: (1) intersecting stratum gradient.**

Rationale: The boundary is a measure-zero set in parameter space, so the convention choice does not affect optimization convergence in exact arithmetic. In finite-precision arithmetic, the convention affects behavior only when a configuration lands within machine epsilon of the boundary -- a probability-zero event in practice. The intersecting formula is the natural "generic" regime (the highest-dimensional stratum in the Whitney stratification), making it the canonical default.

### Implications for boundary proximity evaluation

When sweeping boundary distance epsilon toward 0:

- **Approaching from intersecting side**: Gradient is smooth and continuous at the boundary (the convention matches the interior formula). No discontinuity.
- **Approaching from disjoint side**: At epsilon = 0 exactly, the gradient jumps to the intersecting formula. For epsilon > 0 (however small), the disjoint gradient is used. This produces a step discontinuity at the boundary that is visible in boundary proximity benchmarks.
- **Expected benchmark signature**: Axis 2 (boundary proximity sweep) should show Method (C) with asymmetric error behavior -- smooth from the intersecting side, with a jump from the disjoint side. This asymmetry is a feature of the convention, not a bug. Method (A) smoothing will show symmetric but biased behavior near the boundary.

### Future extension

For N > 2 primitives with more than 3 strata, the "generic stratum" convention generalizes to: use the gradient from the highest-dimensional stratum adjacent to the boundary. This may require a new ADR if the adjacency structure becomes non-trivial.

## Gate Criterion Interpretation

Method comparison must distinguish **interior** from **boundary-proximal** measurements:

- **Interior** (boundary distance >> k): Method (A) smoothing achieves high accuracy here (< 0.01% relative error at k=0.01). Method (C) stratum-aware will also be accurate in the interior. **Do not expect differentiation between methods in the interior** -- both should match analytical gradients closely.
- **Boundary-proximal** (boundary distance in [0.001, 0.1]): This is where the methods diverge. Method (A) accuracy degrades when k > boundary distance (temperature exceeds the geometric scale of the boundary). Method (C) should maintain analytical accuracy regardless of boundary proximity.
- **Gate criterion 1** ("Method (C) error is 1/10 of Method (A) at epsilon in [0.001, 0.1]") applies **only to boundary-proximal measurements**. Comparing methods at interior points is not informative for this criterion.

Empirical confirmation from the Method (A) boundary proximity benchmark:

| boundary dist | k=0.01 rel_err | k=0.1 rel_err | k=1.0 rel_err |
|--------------|----------------|---------------|---------------|
| 0.50 | 0.009% | 0.89% | 2.6% |
| 0.10 | 0.16% | 0.89% | 12.8% |
| 0.01 | 0.06% | **5.2%** | **18.3%** |

Key observation: at boundary distance 0.01, Method (A) with k=0.1 shows 5.2% error. Method (C) must achieve < 0.52% at the same configuration to satisfy the 1/10 criterion.

**Note:** The table above measured the **external tangent** approach only. Internal tangent measurements show different behavior:

| boundary dist | k=0.1 external | k=0.1 internal | k=1.0 external | k=1.0 internal |
|--------------|----------------|----------------|----------------|----------------|
| 0.50 | 0.89% | 0.58% | 2.6% | 12.2% |
| 0.10 | 0.89% | 0.60% | 12.8% | 18.3% |
| 0.01 | **5.2%** | **1.5%** | **18.3%** | **22.1%** |

At k=1.0, internal tangent degradation exceeds external tangent. This occurs because large smoothing distorts the containment structure (one disk's smoothing kernel covers the other).

### Gradient jump order at boundaries

TOI derivation (see `docs/explanation/toi_derivation.md`) revealed that the gradient jump at **external tangent is second-order**: the area gradient is continuous at $d = r_1 + r_2$. At **internal tangent, the jump is first-order**: the contained-stratum gradient ($\nabla f = (0, 0, 2\pi r_1, 0, 0, 0)$) differs from the intersecting-stratum gradient in center components.

**Core insight: Method (C)'s differentiation from Method (A) is not primarily about correcting gradient discontinuities. It is about eliminating smoothing bias.** Method (A) introduces systematic bias from the temperature parameters k and beta, which distort both the SDF field and the area integral. Method (C) uses exact stratum-aware formulas, avoiding this bias entirely. This distinction holds at both external and internal tangent boundaries.

## Consequences

- Method (C) implementation will use `jax.custom_vjp` with `(primal_out, stratum_label)` as residuals
- Gradient validation tests must include near-boundary configurations from both sides
- Benchmark Axis 2 interpretation must account for the asymmetric error signature described above
- The convention is documented here so that gate evaluation can distinguish expected from unexpected behavior
- See `docs/explanation/jax_where_gradient_pitfall.md` for the safe-primal pattern used in the analytical ground truth function
