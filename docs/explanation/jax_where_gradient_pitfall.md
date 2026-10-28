# JAX jnp.where Gradient Pitfall

A recurring pattern in differentiable geometry code using JAX.

## The Problem

`jnp.where(cond, x, y)` evaluates **both** `x` and `y` during forward and backward passes, then selects the result based on `cond`. During gradient computation, the gradient of the **unselected** branch is still computed and can produce NaN or Inf if that branch involves:

- Division by zero (e.g., `1 / d` where `d = 0` in the unselected regime)
- `arccos` at domain boundaries (`arccos(1.0)` has infinite gradient)
- `log(0)` or `sqrt(0)` in unselected branches

The NaN propagates through the selected branch's gradient, corrupting the final result even though the forward pass returns the correct value.

## Example: Two-Disk Intersection Area

The analytical intersection area formula uses:

```python
cos_alpha = (d**2 + r1**2 - r2**2) / (2 * d * r1)  # division by d
alpha = jnp.arccos(cos_alpha)                          # arccos at boundary
lens = r1**2 * (alpha - sin(2*alpha)/2) + ...
return jnp.where(disjoint, 0.0, jnp.where(contained, contained_area, lens))
```

When disks are disjoint (`d >> r1 + r2`), the forward pass correctly returns `0.0`. But the backward pass computes `d(lens)/d(r1)` anyway, which involves `d(arccos(clip(x)))/dx` at `x = 1.0` (infinite gradient), producing NaN.

## The Fix: Safe Primal Values

Replace potentially dangerous intermediate values with safe substitutes in branches that will be masked out:

```python
safe_d = jnp.maximum(d, 1e-10)              # prevent division by zero
cos_alpha = jnp.clip(
    (safe_d**2 + r1**2 - r2**2) / (2 * safe_d * r1),
    -1.0 + eps, 1.0 - eps,                  # keep arccos away from domain boundary
)
```

The `eps` margin keeps `arccos` away from +/-1, and `safe_d` prevents division
by zero. These safeguards can also modify the selected lens branch near
tangency or at very small scales. They therefore change values and derivatives
there; finiteness alone does not establish accuracy. The distance floor has
length units, while the cosine margin is dimensionless. Near-contact checks
need independent limits or stable formulas outside the clipping regime.

## Relation to Custom VJP

Safe primals address invalid intermediate derivatives. A custom VJP permits
a separately specified backward rule, but does not itself make that rule
exact or prevent evaluation of unused expressions. BRepAX currently evaluates
candidate gradients and selects with array operations; its intersecting rule
is a grid STE. See [ADR-0004](../architecture/adr/0004-custom-vjp-at-boundaries.md)
and the [implementation contract](stratification_theory.md#current-brepax-implementation).
