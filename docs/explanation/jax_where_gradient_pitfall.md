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

The `eps` margin ensures that `arccos` never evaluates at exactly +/-1, where its gradient is infinite. The `safe_d` floor prevents division by zero. Both modifications only affect branches that are masked out by `jnp.where`, so forward-pass accuracy is unchanged.

## Relation to Custom VJP

This workaround is a pragmatic fix for the analytical ground truth function. The general solution is `jax.custom_vjp`, which allows computing the correct gradient per stratum without evaluating unused branches. This is the approach adopted for the stratum-aware Boolean method (see ADR-0004).

The safe-primal pattern and the custom_vjp approach are complementary: safe primals prevent NaN in simple functions where custom_vjp would be over-engineering, while custom_vjp provides exact gradients at stratum boundaries where safe primals only suppress NaN without correcting the gradient value.
