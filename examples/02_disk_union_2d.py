# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Disk Union: Method (A) vs Method (C)
#
# Compares smoothing (Method A) and stratum-aware (Method C) gradient
# computation for the union area of two 2D disks.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.analytical.disk_disk import disk_disk_union_area
from brepax.boolean import union_area
from brepax.primitives import Disk

# %% [markdown]
# ## Setup: two overlapping disks

# %%
c1, r1 = jnp.array([0.0, 0.0]), jnp.array(1.0)
c2, r2 = jnp.array([1.5, 0.0]), jnp.array(1.0)

analytical_area = disk_disk_union_area(c1, r1, c2, r2)
analytical_grad = jax.grad(disk_disk_union_area, argnums=1)(c1, r1, c2, r2)
print(f"Analytical union area: {analytical_area:.6f}")
print(f"Analytical d(area)/d(r1): {analytical_grad:.6f}")

# %% [markdown]
# ## Method (A): Smoothing
#
# Uses smooth-min SDF composition with temperature `k` and sigmoid
# area integral with sharpness `beta`.

# %%
for k_beta in [0.01, 0.05, 0.1, 0.5, 1.0]:

    def area_fn(r):
        a = Disk(center=c1, radius=r)
        b = Disk(center=c2, radius=r2)
        return union_area(
            a, b, method="smoothing", k=k_beta, beta=k_beta, resolution=256
        )

    grad_a = jax.grad(area_fn)(r1)
    err = abs(float(grad_a) - float(analytical_grad)) / abs(float(analytical_grad))
    print(f"k=beta={k_beta:.2f}: grad={float(grad_a):.6f}, error={err:.4%}")

# %% [markdown]
# ## Method (C): Stratum-Aware
#
# Uses exact SDF Boolean with stratum-dispatched gradients.
# No temperature parameters.


# %%
def area_fn_c(r):
    a = Disk(center=c1, radius=r)
    b = Disk(center=c2, radius=r2)
    return union_area(a, b, method="stratum")


grad_c = jax.grad(area_fn_c)(r1)
err_c = abs(float(grad_c) - float(analytical_grad)) / abs(float(analytical_grad))
print(f"Method (C): grad={float(grad_c):.6f}, error={err_c:.4%}")

# %% [markdown]
# ## Resolution scaling (Method C)

# %%
for res in [64, 128, 256, 512]:

    def area_fn_res(r):
        a = Disk(center=c1, radius=r)
        b = Disk(center=c2, radius=r2)
        return union_area(a, b, method="stratum", resolution=res)

    grad = jax.grad(area_fn_res)(r1)
    err = abs(float(grad) - float(analytical_grad)) / abs(float(analytical_grad))
    print(f"res={res}: grad={float(grad):.6f}, error={err:.4%}")

# %% [markdown]
# ## Boundary proximity comparison
#
# Near the external tangent boundary (d approaching r1 + r2),
# Method (A) degrades while Method (C) maintains precision.

# %%
print(f"{'eps':>8} {'Method A err':>14} {'Method C err':>14} {'ratio':>8}")
print("-" * 50)

for d_val in [1.5, 1.9, 1.99]:
    eps = 2.0 - d_val
    c2_near = jnp.array([d_val, 0.0])
    exact = jax.grad(disk_disk_union_area, argnums=1)(c1, r1, c2_near, r2)

    def fa(r):
        return union_area(
            Disk(center=c1, radius=r),
            Disk(center=c2_near, radius=r2),
            method="smoothing",
            k=0.1,
            beta=0.1,
            resolution=128,
        )

    def fc(r):
        return union_area(
            Disk(center=c1, radius=r),
            Disk(center=c2_near, radius=r2),
            method="stratum",
        )

    err_a = abs(float(jax.grad(fa)(r1)) - float(exact)) / max(abs(float(exact)), 1e-12)
    err_c = abs(float(jax.grad(fc)(r1)) - float(exact)) / max(abs(float(exact)), 1e-12)
    ratio = err_a / max(err_c, 1e-15)
    print(f"{eps:>8.2f} {err_a:>14.4%} {err_c:>14.4%} {ratio:>8.0f}x")
