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
# # Stratum Tracking
#
# How BRepAX classifies topological strata and dispatches gradients.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.analytical.disk_disk import (
    disk_disk_boundary_distance,
    disk_disk_stratum_label,
)
from brepax.boolean import union_area
from brepax.primitives import Disk

# %% [markdown]
# ## Three strata for two disks
#
# | Label | Condition | Description |
# |-------|-----------|-------------|
# | 0 | d >= r1 + r2 | Disjoint |
# | 1 | \|r1 - r2\| < d < r1 + r2 | Intersecting |
# | 2 | d <= \|r1 - r2\| | Contained |

# %%
configs = {
    "disjoint": (5.0, "d=5 >> r1+r2=2"),
    "intersecting": (1.5, "r1-r2 < d=1.5 < r1+r2"),
    "contained": (0.3, "d=0.3 < |r1-r2|=1"),
}

c1, r1, r2 = jnp.array([0.0, 0.0]), jnp.array(1.5), jnp.array(0.5)

for name, (d_val, desc) in configs.items():
    c2 = jnp.array([d_val, 0.0])
    label = int(disk_disk_stratum_label(c1, r1, c2, r2))
    bdist = float(disk_disk_boundary_distance(c1, r1, c2, r2))
    print(f"{name:>13}: label={label}, boundary_dist={bdist:.2f}  ({desc})")

# %% [markdown]
# ## Gradient behavior per stratum

# %%
print(f"{'stratum':>13} {'d(area)/d(r1)':>14} {'d(area)/d(c2_x)':>16}")
print("-" * 48)

for name, (d_val, _) in configs.items():
    c2 = jnp.array([d_val, 0.0])
    b = Disk(center=c2, radius=r2)

    grad_r = float(
        jax.grad(lambda r: union_area(Disk(center=c1, radius=r), b, method="stratum"))(
            r1
        )
    )

    grad_c = float(
        jax.grad(
            lambda cx: union_area(
                Disk(center=c1, radius=r1),
                Disk(center=jnp.array([cx, 0.0]), radius=r2),
                method="stratum",
            )
        )(jnp.array(d_val))
    )

    print(f"{name:>13} {grad_r:>14.4f} {grad_c:>16.6f}")

# %% [markdown]
# ## Boundary proximity: gradient accuracy sweep
#
# As we approach the external tangent boundary (d -> r1 + r2),
# Method (A) degrades while Method (C) maintains precision.

# %%
r1_test, r2_test = jnp.array(1.0), jnp.array(1.0)
from brepax.analytical.disk_disk import disk_disk_union_area

print(f"{'boundary_dist':>14} {'Method A err':>14} {'Method C err':>14}")
print("-" * 46)

for d_val in [1.5, 1.9, 1.99, 1.999]:
    c2 = jnp.array([d_val, 0.0])
    exact = float(
        jax.grad(disk_disk_union_area, argnums=1)(
            jnp.array([0.0, 0.0]),
            r1_test,
            c2,
            r2_test,
        )
    )
    b = Disk(center=c2, radius=r2_test)

    ga = float(
        jax.grad(
            lambda r: union_area(
                Disk(center=jnp.array([0.0, 0.0]), radius=r),
                b,
                method="smoothing",
                k=0.1,
                beta=0.1,
                resolution=128,
            )
        )(r1_test)
    )
    gc = float(
        jax.grad(
            lambda r: union_area(
                Disk(center=jnp.array([0.0, 0.0]), radius=r),
                b,
                method="stratum",
            )
        )(r1_test)
    )

    err_a = abs(ga - exact) / max(abs(exact), 1e-12)
    err_c = abs(gc - exact) / max(abs(exact), 1e-12)
    bdist = 2.0 - d_val
    print(f"{bdist:>14.3f} {err_a:>14.4%} {err_c:>14.4%}")
