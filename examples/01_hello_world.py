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
# # Hello World
#
# Get started with BRepAX in 5 minutes.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.primitives import Disk, Sphere

# %% [markdown]
# ## 2D: Disk primitive

# %%
disk = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))

# Evaluate SDF at a query point
query = jnp.array([0.5, 0.0])
print(f"SDF at {query}: {disk.sdf(query):.4f}")
print(f"SDF at boundary: {disk.sdf(jnp.array([1.0, 0.0])):.4f}")
print(f"SDF outside: {disk.sdf(jnp.array([2.0, 0.0])):.4f}")

# %% [markdown]
# ## Gradients flow through everything

# %%
# Gradient of SDF w.r.t. query point
grad_fn = jax.grad(lambda x: disk.sdf(x))
gradient = grad_fn(jnp.array([2.0, 0.0]))
print(f"SDF gradient at (2,0): {gradient}")

# Gradient of SDF w.r.t. radius
import equinox as eqx

grad_disk = eqx.filter_grad(lambda d: d.sdf(jnp.array([2.0, 0.0])))(disk)
print(f"d(SDF)/d(radius): {grad_disk.radius:.4f}")

# %% [markdown]
# ## 3D: Sphere primitive

# %%
sphere = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(1.0))

points = jnp.array(
    [
        [0.0, 0.0, 0.0],  # center (inside)
        [1.0, 0.0, 0.0],  # surface
        [2.0, 0.0, 0.0],  # outside
    ]
)

sdfs = eqx.filter_vmap(sphere.sdf)(points)
print(f"SDF values: {sdfs}")
print(f"Analytical volume: {sphere.volume():.4f}")

# %% [markdown]
# ## Boolean operations

# %%
from brepax.boolean import union_area

disk_a = Disk(center=jnp.array([0.0, 0.0]), radius=jnp.array(1.0))
disk_b = Disk(center=jnp.array([1.5, 0.0]), radius=jnp.array(1.0))

area = union_area(disk_a, disk_b, method="stratum")
print(f"Union area: {area:.4f}")

# Gradient of union area w.r.t. radius
grad = jax.grad(
    lambda r: union_area(
        Disk(center=jnp.array([0.0, 0.0]), radius=r),
        disk_b,
        method="stratum",
    )
)(jnp.array(1.0))
print(f"d(union_area)/d(r1): {grad:.4f}")
