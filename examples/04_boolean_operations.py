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
# # Boolean Operations
#
# Union, subtract, and intersect with stratum-dispatched gradients.
# Demonstrates analytical exact gradient in disjoint/contained strata.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.boolean import intersect_volume, subtract_volume, union_volume
from brepax.primitives import Box, Sphere, Torus

# %% [markdown]
# ## Sphere + Sphere: all strata


# %%
def measure_gradient(setup_fn, name):
    """Measure gradient and compare to analytical."""
    grad = float(jax.grad(setup_fn)(jnp.array(1.0)))
    analytical = float(4 * jnp.pi * 1.0**2)
    err = abs(grad - analytical) / analytical
    print(f"  {name}: grad={grad:.6f}, analytical={analytical:.6f}, err={err:.6%}")


# Disjoint: union = vol_a + vol_b
print("Disjoint (d=5):")
measure_gradient(
    lambda r: union_volume(
        Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=r),
        Sphere(center=jnp.array([5.0, 0.0, 0.0]), radius=jnp.array(1.0)),
        resolution=64,
    ),
    "d(union)/d(r1)",
)

# Contained: union = vol_outer
print("\nContained (d=0, r1=2 > r2=0.5):")
print("  d(union)/d(r_outer):")
grad_outer = float(
    jax.grad(
        lambda r: union_volume(
            Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=r),
            Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(0.5)),
            resolution=64,
        ),
    )(jnp.array(2.0))
)
analytical_outer = float(4 * jnp.pi * 2.0**2)
print(
    f"    grad={grad_outer:.4f}, err={abs(grad_outer - analytical_outer) / analytical_outer:.4%}"
)

# %% [markdown]
# ## Sphere - Sphere: subtract strata

# %%
print("Disjoint subtract (d=5):")
print("  subtract = vol_a, d(subtract)/d(r_hole) = 0")
grad_hole = float(
    jax.grad(
        lambda r: subtract_volume(
            Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(2.0)),
            Sphere(center=jnp.array([5.0, 0.0, 0.0]), radius=r),
            resolution=64,
        ),
    )(jnp.array(1.0))
)
print(f"  d(subtract)/d(r_hole) = {grad_hole:.8f}")

print("\nContained B-in-A (r_big=2, r_small=0.3):")
grad_small = float(
    jax.grad(
        lambda r: subtract_volume(
            Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(2.0)),
            Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=r),
            resolution=64,
        ),
    )(jnp.array(0.3))
)
analytical_neg = float(-4 * jnp.pi * 0.3**2)
print(
    f"  d/dr_small={grad_small:.4f}, err={abs(grad_small - analytical_neg) / abs(analytical_neg):.4%}"
)

# %% [markdown]
# ## Heterogeneous pairs: Box + Torus

# %%
box = Box(
    center=jnp.array([0.0, 0.0, 0.0]),
    half_extents=jnp.array([3.0, 3.0, 3.0]),
)
torus = Torus(
    center=jnp.array([0.0, 0.0, 0.0]),
    axis=jnp.array([0.0, 0.0, 1.0]),
    major_radius=jnp.array(1.5),
    minor_radius=jnp.array(0.3),
)

vol_sub = subtract_volume(box, torus, resolution=48)
vol_union = union_volume(box, torus, resolution=48)
vol_inter = intersect_volume(box, torus, resolution=48)

print(f"Box volume:            {float(box.volume()):.2f}")
print(f"Torus volume:          {float(torus.volume()):.2f}")
print(f"Box - Torus:           {float(vol_sub):.2f}")
print(f"Box + Torus (union):   {float(vol_union):.2f}")
print(f"Box & Torus (intersect): {float(vol_inter):.2f}")

# %% [markdown]
# ## Gradient of subtract w.r.t. torus minor radius


# %%
def drilled_box_vol(minor_r):
    t = Torus(
        center=jnp.array([0.0, 0.0, 0.0]),
        axis=jnp.array([0.0, 0.0, 1.0]),
        major_radius=jnp.array(1.5),
        minor_radius=minor_r,
    )
    return subtract_volume(box, t, resolution=48)


grad = jax.grad(drilled_box_vol)(jnp.array(0.3))
print(f"d(box-torus volume)/d(minor_radius) = {float(grad):.4f}")
print("Larger torus -> more material removed -> gradient should be negative")
