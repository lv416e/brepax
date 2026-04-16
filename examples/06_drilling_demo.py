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
# # Drilling Demo
#
# Differentiable Boolean subtraction: optimize a hole's radius to achieve
# a target volume. Demonstrates gradient-based design optimization through
# CAD Boolean operations.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.boolean import subtract_volume
from brepax.primitives import Cylinder, Sphere

# %% [markdown]
# ## Setup: Sphere with a cylindrical hole

# %%
block = Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=jnp.array(2.0))
sphere_vol = float(block.volume())
print(f"Sphere volume: {sphere_vol:.2f}")

# Drill a hole
hole = Cylinder(
    point=jnp.array([0.0, 0.0, 0.0]),
    axis=jnp.array([0.0, 0.0, 1.0]),
    radius=jnp.array(0.5),
)

drilled_vol = float(subtract_volume(block, hole, resolution=64))
print(f"After drilling (r=0.5): {drilled_vol:.2f}")
print(f"Material removed: {sphere_vol - drilled_vol:.2f}")

# %% [markdown]
# ## Gradient direction verification
#
# The gradient tells us how volume changes with each parameter.

# %%
# Larger hole -> less material
grad_hole = float(
    jax.grad(
        lambda r: subtract_volume(
            block,
            Cylinder(
                point=jnp.array([0.0, 0.0, 0.0]),
                axis=jnp.array([0.0, 0.0, 1.0]),
                radius=r,
            ),
            resolution=64,
        )
    )(jnp.array(0.5))
)
print(f"d(vol)/d(hole_radius) = {grad_hole:.4f} (negative: larger hole removes more)")

# Larger block -> more material
grad_block = float(
    jax.grad(
        lambda r: subtract_volume(
            Sphere(center=jnp.array([0.0, 0.0, 0.0]), radius=r),
            hole,
            resolution=64,
        )
    )(jnp.array(2.0))
)
print(f"d(vol)/d(block_radius) = {grad_block:.4f} (positive)")

# %% [markdown]
# ## Optimization: find hole radius for target volume
#
# Target: remove 20% of the sphere volume.

# %%
target_vol = 0.8 * sphere_vol
print(f"Target volume: {target_vol:.2f} (80% of sphere)")


def loss(hole_radius):
    hole = Cylinder(
        point=jnp.array([0.0, 0.0, 0.0]),
        axis=jnp.array([0.0, 0.0, 1.0]),
        radius=hole_radius,
    )
    vol = subtract_volume(block, hole, resolution=64)
    return (vol - target_vol) ** 2


# Gradient descent
r = jnp.array(0.3)
print(f"\n{'step':>4} {'radius':>8} {'volume':>10} {'loss':>12}")
print("-" * 38)
for step in range(80):
    g = jax.grad(loss)(r)
    r = r - 0.001 * g
    r = jnp.maximum(r, 0.01)

    if step % 20 == 0 or step == 79:
        vol = float(
            subtract_volume(
                block,
                Cylinder(
                    point=jnp.array([0.0, 0.0, 0.0]),
                    axis=jnp.array([0.0, 0.0, 1.0]),
                    radius=r,
                ),
                resolution=64,
            )
        )
        print(f"{step:>4} {float(r):>8.4f} {vol:>10.2f} {float(loss(r)):>12.4f}")

final_vol = float(
    subtract_volume(
        block,
        Cylinder(
            point=jnp.array([0.0, 0.0, 0.0]),
            axis=jnp.array([0.0, 0.0, 1.0]),
            radius=r,
        ),
        resolution=64,
    )
)
vol_err = abs(final_vol - float(target_vol)) / float(target_vol)
print(f"\nFinal: radius={float(r):.4f}, volume={final_vol:.2f}, error={vol_err:.2%}")

# %% [markdown]
# ## Hole size comparison

# %%
print(f"{'radius':>8} {'drilled_vol':>12} {'removed':>10} {'% removed':>10}")
print("-" * 45)
for r_val in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
    h = Cylinder(
        point=jnp.array([0.0, 0.0, 0.0]),
        axis=jnp.array([0.0, 0.0, 1.0]),
        radius=jnp.array(r_val),
    )
    vol = float(subtract_volume(block, h, resolution=64))
    removed = sphere_vol - vol
    pct = removed / sphere_vol
    print(f"{r_val:>8.1f} {vol:>12.2f} {removed:>10.2f} {pct:>10.1%}")
