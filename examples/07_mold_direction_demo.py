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
# # Mold Direction Optimization
#
# Find the optimal mold pull direction for a non-convex part using
# gradient-based optimization.  BRepAX computes a smooth undercut metric
# that is differentiable with respect to the pull direction, enabling
# projected gradient descent on the unit sphere S^2.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.experimental.applications.mold_direction import (
    optimize_mold_direction,
    undercut_volume,
)
from brepax.primitives import Box, Cylinder

# %% [markdown]
# ## Shape definition: L-bracket with a through-hole
#
# A rectangular box with a notch cut from the +x/+z corner creates
# an L-bracket.  A cylindrical through-hole along the y-axis adds
# a realistic mounting feature.  The notch breaks the centrosymmetry,
# giving the optimizer a clear preferred pull direction.

# %%
body = Box(
    center=jnp.array([0.0, 0.0, 0.0]),
    half_extents=jnp.array([1.5, 1.0, 1.0]),
)
notch = Box(
    center=jnp.array([1.0, 0.0, 0.5]),
    half_extents=jnp.array([0.6, 1.1, 0.6]),
)
hole = Cylinder(
    point=jnp.array([0.0, 0.0, 0.0]),
    axis=jnp.array([0.0, 1.0, 0.0]),
    radius=jnp.array(0.3),
)

lo = jnp.array([-2.5, -2.0, -2.0])
hi = jnp.array([2.5, 2.0, 2.0])


def bracket_sdf(x):
    """L-bracket with through-hole: body - notch - hole."""
    return jnp.maximum(jnp.maximum(body.sdf(x), -notch.sdf(x)), -hole.sdf(x))


print("Shape: L-bracket with notch and through-hole")
print(f"  Body half-extents: {body.half_extents}")
print(f"  Notch center: {notch.center}, half-extents: {notch.half_extents}")
print(f"  Hole axis: y, radius: {float(hole.radius)}")

# %% [markdown]
# ## Undercut sweep: compare pull directions
#
# The undercut metric uses a surface-weighted softplus formulation:
# surface points whose outward normal opposes the pull direction
# contribute proportionally to the angle of opposition.

# %%
res = 48
directions = {
    "+x": jnp.array([1.0, 0.0, 0.0]),
    "-x": jnp.array([-1.0, 0.0, 0.0]),
    "+y": jnp.array([0.0, 1.0, 0.0]),
    "-y": jnp.array([0.0, -1.0, 0.0]),
    "+z": jnp.array([0.0, 0.0, 1.0]),
    "-z": jnp.array([0.0, 0.0, -1.0]),
    "+x+z (into notch)": jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2.0),
    "-x-z (away)": jnp.array([-1.0, 0.0, -1.0]) / jnp.sqrt(2.0),
}

print(f"\n{'direction':>25} {'undercut':>10}")
print("-" * 38)
for name, d in directions.items():
    uc = float(undercut_volume(bracket_sdf, d, lo=lo, hi=hi, resolution=res))
    print(f"{name:>25} {uc:>10.4f}")

# %% [markdown]
# ## Gradient verification
#
# The gradient of the undercut metric with respect to the pull direction
# should point toward lower-undercut directions.

# %%
d_test = jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2.0)
grad = jax.grad(
    lambda d: undercut_volume(bracket_sdf, d, lo=lo, hi=hi, resolution=res)
)(d_test)
print(
    f"Direction: [{float(d_test[0]):+.3f}, {float(d_test[1]):+.3f}, {float(d_test[2]):+.3f}]"
)
print(
    f"Gradient:  [{float(grad[0]):+.4f}, {float(grad[1]):+.4f}, {float(grad[2]):+.4f}]"
)
print(f"|grad|:    {float(jnp.linalg.norm(grad)):.4f}")

# %% [markdown]
# ## Optimization
#
# Start from a sub-optimal direction (into the notch, +x+z) and let
# projected gradient descent on S^2 find a better pull direction.

# %%
result = optimize_mold_direction(
    bracket_sdf,
    initial_direction=jnp.array([1.0, 0.0, 1.0]),
    lo=lo,
    hi=hi,
    resolution=res,
    steps=200,
    lr=0.02,
    tol=1e-5,
)

print(f"\n{'step':>4} {'loss':>10} {'direction':>30}")
print("-" * 48)
for i, loss in enumerate(result.losses):
    if i % 10 == 0 or i == len(result.losses) - 1:
        d = result.trajectory[min(i, result.trajectory.shape[0] - 1)]
        print(
            f"{i:>4} {loss:>10.4f}   "
            f"[{float(d[0]):+.3f}, {float(d[1]):+.3f}, {float(d[2]):+.3f}]"
        )

# %% [markdown]
# ## Results

# %%
d_init = jnp.array([1.0, 0.0, 1.0]) / jnp.sqrt(2.0)
d_final = result.direction

uc_initial = float(undercut_volume(bracket_sdf, d_init, lo=lo, hi=hi, resolution=res))
uc_final = float(undercut_volume(bracket_sdf, d_final, lo=lo, hi=hi, resolution=res))
reduction = (uc_initial - uc_final) / uc_initial

print(
    f"Initial direction: [{float(d_init[0]):+.4f}, {float(d_init[1]):+.4f}, {float(d_init[2]):+.4f}]"
)
print(f"Initial undercut:  {uc_initial:.4f}")
print()
print(
    f"Final direction:   [{float(d_final[0]):+.4f}, {float(d_final[1]):+.4f}, {float(d_final[2]):+.4f}]"
)
print(f"Final undercut:    {uc_final:.4f}")
print()
print(f"Reduction:         {reduction:.1%}")
print(f"Converged:         {result.converged} (after {len(result.losses)} steps)")
