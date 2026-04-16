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
# # STEP to Primitives
#
# Convert faces from STEP files into BRepAX native primitives,
# then evaluate SDFs at test points using the converted geometry.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp

from brepax.brep.convert import faces_to_primitives
from brepax.io.step import read_step

jax.config.update("jax_enable_x64", True)

fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# %% [markdown]
# ## Box: six planar faces

# %%
box = read_step(fixtures / "sample_box.step")
box_prims = faces_to_primitives(box)

print(f"Box faces: {len(box_prims)}")
for i, p in enumerate(box_prims):
    if p is not None:
        params = p.parameters()
        print(f"  face {i}: {type(p).__name__} -> {params}")

# %% [markdown]
# ## Sphere: single spherical face

# %%
sphere = read_step(fixtures / "sample_sphere.step")
sphere_prims = faces_to_primitives(sphere)

print(f"Sphere faces: {len(sphere_prims)}")
for i, p in enumerate(sphere_prims):
    if p is not None:
        params = p.parameters()
        print(f"  face {i}: {type(p).__name__} -> {params}")

# %% [markdown]
# ## SDF evaluation on converted primitives
#
# Evaluate the sphere SDF at a point outside and a point inside.

# %%
sph = sphere_prims[0]
assert sph is not None

outside = jnp.array([5.0, 0.0, 0.0])
inside = jnp.array([1.0, 0.0, 0.0])

print(f"Sphere SDF at {outside}: {float(sph.sdf(outside)):.4f}")
print(f"Sphere SDF at {inside}: {float(sph.sdf(inside)):.4f}")

# %% [markdown]
# Evaluate a plane SDF from the box at a test point.

# %%
plane = box_prims[0]
assert plane is not None

test_pt = jnp.array([5.0, 10.0, 15.0])
print(f"Plane normal: {plane.parameters()['normal']}")
print(f"Plane offset: {float(plane.parameters()['offset']):.4f}")
print(f"Plane SDF at {test_pt}: {float(plane.sdf(test_pt)):.4f}")
