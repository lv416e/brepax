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
# # From STEP File to Gradient-Based Optimization
#
# This notebook demonstrates BRepAX's end-to-end pipeline:
# load a STEP file, reconstruct its CSG tree, compute a differentiable
# volume, and optimize a design parameter using JAX gradients.
#
# The entire pipeline runs in ~20 lines of code.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# %% [markdown]
# ## 1. Load and Inspect
#
# Read a STEP file containing a 40x30x20 box with two through-holes
# (radii 4 and 3, along the Z axis).

# %%
from brepax.brep.convert import faces_to_primitives, shape_metadata
from brepax.io.step import read_step

shape = read_step(fixtures / "box_with_holes.step")
meta = shape_metadata(shape)

print(f"Faces: {meta.n_faces}  Edges: {meta.n_edges}  Vertices: {meta.n_vertices}")
print(f"Face types: {meta.face_types}")

primitives = faces_to_primitives(shape)
for i, p in enumerate(primitives):
    if p is not None:
        print(f"  face {i}: {type(p).__name__}")

# %% [markdown]
# ## 2. Build Adjacency Graph and Reconstruct CSG Tree
#
# The adjacency graph captures which faces share edges.
# `reconstruct_stock_minus_features` identifies the bounding box
# from planar faces and cylindrical holes from non-planar faces.

# %%
from brepax.brep.csg import CSGOperation, reconstruct_stock_minus_features
from brepax.brep.topology import build_adjacency_graph

graph = build_adjacency_graph(shape)
print(f"Adjacency graph: {graph.n_faces} faces, {graph.n_edges} edges")

tree = reconstruct_stock_minus_features(shape)
assert isinstance(tree, CSGOperation)
print(f"CSG tree root: {tree.op}")

# %% [markdown]
# ## 3. Convert to Differentiable Form
#
# `csg_to_differentiable` wraps the CSG tree in an equinox Module
# where all primitive parameters are JAX-differentiable pytree leaves.

# %%
from brepax.brep.csg_eval import csg_to_differentiable

dcsg = csg_to_differentiable(tree)
print(f"Stock: {type(dcsg.stock).__name__}")
print(f"  center:       {dcsg.stock.parameters()['center']}")
print(f"  half_extents: {dcsg.stock.parameters()['half_extents']}")
print(f"Features: {len(dcsg.features)}")
for i, feat in enumerate(dcsg.features):
    print(
        f"  [{i}] {type(feat).__name__}  radius={float(feat.parameters()['radius']):.1f}"
    )

# %% [markdown]
# ## 4. Differentiable Volume
#
# Evaluate the volume on a grid with sigmoid indicator.
# Compare to the analytical result.

# %%
vol = dcsg.volume(resolution=64)
analytical = 40 * 30 * 20 - jnp.pi * 4**2 * 20 - jnp.pi * 3**2 * 20

print(f"Grid volume (res=64): {float(vol):.1f}")
print(f"Analytical volume:    {float(analytical):.1f}")
print(
    f"Relative error:       {abs(float(vol) - float(analytical)) / float(analytical) * 100:.2f}%"
)

# %% [markdown]
# ## 5. Gradient Computation
#
# Compute the gradient of volume with respect to the hole radius.
# A larger hole means less material, so the gradient should be negative.

# %%
from brepax.brep.csg_eval import DifferentiableCSG
from brepax.primitives import FiniteCylinder

box = dcsg.stock
cyl_params = dcsg.features[0].parameters()


def volume_of_radius(radius):
    """Volume as a function of the first hole's radius."""
    cyl = FiniteCylinder(
        center=cyl_params["center"],
        axis=cyl_params["axis"],
        radius=radius,
        height=cyl_params["height"],
    )
    model = DifferentiableCSG(stock=box, features=(cyl,))
    return model.volume(resolution=32)


radius_0 = cyl_params["radius"]
grad = jax.grad(volume_of_radius)(radius_0)

print(f"d(volume)/d(radius) at r={float(radius_0):.1f}: {float(grad):.1f}")
print(
    f"Sign: {'negative (more hole -> less volume)' if float(grad) < 0 else 'unexpected'}"
)

# %% [markdown]
# ## 6. Optimization: Shrink the Hole to Increase Volume
#
# Gradient descent to find the hole radius that increases volume
# by 100 units from the starting point. The gradient `d(vol)/d(r) < 0`
# drives the radius down, increasing material volume.

# %%
initial_vol = float(volume_of_radius(radius_0))
target_vol = initial_vol + 100.0

print(f"Initial volume: {initial_vol:.1f}  (radius={float(radius_0):.1f})")
print(f"Target volume:  {target_vol:.1f}  (+100)")


def loss_fn(radius):
    """Normalized squared error for stable optimization."""
    return ((volume_of_radius(radius) - target_vol) / target_vol) ** 2


radius = radius_0
lr = 50.0

for step in range(30):
    current_loss = float(loss_fn(radius))
    g = jax.grad(loss_fn)(radius)
    radius = radius - lr * g
    radius = jnp.maximum(radius, 0.1)
    if step % 5 == 0:
        print(
            f"  step {step:3d}: radius={float(radius):.3f}  "
            f"vol={float(volume_of_radius(radius)):.1f}  "
            f"loss={current_loss:.6f}"
        )

final_vol = float(volume_of_radius(radius))
print(f"\nFinal radius: {float(radius):.3f}")
print(f"Final volume: {final_vol:.1f} (target: {target_vol:.1f})")
print(f"Volume improvement: +{final_vol - initial_vol:.1f}")

# %% [markdown]
# ## Summary
#
# In ~20 lines of core logic, BRepAX:
#
# 1. Reads a STEP file via OCCT
# 2. Converts B-Rep faces to analytical primitives (Plane, Cylinder)
# 3. Reconstructs the CSG tree (Box minus cylindrical holes)
# 4. Evaluates a differentiable volume via grid + sigmoid integration
# 5. Computes gradients with `jax.grad` flowing through the entire pipeline
# 6. Runs gradient-based optimization on design parameters
#
# The stratum-aware differentiation developed in earlier work
# (analytical exact gradients in 3/4 strata) provides the foundation
# for this pipeline's gradient quality.
