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
# # STEP File I/O
#
# Load a STEP file, inspect its topological and geometric metadata,
# and render a tessellated 3D view colored by surface type.

# %%
from pathlib import Path

from brepax.brep.convert import shape_metadata
from brepax.io.step import read_step
from brepax.viz.plot3d import plot_shape

# %% [markdown]
# ## Load a sample box

# %%
fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
box_path = fixtures / "sample_box.step"
box = read_step(box_path)
print(f"Loaded: {box_path.name}")
print(f"Shape null? {box.IsNull()}")

# %% [markdown]
# ## Inspect metadata

# %%
meta = shape_metadata(box)
print(f"Faces:    {meta.n_faces}")
print(f"Edges:    {meta.n_edges}")
print(f"Vertices: {meta.n_vertices}")
print(f"Face types: {meta.face_types}")
print(f"Bounding box min: ({meta.bbox_min[0]:.2f}, {meta.bbox_min[1]:.2f}, {meta.bbox_min[2]:.2f})")
print(f"Bounding box max: ({meta.bbox_max[0]:.2f}, {meta.bbox_max[1]:.2f}, {meta.bbox_max[2]:.2f})")

# %% [markdown]
# ## Load a sample cylinder

# %%
cyl_path = fixtures / "sample_cylinder.step"
cyl = read_step(cyl_path)
cyl_meta = shape_metadata(cyl)
print(f"Loaded: {cyl_path.name}")
print(f"Faces: {cyl_meta.n_faces}, types: {cyl_meta.face_types}")

# %% [markdown]
# ## Visualize
#
# Render the box with faces colored by surface type.  All six faces
# are planar, so they share a single color.

# %%
plot_shape(box)

# %% [markdown]
# Render the cylinder.  The lateral face is cylindrical, and the two
# end caps are planar, producing two distinct colors.

# %%
plot_shape(cyl)
