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
# # 3D Primitives
#
# All 7 3D primitive types with SDF evaluation, volume, and gradient.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from brepax.primitives import (
    Box,
    Cone,
    Cylinder,
    FiniteCylinder,
    Plane,
    Sphere,
    Torus,
)

# %% [markdown]
# ## Bounded Primitives (analytical volume)

# %%
primitives = {
    "Sphere": Sphere(
        center=jnp.array([0.0, 0.0, 0.0]),
        radius=jnp.array(1.0),
    ),
    "FiniteCylinder": FiniteCylinder(
        center=jnp.array([0.0, 0.0, 0.0]),
        axis=jnp.array([0.0, 0.0, 1.0]),
        radius=jnp.array(1.0),
        height=jnp.array(2.0),
    ),
    "Torus": Torus(
        center=jnp.array([0.0, 0.0, 0.0]),
        axis=jnp.array([0.0, 0.0, 1.0]),
        major_radius=jnp.array(2.0),
        minor_radius=jnp.array(0.5),
    ),
    "Box": Box(
        center=jnp.array([0.0, 0.0, 0.0]),
        half_extents=jnp.array([1.0, 0.5, 0.25]),
    ),
}

print(f"{'Primitive':>16} {'Volume':>10} {'SDF(origin)':>12}")
print("-" * 42)
for name, prim in primitives.items():
    vol = float(prim.volume())
    sdf = float(prim.sdf(jnp.array([0.0, 0.0, 0.0])))
    print(f"{name:>16} {vol:>10.4f} {sdf:>12.4f}")

# %% [markdown]
# ## Unbounded Primitives (infinite extent)

# %%
unbounded = {
    "Cylinder": Cylinder(
        point=jnp.array([0.0, 0.0, 0.0]),
        axis=jnp.array([0.0, 0.0, 1.0]),
        radius=jnp.array(1.0),
    ),
    "Plane": Plane(
        normal=jnp.array([0.0, 0.0, 1.0]),
        offset=jnp.array(0.0),
    ),
    "Cone": Cone(
        apex=jnp.array([0.0, 0.0, 0.0]),
        axis=jnp.array([0.0, 0.0, 1.0]),
        angle=jnp.array(jnp.pi / 4),
    ),
}

test_point = jnp.array([0.5, 0.0, 1.0])
print(f"SDF at {test_point}:")
for name, prim in unbounded.items():
    sdf = float(prim.sdf(test_point))
    print(f"  {name}: {sdf:.4f}")

# %% [markdown]
# ## Volume gradient (bounded primitives)

# %%
print(f"{'Primitive':>16} {'d(vol)/d(param)':>18} {'param':>10}")
print("-" * 48)

# Sphere: d(vol)/d(r) = 4*pi*r^2
sphere = primitives["Sphere"]
dvdr = jax.grad(lambda s: s.volume())(sphere).radius
print(f"{'Sphere':>16} {float(dvdr):>18.4f} {'radius':>10}")

# Torus: d(vol)/d(R) = 2*pi^2*r^2
torus = primitives["Torus"]
dvd_major = jax.grad(lambda t: t.volume())(torus).major_radius
print(f"{'Torus':>16} {float(dvd_major):>18.4f} {'major_r':>10}")

# Box: d(vol)/d(hx) = 8*hy*hz
box = primitives["Box"]
dvdh = jax.grad(lambda b: b.volume())(box).half_extents
print(
    f"{'Box':>16} [{float(dvdh[0]):.1f},{float(dvdh[1]):.1f},{float(dvdh[2]):.1f}] {'half_ext':>10}"
)
