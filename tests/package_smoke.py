"""Exercise a built wheel with fresh dependencies, without pytest or dev extras.

Run using the isolated installation's Python with ``-I`` so the checkout and
PYTHONPATH cannot hide a missing or broken installed module.
"""

from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import brepax
from brepax._occt.backend import BRepPrimAPI_MakeBox, gp_Pnt
from brepax.brep.gprop import compute_gprop_ground_truth
from brepax.brep.triangulate import (
    divergence_volume,
    mesh_center_of_mass,
    mesh_surface_area,
    triangulate_shape,
)
from brepax.experimental.planar_contact import planar_contact_set_volume_direction
from brepax.experimental.planar_contact_collection_cad import (
    prepare_planar_contact_set_reference,
)
from brepax.io.step import read_step


def check_contact_collection() -> None:
    """Check installed CAD preparation and packed JAX queries with fresh versions."""
    fixed = (
        BRepPrimAPI_MakeBox(gp_Pnt(-1, 0, -1), 2, 1, 1).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(-1, 0, 0), 1, 1, 1).Shape(),
    )
    moving = (BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape(),)
    contacts = prepare_planar_contact_set_reference(
        fixed, moving, mode="zero_residual"
    ).contact_set
    query = jax.jit(
        jax.vmap(
            lambda velocity: planar_contact_set_volume_direction(
                contacts, velocity, jnp.zeros(3), jnp.zeros(3)
            )
        )
    )
    values = query(jnp.array([[-1.0, 0, -1], [1.0, 0, 1], [-1.0, 0, 0]]))
    np.testing.assert_allclose(values, [2, 0, 1], rtol=0, atol=1e-14)
    print("contact collection: actual CAD preparation and JIT/vmap passed")


def main() -> None:
    """Check STEP extraction, analytic/B-spline evaluation, metrics and JAX AD."""
    jax.config.update("jax_enable_x64", True)
    print(f"Installed BRepAX: {brepax.__file__}")
    print({name: version(name) for name in ("brepax", "cadquery-ocp-novtk", "jax")})
    fixtures = Path(__file__).resolve().parent / "fixtures"
    derivative = jax.jit(jax.grad(lambda scale, mesh: divergence_volume(scale * mesh)))
    for name in ("sample_box", "box_with_holes", "nurbs_box"):
        shape = read_step(fixtures / f"{name}.step")
        reference = compute_gprop_ground_truth(shape)
        triangles, _ = triangulate_shape(shape, deflection=0.001)
        volume = divergence_volume(triangles)
        np.testing.assert_allclose(volume, reference["volume"], rtol=1e-3)
        np.testing.assert_allclose(
            mesh_surface_area(triangles), reference["surface_area"], rtol=1e-3
        )
        np.testing.assert_allclose(
            mesh_center_of_mass(triangles), reference["center_of_mass"], atol=1e-3
        )
        # Uniform scaling has derivative 3 V at unit scale, independently of AD.
        np.testing.assert_allclose(
            derivative(1.0, triangles), 3 * reference["volume"], rtol=1e-3
        )
        print(f"{name}: STEP, mesh metrics and JIT gradient passed")
    check_contact_collection()


if __name__ == "__main__":
    main()
