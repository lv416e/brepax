"""Exercise a built wheel with fresh dependencies, without pytest or dev extras.

Run using the isolated installation's Python with ``-I`` so the checkout and
PYTHONPATH cannot hide a missing or broken installed module.
"""

from importlib.metadata import version
from itertools import product
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
from brepax.experimental.planar_contact_placement import (
    place_planar_contact_set,
    placed_planar_contact_set_volume_direction,
)
from brepax.experimental.polyhedral_parameter import (
    AffinePolyhedralCell,
    polyhedral_parameter_position_sensitivity,
    polyhedral_parameter_volume,
    prepare_polyhedral_parameter,
    specialize_polyhedral_parameter,
)
from brepax.experimental.polyhedral_targets import polyhedral_translation_targets
from brepax.experimental.polyhedral_translation import (
    polyhedral_translation_slope,
    polyhedral_translation_volume,
    prepare_polyhedral_translation,
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
    placed = place_planar_contact_set(
        contacts, rotation=np.eye(3), origin=(2, -1, 3), length_scale=0.001
    )
    world_rate = jax.jit(placed_planar_contact_set_volume_direction)(
        placed,
        jnp.array([-0.001, 0, -0.001]),
        jnp.zeros(3),
        jnp.array([2.0, -1, 3]),
    )
    np.testing.assert_allclose(world_rate, 2e-9, rtol=1e-14, atol=0)
    print("contact collection: actual CAD preparation and JIT/vmap passed")


def check_finite_translation() -> None:
    """Check installed exact preparation and finite values across an event."""
    box = np.array(list(product((0.0, 1.0), repeat=3)))
    prepared = prepare_polyhedral_translation(
        [box], [box + np.array([1, 1, 0])], (-1, -1, 0), max_step=2.0
    )
    profile = prepared.profile
    steps = jnp.array([0.0, 0.25, 1.0, 1.25, 2.0])
    values = jax.jit(jax.vmap(polyhedral_translation_volume, in_axes=(None, 0)))(
        profile, steps
    )
    np.testing.assert_allclose(values, [0, 0.0625, 1, 0.5625, 0], atol=1e-14)
    np.testing.assert_allclose(
        [
            polyhedral_translation_slope(profile, 1.0, side="left"),
            polyhedral_translation_slope(profile, 1.0, side="right"),
        ],
        [2, -2],
        atol=1e-14,
    )
    targets = polyhedral_translation_targets(
        prepared, 0.25, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert targets.status == "reachable"
    np.testing.assert_allclose([root.step for root in targets.roots], [0.5, 1.5])
    print("finite translation: JIT/vmap, event slopes and target inversion passed")


def check_design_sensitivity() -> None:
    """Check installed changing-geometry AD and exact host target specialization."""
    box = np.array(list(product((0.0, 1.0), repeat=3)))
    fixed = AffinePolyhedralCell(box, np.zeros((1, 8, 3)))
    moving = AffinePolyhedralCell(
        box + np.array([2, 0, 0]), np.tile(np.array([1.0, 0, 0]), (1, 8, 1))
    )
    prepared = prepare_polyhedral_parameter([fixed], [moving], (-1, 0, 0), max_step=4.0)
    theta = jnp.array([0.125])
    value = jax.jit(polyhedral_parameter_volume)(prepared.model, 1.5, theta)
    gradient = jax.jit(jax.grad(polyhedral_parameter_volume, argnums=2))(
        prepared.model, 1.5, theta
    )
    np.testing.assert_allclose(value, 0.375, rtol=0, atol=1e-14)
    np.testing.assert_allclose(gradient, [-1], rtol=0, atol=1e-14)
    path = specialize_polyhedral_parameter(prepared, theta)
    targets = polyhedral_translation_targets(
        path, 0.25, step_tolerance=1e-12, volume_tolerance=1e-12
    )
    assert targets.status == "reachable"
    np.testing.assert_allclose([root.step for root in targets.roots], [1.375, 2.875])
    sensitivity = jax.jit(polyhedral_parameter_position_sensitivity)(
        prepared.model, 1.375, theta
    )
    np.testing.assert_allclose(sensitivity, [1], rtol=0, atol=1e-14)
    print("design sensitivity: JIT parameter AD and specialized target roots passed")


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
    check_finite_translation()
    check_design_sensitivity()


if __name__ == "__main__":
    main()
