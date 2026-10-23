"""Host mesh preparation and supplied common vertex map for the roof/hole family."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray

# SciPy provides no bundled typing for this host-only neighbor search.
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from brepax.brep.triangulate import (
    divergence_volume,
    mesh_surface_area,
    triangulate_shape,
)
from brepax.experimental.applications.cad_frame import CadFrame

from .geometry import (
    BASE_DIMENSIONS,
    HEIGHT,
    LENGTH,
    WIDTH,
    boundary_tolerance,
    make_shape,
)


class RoofMesh(eqx.Module):
    """Frozen discretization with shared vertex motion; not a generic CAD editor.

    Only dimensions [roof amplitude, hole radius] are edit inputs. The radial
    map is supported within distance 6 of the hole axis and uses reference
    radius 3. It is restricted to the edit box declared in geometry.py.
    Host geometry checks must accompany numerical evaluations at candidate
    designs; JAX tracing does not reject topology changes or invalid edits.
    """

    vertices: Float[Array, "n 3"]
    faces: Int[Array, "m 3"]

    def triangles(self, dimensions: Float[Array, 2]) -> Float[Array, "m 3 3"]:
        """Move the roof, cylinder and their shared trim boundary together."""
        amplitude, radius = dimensions
        xy = self.vertices[:, :2]
        delta = xy - jnp.array([LENGTH / 2, WIDTH / 2])
        distance = jnp.linalg.norm(delta, axis=1)
        weight = jnp.clip((6 - distance) / (6 - 3), 0.0, 1.0)
        moved_xy = xy + (radius - 3) * weight[:, None] * delta / jnp.maximum(
            distance[:, None], 1e-12
        )
        old_u, new_u = xy[:, 0] / LENGTH, moved_xy[:, 0] / LENGTH
        z = (
            self.vertices[:, 2]
            * (HEIGHT + amplitude * 4 * new_u * (1 - new_u))
            / (HEIGHT + 5 * 4 * old_u * (1 - old_u))
        )
        return jnp.column_stack((moved_xy, z))[self.faces]

    def external_measures(
        self, dimensions: Float[Array, 2], frame: CadFrame
    ) -> Float[Array, 2]:
        """[Volume, area] in external units, differentiable in external lengths.

        Compile a closure over this fixed frame/model once per edit run.
        Integrating in working coordinates avoids external-origin cancellation.
        """
        triangles = self.triangles(frame.working_lengths(dimensions))
        return jnp.stack(
            (
                frame.external_measure(divergence_volume(triangles), 3),
                frame.external_measure(mesh_surface_area(triangles), 2),
            )
        )


def prepare_mesh(working_deflection: float) -> tuple[RoofMesh, dict[str, float | int]]:
    """Prepare once, merging coincident trim vertices within measured CAD error.

    KDTree pairs are sorted, and lexicographically smallest points represent
    components. Reject transitive clusters exceeding the measured tolerance;
    this fixture-specific welding is not offered as general mesh repair.
    """
    if not np.isfinite(working_deflection) or working_deflection <= 0:
        raise ValueError("Deflection must be finite and positive")
    shape = make_shape(BASE_DIMENSIONS)
    triangles, _ = triangulate_shape(shape, deflection=working_deflection)
    points = np.asarray(triangles).reshape(-1, 3)
    tolerance = 2 * boundary_tolerance(shape)
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    parents = np.arange(len(unique))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    for i, j in sorted(cKDTree(unique).query_pairs(tolerance)):
        ri, rj = root(i), root(j)
        parents[max(ri, rj)] = min(ri, rj)
    selected, ids = np.unique(
        [root(i) for i in range(len(unique))], return_inverse=True
    )
    vertices, faces = unique[selected], ids[inverse].reshape(-1, 3)
    displacement = np.linalg.norm(points - vertices[faces.reshape(-1)], axis=1).max()
    if displacement > tolerance:
        raise RuntimeError("Vertex weld exceeds CAD-derived distance tolerance")
    model = RoofMesh(jnp.asarray(vertices), jnp.asarray(faces))
    return model, {
        "triangles": len(faces),
        "working_deflection": working_deflection,
        "weld_tolerance": tolerance,
        "max_weld_displacement": float(displacement),
    }


def check_mesh(model: RoofMesh, dimensions: NDArray[np.float64]) -> bool:
    """Check closed oriented connectivity and nondegenerate, locally unflipped faces.

    This does not establish global absence of self-intersection for arbitrary
    edits; the supplied map and parameter box are part of the experiment.
    """
    faces = np.asarray(model.faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    closed = np.all((counts == 2) & (np.bincount(inverse, weights=signs) == 0))
    base = np.asarray(model.triangles(jnp.asarray(BASE_DIMENSIONS)))
    moved = np.asarray(model.triangles(jnp.asarray(dimensions)))
    normal0 = np.cross(base[:, 1] - base[:, 0], base[:, 2] - base[:, 0])
    normal1 = np.cross(moved[:, 1] - moved[:, 0], moved[:, 2] - moved[:, 0])
    nondegenerate = np.all(
        np.linalg.norm(normal1, axis=1) > 1e-10 * np.linalg.norm(normal0, axis=1)
    )
    return bool(
        closed and nondegenerate and np.all(np.sum(normal0 * normal1, axis=1) > 0)
    )
