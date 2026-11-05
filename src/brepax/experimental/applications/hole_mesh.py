"""Frozen watertight mesh moved by through-hole edits, for responses without a law.

Volume, surface area and first moment of a hole edit have exact laws in
:class:`~brepax.experimental.applications.hole_edit.HoleEditModel`. This layer
is for responses that have none, such as metrics evaluated on triangles. Its
derivatives are derivatives of the frozen discretization, which limits accuracy
and can restrict the usable radii; neither affects the prepared edit or its
rebuild.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
from numpy.typing import NDArray

from brepax.brep.triangulate import (
    divergence_volume,
    mesh_center_of_mass,
    mesh_surface_area,
    triangulate_shape_watertight,
)
from brepax.experimental.applications._hole_geometry import (
    ThroughHole,
    shape_tolerance,
)
from brepax.experimental.applications.hole_edit import (
    HoleEditModel,
    PreparedHoleEdit,
)


def _displacement_fields(
    vertices: NDArray[np.float64], holes: tuple[ThroughHole, ...], tolerance: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Radial fields and translation weights of each hole on the shared vertices."""
    fields = []
    weights = []
    for hole in holes:
        axis = np.asarray(hole.axis)
        delta = vertices - hole.base
        axial = delta @ axis
        radial = delta - axial[:, None] * axis
        distance = np.linalg.norm(radial, axis=-1)
        # Disjoint support (less than half the clearance) moves the planar
        # annulus without reaching any other face or another hole's support.
        support = 0.4 * hole.clearance
        weight = np.clip((hole.radius + support - distance) / support, 0, 1)
        weight *= (axial >= -tolerance) & (axial <= hole.height + tolerance)
        direction = radial / np.maximum(distance[:, None], hole.radius * 1e-12)
        fields.append(weight[:, None] * direction)
        weights.append(weight)
    return np.asarray(fields), np.asarray(weights)


class HoleMeshModel(eqx.Module):
    """Pure array mesh; differentiate radii and shifts with the mesh frozen.

    Parameters are those of ``edit``: external-unit ``radii`` and optional
    world-coordinate ``center_shifts``. Cylinder samples and neighboring planar
    samples move together, tapering to zero before another feature, so the mesh
    stays connected. Methods support ``jit``, ``vmap`` and parameter
    differentiation; the Boolean validity predicate supports ``jit``/``vmap``.

    ``is_valid`` adds to the edit bounds a check that no triangle collapses or
    reverses. It rejects edits of this mesh only: a rejected edit can still
    rebuild, and the check is not a general self-intersection test. Invalid
    edits produce NaN triangles; derivatives there are undefined. Values and
    gradients carry discretization error and depend on the deflection.
    """

    edit: HoleEditModel
    vertices: Float[Array, "v 3"]
    faces: Int[Array, "t 3"]
    fields: Float[Array, "h v 3"]
    translation_weights: Float[Array, "h v"]

    def _triangles(
        self, radii: Float[Array, " h"], shifts: Float[Array, "h 3"]
    ) -> Float[Array, "t 3 3"]:
        delta = (radii - self.edit.reference_radii) / self.edit.scale
        vertices = self.vertices + jnp.einsum("h,hvi->vi", delta, self.fields)
        vertices = vertices + jnp.einsum(
            "hv,hi->vi",
            self.translation_weights,
            self.edit.transverse(shifts) / self.edit.scale,
        )
        return vertices[self.faces]

    def _valid(
        self,
        radii: Float[Array, " h"],
        shifts: Float[Array, "h 3"],
        triangles: Float[Array, "t 3 3"],
    ) -> Bool[Array, ""]:
        original = self.vertices[self.faces]
        normals = jnp.cross(
            original[:, 1] - original[:, 0], original[:, 2] - original[:, 0]
        )
        changed = jnp.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        # Relative oriented-area margin, independent of length units.
        orientation = jnp.sum(normals * changed, axis=-1)
        area_squared = jnp.sum(normals**2, axis=-1)
        return self.edit.is_valid(radii, shifts) & jnp.all(
            orientation > 1e-8 * area_squared
        )

    def is_valid(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Bool[Array, ""]:
        """Check the edit bounds and the mesh orientation."""
        shifts = self.edit.shifts_or_zero(center_shifts)
        if radii.shape != self.edit.reference_radii.shape:
            raise ValueError("Expected one radius per selected hole")
        return self._valid(radii, shifts, self._triangles(radii, shifts))

    def working_triangles(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, "t 3 3"]:
        """Return normalized triangles; both parameters still use input units."""
        shifts = self.edit.shifts_or_zero(center_shifts)
        if radii.shape != self.edit.reference_radii.shape:
            raise ValueError("Expected one radius per selected hole")
        triangles = self._triangles(radii, shifts)
        return jnp.where(self._valid(radii, shifts, triangles), triangles, jnp.nan)

    def triangles(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, "t 3 3"]:
        """Return external-coordinate triangles, or NaNs for invalid edits."""
        return (
            self.working_triangles(radii, center_shifts) * self.edit.scale
            + self.edit.origin
        )

    def volume(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, ""]:
        """Signed mesh volume in input length units cubed."""
        return (
            divergence_volume(self.working_triangles(radii, center_shifts))
            * self.edit.scale**3
        )

    def surface_area(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, ""]:
        """Mesh surface area in input length units squared."""
        return (
            mesh_surface_area(self.working_triangles(radii, center_shifts))
            * self.edit.scale**2
        )

    def center_of_mass(
        self,
        radii: Float[Array, " h"],
        center_shifts: Float[Array, "h 3"] | None = None,
    ) -> Float[Array, "3"]:
        """Uniform-density mesh center of mass in external coordinates."""
        return (
            mesh_center_of_mass(self.working_triangles(radii, center_shifts))
            * self.edit.scale
            + self.edit.origin
        )


def prepare_hole_mesh(
    edit: PreparedHoleEdit, *, deflection: float = 0.001
) -> HoleMeshModel:
    """Build the frozen watertight mesh of a prepared hole edit.

    ``deflection`` is an absolute WORKING-unit meshing tolerance, finite and
    positive. Meshing accuracy is not certified.

    Raises ValueError for an invalid deflection, a mesh that cannot be welded
    into a closed oriented surface, incident faces whose mesh nodes disagree by
    more than twice the shape's CAD tolerance (``WatertightMesh.max_gap``), or
    degenerate reference geometry. Unsupported surface evaluators raise
    NotImplementedError. The prepared edit stays usable after any of these.

    Example:
        >>> import jax
        >>> from brepax.experimental.applications.hole_edit import prepare_hole_edit
        >>> from brepax.io.step import read_step
        >>> jax.config.update("jax_enable_x64", True)
        >>> edit = prepare_hole_edit(read_step("plate_with_holes.step"))
        >>> mesh = prepare_hole_mesh(edit, deflection=0.0005)
        >>> triangles = mesh.triangles(edit.model.reference_radii)
    """
    if not np.isfinite(deflection) or deflection <= 0:
        raise ValueError("deflection must be finite and positive in working units")
    working = edit.working_shape
    # Two faces sharing an edge each lie within the CAD tolerance of it, so
    # their mesh nodes there may disagree by up to twice that tolerance.
    tolerance = shape_tolerance(working)
    triangulated = triangulate_shape_watertight(
        working, deflection=deflection, tolerance=2 * tolerance
    )
    vertices = np.asarray(triangulated.vertices)
    fields, weights = _displacement_fields(vertices, edit.working_holes, tolerance)
    model = HoleMeshModel(
        edit=edit.model,
        vertices=jnp.asarray(vertices),
        faces=jnp.asarray(np.asarray(triangulated.faces)),
        fields=jnp.asarray(fields),
        translation_weights=jnp.asarray(weights),
    )
    reference = edit.model.reference_radii
    if not bool(model.is_valid(reference)) or float(model.volume(reference)) <= 0:
        raise ValueError("Reference mesh contains degenerate or inverted geometry")
    return model


__all__ = ["HoleMeshModel", "prepare_hole_mesh"]
