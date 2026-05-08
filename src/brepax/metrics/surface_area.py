"""Differentiable surface area via SDF boundary integral.

Approximates surface area as the integral of a sigmoid-derivative
delta function over a 3D grid.  For a signed distance field *f*,
the surface area is:

    A = integral of delta(f(x)) dx

where delta is approximated by sigma(-f/eps) * (1 - sigma(-f/eps)) / eps
with eps = cell_width (geometric mean of axis spacings).  This is the
derivative of the sigmoid Heaviside used in volume integration, ensuring
consistent sharpness scaling across metrics.

A face-level surface area path (:func:`surface_area_per_face`) is also
provided.  It bypasses the SDF grid entirely and reduces directly over
the per-face triangle slices that :func:`~brepax.brep.triangulate.triangulate_shape`
already produces.  Trim awareness comes from OCCT's BRepMesh, which
respects the trim curves when building the triangulation; the metric
reduction is a polynomial sum over triangle vertex positions and is
therefore differentiable through the JAX-side vertex re-evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax._occt.types import TopoDS_Shape
from brepax.brep.csg_eval import make_grid_3d
from brepax.brep.triangulate import mesh_surface_area, triangulate_shape


def integrate_sdf_surface_area(
    sdf: Float[Array, ...],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate SDF values on a grid to compute surface area.

    Uses a sigmoid-derivative delta function with sharpness
    ``1 / cell_width``, matching the convention in
    :func:`~brepax.brep.csg_eval.integrate_sdf_volume`.

    This assumes the input is a proper signed distance field where
    ``||grad(f)|| = 1``.  CSG Boolean SDFs (min/max compositions)
    satisfy this almost everywhere except at the Boolean boundary
    itself, where the kink does not affect the integral.

    Args:
        sdf: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Scalar surface area estimate.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Sphere
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> sdf = sphere.sdf(grid)
        >>> area = integrate_sdf_surface_area(sdf, lo, hi, 64)
    """
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    indicator = jax.nn.sigmoid(-sdf / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width
    return jnp.sum(delta) * cell_vol


def surface_area(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute differentiable surface area of a shape defined by its SDF.

    Evaluates the SDF on a cell-centered grid and integrates a
    sigmoid-derivative delta function to approximate the area of the
    zero level-set.  Assumes ``sdf_fn`` returns a proper signed
    distance field (``||grad(f)|| = 1``).

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar surface area estimate, differentiable w.r.t. the SDF
        function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Sphere
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> area = surface_area(sphere.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    return integrate_sdf_surface_area(sdf_vals, lo, hi, resolution)


def surface_area_per_face(
    shape: TopoDS_Shape,
    *,
    deflection: float = 0.5,
) -> tuple[Float[Array, " n_faces"], list[dict[str, object]]]:
    """Compute mesh-based surface area for each face of a shape.

    Tessellates ``shape`` once via :func:`~brepax.brep.triangulate.triangulate_shape`,
    then reduces :func:`~brepax.brep.triangulate.mesh_surface_area` over
    each face's triangle slice.  The per-face triangle counts come from
    the ``n_triangles`` entry in the params list, so no second pass over
    OCCT is needed.

    Trim awareness is delegated to OCCT's BRepMesh: the triangulation
    only covers the face's trimmed region, so the polynomial triangle
    area sum is already trim-aware.  Verified per face against
    ``BRepGProp.SurfaceProperties_s(face)`` on the standard fixture set
    (max abs error 2.25%, median 0.07%).

    Args:
        shape: An OCCT topological shape.  Faces are iterated in the
            same per-Solid order as
            :func:`~brepax.brep.triangulate.triangulate_shape`.
        deflection: Mesh deflection passed to OCCT BRepMesh.

    Returns:
        Tuple ``(areas, params_list)`` where ``areas`` has shape
        ``(n_faces,)`` with the mesh surface area for each face in
        traversal order, and ``params_list`` is the same list returned
        by :func:`~brepax.brep.triangulate.triangulate_shape` (each
        entry includes ``surface_type`` and ``n_triangles``).

    Examples:
        >>> from brepax.io.step import read_step
        >>> from brepax.metrics.surface_area import surface_area_per_face
        >>> # shape = read_step("part.step")
        >>> # areas, params = surface_area_per_face(shape)
        >>> # assert areas.shape == (len(params),)
    """
    triangles, params_list = triangulate_shape(shape, deflection=deflection)
    n_faces = len(params_list)

    if n_faces == 0:
        return jnp.zeros((0,)), params_list

    # Cumulative triangle offsets per face (no shared edges across face
    # slices in the global ``triangles`` array).
    n_tris = jnp.asarray([p["n_triangles"] for p in params_list])
    offsets = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(n_tris).astype(jnp.int32)]
    )

    # Face slices may have different triangle counts, so the per-face
    # reduction is a Python loop over jax.lax.dynamic_slice rather than
    # a vmap.  The reduction inside each slice is jit-friendly.
    areas = jnp.stack(
        [
            mesh_surface_area(
                jax.lax.dynamic_slice_in_dim(
                    triangles, int(offsets[i]), int(n_tris[i]), axis=0
                )
            )
            for i in range(n_faces)
        ]
    )
    return areas, params_list


__all__ = [
    "integrate_sdf_surface_area",
    "surface_area",
    "surface_area_per_face",
]
