"""Differentiable draft angle metrics.

Draft angle = ``arcsin(|unit_normal . mold_direction|)``.  A surface
parallel to the mold pull direction (vertical wall ejected vertically)
has zero draft and causes ejection drag; a surface perpendicular to
the pull direction has 90 degrees of draft and ejects cleanly.

Two paths are exposed:

- The SDF-based volume/surface integration in
  :func:`integrate_sdf_draft_angle_violation` /
  :func:`draft_angle_violation` reports the *surface area* that
  violates a minimum draft angle, integrated over a sigmoid surface
  delta on a 3D grid.  Suited to global DFM constraint terms in
  optimisation losses.
- The face-level mesh path in :func:`min_draft_angle_per_face`
  reports the worst-case draft angle for each face's tessellation,
  trim-aware via OCCT BRepMesh.  Suited to face-by-face inspection
  and to constraint terms expressed per face.

The surface normal in the SDF path is estimated from the SDF gradient
via central finite differences (avoiding NaN at degenerate SDF
points).  The face-level path computes per-triangle normals from
``cross(e1, e2)`` directly.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax._occt.types import TopoDS_Shape
from brepax.brep.csg_eval import make_grid_3d
from brepax.brep.triangulate import _DEFAULT_DEFLECTION, triangulate_shape


def _grid_normals(
    sdf: Float[Array, "R R R"],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, "R R R 3"]:
    """Numerical gradient of SDF on a structured grid.

    Uses ``jnp.gradient`` which handles domain boundaries with
    forward/backward differences, avoiding wrap-around artifacts
    from ``jnp.roll`` on non-periodic domains.
    """
    dx = (hi - lo) / resolution
    grads = jnp.gradient(sdf, dx[0], dx[1], dx[2])
    return jnp.stack(grads, axis=-1)


def integrate_sdf_draft_angle_violation(
    sdf: Float[Array, "R R R"],
    normals: Float[Array, "R R R 3"],
    mold_direction: Float[Array, 3],
    min_angle: Float[Array, ""] | float,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate draft angle violation over the surface.

    For each surface point, the draft angle is the angle between the
    surface normal and the mold pull direction.  Points where the
    draft angle is less than ``min_angle`` contribute to the violation
    surface area.

    Args:
        sdf: Pre-evaluated SDF values with shape ``(R, R, R)``.
        normals: SDF gradient vectors with shape ``(R, R, R, 3)``.
        mold_direction: Unit mold pull direction ``(3,)``.
        min_angle: Minimum acceptable draft angle in radians.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Scalar surface area with insufficient draft angle.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Box
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> sdf = box.sdf(grid)
        >>> normals = _grid_normals(sdf, lo, hi, 64)
        >>> d = jnp.array([0.0, 0.0, 1.0])
        >>> violation = integrate_sdf_draft_angle_violation(
        ...     sdf, normals, d, 0.1, lo, hi, 64,
        ... )
    """
    min_angle = jnp.asarray(min_angle)
    mold_direction = mold_direction / (jnp.linalg.norm(mold_direction) + 1e-10)

    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf / cell_width)
    surface_delta = indicator * (1.0 - indicator) / cell_width

    # Absolute dot product: n and -n represent the same surface
    normal_norm = jnp.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
    unit_normals = normals / normal_norm
    cos_angle = jnp.abs(jnp.sum(unit_normals * mold_direction, axis=-1))

    # draft_angle = arcsin(|n.d|): violation when |n.d| < sin(min_angle)
    sin_threshold = jnp.sin(min_angle)
    # Dimensionless epsilon: sin_threshold and cos_angle are both unitless
    violation = jax.nn.sigmoid((sin_threshold - cos_angle) / 0.01)

    return jnp.sum(surface_delta * violation) * cell_vol


def draft_angle_violation(
    sdf_fn: Callable[..., Float[Array, ...]],
    mold_direction: Float[Array, 3],
    min_angle: Float[Array, ""] | float,
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Surface area with draft angle below the manufacturing threshold.

    Draft angle is the angle between the surface tangent plane and the
    mold pull direction.  A surface normal perpendicular to the pull
    direction has zero draft angle (worst case for ejection).  A normal
    parallel to the pull direction has 90 degrees of draft (ideal).

    This function computes the surface area where the draft angle is
    less than ``min_angle``, providing a differentiable manufacturing
    constraint.  Minimize ``draft_angle_violation(sdf, d, min_angle)``
    to ensure all surfaces have sufficient draft for mold ejection.

    Assumes ``sdf_fn`` returns a proper signed distance field
    (``||grad(f)|| = 1``).

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        mold_direction: Unit mold pull direction ``(3,)``.
        min_angle: Minimum acceptable draft angle in radians.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar surface area with insufficient draft angle,
        differentiable w.r.t. shape parameters and ``mold_direction``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> d = jnp.array([0.0, 0.0, 1.0])
        >>> violation = draft_angle_violation(
        ...     box.sdf, d, 0.1, lo=lo, hi=hi, resolution=64,
        ... )
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    normals = _grid_normals(sdf_vals, lo, hi, resolution)
    return integrate_sdf_draft_angle_violation(
        sdf_vals, normals, mold_direction, min_angle, lo, hi, resolution
    )


_EPS_SQ_TRI_NORMAL = 1e-24
# arcsin saturates at 1.0; pull just inside the unit interval so the
# autodiff branch stays finite at exactly perpendicular surfaces.
_ASIN_CLIP = 1.0 - 1e-7


def _per_triangle_draft_angle(
    triangle: Float[Array, "3 3"],
    mold_direction: Float[Array, 3],
) -> Float[Array, ""]:
    """Draft angle of a single triangle in radians.

    ``draft_angle = arcsin(|unit_normal . mold_direction|)``.  The
    safe-square-then-sqrt pattern is used because zero-area triangles
    (a degenerate but legal artefact of OCCT BRepMesh on slivers) have
    an undefined gradient through ``cross(e1, e2) / |...|``.  Such
    triangles return zero draft, which is the conservative answer for
    a "min draft angle" reduction (they cannot be the only triangle
    on a non-degenerate face, so the min stays meaningful).
    """
    e1 = triangle[1] - triangle[0]
    e2 = triangle[2] - triangle[0]
    cross = jnp.cross(e1, e2)
    norm_sq = jnp.sum(cross**2)
    is_off = norm_sq > _EPS_SQ_TRI_NORMAL
    safe_normal = jnp.where(
        is_off,
        cross / jnp.sqrt(norm_sq + _EPS_SQ_TRI_NORMAL),
        jnp.zeros_like(cross),
    )
    sin_d = jnp.where(is_off, jnp.abs(jnp.sum(safe_normal * mold_direction)), 0.0)
    sin_d = jnp.clip(sin_d, 0.0, _ASIN_CLIP)
    return jnp.arcsin(sin_d)


def min_draft_angle_per_face(
    shape: TopoDS_Shape,
    mold_direction: Float[Array, 3],
    *,
    deflection: float = _DEFAULT_DEFLECTION,
) -> tuple[Float[Array, " n_faces"], list[dict[str, object]]]:
    """Worst-case draft angle for each face of a shape, in radians.

    Tessellates ``shape`` once via
    :func:`~brepax.brep.triangulate.triangulate_shape`, computes the
    draft angle per triangle, and reduces with ``min`` over each
    face's triangle slice.  The ``min`` is the worst-case draft angle
    on the face — the angle that determines whether the face is a DFM
    violation under a given threshold.

    Trim awareness is delegated to OCCT's BRepMesh (the triangulation
    only covers the trimmed region).  The reduction is differentiable
    via ``jax.grad`` through the JAX-side vertex re-evaluation in
    ``triangulate_shape``; ``min`` has the standard subgradient at
    ties.

    Args:
        shape: An OCCT topological shape.  Faces are iterated in the
            same per-Solid order as
            :func:`~brepax.brep.triangulate.triangulate_shape`.
        mold_direction: 3D mold pull direction.  Need not be a unit
            vector; it is normalised internally.
        deflection: Mesh deflection passed to OCCT BRepMesh.  Default
            matches ``triangulate_shape``'s own default so the per-face
            sum invariant of ``surface_area_per_face`` carries over.

    Returns:
        Tuple ``(angles, params_list)`` where ``angles`` has shape
        ``(n_faces,)`` with the worst-case (minimum) draft angle in
        radians for each face in traversal order, and ``params_list``
        is the list returned by
        :func:`~brepax.brep.triangulate.triangulate_shape`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.io.step import read_step
        >>> from brepax.metrics.draft_angle import min_draft_angle_per_face
        >>> # shape = read_step("part.step")
        >>> # mold = jnp.array([0.0, 0.0, 1.0])
        >>> # angles, _ = min_draft_angle_per_face(shape, mold)
        >>> # # angles[i] is in [0, pi/2]; 0 = wall parallel to ejection,
        >>> # # pi/2 = surface perpendicular to ejection.
    """
    triangles, params_list = triangulate_shape(shape, deflection=deflection)
    n_faces = len(params_list)
    if n_faces == 0:
        return jnp.zeros((0,)), params_list

    n_tris_py: list[int] = [int(p["n_triangles"]) for p in params_list]
    offsets_py: list[int] = [0]
    for n in n_tris_py:
        offsets_py.append(offsets_py[-1] + n)

    mold_dir = mold_direction / (jnp.linalg.norm(mold_direction) + 1e-30)
    drafts = jax.vmap(lambda tri: _per_triangle_draft_angle(tri, mold_dir))(triangles)

    return jnp.stack(
        [
            jnp.min(drafts[offsets_py[i] : offsets_py[i] + n_tris_py[i]])
            for i in range(n_faces)
        ]
    ), params_list


__all__ = [
    "draft_angle_violation",
    "integrate_sdf_draft_angle_violation",
    "min_draft_angle_per_face",
]
