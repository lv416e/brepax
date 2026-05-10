"""Differentiable wall thickness metrics.

For a proper signed distance field, the absolute value at any interior
point equals the distance to the nearest surface boundary.  Two
SDF-grid paths are exposed for global descriptors:

- :func:`thin_wall_volume` counts the volume of material closer to a
  surface than a given threshold -- directly useful as a DFM constraint
  (minimize to enforce minimum wall thickness).
- :func:`min_wall_thickness` estimates the minimum wall thickness as a
  differentiable scalar via soft-minimum over interior SDF values.

Both integrate the same sigmoid framework used by volume and surface
area metrics, with sharpness ``1 / cell_width``.

A face-level mesh path is also exposed:

- :func:`min_wall_thickness_per_face` returns, for each face, the
  minimum distance from that face's centroid to triangles on every
  *other* face.  Trim awareness is delegated to OCCT BRepMesh (the
  triangulation only covers the trimmed region of each face).  The
  metric is well-defined even when faces share edges because the
  sample point is the per-face centroid, not the boundary triangles.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax._occt.types import TopoDS_Shape
from brepax.brep.csg_eval import make_grid_3d
from brepax.brep.mesh_sdf import point_triangle_distance
from brepax.brep.triangulate import _DEFAULT_DEFLECTION, triangulate_shape


def integrate_sdf_thin_wall_volume(
    sdf: Float[Array, ...],
    threshold: Float[Array, ""] | float,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate SDF values to compute volume of thin-wall material.

    Selects interior points whose distance to the nearest surface is
    less than ``threshold``.  Both the interior membership and the
    distance test use sigmoid indicators with sharpness ``1 / cell_width``.

    This assumes the input is a proper signed distance field where
    ``||grad(f)|| = 1``.

    Args:
        sdf: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        threshold: Distance threshold.  Points inside the shape
            and within this distance of the boundary contribute.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.

    Returns:
        Scalar volume of material thinner than ``threshold``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg_eval import make_grid_3d
        >>> from brepax.primitives import Box
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> grid, _ = make_grid_3d(lo, hi, 64)
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> sdf = box.sdf(grid)
        >>> vol = integrate_sdf_thin_wall_volume(sdf, 0.5, lo, hi, 64)
    """
    threshold = jnp.asarray(threshold)
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    # Interior membership: sigmoid(d/eps) where d = -sdf
    inside = jax.nn.sigmoid(-sdf / cell_width)
    # Thin-wall membership: points within threshold of boundary
    # sdf + threshold > 0 means |sdf| < threshold for interior points
    thin = jax.nn.sigmoid((sdf + threshold) / cell_width)
    return jnp.sum(inside * thin) * cell_vol


def thin_wall_volume(
    sdf_fn: Callable[..., Float[Array, ...]],
    threshold: Float[Array, ""] | float,
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Volume of material with wall thickness below a given threshold.

    For each interior point, the SDF value gives the distance to the
    nearest surface.  This function counts the volume of material
    where that distance is less than ``threshold``, i.e. the material
    that would violate a minimum wall thickness requirement.

    Useful as a manufacturing constraint: minimize
    ``thin_wall_volume(sdf, 1.5)`` to ensure no wall is thinner
    than 1.5 units.  Assumes ``sdf_fn`` returns a proper signed
    distance field (``||grad(f)|| = 1``).

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        threshold: Minimum acceptable wall thickness.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar volume of thin-wall material, differentiable w.r.t.
        both the SDF function's parameters and ``threshold``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3))
        >>> lo, hi = jnp.array([-3.0]*3), jnp.array([3.0]*3)
        >>> vol = thin_wall_volume(box.sdf, 0.5, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)
    return integrate_sdf_thin_wall_volume(sdf_vals, threshold, lo, hi, resolution)


def integrate_sdf_min_wall_thickness(
    sdf: Float[Array, ...],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Integrate SDF values to estimate minimum wall thickness.

    Uses a normalized soft-max (log-mean-exp) over interior SDF values
    weighted by interior membership.  The normalization ensures the
    estimate is invariant to grid resolution and domain size.

    This assumes the input is a proper signed distance field where
    ``||grad(f)|| = 1``.

    Args:
        sdf: Pre-evaluated SDF values on a cell-centered grid
            with shape ``(R, R, R)`` from :func:`make_grid_3d`.
        lo: Grid lower bound ``(3,)``.
        hi: Grid upper bound ``(3,)``.
        resolution: Number of grid points per axis.
        temperature: Soft-max temperature; lower values approximate
            the true maximum more closely.

    Returns:
        Scalar estimate of minimum wall thickness.
    """
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    interior_dist = jnp.clip(-sdf, 0.0, None)
    weight = jax.nn.sigmoid(-sdf / cell_width)

    flat_dist = interior_dist.ravel()
    flat_weight = weight.ravel()
    # Normalized soft-max: subtract log(sum(w)) for resolution invariance
    max_dist = temperature * (
        jax.nn.logsumexp(flat_dist / temperature, b=flat_weight)
        - jnp.log(jnp.sum(flat_weight) + 1e-10)
    )
    return 2.0 * max_dist


def min_wall_thickness(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Differentiable estimate of minimum wall thickness.

    For a convex shape, the minimum wall thickness equals twice the
    maximum inscribed distance (the SDF value at the deepest interior
    point).  This function returns a differentiable approximation via
    normalized soft-max (log-mean-exp) over interior SDF values,
    ensuring the estimate is invariant to grid resolution.

    For shapes with varying wall thickness (e.g. a box with holes
    near an edge), this returns a global estimate that may not reflect
    the thinnest local section.  Use :func:`thin_wall_volume` with an
    explicit threshold for manufacturing constraint enforcement.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).
        temperature: Soft-max temperature; lower values approximate
            the true maximum more closely but may have sharper gradients.

    Returns:
        Scalar estimate of minimum wall thickness (twice the maximum
        inscribed distance), differentiable w.r.t. the SDF function's
        parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Box
        >>> box = Box(
        ...     center=jnp.zeros(3),
        ...     half_extents=jnp.array([2.0, 1.5, 1.0]),
        ... )
        >>> lo, hi = jnp.array([-4.0]*3), jnp.array([4.0]*3)
        >>> thickness = min_wall_thickness(box.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)

    # Sub-grid refinement: soft-argmax gives continuous position
    # of the deepest interior point, then re-evaluate SDF there
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    interior_dist = jnp.clip(-sdf_vals, 0.0, None)
    weight = jax.nn.sigmoid(-sdf_vals / cell_width)

    flat_dist = interior_dist.ravel()
    flat_weight = weight.ravel()
    flat_grid = grid.reshape(-1, 3)

    # Softmax over interior distances: weighted average position
    log_w = flat_dist / temperature + jnp.log(flat_weight + 1e-20)
    softmax_w = jax.nn.softmax(log_w)
    x_refined = jnp.sum(flat_grid * softmax_w[:, None], axis=0)

    # Re-evaluate SDF at the refined sub-grid position
    refined_sdf = sdf_fn(x_refined[None, None, None, :]).squeeze()
    refined_dist = jnp.clip(-refined_sdf, 0.0, None)

    return 2.0 * refined_dist


def min_wall_thickness_per_face(
    shape: TopoDS_Shape,
    *,
    deflection: float = _DEFAULT_DEFLECTION,
) -> tuple[Float[Array, " n_faces"], list[dict[str, object]]]:
    """Per-face minimum wall thickness from each face's centroid
    (centroid-based approximation, not manufacturing-grade minimum).

    This is a single-sample-per-face estimator: it samples one point
    per face (the centroid of that face's triangle vertices) and
    measures the distance to the nearest triangle on any other face.
    A true manufacturing-grade minimum wall thickness would sample
    every point on each face (or a dense covering) and take the
    pointwise minimum, which would also catch local thinning at face
    edges, ribs, fillets, and small features that the single
    centroid sample is blind to.  Use this metric for DFM screening
    on smooth, near-uniform faces; do not use it as a release-gate
    minimum on geometry with sharp local thinning.

    Tessellates ``shape`` once via
    :func:`~brepax.brep.triangulate.triangulate_shape`, computes the
    centroid of each face's triangle slice, and returns the distance
    from that centroid to the nearest triangle on **any other face**.
    The "other" qualifier matters: a centroid sampled on face F is by
    construction at distance zero from F's own triangles, so the
    self-face distance is uninteresting; the per-other-face minimum
    is the metric of practical interest for DFM.

    A face whose triangles share edges with adjacent faces has
    boundary triangles arbitrarily close to those neighbours.
    Sampling at the face *centroid* (mean of all the face's
    triangle vertices) keeps the sample point off the boundary and
    produces a stable wall-thickness estimate.  For an axis-aligned
    box of half-extents ``(a, b, c)``, this gives ``min(2*a, b, c)``
    for each ``+/- x`` face, ``min(a, 2*b, c)`` for each ``+/- y``
    face, and ``min(a, b, 2*c)`` for each ``+/- z`` face — the minimum
    of the distance to the opposite face and the half-extents of the
    perpendicular dimensions.

    Trim awareness is delegated to OCCT BRepMesh: the triangulation
    only covers the trimmed region of each face, so the centroid and
    the cross-face distance both respect trim.  Differentiability is
    inherited from :func:`~brepax.brep.mesh_sdf.point_triangle_distance`,
    which is differentiable through triangle vertex positions.

    Single-face shapes (e.g. one full sphere) have no "other face",
    so the returned thickness is ``+inf`` for that slot — an honest
    "no other surface to measure to" signal rather than 0.

    Args:
        shape: An OCCT topological shape.  Faces are iterated in the
            same per-Solid order as
            :func:`~brepax.brep.triangulate.triangulate_shape`.
        deflection: Mesh deflection passed to OCCT BRepMesh.  Default
            matches ``triangulate_shape``'s own default.

    Returns:
        Tuple ``(thicknesses, params_list)`` where ``thicknesses`` has
        shape ``(n_faces,)`` with the per-face min wall thickness in
        traversal order (``+inf`` if there is no other face), and
        ``params_list`` is the list returned by
        :func:`~brepax.brep.triangulate.triangulate_shape`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax._occt.backend import BRepPrimAPI_MakeBox
        >>> from brepax.metrics.wall_thickness import min_wall_thickness_per_face
        >>> shape = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
        >>> thicknesses, _ = min_wall_thickness_per_face(shape)
        >>> # 6 faces; for half-extents (0.5, 1.0, 1.5) the +/- x
        >>> # faces give min(2*0.5, 1.0, 1.5)=1.0, the +/- y faces
        >>> # give min(0.5, 2*1.0, 1.5)=0.5, the +/- z faces give
        >>> # min(0.5, 1.0, 2*1.5)=0.5 — counted at the face centroid.
        >>> bool(jnp.all(thicknesses > 0))
        True
    """
    triangles, params_list = triangulate_shape(shape, deflection=deflection)
    n_faces = len(params_list)
    if n_faces == 0:
        return jnp.zeros((0,)), params_list

    # Per-face triangle slice offsets.  Python ints stay on the host;
    # ``triangles[a:b]`` is differentiable through the slice.
    n_tris_py: list[int] = [int(p["n_triangles"]) for p in params_list]
    offsets_py: list[int] = [0]
    for n in n_tris_py:
        offsets_py.append(offsets_py[-1] + n)

    def _per_centroid_min_other_face_dist(
        centroid: Float[Array, 3],
        others: Float[Array, "m 3 3"],
    ) -> Float[Array, ""]:
        per_tri = jax.vmap(
            lambda t: point_triangle_distance(centroid, t[0], t[1], t[2])
        )(others)
        return jnp.min(per_tri)

    thicknesses_per_face: list[Float[Array, ""]] = []
    for i in range(n_faces):
        a = offsets_py[i]
        b = a + n_tris_py[i]
        if n_tris_py[i] == 0:
            # Degenerate face that BRepMesh failed to triangulate: no
            # centroid to sample from.  Return the same +inf sentinel
            # used for the structurally analogous "no other surface"
            # case so callers can still distinguish missing-data from
            # a real numeric blowup.
            thicknesses_per_face.append(jnp.asarray(jnp.inf))
            continue
        face_tris = triangles[a:b]
        # Face centroid: mean of all triangle vertices on F.  Off the
        # boundary by construction, so distance to neighbouring faces
        # is not contaminated by the shared-edge artifact.
        centroid = jnp.mean(face_tris.reshape(-1, 3), axis=0)

        # Other-face triangles: everything outside the [a, b) slice.
        # ``jnp.concatenate`` of two slices of the same flat array is
        # cheap and stays differentiable.
        if a == 0:
            others = triangles[b:]
        elif b == triangles.shape[0]:
            others = triangles[:a]
        else:
            others = jnp.concatenate(
                [triangles[:a], triangles[b:]],
                axis=0,
            )

        if others.shape[0] == 0:
            # Single-face shape: no other surface to measure against.
            thicknesses_per_face.append(jnp.asarray(jnp.inf))
        else:
            thicknesses_per_face.append(
                _per_centroid_min_other_face_dist(centroid, others)
            )

    return jnp.stack(thicknesses_per_face), params_list


__all__ = [
    "integrate_sdf_min_wall_thickness",
    "integrate_sdf_thin_wall_volume",
    "min_wall_thickness",
    "min_wall_thickness_per_face",
    "thin_wall_volume",
]
