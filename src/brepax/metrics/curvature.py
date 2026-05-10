"""Differentiable curvature metrics.

For a proper signed distance field where ``||grad(f)|| = 1``, the mean
curvature at the zero level-set equals the Laplacian of the SDF:

    kappa = div(grad(f) / ||grad(f)||) = laplacian(f)

This equals the sum of principal curvatures (kappa_1 + kappa_2).  Two
SDF-grid paths are exposed for global descriptors:

- :func:`mean_curvature` computes the delta-weighted average of the
  Laplacian over the surface.
- :func:`max_curvature` returns a soft-max estimate.

Both use the sigmoid-derivative delta framework (consistent with
:func:`~brepax.metrics.surface_area.surface_area`) and Newton-refine
grid points toward the zero level-set before evaluating the AD Hessian.

A face-level mesh path is also exposed:

- :func:`mean_curvature_per_face` returns a per-face mean curvature
  using the analytical closed form for the underlying primitive
  surface (plane, sphere, cylinder).  Cone, torus and BSpline faces
  return NaN until their analytical handlers are added in a follow-up
  PR.  Trim awareness is automatic because each face's primitive
  parameters describe the face's geometry, not its parametric extent.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax._occt.types import TopoDS_Shape
from brepax.brep.csg_eval import make_grid_3d
from brepax.brep.triangulate import per_face_geometric_params


def _newton_refine(
    sdf_fn: Callable[..., Float[Array, ...]],
    points: Float[Array, "N 3"],
    n_steps: int = 2,
) -> Float[Array, "N 3"]:
    """Project points toward the SDF=0 surface via Newton steps.

    Each step moves a point along the SDF gradient direction to reduce
    the residual SDF value.  Points far from the surface will not
    converge in a few steps, but their sigmoid delta weight is
    negligible so they do not affect the result.

    Args:
        sdf_fn: Signed distance function ``(3,) -> ()``.
        points: Initial points of shape ``(N, 3)``.
        n_steps: Number of Newton iterations (default 2).

    Returns:
        Refined points of shape ``(N, 3)``.
    """

    def _step(x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        f = sdf_fn(x)
        g = jax.grad(sdf_fn)(x)
        g_norm_sq = jnp.sum(g**2) + 1e-10
        refined: Float[Array, " 3"] = x - f * g / g_norm_sq
        return refined

    step_vmapped = jax.vmap(_step)
    result = points
    for _ in range(n_steps):
        result = step_vmapped(result)
    return result


def mean_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
) -> Float[Array, ""]:
    """Compute differentiable mean curvature via sigmoid-weighted Laplacian.

    Evaluates the SDF on a cell-centered grid, computes a sigmoid-derivative
    delta to identify the surface, Newton-refines grid points toward the
    zero level-set, and returns the delta-weighted average of the AD Hessian
    Laplacian at the refined points.

    For a sphere of radius R, returns approximately 2/R.
    For a plane, returns approximately 0.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).

    Returns:
        Scalar mean curvature, differentiable w.r.t. the SDF
        function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Sphere
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> kappa = mean_curvature(sphere.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)

    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width

    flat_grid = grid.reshape(-1, 3)
    refined = _newton_refine(sdf_fn, flat_grid, n_steps=2)

    curvatures = jax.vmap(lambda x: jnp.trace(jax.jacfwd(jax.grad(sdf_fn))(x)))(refined)
    curvatures = jnp.where(jnp.isfinite(curvatures), curvatures, 0.0)
    curvatures = curvatures.reshape(grid.shape[:-1])

    delta_sum = jnp.sum(delta) * cell_vol
    weighted = jnp.sum(curvatures * delta) * cell_vol
    return weighted / (delta_sum + 1e-20)


def max_curvature(
    sdf_fn: Callable[..., Float[Array, ...]],
    *,
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int = 64,
    temperature: float = 0.01,
) -> Float[Array, ""]:
    """Differentiable estimate of maximum surface curvature.

    Uses the same sigmoid delta framework as :func:`mean_curvature` but
    returns a soft-max estimate of the maximum absolute curvature over
    the surface, weighted by the delta function.

    For a sphere of radius R, returns approximately 2/R.

    Args:
        sdf_fn: Signed distance function accepting points of shape
            ``(..., 3)`` and returning SDF values of shape ``(...)``.
        lo: Lower corner of the evaluation domain.
        hi: Upper corner of the evaluation domain.
        resolution: Grid resolution per axis (default 64).
        temperature: Soft-max temperature; lower values approximate
            the true maximum more closely but may have sharper gradients.

    Returns:
        Scalar estimate of maximum surface curvature, differentiable
        w.r.t. the SDF function's parameters.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.primitives import Sphere
        >>> sphere = Sphere(center=jnp.zeros(3), radius=jnp.array(1.0))
        >>> lo, hi = jnp.array([-2.0]*3), jnp.array([2.0]*3)
        >>> kappa_max = max_curvature(sphere.sdf, lo=lo, hi=hi, resolution=64)
    """
    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)
    grid, _ = make_grid_3d(lo, hi, resolution)
    sdf_vals = sdf_fn(grid)

    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)

    indicator = jax.nn.sigmoid(-sdf_vals / cell_width)
    delta = indicator * (1.0 - indicator) / cell_width

    flat_grid = grid.reshape(-1, 3)
    refined = _newton_refine(sdf_fn, flat_grid, n_steps=2)

    curvatures = jax.vmap(lambda x: jnp.trace(jax.jacfwd(jax.grad(sdf_fn))(x)))(refined)
    curvatures = jnp.where(jnp.isfinite(curvatures), curvatures, 0.0)
    curvatures = curvatures.reshape(grid.shape[:-1])

    abs_curv = jnp.abs(curvatures)
    flat_delta = delta.ravel()
    flat_abs_curv = abs_curv.ravel()

    # Delta-weighted soft-max: use log(delta) to focus on surface points
    log_w = flat_abs_curv / temperature + jnp.log(flat_delta + 1e-30)
    softmax_w = jax.nn.softmax(log_w)
    return jnp.sum(flat_abs_curv * softmax_w)


def _per_face_analytical_mean_curvature(
    surface_type: str,
    params: dict[str, object],
) -> Float[Array, ""]:
    """Analytical mean curvature for one face's primitive type.

    The mean curvature ``H = (k1 + k2) / 2`` is constant across plane,
    sphere and cylinder faces; the closed forms below match the
    standard convention with the surface's outward normal pointing
    away from a convex region.  Cone, torus and BSpline faces have
    spatially varying curvature and return ``NaN`` until their
    handlers are added.

    Sign convention is intrinsic to the underlying surface: the value
    is non-negative for the supported primitive types.  Callers that
    need an outward-of-solid sign should multiply by the face
    orientation (``+1`` for ``TopAbs_FORWARD``, ``-1`` for
    ``TopAbs_REVERSED``); ``triangulate_shape`` does not currently
    surface that flag in ``params_list`` and a follow-up PR can add
    it once a metric needs it.
    """
    if surface_type == "plane":
        return jnp.asarray(0.0)
    if surface_type == "sphere":
        radius = params["radius"]
        return jnp.asarray(1.0) / jnp.asarray(radius)
    if surface_type == "cylinder":
        radius = params["radius"]
        return jnp.asarray(1.0) / (2.0 * jnp.asarray(radius))
    # Cone / torus / BSpline analytical closed forms vary spatially or
    # require parametric integration; deferred to a follow-up PR.
    return jnp.asarray(jnp.nan)


def mean_curvature_per_face(
    shape: TopoDS_Shape,
) -> tuple[Float[Array, " n_faces"], list[dict[str, object]]]:
    """Analytical mean curvature for each face of a shape.

    Reads each face's primitive parameters via
    :func:`~brepax.brep.triangulate.per_face_geometric_params` (an
    OCCT face traversal that does not build a triangulation) and
    computes the mean curvature analytically: 0 for plane, ``1 / r``
    for sphere, ``1 / (2 r)`` for cylinder.  Cone, torus and BSpline
    faces return ``NaN``; those handlers are deferred to a follow-up
    PR because their mean curvature is not constant across the face
    and a single representative value requires either parametric
    integration or a face-centroid convention that this PR does not
    yet pin.

    Skipping the mesh build is what distinguishes this function from
    :func:`surface_area_per_face` and :func:`min_draft_angle_per_face`,
    which both need the actual triangle vertex array.  Analytical mean
    curvature only depends on the primitive's parameters, so paying
    for ``BRepMesh`` plus the JAX-side vertex evaluation would be
    wasted work.

    Trim awareness is automatic: the per-face mean curvature is a
    property of the primitive surface, not its trim region, so the
    function reports the same value for any trimmed sub-region of
    the same surface.

    Differentiability: the closed forms flow through the primitive's
    ``radius`` field, which is a JAX scalar populated by
    :func:`~brepax.brep.triangulate._extract_face_geometric_params`.
    ``jax.grad`` over a per-face curvature sum gives finite gradients
    on the radius; NaN faces have undefined gradient by construction.

    Args:
        shape: An OCCT topological shape.  Faces are iterated in the
            same per-Solid order as
            :func:`~brepax.brep.triangulate.triangulate_shape`.

    Returns:
        Tuple ``(curvatures, params_list)`` where ``curvatures`` has
        shape ``(n_faces,)`` with the mean curvature for each face in
        traversal order (NaN for unsupported types), and
        ``params_list`` is the list returned by
        :func:`~brepax.brep.triangulate.per_face_geometric_params`
        (each entry includes ``surface_type`` and the analytical
        primitive parameters; no ``n_triangles`` because no
        triangulation is performed).

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax._occt.backend import BRepPrimAPI_MakeBox
        >>> from brepax.metrics.curvature import mean_curvature_per_face
        >>> shape = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        >>> kappas, _ = mean_curvature_per_face(shape)
        >>> # All 6 faces are planes => H = 0.
        >>> bool(jnp.all(kappas == 0.0))
        True
    """
    params_list = per_face_geometric_params(shape)
    n_faces = len(params_list)
    if n_faces == 0:
        return jnp.zeros((0,)), params_list
    kappas = jnp.stack(
        [_per_face_analytical_mean_curvature(p["surface_type"], p) for p in params_list]
    )
    return kappas, params_list


__all__ = [
    "max_curvature",
    "mean_curvature",
    "mean_curvature_per_face",
]
