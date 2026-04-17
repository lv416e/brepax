"""CSG-Stump: disjunctive normal form representation of CSG expressions.

A CSG-Stump encodes a solid as a union of intersections of half-spaces.
This is the normal form (DNF) for CSG, capable of representing any
solid bounded by the given primitive surfaces.

The representation uses two matrices:

- ``intersection_matrix`` of shape ``(m, n)`` with values in {+1, -1, 0}:
  for each of ``m`` intersection terms, ``+1`` means the primitive's
  inside, ``-1`` means outside, ``0`` means unused.
- ``union_mask`` of shape ``(m,)`` with values in {0, 1}: selects
  which intersection terms participate in the final union.

The stock-minus-features pattern from :mod:`brepax.brep.csg` is a
special case where ``m=1`` and the single intersection term combines
the stock's inside with each feature's outside.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from brepax._occt.backend import BRepClass3d_SolidClassifier, TopAbs_IN, gp_Pnt
from brepax._occt.types import TopoDS_Shape
from brepax.brep.csg import CSGLeaf, CSGNode, CSGOperation
from brepax.brep.csg_eval import integrate_sdf_volume, make_grid_3d, primitive_bounds
from brepax.primitives._base import Primitive


@dataclass
class CSGStump:
    """Disjunctive normal form of a CSG expression.

    Attributes:
        primitives: The ``n`` half-space primitives.
        intersection_matrix: ``(m, n)`` matrix with values +1, -1, or 0.
        union_mask: ``(m,)`` binary mask selecting active intersection terms.
    """

    primitives: list[Primitive]
    intersection_matrix: Array
    union_mask: Array
    face_ids: list[list[int]] = field(default_factory=list)
    bbox_lo: Array | None = None
    bbox_hi: Array | None = None


def evaluate_stump_sdf(
    stump: CSGStump,
    x: Float[Array, "... 3"],
) -> Float[Array, ...]:
    """Evaluate the composite SDF of a CSG-Stump at query points.

    For each intersection term, computes ``max(sign_j * sdf_j)`` over
    active primitives.  The final result is ``min`` over active terms.

    Args:
        stump: A CSG-Stump representation.
        x: Query points with shape ``(..., 3)``.

    Returns:
        Signed distance values with shape ``(...)``.
    """
    sdfs = jnp.stack([p.sdf(x) for p in stump.primitives], axis=-1)
    return _evaluate_dnf_sdf(sdfs, stump.intersection_matrix, stump.union_mask)


def evaluate_stump_volume(
    stump: CSGStump,
    *,
    resolution: int = 64,
    lo: Float[Array, 3] | None = None,
    hi: Float[Array, 3] | None = None,
) -> Float[Array, ""]:
    """Evaluate volume of a CSG-Stump via grid integration.

    Args:
        stump: A CSG-Stump representation.
        resolution: Number of grid points per axis.
        lo: Grid lower bound. Auto-computed if None.
        hi: Grid upper bound. Auto-computed if None.

    Returns:
        Scalar volume estimate.
    """
    if lo is None:
        lo = stump.bbox_lo if stump.bbox_lo is not None else _stump_bounds(stump)[0]
        lo = lo - 0.5
    if hi is None:
        hi = stump.bbox_hi if stump.bbox_hi is not None else _stump_bounds(stump)[1]
        hi = hi + 0.5

    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)

    grid = make_grid_3d(lo, hi, resolution)[0]
    sdf = evaluate_stump_sdf(stump, grid)
    return integrate_sdf_volume(sdf, lo, hi, resolution)


class DifferentiableCSGStump(eqx.Module):
    """CSG-Stump wrapped for differentiable evaluation via equinox.

    Primitive parameters are pytree leaves (differentiable).
    The intersection matrix and union mask are static (not differentiated).

    Attributes:
        primitives: Tuple of primitives (pytree leaves).
        intersection_matrix: ``(m, n)`` sign matrix (static).
        union_mask: ``(m,)`` binary mask (static).
    """

    primitives: tuple[Primitive, ...]
    intersection_matrix: np.ndarray = eqx.field(static=True)
    union_mask: np.ndarray = eqx.field(static=True)

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, ...]:
        """Composite SDF of the CSG-Stump."""
        sdfs = jnp.stack([p.sdf(x) for p in self.primitives], axis=-1)
        return _evaluate_dnf_sdf(sdfs, self.intersection_matrix, self.union_mask)

    def volume(
        self,
        *,
        resolution: int = 64,
        lo: Float[Array, 3] | None = None,
        hi: Float[Array, 3] | None = None,
    ) -> Float[Array, ""]:
        """Differentiable volume via grid integration."""
        if lo is None or hi is None:
            lo_auto, hi_auto = _primitives_bounds(self.primitives)
            margin = 0.5
            if lo is None:
                lo = lo_auto - margin
            if hi is None:
                hi = hi_auto + margin

        lo = jax.lax.stop_gradient(lo)
        hi = jax.lax.stop_gradient(hi)

        grid = make_grid_3d(lo, hi, resolution)[0]
        sdf = self.sdf(grid)
        return integrate_sdf_volume(sdf, lo, hi, resolution)


def csg_tree_to_stump(node: CSGNode) -> CSGStump:
    """Convert a left-leaning subtract tree to a CSG-Stump.

    A tree of the form ``(((stock - f1) - f2) - f3)`` becomes a
    single intersection term: stock inside, all features outside.

    Args:
        node: A CSG tree from :func:`reconstruct_stock_minus_features`.

    Returns:
        Equivalent CSG-Stump representation.

    Raises:
        ValueError: If the tree is not a left-leaning subtract chain.
    """
    primitives: list[Primitive] = []
    face_ids: list[list[int]] = []
    feature_indices: list[int] = []

    current: CSGNode = node
    while isinstance(current, CSGOperation) and current.op == "subtract":
        if isinstance(current.right, CSGLeaf):
            idx = len(primitives)
            primitives.append(current.right.primitive)
            face_ids.append(list(current.right.face_ids))
            feature_indices.append(idx)
        current = current.left

    if not isinstance(current, CSGLeaf):
        msg = "Cannot convert: root of the tree is not a CSGLeaf"
        raise ValueError(msg)

    stock_idx = len(primitives)
    primitives.append(current.primitive)
    face_ids.append(list(current.face_ids))

    n = len(primitives)
    row = jnp.zeros(n)
    row = row.at[stock_idx].set(1.0)
    for fi in feature_indices:
        row = row.at[fi].set(-1.0)

    return CSGStump(
        primitives=primitives,
        intersection_matrix=row[jnp.newaxis, :],
        union_mask=jnp.array([1.0]),
        face_ids=face_ids,
    )


def stump_to_differentiable(stump: CSGStump) -> DifferentiableCSGStump:
    """Convert a CSGStump to a :class:`DifferentiableCSGStump`.

    Args:
        stump: A CSG-Stump representation.

    Returns:
        An equinox Module ready for gradient-based optimization.
    """
    # equinox warns about JAX arrays in non-static fields when constructing
    # a Module with static fields; the warning is spurious here since the
    # static fields receive numpy arrays, not JAX arrays.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A JAX array is being set as static")
        return DifferentiableCSGStump(
            primitives=tuple(stump.primitives),
            intersection_matrix=np.asarray(stump.intersection_matrix),
            union_mask=np.asarray(stump.union_mask),
        )


# --- Bounds utilities ---


def _stump_bounds(stump: CSGStump) -> tuple[Array, Array]:
    """Compute bounding box for a CSG-Stump."""
    return _primitives_bounds(stump.primitives)


def _evaluate_dnf_sdf(
    sdfs: Float[Array, "... n"],
    intersection_matrix: Array | np.ndarray,
    union_mask: Array | np.ndarray,
) -> Float[Array, ...]:
    """Core DNF SDF evaluation shared by CSGStump and DifferentiableCSGStump.

    For each active intersection term, computes max(sign * sdf) over
    participating half-spaces.  Returns min over active terms.
    """
    m = intersection_matrix.shape[0]
    term_sdfs = []
    for i in range(m):
        row = intersection_matrix[i]
        active = jnp.abs(row) > 0.5
        signed = row * sdfs
        masked = jnp.where(active, signed, -jnp.inf)
        term_sdf = jnp.max(masked, axis=-1)
        # Weight by union_mask: inactive terms get +inf (ignored by min)
        term_sdf = jnp.where(union_mask[i] > 0.5, term_sdf, jnp.inf)
        term_sdfs.append(term_sdf)

    if not term_sdfs:
        return jnp.full(sdfs.shape[:-1], jnp.inf)
    return jnp.min(jnp.stack(term_sdfs, axis=-1), axis=-1)


def _primitives_bounds(
    primitives: tuple[Primitive, ...] | list[Primitive],
) -> tuple[Array, Array]:
    """Compute bounding box enclosing all primitives."""
    lo = jnp.full(3, jnp.inf)
    hi = jnp.full(3, -jnp.inf)
    for p in primitives:
        plo, phi = primitive_bounds(p)
        lo = jnp.minimum(lo, plo)
        hi = jnp.maximum(hi, phi)
    return lo, hi


def reconstruct_csg_stump(
    shape: TopoDS_Shape,
    *,
    samples_per_round: int = 5000,
    max_rounds: int = 10,
    convergence_rounds: int = 3,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> CSGStump | None:
    """Reconstruct a CSG-Stump from a B-Rep shape via point membership classification.

    Enumerates spatial cells by sampling random points and computing
    sign vectors (signs of each primitive's SDF).  Each unique sign
    vector is classified as inside or outside using OCCT's solid
    classifier.  Inside cells form the rows of the intersection matrix.

    Sampling runs in rounds; if no new sign vectors are found for
    ``convergence_rounds`` consecutive rounds, enumeration stops.

    Args:
        shape: An OCCT topological shape (must be a closed solid).
        samples_per_round: Random points per sampling round.
        max_rounds: Maximum number of sampling rounds.
        convergence_rounds: Stop after this many rounds with no new cells.
        tolerance: Tolerance for OCCT solid classifier.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`CSGStump`, or ``None`` if reconstruction fails.
    """
    # Lazy imports to avoid circular dependency
    from brepax.brep.convert import face_to_primitive
    from brepax.brep.csg import _extract_indexed_faces

    faces = _extract_indexed_faces(shape)
    primitives = [face_to_primitive(f) for f in faces]
    valid_primitives: list[Primitive] = []
    face_id_map: list[int] = []
    for i, p in enumerate(primitives):
        if p is not None:
            face_id_map.append(i)
            valid_primitives.append(p)

    if not valid_primitives:
        return None

    n = len(valid_primitives)

    # Use OCCT bounding box (handles unbounded primitives like Plane)
    from brepax.brep.convert import shape_metadata

    meta = shape_metadata(shape)
    lo_np = np.array(meta.bbox_min) - 0.5
    hi_np = np.array(meta.bbox_max) + 0.5

    # Cell enumeration via random sampling with convergence check
    rng = np.random.default_rng(seed)
    found_sign_vectors: set[tuple[float, ...]] = set()
    no_new_count = 0

    for _round in range(max_rounds):
        pts = rng.uniform(lo_np, hi_np, size=(samples_per_round, 3))
        sign_matrix = np.zeros((samples_per_round, n))
        for j, prim in enumerate(valid_primitives):
            sdf_vals = np.asarray(prim.sdf(jnp.array(pts)))
            sign_matrix[:, j] = np.sign(sdf_vals)

        new_vectors = {tuple(row) for row in sign_matrix} - found_sign_vectors
        if not new_vectors:
            no_new_count += 1
            if no_new_count >= convergence_rounds:
                break
        else:
            no_new_count = 0
            found_sign_vectors.update(new_vectors)

    if not found_sign_vectors:
        return None

    # PMC: classify each sign vector as IN or OUT
    classifier = BRepClass3d_SolidClassifier()
    classifier.Load(shape)

    inside_sign_vectors: list[tuple[float, ...]] = []

    for sv in found_sign_vectors:
        # Find a representative point for this sign vector
        rep_point = _find_representative_point(sv, valid_primitives, lo_np, hi_np, rng)
        if rep_point is None:
            continue

        gp = gp_Pnt(float(rep_point[0]), float(rep_point[1]), float(rep_point[2]))
        classifier.Perform(gp, tolerance)
        if classifier.State() == TopAbs_IN:
            inside_sign_vectors.append(sv)

    if not inside_sign_vectors:
        return None

    # Build T matrix: T[j] = -sign_vector[j]
    t_rows = []
    for sv in inside_sign_vectors:
        t_rows.append([-s for s in sv])

    intersection_matrix = jnp.array(t_rows)
    union_mask = jnp.ones(len(t_rows))

    face_ids = [[fid] for fid in face_id_map]

    return CSGStump(
        primitives=valid_primitives,
        intersection_matrix=intersection_matrix,
        union_mask=union_mask,
        face_ids=face_ids,
        bbox_lo=jnp.array(meta.bbox_min),
        bbox_hi=jnp.array(meta.bbox_max),
    )


def _find_representative_point(
    sign_vector: tuple[float, ...],
    primitives: list[Primitive],
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> np.ndarray | None:
    """Find a point whose SDF signs match the given sign vector."""
    pts = rng.uniform(lo, hi, size=(max_attempts, 3))
    for pt in pts:
        signs = tuple(
            float(np.sign(np.asarray(p.sdf(jnp.array(pt))))) for p in primitives
        )
        if signs == sign_vector:
            return np.asarray(pt)
    return None


__all__ = [
    "CSGStump",
    "DifferentiableCSGStump",
    "csg_tree_to_stump",
    "evaluate_stump_sdf",
    "evaluate_stump_volume",
    "reconstruct_csg_stump",
    "stump_to_differentiable",
]
