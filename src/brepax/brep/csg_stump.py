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
    if lo is None or hi is None:
        if stump.bbox_lo is not None and stump.bbox_hi is not None:
            lo_auto, hi_auto = stump.bbox_lo, stump.bbox_hi
        else:
            lo_auto, hi_auto = _stump_bounds(stump)
        if lo is None:
            lo = lo_auto - 0.5
        if hi is None:
            hi = hi_auto + 0.5

    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)

    grid = make_grid_3d(lo, hi, resolution)[0]
    sdf = evaluate_stump_sdf(stump, grid)
    return integrate_sdf_volume(sdf, lo, hi, resolution)


def evaluate_stump_volume_stratum(
    stump: CSGStump,
    *,
    resolution: int = 64,
) -> Float[Array, ""] | None:
    """Evaluate volume using analytical methods where possible.

    Supports two analytical paths:

    1. **Stratum dispatch** (single-term, all bounded primitives):
       Decomposes into pairwise Boolean operations with
       stratum dispatch for analytical exact gradients.

    2. **Clipped-box** (Box + axis-aligned Planes):
       Each term is a Box clipped by axis-aligned half-spaces,
       producing a smaller Box whose volume is analytically computable.
       PMC cells are disjoint, so term volumes sum correctly.

    Returns ``None`` if neither path applies.

    Args:
        stump: A CSG-Stump (typically after :func:`group_stump_primitives`).
        resolution: Grid resolution for the intersecting stratum.

    Returns:
        Scalar volume, or ``None`` if analytical evaluation is not applicable.
    """
    from brepax.primitives import Box as BoxPrim
    from brepax.primitives import Plane as PlanePrim

    all_bounded = all(jnp.isfinite(p.volume()) for p in stump.primitives)
    t_mat = np.asarray(stump.intersection_matrix)
    m = t_mat.shape[0]

    # Path 1: single-term + all bounded → stratum dispatch
    if all_bounded:
        from brepax.boolean import intersect_volume

        term_volumes: list[Float[Array, ""]] = []
        for k in range(m):
            if float(stump.union_mask[k]) < 0.5:
                continue
            row = t_mat[k]
            inside = [stump.primitives[j] for j in range(len(row)) if row[j] > 0.5]
            outside = [stump.primitives[j] for j in range(len(row)) if row[j] < -0.5]
            if not inside:
                continue
            vol = inside[0].volume()
            base = inside[0]
            for p in inside[1:]:
                vol = intersect_volume(base, p, resolution=resolution)
                base = p
            for p in outside:
                vol = vol - intersect_volume(base, p, resolution=resolution)
            term_volumes.append(vol)

        if len(term_volumes) == 1:
            return term_volumes[0]

    # Path 2: Box + axis-aligned Planes → clipped-box analytical
    boxes = [(j, p) for j, p in enumerate(stump.primitives) if isinstance(p, BoxPrim)]
    if len(boxes) != 1:
        return None
    box_idx, box = boxes[0]
    non_box_are_planes = all(
        isinstance(p, PlanePrim) for j, p in enumerate(stump.primitives) if j != box_idx
    )
    if not non_box_are_planes:
        return None

    box_lo = box.center - box.half_extents
    box_hi = box.center + box.half_extents

    total = jnp.array(0.0)
    for k in range(m):
        if float(stump.union_mask[k]) < 0.5:
            continue
        if t_mat[k, box_idx] < 0.5:
            continue
        lo = box_lo
        hi = box_hi
        for j in range(len(stump.primitives)):
            if j == box_idx or t_mat[k, j] == 0:
                continue
            p = stump.primitives[j]
            if not isinstance(p, PlanePrim):
                continue
            n = np.asarray(p.normal)
            axis = int(np.argmax(np.abs(n)))
            sign_n = float(np.sign(n[axis]))
            plane_pos = p.offset / sign_n
            if t_mat[k, j] * sign_n > 0:
                hi = hi.at[axis].set(jnp.minimum(hi[axis], plane_pos))
            else:
                lo = lo.at[axis].set(jnp.maximum(lo[axis], plane_pos))
        cell_vol = jnp.prod(jnp.maximum(hi - lo, 0.0))
        total = total + cell_vol

    return total


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


def group_stump_primitives(
    stump: CSGStump,
    shape: TopoDS_Shape,
) -> CSGStump:
    """Group face-level primitives into bounded primitives (Box, FiniteCylinder).

    Uses the same face classification logic as
    :func:`~brepax.brep.csg.reconstruct_stock_minus_features` to
    identify which faces form a Box (3 pairs of antiparallel planes)
    and which cylindrical faces form FiniteCylinders.

    The T matrix is re-indexed so that grouped primitives replace
    their constituent face-level columns.  This enables stratum
    dispatch (which requires bounded primitives with finite volume).

    Args:
        stump: A CSG-Stump with face-level primitives.
        shape: The OCCT shape (needed for face parametric bounds).

    Returns:
        A CSG-Stump with grouped (bounded) primitives.
    """
    from brepax.brep.csg import (
        _classify_faces_and_build_box,
        _extract_indexed_faces,
        _group_connected_faces,
        _reconstruct_cylinder_feature,
    )
    from brepax.brep.topology import build_adjacency_graph

    prims_nullable: list[Primitive | None] = list(stump.primitives)
    result = _classify_faces_and_build_box(prims_nullable)
    if result is None:
        return stump

    box, box_face_ids, feature_face_ids = result

    # Group cylinder features via adjacency
    faces = _extract_indexed_faces(shape)
    graph = build_adjacency_graph(shape)
    feature_groups = _group_connected_faces(feature_face_ids, graph)

    grouped_primitives: list[Primitive] = [box]
    # Mapping: old face index → new primitive index
    face_to_group: dict[int, int] = {}
    for fid in box_face_ids:
        face_to_group[fid] = 0

    for group in feature_groups:
        cyl_leaf = _reconstruct_cylinder_feature(group, faces, prims_nullable)
        if cyl_leaf is not None:
            gid = len(grouped_primitives)
            grouped_primitives.append(cyl_leaf.primitive)
            for fid in group:
                face_to_group[fid] = gid
        else:
            for fid in group:
                gid = len(grouped_primitives)
                grouped_primitives.append(stump.primitives[fid])
                face_to_group[fid] = gid

    n_new = len(grouped_primitives)
    t_old = np.asarray(stump.intersection_matrix)
    m = t_old.shape[0]

    # Re-derive T values using representative points from each cell.
    # This avoids face-level → grouped-level mapping errors when
    # multiple faces with conflicting T values map to the same group.
    rng = np.random.default_rng(123)
    lo_np = np.asarray(stump.bbox_lo if stump.bbox_lo is not None else -np.ones(3) * 10)
    hi_np = np.asarray(stump.bbox_hi if stump.bbox_hi is not None else np.ones(3) * 10)

    t_new = np.zeros((m, n_new))
    for k in range(m):
        # Find a representative point for this cell
        rep = _find_cell_representative(t_old[k], stump.primitives, lo_np, hi_np, rng)
        if rep is None:
            continue
        # Evaluate grouped primitives' SDF at the representative point
        rep_jax = jnp.array(rep)
        for g, gp in enumerate(grouped_primitives):
            sdf_val = float(np.asarray(gp.sdf(rep_jax)))
            t_new[k, g] = -float(np.sign(sdf_val)) if abs(sdf_val) > 1e-10 else 0.0

    # Remove duplicate rows and all-zero rows
    nonzero_mask = np.any(t_new != 0, axis=1)
    t_new = t_new[nonzero_mask]
    if len(t_new) > 0:
        t_new = np.unique(t_new, axis=0)

    if len(t_new) == 0:
        return stump

    return CSGStump(
        primitives=grouped_primitives,
        intersection_matrix=jnp.array(t_new),
        union_mask=jnp.ones(len(t_new)),
        face_ids=stump.face_ids,
        bbox_lo=stump.bbox_lo,
        bbox_hi=stump.bbox_hi,
    )


def compact_stump(stump: CSGStump) -> CSGStump:
    """Compact a CSG-Stump by merging redundant intersection terms.

    Applies don't-care elimination: two rows that differ in exactly one
    column are merged by setting that column to 0 (don't-care).  This is
    the basic step of the Quine-McCluskey algorithm.  Duplicate rows are
    also removed.

    The result is a semantically equivalent CSG-Stump with fewer
    intersection terms, improving evaluation speed and volume accuracy.

    Args:
        stump: A CSG-Stump (typically from :func:`reconstruct_csg_stump`).

    Returns:
        A compact CSG-Stump with reduced intersection terms.
    """
    t_mat = np.asarray(stump.intersection_matrix)

    # Iterative don't-care merge
    changed = True
    while changed:
        changed = False
        new_rows: list[np.ndarray] = []
        used: set[int] = set()
        for i in range(len(t_mat)):
            if i in used:
                continue
            merged = False
            for j in range(i + 1, len(t_mat)):
                if j in used:
                    continue
                diff_mask = t_mat[i] != t_mat[j]
                if diff_mask.sum() == 1:
                    merged_row = t_mat[i].copy()
                    merged_row[diff_mask] = 0.0
                    new_rows.append(merged_row)
                    used.add(i)
                    used.add(j)
                    merged = True
                    changed = True
                    break
            if not merged and i not in used:
                new_rows.append(t_mat[i])
        t_mat = np.array(new_rows) if new_rows else t_mat

    # Remove duplicate rows
    t_mat = np.unique(t_mat, axis=0)

    return CSGStump(
        primitives=stump.primitives,
        intersection_matrix=jnp.array(t_mat),
        union_mask=jnp.ones(len(t_mat)),
        face_ids=stump.face_ids,
        bbox_lo=stump.bbox_lo,
        bbox_hi=stump.bbox_hi,
    )


def _find_cell_representative(
    row: np.ndarray,
    primitives: list[Primitive],
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 5000,
) -> np.ndarray | None:
    """Find a point whose face-level SDF signs match the given T row."""
    pts = rng.uniform(lo, hi, size=(n_samples, 3))
    pts_jax = jnp.array(pts)
    for j, prim in enumerate(primitives):
        if row[j] == 0:
            continue
        sdf_vals = np.asarray(prim.sdf(pts_jax))
        expected_sign = -row[j]
        mask = np.sign(sdf_vals) == expected_sign
        pts = pts[mask]
        pts_jax = jnp.array(pts)
        if len(pts) == 0:
            return None
    return pts[0] if len(pts) > 0 else None


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

    # Cell enumeration via random sampling with convergence check.
    # Store a representative point for each sign vector to avoid
    # expensive re-search later.
    rng = np.random.default_rng(seed)
    found_cells: dict[tuple[float, ...], np.ndarray] = {}
    no_new_count = 0

    for _round in range(max_rounds):
        pts = rng.uniform(lo_np, hi_np, size=(samples_per_round, 3))
        pts_jax = jnp.array(pts)
        sign_matrix = np.zeros((samples_per_round, n))
        for j, prim in enumerate(valid_primitives):
            sdf_vals = np.asarray(prim.sdf(pts_jax))
            sign_matrix[:, j] = np.sign(sdf_vals)

        any_new = False
        unique_svs, first_indices = np.unique(sign_matrix, axis=0, return_index=True)
        for sv_arr, idx in zip(unique_svs, first_indices, strict=True):
            sv = tuple(float(v) for v in sv_arr)
            if sv not in found_cells:
                found_cells[sv] = pts[idx]
                any_new = True

        if not any_new:
            no_new_count += 1
            if no_new_count >= convergence_rounds:
                break
        else:
            no_new_count = 0

    if not found_cells:
        return None

    # PMC: classify each cell as IN or OUT
    classifier = BRepClass3d_SolidClassifier()
    classifier.Load(shape)

    inside_sign_vectors: list[tuple[float, ...]] = []

    for sv, rep_point in found_cells.items():
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


__all__ = [
    "CSGStump",
    "DifferentiableCSGStump",
    "compact_stump",
    "csg_tree_to_stump",
    "evaluate_stump_sdf",
    "evaluate_stump_volume",
    "evaluate_stump_volume_stratum",
    "group_stump_primitives",
    "reconstruct_csg_stump",
    "stump_to_differentiable",
]
