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
        lo_auto, hi_auto = _stump_bounds(stump)
        margin = 0.5
        if lo is None:
            lo = lo_auto - margin
        if hi is None:
            hi = hi_auto + margin

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


__all__ = [
    "CSGStump",
    "DifferentiableCSGStump",
    "csg_tree_to_stump",
    "evaluate_stump_sdf",
    "evaluate_stump_volume",
    "stump_to_differentiable",
]
