"""Differentiable evaluation of CSG trees.

Provides composite SDF evaluation, grid-based volume integration,
and an equinox Module wrapper for gradient-based optimization of
primitive parameters.
"""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from brepax.brep.csg import CSGLeaf, CSGNode, CSGOperation
from brepax.primitives._base import Primitive


def evaluate_csg_sdf(
    node: CSGNode,
    x: Float[Array, "... 3"],
) -> Float[Array, ...]:
    """Evaluate the composite SDF of a CSG tree at query points.

    Recursively combines leaf primitive SDFs using the Boolean operations
    defined in the tree: min for union, max(left, -right) for subtract,
    max for intersect.

    Args:
        node: Root of a CSG tree.
        x: Query points with shape ``(..., 3)``.

    Returns:
        Signed distance values with shape ``(...)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from brepax.brep.csg import CSGLeaf
        >>> from brepax.primitives import Sphere
        >>> leaf = CSGLeaf(primitive=Sphere(center=jnp.zeros(3), radius=jnp.array(1.0)))
        >>> float(evaluate_csg_sdf(leaf, jnp.array([2.0, 0.0, 0.0])))
        1.0
    """
    if isinstance(node, CSGLeaf):
        return node.primitive.sdf(x)
    left = evaluate_csg_sdf(node.left, x)
    right = evaluate_csg_sdf(node.right, x)
    if node.op == "union":
        return jnp.minimum(left, right)
    if node.op == "subtract":
        return jnp.maximum(left, -right)
    return jnp.maximum(left, right)


def evaluate_csg_volume(
    node: CSGNode,
    *,
    resolution: int = 64,
    lo: Float[Array, 3] | None = None,
    hi: Float[Array, 3] | None = None,
) -> Float[Array, ""]:
    """Evaluate volume of a CSG tree via grid integration.

    Uses a sigmoid indicator for differentiability. The sharpness is
    set to ``1 / cell_width`` (same convention as the stratum method),
    so precision improves with resolution.

    Args:
        node: Root of a CSG tree.
        resolution: Number of grid points per axis.
        lo: Grid lower bound ``(3,)``. Auto-computed from primitives if None.
        hi: Grid upper bound ``(3,)``. Auto-computed from primitives if None.

    Returns:
        Scalar volume estimate.
    """
    if lo is None or hi is None:
        lo_auto, hi_auto = _tree_bounds(node)
        margin = 0.5
        if lo is None:
            lo = lo_auto - margin
        if hi is None:
            hi = hi_auto + margin

    lo = jax.lax.stop_gradient(lo)
    hi = jax.lax.stop_gradient(hi)

    sdf = evaluate_csg_sdf(node, _make_grid_3d(lo, hi, resolution)[0])
    return _integrate_sdf_volume(sdf, lo, hi, resolution)


class DifferentiableCSG(eqx.Module):
    """CSG tree wrapped for differentiable evaluation via equinox.

    Stores the stock primitive and subtractive features as pytree
    leaves, enabling ``eqx.filter_grad`` through volume evaluation.

    Attributes:
        stock: The bounding stock primitive (typically a Box).
        features: Subtractive feature primitives (typically FiniteCylinders).

    Examples:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> from brepax.primitives import Box, FiniteCylinder
        >>> box = Box(center=jnp.zeros(3), half_extents=jnp.ones(3) * 5)
        >>> cyl = FiniteCylinder(
        ...     center=jnp.zeros(3), axis=jnp.array([0., 0., 1.]),
        ...     radius=jnp.array(1.0), height=jnp.array(10.0),
        ... )
        >>> dcsg = DifferentiableCSG(stock=box, features=(cyl,))
        >>> vol = dcsg.volume(resolution=32)
    """

    stock: Primitive
    features: tuple[Primitive, ...]

    def sdf(self, x: Float[Array, "... 3"]) -> Float[Array, ...]:
        """Composite SDF: stock minus all features."""
        sdf_val = self.stock.sdf(x)
        for feat in self.features:
            sdf_val = jnp.maximum(sdf_val, -feat.sdf(x))
        return sdf_val

    def volume(
        self,
        *,
        resolution: int = 64,
        lo: Float[Array, 3] | None = None,
        hi: Float[Array, 3] | None = None,
    ) -> Float[Array, ""]:
        """Differentiable volume via grid integration with sigmoid indicator.

        Args:
            resolution: Number of grid points per axis.
            lo: Grid lower bound. Auto-computed from stock if None.
            hi: Grid upper bound. Auto-computed from stock if None.

        Returns:
            Scalar volume estimate.
        """
        if lo is None or hi is None:
            lo_auto, hi_auto = _primitive_bounds(self.stock)
            margin = 0.5
            if lo is None:
                lo = lo_auto - margin
            if hi is None:
                hi = hi_auto + margin

        lo = jax.lax.stop_gradient(lo)
        hi = jax.lax.stop_gradient(hi)

        sdf = self.sdf(_make_grid_3d(lo, hi, resolution)[0])
        return _integrate_sdf_volume(sdf, lo, hi, resolution)


def csg_to_differentiable(node: CSGNode) -> DifferentiableCSG:
    """Convert a CSG tree to a :class:`DifferentiableCSG`.

    Extracts the stock primitive and feature primitives from a
    left-leaning subtract tree produced by
    :func:`~brepax.brep.csg.reconstruct_stock_minus_features`.

    Args:
        node: A CSG tree (typically from ``reconstruct_stock_minus_features``).

    Returns:
        A :class:`DifferentiableCSG` ready for gradient-based optimization.

    Raises:
        ValueError: If the tree is not a left-leaning subtract chain.
    """
    features: list[Primitive] = []
    current: CSGNode = node
    while isinstance(current, CSGOperation) and current.op == "subtract":
        if isinstance(current.right, CSGLeaf):
            features.append(current.right.primitive)
        current = current.left

    if not isinstance(current, CSGLeaf):
        msg = "Cannot convert: root of the tree is not a CSGLeaf"
        raise ValueError(msg)

    return DifferentiableCSG(stock=current.primitive, features=tuple(features))


# --- Shared integration helper ---


def _integrate_sdf_volume(
    sdf: Float[Array, ...],
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> Float[Array, ""]:
    """Integrate SDF values on a grid to compute volume.

    Uses sigmoid indicator with sharpness = 1 / cell_width,
    where cell_width is the geometric mean of axis spacings.
    """
    cell_vol = jnp.prod((hi - lo) / resolution)
    cell_width = jnp.power(cell_vol, 1.0 / 3.0)
    indicator = jax.nn.sigmoid(-sdf / cell_width)
    return jnp.sum(indicator) * cell_vol


# --- Grid utilities ---


def _make_grid_3d(
    lo: Float[Array, 3],
    hi: Float[Array, 3],
    resolution: int,
) -> tuple[Float[Array, "R R R 3"], Float[Array, ""]]:
    """Create a cell-centered 3D grid over the given domain."""
    dx = (hi - lo) / resolution
    axes = [
        jnp.linspace(lo[i] + dx[i] / 2.0, hi[i] - dx[i] / 2.0, resolution)
        for i in range(3)
    ]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    grid = jnp.stack(mesh, axis=-1)
    cell_vol = jnp.prod(dx)
    return grid, cell_vol


def _primitive_bounds(p: Primitive) -> tuple[Array, Array]:
    """Estimate axis-aligned bounding box for a primitive."""
    params = p.parameters()
    if "center" in params:
        c = params["center"]
        he = params.get("half_extents", None)
        if he is not None:
            return c - he, c + he
        r = params.get("radius", jnp.array(1.0))
        h = params.get("height", jnp.array(0.0))
        extent = jnp.maximum(r, h / 2.0)
        return c - extent, c + extent
    if "point" in params:
        pt = params["point"]
        r = params["radius"]
        return pt - r - 2.0, pt + r + 2.0
    warnings.warn(
        f"Cannot determine bounds for {type(p).__name__}, using default",
        stacklevel=2,
    )
    return -jnp.ones(3) * 10.0, jnp.ones(3) * 10.0


def _tree_bounds(node: CSGNode) -> tuple[Array, Array]:
    """Compute bounding box enclosing all primitives in the tree."""
    leaves = _collect_leaves(node)
    lo = jnp.full(3, jnp.inf)
    hi = jnp.full(3, -jnp.inf)
    for leaf in leaves:
        plo, phi = _primitive_bounds(leaf.primitive)
        lo = jnp.minimum(lo, plo)
        hi = jnp.maximum(hi, phi)
    return lo, hi


def _collect_leaves(node: CSGNode) -> list[CSGLeaf]:
    """Collect all leaf nodes from a CSG tree."""
    if isinstance(node, CSGLeaf):
        return [node]
    leaves: list[CSGLeaf] = []
    leaves.extend(_collect_leaves(node.left))
    leaves.extend(_collect_leaves(node.right))
    return leaves


__all__ = [
    "DifferentiableCSG",
    "csg_to_differentiable",
    "evaluate_csg_sdf",
    "evaluate_csg_volume",
]
