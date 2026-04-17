"""Tests for CSG tree reconstruction from B-Rep topology."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from brepax.brep.csg import (
    CSGLeaf,
    CSGOperation,
    reconstruct_stock_minus_features,
)
from brepax.io.step import read_step
from brepax.primitives import Box, FiniteCylinder

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture()
def box_with_holes_tree() -> CSGOperation:
    shape = read_step(FIXTURES / "box_with_holes.step")
    tree = reconstruct_stock_minus_features(shape)
    assert isinstance(tree, CSGOperation)
    return tree


class TestBoxWithHolesReconstruction:
    """Reconstruction of a 40x30x20 box with two through-holes."""

    def test_root_is_subtract(self, box_with_holes_tree: CSGOperation) -> None:
        assert box_with_holes_tree.op == "subtract"

    def test_tree_depth(self, box_with_holes_tree: CSGOperation) -> None:
        """Two features produce a left-leaning tree of depth 2."""
        root = box_with_holes_tree
        assert root.op == "subtract"
        assert isinstance(root.right, CSGLeaf)
        assert isinstance(root.left, CSGOperation)
        assert root.left.op == "subtract"
        assert isinstance(root.left.left, CSGLeaf)
        assert isinstance(root.left.right, CSGLeaf)

    def test_stock_is_box(self, box_with_holes_tree: CSGOperation) -> None:
        inner = box_with_holes_tree.left
        assert isinstance(inner, CSGOperation)
        stock = inner.left
        assert isinstance(stock, CSGLeaf)
        assert isinstance(stock.primitive, Box)

    def test_box_dimensions(self, box_with_holes_tree: CSGOperation) -> None:
        """Stock box should match the 40x30x20 fixture."""
        inner = box_with_holes_tree.left
        assert isinstance(inner, CSGOperation)
        stock = inner.left
        assert isinstance(stock, CSGLeaf)
        box = stock.primitive
        assert isinstance(box, Box)
        assert jnp.allclose(box.center, jnp.array([20.0, 15.0, 10.0]), atol=0.1)
        assert jnp.allclose(box.half_extents, jnp.array([20.0, 15.0, 10.0]), atol=0.1)

    def test_features_are_finite_cylinders(
        self, box_with_holes_tree: CSGOperation
    ) -> None:
        features = _collect_features(box_with_holes_tree)
        assert len(features) == 2
        for feat in features:
            assert isinstance(feat.primitive, FiniteCylinder)

    def test_cylinder_radii(self, box_with_holes_tree: CSGOperation) -> None:
        features = _collect_features(box_with_holes_tree)
        radii = sorted(float(f.primitive.radius) for f in features)
        assert radii[0] == pytest.approx(3.0, abs=0.01)
        assert radii[1] == pytest.approx(4.0, abs=0.01)

    def test_cylinder_heights(self, box_with_holes_tree: CSGOperation) -> None:
        """Through-holes should span the full box height (20)."""
        features = _collect_features(box_with_holes_tree)
        for feat in features:
            assert isinstance(feat.primitive, FiniteCylinder)
            assert float(feat.primitive.height) == pytest.approx(20.0, abs=0.1)

    def test_cylinder_axes(self, box_with_holes_tree: CSGOperation) -> None:
        features = _collect_features(box_with_holes_tree)
        for feat in features:
            assert isinstance(feat.primitive, FiniteCylinder)
            axis = feat.primitive.axis
            assert jnp.allclose(jnp.abs(axis), jnp.array([0.0, 0.0, 1.0]), atol=0.01)

    def test_cylinder_centers(self, box_with_holes_tree: CSGOperation) -> None:
        features = _collect_features(box_with_holes_tree)
        centers_x = sorted(float(f.primitive.center[0]) for f in features)
        assert centers_x[0] == pytest.approx(10.0, abs=0.1)
        assert centers_x[1] == pytest.approx(30.0, abs=0.1)
        for feat in features:
            assert isinstance(feat.primitive, FiniteCylinder)
            assert float(feat.primitive.center[2]) == pytest.approx(10.0, abs=0.1)

    def test_face_ids_cover_all_faces(self, box_with_holes_tree: CSGOperation) -> None:
        all_ids = _collect_all_face_ids(box_with_holes_tree)
        assert sorted(all_ids) == list(range(8))

    def test_stock_has_six_faces(self, box_with_holes_tree: CSGOperation) -> None:
        inner = box_with_holes_tree.left
        assert isinstance(inner, CSGOperation)
        stock = inner.left
        assert isinstance(stock, CSGLeaf)
        assert len(stock.face_ids) == 6


class TestPlainBox:
    """A plain box (no features) should return a CSGLeaf."""

    def test_plain_box_returns_leaf(self) -> None:
        shape = read_step(FIXTURES / "sample_box.step")
        tree = reconstruct_stock_minus_features(shape)
        assert isinstance(tree, CSGLeaf)
        assert isinstance(tree.primitive, Box)

    def test_plain_box_dimensions(self) -> None:
        """sample_box.step is 10x20x30 at corner origin."""
        shape = read_step(FIXTURES / "sample_box.step")
        tree = reconstruct_stock_minus_features(shape)
        assert isinstance(tree, CSGLeaf)
        box = tree.primitive
        assert isinstance(box, Box)
        assert jnp.allclose(box.center, jnp.array([5.0, 10.0, 15.0]), atol=0.1)
        assert jnp.allclose(box.half_extents, jnp.array([5.0, 10.0, 15.0]), atol=0.1)


class TestUnsupportedShapes:
    """Shapes that don't match the stock-minus-features pattern."""

    def test_sphere_returns_none(self) -> None:
        shape = read_step(FIXTURES / "sample_sphere.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is None

    def test_cylinder_returns_none(self) -> None:
        shape = read_step(FIXTURES / "sample_cylinder.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is None


def _collect_features(tree: CSGNode) -> list[CSGLeaf]:
    """Walk down the left-leaning subtract tree collecting right-hand features."""
    features: list[CSGLeaf] = []
    node: CSGNode = tree
    while isinstance(node, CSGOperation) and node.op == "subtract":
        assert isinstance(node.right, CSGLeaf)
        features.append(node.right)
        node = node.left
    return features


def _collect_all_face_ids(node: CSGNode) -> list[int]:
    """Recursively collect all face_ids from a CSG tree."""
    if isinstance(node, CSGLeaf):
        return list(node.face_ids)
    assert isinstance(node, CSGOperation)
    return _collect_all_face_ids(node.left) + _collect_all_face_ids(node.right)


# Re-export CSGNode for the type alias used in helpers
CSGNode = CSGLeaf | CSGOperation
