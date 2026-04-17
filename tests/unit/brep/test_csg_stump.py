"""Tests for CSG-Stump representation and evaluation."""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.brep.csg import reconstruct_stock_minus_features
from brepax.brep.csg_eval import evaluate_csg_sdf, evaluate_csg_volume
from brepax.brep.csg_stump import (
    CSGStump,
    DifferentiableCSGStump,
    csg_tree_to_stump,
    evaluate_stump_sdf,
    evaluate_stump_volume,
    reconstruct_csg_stump,
    stump_to_differentiable,
)
from brepax.io.step import read_step
from brepax.primitives import Box, FiniteCylinder

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class TestCSGTreeToStump:
    """Conversion from CSG tree to CSG-Stump."""

    def test_box_with_holes_shape(self) -> None:
        """Intersection matrix has correct shape for 2-hole box."""
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        assert len(stump.primitives) == 3
        assert stump.intersection_matrix.shape == (1, 3)
        assert stump.union_mask.shape == (1,)

    def test_single_intersection_term(self) -> None:
        """Stock-minus-features produces exactly one intersection term."""
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        assert float(stump.union_mask[0]) == 1.0

    def test_matrix_signs(self) -> None:
        """Stock is +1 (inside), features are -1 (outside)."""
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        row = stump.intersection_matrix[0]
        signs = sorted(float(s) for s in row)
        assert signs == [-1.0, -1.0, 1.0]

    def test_plain_box(self) -> None:
        """Plain box (no features) has one primitive, T=[[+1]], U=[1]."""
        shape = read_step(FIXTURES / "sample_box.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        assert len(stump.primitives) == 1
        assert float(stump.intersection_matrix[0, 0]) == 1.0


class TestStumpSdfMatchesTree:
    """CSG-Stump SDF must match tree-based SDF at all points."""

    @pytest.fixture()
    def stump_and_tree(self):
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        return stump, tree

    def test_sdf_inside_box(self, stump_and_tree) -> None:
        stump, tree = stump_and_tree
        pt = jnp.array([5.0, 5.0, 10.0])
        assert float(evaluate_stump_sdf(stump, pt)) == pytest.approx(
            float(evaluate_csg_sdf(tree, pt)), abs=0.01
        )

    def test_sdf_inside_hole(self, stump_and_tree) -> None:
        stump, tree = stump_and_tree
        pt = jnp.array([10.0, 15.0, 10.0])
        assert float(evaluate_stump_sdf(stump, pt)) == pytest.approx(
            float(evaluate_csg_sdf(tree, pt)), abs=0.01
        )

    def test_sdf_outside(self, stump_and_tree) -> None:
        stump, tree = stump_and_tree
        pt = jnp.array([50.0, 50.0, 50.0])
        assert float(evaluate_stump_sdf(stump, pt)) == pytest.approx(
            float(evaluate_csg_sdf(tree, pt)), abs=0.01
        )

    def test_sdf_batch(self, stump_and_tree) -> None:
        stump, tree = stump_and_tree
        pts = jnp.array(
            [
                [5.0, 5.0, 10.0],
                [10.0, 15.0, 10.0],
                [50.0, 50.0, 50.0],
            ]
        )
        stump_sdf = evaluate_stump_sdf(stump, pts)
        tree_sdf = evaluate_csg_sdf(tree, pts)
        assert jnp.allclose(stump_sdf, tree_sdf, atol=0.01)


class TestStumpVolumeMatchesTree:
    """CSG-Stump volume must match tree-based volume."""

    def test_box_with_holes_volume(self) -> None:
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        vol_tree = float(evaluate_csg_volume(tree, resolution=32))
        vol_stump = float(evaluate_stump_volume(stump, resolution=32))
        assert vol_stump == pytest.approx(vol_tree, rel=0.01)

    def test_plain_box_volume(self) -> None:
        shape = read_step(FIXTURES / "sample_box.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        vol_tree = float(evaluate_csg_volume(tree, resolution=32))
        vol_stump = float(evaluate_stump_volume(stump, resolution=32))
        assert vol_stump == pytest.approx(vol_tree, rel=0.01)


class TestMultiTermStump:
    """CSG-Stump with multiple intersection terms (union of regions)."""

    def test_two_disjoint_boxes(self) -> None:
        """Union of two boxes via 2-term CSG-Stump."""
        box_a = Box(
            center=jnp.array([0.0, 0.0, 0.0]),
            half_extents=jnp.array([1.0, 1.0, 1.0]),
        )
        box_b = Box(
            center=jnp.array([5.0, 0.0, 0.0]),
            half_extents=jnp.array([1.0, 1.0, 1.0]),
        )
        stump = CSGStump(
            primitives=[box_a, box_b],
            intersection_matrix=jnp.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            union_mask=jnp.array([1.0, 1.0]),
        )
        vol = float(evaluate_stump_volume(stump, resolution=64))
        assert vol == pytest.approx(8.0 + 8.0, rel=0.10)

    def test_two_disjoint_boxes_sdf(self) -> None:
        """SDF at midpoint between two disjoint boxes is positive."""
        box_a = Box(
            center=jnp.array([0.0, 0.0, 0.0]),
            half_extents=jnp.array([1.0, 1.0, 1.0]),
        )
        box_b = Box(
            center=jnp.array([5.0, 0.0, 0.0]),
            half_extents=jnp.array([1.0, 1.0, 1.0]),
        )
        stump = CSGStump(
            primitives=[box_a, box_b],
            intersection_matrix=jnp.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ),
            union_mask=jnp.array([1.0, 1.0]),
        )
        mid = jnp.array([2.5, 0.0, 0.0])
        assert float(evaluate_stump_sdf(stump, mid)) > 0

        inside_a = jnp.array([0.0, 0.0, 0.0])
        assert float(evaluate_stump_sdf(stump, inside_a)) < 0

        inside_b = jnp.array([5.0, 0.0, 0.0])
        assert float(evaluate_stump_sdf(stump, inside_b)) < 0


class TestDifferentiableCSGStump:
    """Gradient computation through CSG-Stump."""

    def test_gradient_radius_negative(self) -> None:
        """d(vol)/d(cylinder_radius) < 0 through CSG-Stump."""
        box = Box(
            center=jnp.array([20.0, 15.0, 10.0]),
            half_extents=jnp.array([20.0, 15.0, 10.0]),
        )

        def volume_fn(radius):
            cyl = FiniteCylinder(
                center=jnp.array([20.0, 15.0, 10.0]),
                axis=jnp.array([0.0, 0.0, 1.0]),
                radius=radius,
                height=jnp.array(20.0),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="A JAX array is being set as static"
                )
                dstump = DifferentiableCSGStump(
                    primitives=(box, cyl),
                    intersection_matrix=np.array([[1.0, -1.0]]),
                    union_mask=np.array([1.0]),
                )
            return dstump.volume(resolution=32)

        grad = jax.grad(volume_fn)(jnp.array(4.0))
        assert float(grad) < 0

    def test_stump_to_differentiable(self) -> None:
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        stump = csg_tree_to_stump(tree)
        dstump = stump_to_differentiable(stump)
        assert len(dstump.primitives) == 3
        assert dstump.intersection_matrix.shape == (1, 3)


class TestReconstructCsgStump:
    """PMC-based CSG-Stump reconstruction from B-Rep shapes."""

    def test_box_with_holes_returns_stump(self) -> None:
        shape = read_step(FIXTURES / "box_with_holes.step")
        stump = reconstruct_csg_stump(shape)
        assert stump is not None
        assert len(stump.primitives) == 8

    def test_box_with_holes_single_inside_cell(self) -> None:
        """Only one cell is inside the solid."""
        shape = read_step(FIXTURES / "box_with_holes.step")
        stump = reconstruct_csg_stump(shape)
        assert stump is not None
        assert stump.intersection_matrix.shape[0] == 1

    def test_box_with_holes_sdf_correctness(self) -> None:
        shape = read_step(FIXTURES / "box_with_holes.step")
        stump = reconstruct_csg_stump(shape)
        assert stump is not None
        inside = jnp.array([5.0, 5.0, 10.0])
        hole = jnp.array([10.0, 15.0, 10.0])
        outside = jnp.array([50.0, 50.0, 50.0])
        assert float(evaluate_stump_sdf(stump, inside)) < 0
        assert float(evaluate_stump_sdf(stump, hole)) > 0
        assert float(evaluate_stump_sdf(stump, outside)) > 0

    def test_box_with_holes_volume(self) -> None:
        shape = read_step(FIXTURES / "box_with_holes.step")
        stump = reconstruct_csg_stump(shape)
        assert stump is not None
        vol = float(evaluate_stump_volume(stump, resolution=64))
        analytical = 40 * 30 * 20 - jnp.pi * 16 * 20 - jnp.pi * 9 * 20
        assert vol == pytest.approx(float(analytical), rel=0.05)

    def test_plain_box(self) -> None:
        shape = read_step(FIXTURES / "sample_box.step")
        stump = reconstruct_csg_stump(shape)
        assert stump is not None
        vol = float(evaluate_stump_volume(stump, resolution=64))
        assert vol == pytest.approx(6000.0, rel=0.05)

    def test_sphere_returns_stump(self) -> None:
        """Sphere has 1 face → 1 primitive → should reconstruct."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        stump = reconstruct_csg_stump(shape)
        assert stump is not None
        assert len(stump.primitives) == 1

    def test_bbox_is_set(self) -> None:
        shape = read_step(FIXTURES / "box_with_holes.step")
        stump = reconstruct_csg_stump(shape)
        assert stump is not None
        assert stump.bbox_lo is not None
        assert stump.bbox_hi is not None
