"""Integration tests for CSG tree differentiable evaluation.

Tests the end-to-end pipeline: STEP file -> CSG tree -> differentiable
volume -> gradient computation -> optimization convergence.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.csg import CSGLeaf, CSGOperation, reconstruct_stock_minus_features
from brepax.brep.csg_eval import (
    DifferentiableCSG,
    csg_to_differentiable,
    evaluate_csg_sdf,
    evaluate_csg_volume,
)
from brepax.io.step import read_step
from brepax.primitives import Box, FiniteCylinder

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# Analytical volume: 40*30*20 - pi*4^2*20 - pi*3^2*20
_BOX_VOL = 40.0 * 30.0 * 20.0
_HOLE1_VOL = jnp.pi * 16.0 * 20.0
_HOLE2_VOL = jnp.pi * 9.0 * 20.0
_ANALYTICAL_VOL = float(_BOX_VOL - _HOLE1_VOL - _HOLE2_VOL)


class TestEvaluateCsgSdf:
    """Composite SDF evaluation on the CSG tree."""

    @pytest.fixture()
    def tree(self) -> CSGOperation:
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert isinstance(tree, CSGOperation)
        return tree

    def test_inside_box_outside_holes(self, tree: CSGOperation) -> None:
        """Point inside stock and away from holes has negative SDF."""
        pt = jnp.array([5.0, 5.0, 10.0])
        assert float(evaluate_csg_sdf(tree, pt)) < 0

    def test_inside_hole_positive(self, tree: CSGOperation) -> None:
        """Point inside a through-hole has positive SDF."""
        pt = jnp.array([10.0, 15.0, 10.0])
        assert float(evaluate_csg_sdf(tree, pt)) > 0

    def test_outside_box_positive(self, tree: CSGOperation) -> None:
        """Point outside the bounding box has positive SDF."""
        pt = jnp.array([50.0, 50.0, 50.0])
        assert float(evaluate_csg_sdf(tree, pt)) > 0

    def test_batch_evaluation(self, tree: CSGOperation) -> None:
        """SDF evaluates correctly on a batch of points."""
        pts = jnp.array(
            [
                [5.0, 5.0, 10.0],
                [10.0, 15.0, 10.0],
                [50.0, 50.0, 50.0],
            ]
        )
        sdf = evaluate_csg_sdf(tree, pts)
        assert sdf.shape == (3,)
        assert float(sdf[0]) < 0
        assert float(sdf[1]) > 0
        assert float(sdf[2]) > 0


class TestEvaluateCsgVolume:
    """Grid-based volume integration of CSG trees."""

    def test_box_with_holes_volume(self) -> None:
        """Volume matches analytical result within grid tolerance."""
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        vol = evaluate_csg_volume(tree, resolution=64)
        assert float(vol) == pytest.approx(_ANALYTICAL_VOL, rel=0.05)

    def test_plain_box_volume(self) -> None:
        """Plain box volume matches analytical 10*20*30 = 6000."""
        shape = read_step(FIXTURES / "sample_box.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        vol = evaluate_csg_volume(tree, resolution=64)
        assert float(vol) == pytest.approx(6000.0, rel=0.05)

    def test_resolution_improves_accuracy(self) -> None:
        """Higher resolution yields more accurate volume."""
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        vol_32 = float(evaluate_csg_volume(tree, resolution=32))
        vol_64 = float(evaluate_csg_volume(tree, resolution=64))
        err_32 = abs(vol_32 - _ANALYTICAL_VOL) / _ANALYTICAL_VOL
        err_64 = abs(vol_64 - _ANALYTICAL_VOL) / _ANALYTICAL_VOL
        assert err_64 < err_32


class TestDifferentiableCSG:
    """DifferentiableCSG equinox Module: gradient computation and optimization."""

    @pytest.fixture()
    def box(self) -> Box:
        return Box(
            center=jnp.array([20.0, 15.0, 10.0]),
            half_extents=jnp.array([20.0, 15.0, 10.0]),
        )

    def test_csg_to_differentiable(self) -> None:
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        dcsg = csg_to_differentiable(tree)
        assert isinstance(dcsg.stock, Box)
        assert len(dcsg.features) == 2
        for feat in dcsg.features:
            assert isinstance(feat, FiniteCylinder)

    def test_dcsg_volume_matches_tree(self) -> None:
        """DifferentiableCSG.volume matches evaluate_csg_volume."""
        shape = read_step(FIXTURES / "box_with_holes.step")
        tree = reconstruct_stock_minus_features(shape)
        assert tree is not None
        dcsg = csg_to_differentiable(tree)
        vol_tree = float(evaluate_csg_volume(tree, resolution=32))
        vol_dcsg = float(dcsg.volume(resolution=32))
        assert vol_dcsg == pytest.approx(vol_tree, rel=0.01)

    def test_gradient_radius_negative(self, box: Box) -> None:
        """d(vol)/d(cylinder_radius) < 0: bigger hole means less volume."""
        center = jnp.array([20.0, 15.0, 10.0])
        axis = jnp.array([0.0, 0.0, 1.0])
        height = jnp.array(20.0)

        def volume_fn(radius):
            cyl = FiniteCylinder(center=center, axis=axis, radius=radius, height=height)
            dcsg = DifferentiableCSG(stock=box, features=(cyl,))
            return dcsg.volume(resolution=32)

        grad = jax.grad(volume_fn)(jnp.array(4.0))
        assert float(grad) < 0

    def test_gradient_half_extent_positive(self) -> None:
        """d(vol)/d(box_half_x) > 0: bigger box means more volume."""
        center = jnp.array([20.0, 15.0, 10.0])

        def volume_fn(half_x):
            b = Box(
                center=center,
                half_extents=jnp.array([half_x, 15.0, 10.0]),
            )
            dcsg = DifferentiableCSG(stock=b, features=())
            return dcsg.volume(resolution=32)

        grad = jax.grad(volume_fn)(jnp.array(20.0))
        assert float(grad) > 0

    def test_optimization_converges(self, box: Box) -> None:
        """Gradient descent on cylinder radius converges to target volume."""
        axis = jnp.array([0.0, 0.0, 1.0])
        height = jnp.array(20.0)
        cyl_center = jnp.array([20.0, 15.0, 10.0])
        target_vol = 23000.0

        def loss(radius):
            cyl = FiniteCylinder(
                center=cyl_center, axis=axis, radius=radius, height=height
            )
            dcsg = DifferentiableCSG(stock=box, features=(cyl,))
            vol = dcsg.volume(resolution=32)
            return (vol - target_vol) ** 2

        radius = jnp.array(2.0)
        lr = 0.001
        for _ in range(100):
            g = jax.grad(loss)(radius)
            radius = radius - lr * g

        final_loss = float(loss(radius))
        assert final_loss < 1e4


class TestCsgToLeaf:
    """Edge case: CSGLeaf (no features) converts correctly."""

    def test_leaf_to_differentiable(self) -> None:
        box = Box(
            center=jnp.array([5.0, 10.0, 15.0]),
            half_extents=jnp.array([5.0, 10.0, 15.0]),
        )
        leaf = CSGLeaf(primitive=box, face_ids=[0, 1, 2, 3, 4, 5])
        dcsg = csg_to_differentiable(leaf)
        assert isinstance(dcsg.stock, Box)
        assert len(dcsg.features) == 0
