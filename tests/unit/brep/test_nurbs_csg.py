"""Tests for CSG reconstruction with NURBS faces."""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from brepax.brep.convert import faces_to_primitives
from brepax.brep.csg_stump import (
    evaluate_stump_volume,
    reconstruct_csg_stump,
    stump_to_differentiable,
)
from brepax.io.step import read_step
from brepax.primitives import BSplineSurface

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.mark.slow
class TestNurbsCsgReconstruction:
    """Tests for CSG-Stump reconstruction on NURBS shapes."""

    def test_nurbs_box_all_bspline(self) -> None:
        """NURBS box has all BSplineSurface faces."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        prims = faces_to_primitives(shape)
        assert all(isinstance(p, BSplineSurface) for p in prims)
        assert len(prims) == 6

    def test_reconstruct_stump(self) -> None:
        """PMC-based reconstruction produces a valid stump."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        stump = reconstruct_csg_stump(shape)
        assert len(stump.primitives) == 6
        assert stump.intersection_matrix.shape[1] == 6
        assert all(isinstance(p, BSplineSurface) for p in stump.primitives)

    def test_volume_accuracy(self) -> None:
        """Stump volume converges to analytical box volume."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        stump = reconstruct_csg_stump(shape)
        expected = 10.0 * 8.0 * 6.0
        vol = evaluate_stump_volume(stump, resolution=32)
        assert jnp.isclose(vol, expected, rtol=0.02), (
            f"vol={float(vol):.2f}, expected={expected}"
        )

    def test_differentiable_volume(self) -> None:
        """DifferentiableCSGStump volume is finite."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        stump = reconstruct_csg_stump(shape)
        diff = stump_to_differentiable(stump)
        vol = diff.volume(resolution=16)
        assert jnp.isfinite(vol)
        assert float(vol) > 0.0

    def test_gradient_finite(self) -> None:
        """Gradient of volume w.r.t. control points is finite."""
        shape = read_step(FIXTURES / "nurbs_box.step")
        stump = reconstruct_csg_stump(shape)
        diff = stump_to_differentiable(stump)

        def loss(ds: object) -> jnp.ndarray:
            return ds.volume(resolution=8)

        grad = eqx.filter_grad(loss)(diff)
        for p in grad.primitives:
            if hasattr(p, "control_points"):
                assert jnp.all(jnp.isfinite(p.control_points)), (
                    "Non-finite gradient in control points"
                )
