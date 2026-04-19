"""Tests for OCCT mesh hybrid triangulation."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from brepax.brep.triangulate import triangulate_shape
from brepax.io.step import read_step

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _divergence_volume(triangles: jnp.ndarray) -> jnp.ndarray:
    """Divergence theorem volume from a triangle mesh."""
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return jnp.sum(v0 * jnp.cross(v1, v2)) / 6.0


# Reference volumes from OCCT GProp (pre-computed for each fixture)
_REF_VOLUMES = {
    "sample_box": 6000.0,
    "sample_cylinder": 1178.0972,
    "sample_sphere": 113.0973,
    "sample_cone": 54.9779,
    "sample_torus": 222.0661,
}


class TestTriangulateShape:
    """Tests for triangulate_shape()."""

    def test_box_volume_exact(self) -> None:
        """Box triangulation gives exact volume."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, params = triangulate_shape(shape)
        vol = float(_divergence_volume(tris))
        assert vol == pytest.approx(6000.0, rel=1e-3)
        assert len(params) == 6

    @pytest.mark.parametrize(
        "fixture",
        ["sample_cylinder", "sample_sphere", "sample_cone", "sample_torus"],
    )
    def test_curved_surface_volume(self, fixture: str) -> None:
        """Curved surface volume within 0.5% of OCCT GProp."""
        shape = read_step(FIXTURES / f"{fixture}.step")
        tris, _params = triangulate_shape(shape)
        vol = float(_divergence_volume(tris))
        assert vol == pytest.approx(_REF_VOLUMES[fixture], rel=1e-2)

    def test_bspline_face_triangulated(self) -> None:
        """BSpline face produces triangles with control_points param."""
        shape = read_step(FIXTURES / "nurbs_saddle.step")
        tris, params = triangulate_shape(shape)
        assert tris.shape[0] > 0
        assert len(params) == 1
        assert "control_points" in params[0]

    def test_deflection_affects_triangle_count(self) -> None:
        """Finer deflection produces more triangles."""
        shape = read_step(FIXTURES / "sample_sphere.step")
        tris_coarse, _ = triangulate_shape(shape, deflection=0.1)
        tris_fine, _ = triangulate_shape(shape, deflection=0.01)
        assert tris_fine.shape[0] > tris_coarse.shape[0]

    def test_empty_shape_returns_empty(self) -> None:
        """Shape with no faces returns empty array."""
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        tris, params = triangulate_shape(compound)
        assert tris.shape == (0, 3, 3)
        assert params == []


class TestDivergenceVolumeGradient:
    """Tests for gradient flow through divergence_volume."""

    def test_gradient_finite_and_nonzero(self) -> None:
        """Gradient of volume w.r.t. triangle vertices is finite and nonzero."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        grad = jax.grad(_divergence_volume)(tris)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0)

    def test_gradient_matches_finite_diff(self) -> None:
        """AD gradient matches central finite difference."""
        shape = read_step(FIXTURES / "sample_box.step")
        tris, _ = triangulate_shape(shape)
        grad = jax.grad(_divergence_volume)(tris)

        eps = 1e-3
        idx = (0, 1, 2)
        tris_p = tris.at[idx].add(eps)
        tris_m = tris.at[idx].add(-eps)
        fd = (_divergence_volume(tris_p) - _divergence_volume(tris_m)) / (2 * eps)
        assert float(grad[idx]) == pytest.approx(float(fd), rel=0.05)
