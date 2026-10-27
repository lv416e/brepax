"""CAD acceptance must not be inferred from a converged mesh surrogate."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.cad_editing.edit import run
from benchmarks.cad_editing.mesh import RoofMesh
from benchmarks.cad_editing.optimization import EditSolver


@pytest.mark.parametrize("deflection", [-1.0, float("nan")])
def test_invalid_setup_retains_serializable_failure(deflection, tmp_path):
    report = run(deflection, tmp_path)
    assert report["passed"] is False
    assert "setup_error" in report
    json.dumps(report, allow_nan=False)


def test_coarse_mesh_fit_is_rejected_by_independent_cad(tmp_path):
    """The solver can hit its target while the actual CAD misses its budget."""
    report = run(0.01, tmp_path)
    assert report["passed"] is False
    assert len(report["cases"]) == 3
    assert all(row["optimization"]["status"] == "converged" for row in report["cases"])
    assert all(not row["passed"] for row in report["cases"])
    for row in report["cases"]:
        steps = row["optimization"]["steps"]
        assert np.max(np.abs(steps[-1]["mesh_residual"])) <= 1e-10
        assert np.max(np.abs(steps[-1]["cad_residual"])) > 1e-4
        assert len(row["geometry"]) == 4
        assert all(check["error"] is None for check in row["geometry"])
        assert all(not check["passed"] for check in row["geometry"])
    assert {row["optimization"]["status"] for row in report["controls"]} == {
        "invalid_input",
        "no_descent",
        "iteration_limit",
    }
    assert all(row["passed"] for row in report["controls"])
    # The rejected zero-budget attempt still exposes its evaluated starting point.
    limited = next(row for row in report["controls"] if row["name"] == "zero_budget")
    assert len(limited["optimization"]["steps"]) == 1
    assert len(list(tmp_path.rglob("*.step"))) == 12
    json.dumps(report, allow_nan=False)


def test_failed_geometry_keeps_attempted_dimensions():
    """A failed edit must identify the design even before a valid step exists."""
    # An open triangle cannot serve as the closed CAD editing mesh.
    model = RoofMesh(
        jnp.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 20.0, 0.0]]),
        jnp.array([[0, 1, 2]]),
    )
    result = EditSolver(model).solve(np.array([5.0, 3.0]), np.array([10000.0, 3000.0]))
    assert result.status == "evaluation_failed"
    assert result.failed_dimensions == [5.0, 3.0]
    assert result.failed_mesh_residual is not None
    assert result.error is not None
