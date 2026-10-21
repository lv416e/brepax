"""Regression checks for evaluation failures and comparable path timing."""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from benchmarks.abc_evaluation import evaluate_step as evaluation


def test_empty_exception_is_recorded() -> None:
    def fail():
        raise RuntimeError

    value, elapsed, note = evaluation._safe(fail)
    assert value is None
    assert elapsed >= 0
    assert note.startswith("RuntimeError")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_volume_is_recorded_as_failure(value) -> None:
    result, _, note = evaluation._safe(lambda: value)
    assert result is None
    assert "non-finite" in note


def test_invalid_resolution_is_rejected_before_reading() -> None:
    with pytest.raises(ValueError, match="resolution"):
        evaluation.evaluate_step_file(Path("missing.step"), resolution=0)


@pytest.mark.parametrize("grid_fails", [False, True])
def test_elapsed_times_include_required_preparation(monkeypatch, grid_fails) -> None:
    clock = [0.0]

    def work(value, seconds):
        def run(*args, **kwargs):
            clock[0] += seconds
            return value

        return run

    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        evaluation, "triangulate_shape", work((jnp.zeros((1, 3, 3)), []), 7)
    )
    monkeypatch.setattr(
        evaluation, "_shape_grid_bounds", work((jnp.zeros(3), jnp.ones(3)), 2)
    )
    monkeypatch.setattr(
        evaluation,
        "make_grid_3d",
        work((jnp.zeros((2, 2, 2, 3)), jnp.array(0.125)), 3),
    )
    if grid_fails:

        def fail_grid(*args):
            clock[0] += 3
            raise RuntimeError

        monkeypatch.setattr(evaluation, "make_grid_3d", fail_grid)
    monkeypatch.setattr(evaluation, "divergence_volume", work(jnp.array(1.0), 11))
    monkeypatch.setattr(evaluation, "mesh_sdf", work(jnp.zeros(8), 13))
    monkeypatch.setattr(evaluation, "integrate_sdf_volume", work(jnp.array(1.0), 17))
    monkeypatch.setattr(evaluation, "winding_number", lambda p, triangles: 1.0)
    monkeypatch.setattr(evaluation, "reconstruct_csg_stump", work(object(), 5))
    diff = SimpleNamespace(volume=work(jnp.array(1.0), 29))
    trimmed = SimpleNamespace(volume=work(jnp.array(1.0), 37))
    monkeypatch.setattr(evaluation, "stump_to_differentiable", work(diff, 23))
    monkeypatch.setattr(evaluation, "enrich_with_trim_frames", work(trimmed, 31))

    results = evaluation._evaluate_volume_paths(None, resolution=2)
    elapsed = {result.path: result.elapsed_s for result in results}
    assert elapsed == {
        "divergence": 7 + 11,
        "mesh_sdf": 7 + 2 + 3 + (0 if grid_fails else 13 + 17),
        "gwn": 7 + 2 + 3,
        "csg_stump": 2 + 5 + 23 + 29,
        "trimmed_csg": 2 + 5 + 31 + 37,
    }
    if grid_fails:
        assert [result.path for result in results if result.value is None] == [
            "mesh_sdf",
            "gwn",
        ]
        assert all("grid: RuntimeError" in result.note for result in results[1:3])
