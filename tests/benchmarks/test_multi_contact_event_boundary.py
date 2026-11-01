"""Event-crossing edits retain exact topology and measured CAD accuracy separately."""

from fractions import Fraction
from time import perf_counter

import numpy as np
import pytest

from benchmarks.face_contact.cad import boundary_tolerance, cad_volume, transform_shape
from benchmarks.multi_contact.event_boundary_cad import (
    CadCallCounts,
    observe_scene,
    prepare_scene,
)
from benchmarks.multi_contact.event_boundary_evaluation import reference_audit, run
from benchmarks.multi_contact.event_boundary_fixtures import FAMILIES
from benchmarks.multi_contact.event_boundary_selection import BoundaryCache, evaluate
from brepax._occt.backend import BRepAlgoAPI_Common, BRepCheck_Analyzer


@pytest.mark.parametrize("name", ["ramp_full", "tetra_full", "independent"])
def test_boundary_specialization_matches_independent_sections_and_rebuilt_cad(name):
    family = next(item for item in FAMILIES if item.name == name)
    for theta in (Fraction(-1, 32), Fraction(0), Fraction(1, 32)):
        result = reference_audit(family, theta)
        assert result["passed"], result
        assert result["boundary_inventory_matches"]
        assert result["max_boundary_profile_error"] == 0


def test_exact_unreachable_target_still_allows_the_same_tolerance_qualified_edit():
    family = next(item for item in FAMILIES if item.name == "tetra_full")
    for method in ("rebuild_profile", "cad_search", "reusable_boundary"):
        attempt = evaluate(family, Fraction(-1, 32), method)
        assert attempt.outcome is not None and attempt.outcome.operational_success
        assert attempt.outcome.relative_removed_error <= 0.001
        if attempt.selection.exact_target_query is not None:
            assert not attempt.selection.exact_target_query.exact_reachable
            assert attempt.selection.decision == "profile_extremum_band_hit"


def test_sequential_boundary_reuse_preserves_target_after_seed_misses():
    family = next(item for item in FAMILIES if item.name == "independent")
    cache = BoundaryCache()
    seed = None
    attempts = []
    for theta in (Fraction(-1, 16), Fraction(0), Fraction(1, 16)):
        attempt = evaluate(
            family, theta, "reusable_boundary", seed=seed, boundary_cache=cache
        )
        assert attempt.outcome is not None and attempt.outcome.operational_success
        assert not attempt.selection.seed_hit
        assert attempt.selection.model.boundary_specialized
        assert not attempt.selection.model.boundary_fallback
        seed = attempt.selection.selected_step
        attempts.append(attempt)
    assert sum(a.selection.model.boundary_preparation_attempted for a in attempts) == 1
    assert all(a.calls.cut == 1 for a in attempts)


def test_root_birth_control_retains_explicit_boundary_fallback():
    family = next(item for item in FAMILIES if item.name == "ramp_birth")
    attempt = evaluate(family, Fraction(1, 32), "reusable_boundary")
    assert attempt.outcome is not None and attempt.outcome.operational_success
    assert attempt.selection.model.boundary_fallback
    assert attempt.selection.model.boundary_fallback_reason is not None
    assert attempt.selection.model.profile_rebuilt


def test_invalid_subcohorts_cannot_be_reported_as_success(tmp_path):
    output = tmp_path / "unwritten.json"
    with pytest.raises(ValueError, match="Unknown family"):
        run(output, names=("missing",), audits=False)
    with pytest.raises(ValueError, match="outside"):
        run(output, modes=("cold",), theta_index=100, audits=False)
    with pytest.raises(ValueError, match="cannot omit"):
        run(output, modes=("sequential",), theta_index=1, audits=False)
    assert not output.exists()


def test_repeated_near_contact_queries_preserve_source_and_match_fresh_cad():
    family = next(item for item in FAMILIES if item.name == "ramp_full")
    cases = (
        # These consecutive probes used to increase the reusable source tolerance.
        (Fraction(-1, 4), (1.2499996985976352, 1.25000038935039, 1.249999271688955)),
        # Non-destructive Boolean mode instead returned an incorrect empty Common.
        (Fraction(-1, 8), (1.2499994970386978,)),
    )
    for theta, steps in cases:
        scene, _ = prepare_scene(family, theta, CadCallCounts())
        before = tuple(
            (boundary_tolerance(shape), cad_volume(shape))
            for shape in (scene.fixed, scene.moving)
        )
        for step in steps:
            observed = observe_scene(scene, step, deadline_started=perf_counter())
            fresh, _ = prepare_scene(family, theta, CadCallCounts())
            moving = transform_shape(fresh.moving, np.eye(3), step * fresh.velocity)
            independent = BRepAlgoAPI_Common(fresh.fixed, moving)
            assert (
                independent.IsDone()
                and BRepCheck_Analyzer(independent.Shape()).IsValid()
            )
            assert observed.error is None and observed.volume is not None
            assert 0.46 < observed.volume < 0.5
            assert observed.volume == pytest.approx(
                cad_volume(independent.Shape()), abs=1e-14
            )
            after = tuple(
                (boundary_tolerance(shape), cad_volume(shape))
                for shape in (scene.fixed, scene.moving)
            )
            assert after == before
