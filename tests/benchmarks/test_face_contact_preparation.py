"""Preparation measurements keep failures without losing measured groups."""

import numpy as np

from benchmarks.face_contact import preparation
from benchmarks.face_contact.fixtures import audit_directions, selection_directions


def test_reference_failure_retains_all_directions_and_previous_group(monkeypatch):
    population = preparation.fixtures()[:2]
    identity = preparation.placements()[:1]
    original_reference = preparation.directional_reference
    failed_vertices = np.asarray(population[1].vertices)

    def reference(vertices, velocity, angular_velocity, pivot):
        if np.array_equal(vertices, failed_vertices):
            raise ArithmeticError("Independent reference unavailable")
        return original_reference(vertices, velocity, angular_velocity, pivot)

    monkeypatch.setattr(preparation, "fixtures", lambda: population)
    monkeypatch.setattr(preparation, "placements", lambda: identity)
    monkeypatch.setattr(preparation, "directional_reference", reference)
    measured = preparation.collect_prepared_contact()
    direction_count = len(audit_directions()) + len(selection_directions())
    assert len(measured["preparations"]) == 2
    assert len(measured["rows"]) == 2 * direction_count
    good = [r for r in measured["rows"] if r["group_id"] == population[0].group_id]
    failed = [r for r in measured["rows"] if r["group_id"] == population[1].group_id]
    assert len(good) == len(failed) == direction_count
    assert all(
        "error" not in row and row["normalized_error"] < 1e-12 for row in good
    ), good[0]
    assert all("Independent reference unavailable" in row["error"] for row in failed)
    assert all("working_value" not in row for row in failed)
    first = measured["preparations"][0]
    assert first["source_validity"] == [True, True]
    assert first["diagnostics"]["original_cad_contact_certified"] is False
    assert set(first["candidate_panels"]) == {
        "bounded_tilt_choice",
        "unconstrained_controls",
    }
