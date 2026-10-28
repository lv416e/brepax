"""Heldout isolation, physical tie policy, and failed-evidence rejection."""

from copy import deepcopy

import numpy as np
import pytest

from benchmarks.multi_contact.evaluation import evaluate, select_action
from benchmarks.multi_contact.fixtures import ACTION_STEPS


def _observations():
    geometry = {"passed": True, "rows": []}
    sampled = {"passed": True, "rows": []}
    for index in range(5):
        group = f"family_{index}"
        rates = [2.3, 0.7]
        geometry["rows"].append(
            {
                "group_id": group,
                "placement_name": "identity",
                "fixture": {"family": "family"},
                "target_volume": 0.021,
                "volume_tie_tolerance": 1e-12,
                "motions": [{"name": "first"}, {"name": "second"}],
                "rates": dict.fromkeys(
                    (
                        "full_contact",
                        "first_patch",
                        "largest_patch",
                        "signed_linear_flux",
                        "central_coordinate_fd",
                        "cad_directional_fd",
                    ),
                    rates,
                ),
                "endpoints": [
                    {
                        "direction": direction,
                        "step": step,
                        "measurement": {
                            "error": None,
                            "source_valid": True,
                            "valid": True,
                            "working_volume": rate * step,
                        },
                    }
                    for direction, rate in enumerate(rates)
                    for step in ACTION_STEPS
                ],
            }
        )
        for representation in ("exterior", "cell_min"):
            sampled["rows"].append(
                {
                    "group_id": group,
                    "placement_name": "identity",
                    "representation": representation,
                    "candidate_names": ["first", "second"],
                    "records": [
                        {
                            "method": method,
                            "k_over_h": float(config),
                            "beta_over_h": 1.0,
                            "error": None,
                            "linear_rates": rates if config == 1 else rates[::-1],
                            "directional_fd_rates": rates
                            if config == 1
                            else rates[::-1],
                        }
                        for method in ("stratum", "sigmoid", "smoothing")
                        for config in (1, 2)
                    ],
                }
            )
    return geometry, sampled


def test_physical_volume_ties_are_stable_under_unit_change():
    values = np.array([[0.005, 0.01, 0.02, 0.04], [0.006, 0.011, 0.021, 0.041]])
    for factor in (1e-18, 1, 1e18):
        assert select_action(
            values * factor, 0.0105 * factor, 1e-12 * factor, "target"
        ) == (
            0,
            1,
        )
        assert select_action(
            values * factor, 0.0105 * factor, 1e-12 * factor, "fixed_step"
        ) == (1, 2)


def test_changed_heldout_truth_cannot_change_its_calibration():
    geometry, sampled = _observations()
    before = evaluate(geometry, sampled)
    seed = before["split_seeds"][0]
    original = [c for c in before["calibrations"] if c["split_seed"] == seed]
    held = original[0]["heldout_groups"]
    assert set(held).isdisjoint(original[0]["training_groups"])
    changed = deepcopy(geometry)
    for row in changed["rows"]:
        if row["group_id"] in held:
            for endpoint in row["endpoints"]:
                endpoint["measurement"]["working_volume"] = (
                    1000.0 if endpoint["direction"] == 0 else 0.021
                )
    after = evaluate(changed, sampled)
    assert [c for c in after["calibrations"] if c["split_seed"] == seed] == original
    assert (
        after["summary"]["target"]["full_contact"]["mean_relative_error"]
        > before["summary"]["target"]["full_contact"]["mean_relative_error"]
    )


@pytest.mark.parametrize("failure", ["invalid_source", "missing", "duplicate"])
def test_bad_endpoint_evidence_is_not_silently_removed(failure):
    geometry, sampled = _observations()
    endpoints = geometry["rows"][0]["endpoints"]
    if failure == "invalid_source":
        endpoints[0]["measurement"]["source_valid"] = False
    elif failure == "missing":
        endpoints.pop()
    else:
        endpoints.append(deepcopy(endpoints[0]))
    with pytest.raises(ValueError):
        evaluate(geometry, sampled)
