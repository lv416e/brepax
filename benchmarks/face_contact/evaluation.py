"""Grouped calibration and true-CAD finite action evaluation.

A gradient dotted with a direction is labeled a linear proxy. Its mismatch
with a nonlinear exact directional response is not an AD implementation defect.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .fixtures import (
    audit_directions,
    fixtures,
    grouped_splits,
    placements,
    selection_directions,
)


def select_direction(scores: list[float]) -> int:
    """First maximum within 64 roundoff units; do not rank numerical noise."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite candidate scores")
    tolerance = 64 * np.finfo(float).eps * float(np.max(np.abs(values)))
    return int(np.flatnonzero(values >= np.max(values) - tolerance)[0])


def mean_group(values: dict[str, list[float]], groups: tuple[str, ...]) -> float:
    """Equal polygon-group weight, independent of placement/direction counts."""
    return float(np.mean([np.mean(values[group]) for group in groups]))


def evaluate(
    sampled: dict[str, Any],
    analytic: dict[str, Any],
    cad: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    """Freeze temperatures using calibration only, then report all five splits."""
    failures = [row for row in sampled["records"] if row.get("error")]
    failures += [
        row
        for row in [*cad["baselines"], *cad["rows"]]
        if row.get("error")
        or not all(row[key] for key in ("valid", "fixed_valid", "moving_valid"))
        or row.get("baseline_error")
    ]
    failures += [row for row in analytic["rows"] if row.get("error")]
    failures += [row for row in prepared["rows"] if row.get("error")]
    failures += [
        row
        for row in analytic["rows"]
        if not row.get("error")
        and not np.isfinite(
            [
                row[key]
                for key in ("value", "reference", "absolute_error", "normalization")
            ]
        ).all()
    ]
    failures += [
        row
        for row in prepared["rows"]
        if not row.get("error") and not np.isfinite(row["working_value"])
    ]
    failures += [
        row for row in sampled["own_forward_generic_pose_checks"] if row.get("error")
    ]
    if failures:
        return {
            "assessed": False,
            "passed": False,
            "failed_cases": len(failures),
            "reason": "See retained raw measurement failures",
        }
    reference = {(row["group_id"], row["direction"]): row for row in analytic["rows"]}
    cad_rows = {
        (row["group_id"], row["placement"], row["direction"], row["step"]): row
        for row in cad["rows"]
    }
    directions = (*audit_directions(), *selection_directions())
    choices = selection_directions()
    vectors = {d.name: np.asarray((*d.translation, *d.angular)) for d in directions}
    predictions: dict[
        tuple[str, tuple[float, ...] | None], dict[tuple[str, str, str], float]
    ] = defaultdict(dict)
    for fixture in fixtures():
        for direction in directions:
            ref = reference[fixture.group_id, direction.name]
            for placement in placements():
                key = (fixture.group_id, placement.name, direction.name)
                predictions["known_patch_flux", None][key] = ref["value"]
                for name, ablation in ref["ablations"].items():
                    predictions[name, None][key] = ablation
    for row in prepared["rows"]:
        key = (row["group_id"], row["placement"], row["direction"])
        predictions["positive_spatial_flux", None][key] = row["working_value"]
    for row in sampled["records"]:
        if "error" in row:
            raise RuntimeError(f"Failed sampled case retained in source: {row}")
        method = row["method"]
        temperature = tuple(row["temperature"]) if method != "stratum" else None
        gradient = np.asarray(row["gradient"])
        for direction in directions:
            key = (row["group_id"], row["placement"], direction.name)
            predictions[method + "_gradient_dot", temperature][key] = float(
                gradient @ vectors[direction.name]
            )
        for direction in choices:
            key = (row["group_id"], row["placement"], direction.name)
            for step in (0.02, 0.005):
                predictions[f"{method}_same_forward_fd_{step}", temperature][key] = row[
                    "selection_forward"
                ][direction.name][str(step)]["positive_slope"]
    for row in cad["rows"]:
        if row["error"] or not row["valid"] or row["baseline_error"]:
            raise RuntimeError(f"Failed CAD case retained in source: {row}")
        name = (
            "cad_action_scan"
            if row["step"] == 0.02
            else f"cad_positive_fd_{row['step']}"
        )
        predictions[name, None][row["group_id"], row["placement"], row["direction"]] = (
            row["working_difference"]
        )
    errors = {}
    selections = {}
    for variant, values in predictions.items():
        by_group: dict[str, list[float]] = defaultdict(list)
        panels: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        selection_rows = []
        for fixture in fixtures():
            for placement in placements():
                for direction in choices:
                    key = (fixture.group_id, placement.name, direction.name)
                    ref = reference[fixture.group_id, direction.name]
                    by_group[fixture.group_id].append(
                        abs(values[key] - ref["reference"]) / ref["normalization"]
                    )
                for panel, subset in (
                    ("bounded_tilt_choice", choices[:8]),
                    ("unconstrained_controls", choices),
                ):
                    predicted = [
                        values[fixture.group_id, placement.name, d.name] for d in subset
                    ]
                    true = [
                        cad_rows[fixture.group_id, placement.name, d.name, 0.02][
                            "working_volume"
                        ]
                        for d in subset
                    ]
                    chosen = select_direction(predicted)
                    best = float(max(true))
                    regret = best - true[chosen]
                    relative = regret / best if best else 0.0
                    panels[panel][fixture.group_id].append(relative)
                    selection_rows.append(
                        {
                            "group_id": fixture.group_id,
                            "placement": placement.name,
                            "panel": panel,
                            "selected": subset[chosen].name,
                            "selected_true_volume": true[chosen],
                            "best_true_volume": best,
                            "working_regret": regret,
                            "relative_regret": relative,
                        }
                    )
        errors[variant] = by_group
        selections[variant] = (panels, selection_rows)
    methods = list(dict.fromkeys(method for method, temperature in predictions))
    split_rows = []
    for split in grouped_splits(fixtures()):
        for method in methods:
            variants = [key for key in predictions if key[0] == method]
            # Selection uses only calibration response MAE over the same twelve
            # finite candidate directions. Heldout outcomes never tune a method.
            selected = min(
                variants,
                key=lambda key: mean_group(errors[key], split.calibration_groups),
            )
            panels, rows = selections[selected]
            split_rows.append(
                {
                    "seed": split.seed,
                    "method": method,
                    "selected_temperature": selected[1],
                    "calibration_normalized_directional_mae": mean_group(
                        errors[selected], split.calibration_groups
                    ),
                    "heldout_normalized_directional_mae": mean_group(
                        errors[selected], split.heldout_groups
                    ),
                    "heldout_mean_relative_regret": {
                        panel: mean_group(values, split.heldout_groups)
                        for panel, values in panels.items()
                    },
                    "heldout_groups": list(split.heldout_groups),
                }
            )
    summary = []
    for method in methods:
        rows = [row for row in split_rows if row["method"] == method]
        mae = [row["heldout_normalized_directional_mae"] for row in rows]
        summary.append(
            {
                "method": method,
                "directional_mae_mean": float(np.mean(mae)),
                "directional_mae_range": [min(mae), max(mae)],
                "relative_regret": {
                    panel: {
                        "mean": float(
                            np.mean(
                                [
                                    row["heldout_mean_relative_regret"][panel]
                                    for row in rows
                                ]
                            )
                        ),
                        "range": [
                            min(
                                row["heldout_mean_relative_regret"][panel]
                                for row in rows
                            ),
                            max(
                                row["heldout_mean_relative_regret"][panel]
                                for row in rows
                            ),
                        ],
                    }
                    for panel in ("bounded_tilt_choice", "unconstrained_controls")
                },
            }
        )
    candidate_error = max(row["absolute_error"] for row in analytic["rows"])
    prepared_error = max(
        abs(
            row["working_value"]
            - reference[row["group_id"], row["direction"]]["reference"]
        )
        / reference[row["group_id"], row["direction"]]["normalization"]
        for row in prepared["rows"]
    )
    ordinary_ad_errors = [
        row["central_differences"][-1]["max_abs_difference_to_ad"]
        for row in sampled["own_forward_generic_pose_checks"]
        if row["method"] != "stratum"
    ]
    c_sigmoid_error = sampled["intersecting_C_same_sigmoid_max_abs_difference"]
    intersecting_control = all(
        row["label"] == 1.0 for row in sampled["own_forward_generic_pose_checks"]
    )
    derivative_evidence_finite = bool(np.isfinite(ordinary_ad_errors).all())
    if intersecting_control:
        derivative_evidence_finite &= bool(np.isfinite(c_sigmoid_error))
    return {
        "assessed": True,
        "passed": candidate_error < 1e-10
        and prepared_error < 1e-10
        and derivative_evidence_finite
        and max(ordinary_ad_errors) < 1e-5
        and (not intersecting_control or c_sigmoid_error < 1e-10),
        "candidate_max_absolute_error": candidate_error,
        "prepared_max_normalized_error": prepared_error,
        "same_forward_ordinary_ad_max_error": max(ordinary_ad_errors),
        "C_same_sigmoid_control_assessed": intersecting_control,
        "scope": (
            "Five repeated grouped robustness evaluations on fixed "
            "synthetic convex prisms; no trained analytic model or "
            "unseen-family generalization claim"
        ),
        "temperature_selection": (
            "Calibration group-macro directional MAE over the twelve "
            "finite candidate directions; deterministic temperature "
            "order breaks exact ties"
        ),
        "tie_policy": (
            "First predicted maximum within 64 * machine epsilon * max absolute score"
        ),
        "summary": summary,
        "splits": split_rows,
        "selections_all_variants": [
            {"method": key[0], "temperature": key[1], "rows": rows}
            for key, (_, rows) in selections.items()
        ],
        "cost_contract": {
            "cad_action_scan": (
                "K+1 Common calls, zero regret by definition relative "
                "to these observations"
            ),
            "cad_smaller_positive_fd": (
                "K+1 Common calls; one selected-action endpoint "
                "validation additionally if not already evaluated"
            ),
            "gradient_proxy": (
                "One six-coordinate parameter gradient and K dot "
                "products; one selected-action CAD validation"
            ),
            "positive_spatial_flux": (
                "K directional evaluations or one K batch; "
                "known-contact preparation timed separately; one "
                "selected-action CAD validation"
            ),
            "same_forward_positive_fd": (
                "K+1 scalar forward calls and one selected-action CAD validation"
            ),
            "reuse": (
                "Measure each physical case once; reuse immutable "
                "outcomes across panels and five splits"
            ),
        },
    }
