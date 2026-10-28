"""Group-separated calibration and true finite-volume selection metrics.

The shapes are developmental examples, not an untouched generalization set.
Every placement/direction of a shape stays in its group. Only training groups
choose smoothing temperatures and scalar gains; all methods use the same
candidate grid, physical-volume tie allowance, and actual CAD endpoint values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from benchmarks.face_contact.fixtures import SPLIT_SEEDS

from .fixtures import ACTION_STEPS

Objective = Literal["target", "fixed_step"]
_OBJECTIVES: tuple[Objective, ...] = ("target", "fixed_step")
_FIXED_STEP = 0.02
_CORE_METHODS = (
    "full_contact",
    "first_patch",
    "largest_patch",
    "signed_linear_flux",
    "central_coordinate_fd",
    "cad_directional_fd",
    "endpoint_scan",
)


@dataclass(frozen=True)
class SampledMethod:
    """One comparator's fixed representation and training-only gain search."""

    name: str
    method: str
    representation: str = "exterior"
    gains: tuple[float, ...] = (1.0,)
    rate_field: str = "linear_rates"


_SAMPLED_METHODS = (
    SampledMethod("method_c", "stratum"),
    SampledMethod("sigmoid_raw", "sigmoid"),
    SampledMethod("smoothing_raw", "smoothing"),
    SampledMethod("sigmoid_calibrated", "sigmoid", gains=(0.5, 1.0, 2.0, 4.0)),
    SampledMethod("smoothing_calibrated", "smoothing", gains=(0.5, 1.0, 2.0, 4.0)),
    SampledMethod(
        "sigmoid_directional_fd",
        "sigmoid",
        gains=(0.5, 1.0, 2.0, 4.0),
        rate_field="directional_fd_rates",
    ),
    SampledMethod(
        "smoothing_directional_fd",
        "smoothing",
        gains=(0.5, 1.0, 2.0, 4.0),
        rate_field="directional_fd_rates",
    ),
    SampledMethod("cell_min_smoothing", "smoothing", "cell_min", (0.5, 1.0, 2.0, 4.0)),
)


def select_action(
    predictions: NDArray[np.float64],
    target: float,
    tolerance: float,
    objective: Objective,
) -> tuple[int, int]:
    """Use the first index within a shared dimensional score allowance.

    The allowance is a reproducible numerical tie policy, not a certified CAD
    error bound. Predictions must be finite and have the full candidate grid.
    All removal proxies are clipped below at zero before this operation.
    """
    if (
        predictions.ndim != 2
        or predictions.shape[1] != len(ACTION_STEPS)
        or predictions.shape[0] == 0
        or not np.isfinite(predictions).all()
        or not np.isfinite([target, tolerance]).all()
        or target <= 0
        or tolerance < 0
    ):
        raise ValueError("Expected finite candidate volumes and positive target")
    if objective == "fixed_step":
        step_index = ACTION_STEPS.index(_FIXED_STEP)
        values = predictions[:, step_index]
        candidates = np.flatnonzero(values >= np.max(values) - tolerance)
        return int(candidates[0]), step_index
    if objective != "target":
        raise ValueError("Unknown selection objective")
    losses = np.abs(predictions - target)
    grid_candidates = np.argwhere(losses <= np.min(losses) + tolerance)
    return int(grid_candidates[0, 0]), int(grid_candidates[0, 1])


def _true_volumes(row: dict[str, Any]) -> NDArray[np.float64]:
    values = np.full((len(row["motions"]), len(ACTION_STEPS)), np.nan)
    seen = set()
    for endpoint in row["endpoints"]:
        key = (int(endpoint["direction"]), ACTION_STEPS.index(endpoint["step"]))
        if not 0 <= key[0] < len(row["motions"]):
            raise ValueError("Invalid finite CAD candidate index")
        if key in seen:
            raise ValueError("Duplicate finite CAD endpoint")
        seen.add(key)
        measurement = endpoint["measurement"]
        if (
            measurement["error"]
            or not measurement["valid"]
            or not measurement["source_valid"]
        ):
            raise ValueError("Failed or invalid finite CAD endpoint")
        values[key] = measurement["working_volume"]
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite finite CAD endpoint")
    if np.min(values) < -row["volume_tie_tolerance"]:
        raise ValueError("Negative finite CAD volume beyond the tie allowance")
    return values


def _metrics(
    row: dict[str, Any],
    actual: NDArray[np.float64],
    predictions: NDArray[np.float64],
    objective: Objective,
) -> dict[str, Any]:
    target, tolerance = row["target_volume"], row["volume_tie_tolerance"]
    if predictions.shape != actual.shape:
        raise ValueError("Predicted and actual finite candidate grids disagree")
    direction, step_index = select_action(predictions, target, tolerance, objective)
    value = float(actual[direction, step_index])
    record: dict[str, Any] = {
        "group_id": row["group_id"],
        "placement_name": row["placement_name"],
        "direction": direction,
        "step": ACTION_STEPS[step_index],
        "predicted_volume": float(predictions[direction, step_index]),
        "actual_volume": value,
        "target_volume": target,
        "objective": objective,
    }
    if objective == "target":
        loss = abs(value - target)
        best = float(np.min(np.abs(actual - target)))
        record.update(
            relative_error=loss / target,
            excess_relative_error=max(0.0, (loss - best) / target),
            best_candidate_relative_error=best / target,
            finite_ranking_miss=bool(loss > best + tolerance),
        )
    else:
        best = float(np.max(actual[:, ACTION_STEPS.index(_FIXED_STEP)]))
        loss = max(0.0, best - value)
        record.update(
            relative_error=0.0 if best <= tolerance else loss / best,
            best_candidate_volume=best,
        )
    return record


def _prediction(rates: Any) -> NDArray[np.float64]:
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite directional rates")
    # The true initial removal response is nonnegative. Give every proxy this
    # same known constraint; retain its unclipped values in the measurements.
    return np.asarray(
        np.maximum(values, 0)[:, None] * np.asarray(ACTION_STEPS)[None, :],
        dtype=np.float64,
    )


def _sampled_record(
    row: dict[str, Any], method: str, k: float, beta: float
) -> dict[str, Any]:
    records: list[dict[str, Any]] = row["records"]
    matches = [
        record
        for record in records
        if (record["method"], record["k_over_h"], record["beta_over_h"])
        == (method, k, beta)
    ]
    if len(matches) != 1 or matches[0]["error"]:
        raise ValueError("Missing, duplicate or failed sampled configuration")
    return matches[0]


def _split(rows: list[dict[str, Any]], seed: int) -> tuple[list[str], list[str]]:
    families: dict[str, set[str]] = {}
    for row in rows:
        families.setdefault(row["fixture"]["family"], set()).add(row["group_id"])
    random = np.random.default_rng(seed)
    training, held = [], []
    for family in sorted(families):
        groups = sorted(families[family])
        if len(groups) < 2:
            raise ValueError("Grouped evaluation requires two shapes per family")
        shuffled = random.permutation(groups).tolist()
        count = min(len(groups) - 1, max(1, int(0.6 * len(groups))))
        training.extend(shuffled[:count])
        held.extend(shuffled[count:])
    return sorted(training), sorted(held)


@dataclass(frozen=True)
class Calibration:
    """One configuration chosen without reading heldout endpoint outcomes."""

    training_loss: float
    k_over_h: float
    beta_over_h: float
    gain: float


def _calibrate(
    method: SampledMethod,
    train_keys: list[tuple[str, str]],
    rows: dict[tuple[str, str], dict[str, Any]],
    actuals: dict[tuple[str, str], NDArray[np.float64]],
    samples: dict[tuple[str, str, str], dict[str, Any]],
    objective: Objective,
) -> Calibration:
    sample = samples[(*train_keys[0], method.representation)]
    configurations = sorted(
        {
            (record["k_over_h"], record["beta_over_h"], gain)
            for record in sample["records"]
            if record["method"] == method.method
            for gain in method.gains
        }
    )
    scored = []
    for k, beta, gain in configurations:
        losses = []
        for key in train_keys:
            record = _sampled_record(
                samples[(*key, method.representation)], method.method, k, beta
            )
            prediction = _prediction(np.asarray(record[method.rate_field]) * gain)
            losses.append(
                _metrics(rows[key], actuals[key], prediction, objective)[
                    "relative_error"
                ]
            )
        scored.append((float(np.mean(losses)), k, beta, gain))
    if not scored:
        raise ValueError("No sampled calibration candidates")
    return Calibration(*min(scored))


def evaluate(geometry: dict[str, Any], sampled: dict[str, Any]) -> dict[str, Any]:
    """Score complete observations; failed measurements cannot become passes."""
    if not geometry["passed"] or not sampled["passed"]:
        return {
            "assessed": False,
            "passed": False,
            "error": "Geometry or sampled measurement stage failed",
        }
    rows = geometry["rows"]
    by_key = {(row["group_id"], row["placement_name"]): row for row in rows}
    if len(by_key) != len(rows) or not rows:
        raise ValueError("Expected unique, nonempty geometry rows")
    actuals = {key: _true_volumes(row) for key, row in by_key.items()}
    samples = {
        (row["group_id"], row["placement_name"], row["representation"]): row
        for row in sampled["rows"]
    }
    if len(samples) != len(sampled["rows"]):
        raise ValueError("Duplicate sampled geometry rows")
    for key, row in by_key.items():
        for representation in ("exterior", "cell_min"):
            candidate = samples[(*key, representation)]
            if candidate["candidate_names"] != [m["name"] for m in row["motions"]]:
                raise ValueError("Sampled and CAD candidate identities disagree")
    selections = []
    calibrations = []
    for seed in SPLIT_SEEDS:
        training, held = _split(rows, seed)
        train_keys = [key for key in by_key if key[0] in training]
        held_keys = [key for key in by_key if key[0] in held]
        for objective in _OBJECTIVES:
            for method in _CORE_METHODS:
                for key in held_keys:
                    prediction = (
                        actuals[key]
                        if method == "endpoint_scan"
                        else _prediction(by_key[key]["rates"][method])
                    )
                    selections.append(
                        {
                            **_metrics(
                                by_key[key], actuals[key], prediction, objective
                            ),
                            "method": method,
                            "split_seed": seed,
                        }
                    )
            for method_spec in _SAMPLED_METHODS:
                chosen = _calibrate(
                    method_spec, train_keys, by_key, actuals, samples, objective
                )
                k, beta, gain = chosen.k_over_h, chosen.beta_over_h, chosen.gain
                calibrations.append(
                    {
                        "split_seed": seed,
                        "objective": objective,
                        "method": method_spec.name,
                        "training_groups": training,
                        "heldout_groups": held,
                        "k_over_h": k,
                        "beta_over_h": beta,
                        "gain": gain,
                        "training_loss": chosen.training_loss,
                    }
                )
                for key in held_keys:
                    record = _sampled_record(
                        samples[(*key, method_spec.representation)],
                        method_spec.method,
                        k,
                        beta,
                    )
                    prediction = _prediction(
                        np.asarray(record[method_spec.rate_field]) * gain
                    )
                    selections.append(
                        {
                            **_metrics(
                                by_key[key], actuals[key], prediction, objective
                            ),
                            "method": method_spec.name,
                            "split_seed": seed,
                            "k_over_h": k,
                            "beta_over_h": beta,
                            "gain": gain,
                        }
                    )
    summary: dict[str, dict[str, Any]] = {}
    methods = (*_CORE_METHODS, *(item.name for item in _SAMPLED_METHODS))
    for objective in _OBJECTIVES:
        summary[objective] = {}
        for method in methods:
            values = [
                item
                for item in selections
                if item["objective"] == objective and item["method"] == method
            ]
            split_means = [
                float(
                    np.mean(
                        [
                            item["relative_error"]
                            for item in values
                            if item["split_seed"] == seed
                        ]
                    )
                )
                for seed in SPLIT_SEEDS
            ]
            result: dict[str, Any] = {
                "mean_relative_error": float(np.mean(split_means)),
                "split_min": min(split_means),
                "split_max": max(split_means),
                "selection_count": len(values),
            }
            if objective == "target":
                result["mean_excess_relative_error"] = float(
                    np.mean([item["excess_relative_error"] for item in values])
                )
                result["finite_ranking_miss_groups"] = sorted(
                    {item["group_id"] for item in values if item["finite_ranking_miss"]}
                )
            summary[objective][method] = result
    return {
        "passed": True,
        "assessed": True,
        "passed_scope": (
            "Complete finite comparisons; accuracy is measured, not guaranteed"
        ),
        "group_count": len({key[0] for key in by_key}),
        "split_seeds": SPLIT_SEEDS,
        "calibrations": calibrations,
        "selections": selections,
        "summary": summary,
        "interpretation": (
            "Target error includes finite-model misranking and the discrete candidate "
            "grid's nonzero best error. Fixed-step error is relative volume regret. "
            "These are developmental synthetic shape groups, "
            "not external generalization. "
            "All methods receive the same nonnegative-removal and tie policies."
        ),
    }
