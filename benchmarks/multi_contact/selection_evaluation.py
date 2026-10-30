"""Shape-grouped comparisons of observed finite edit choices.

The fresh cohort is synthetic and cross-validated, not an external test set.
All targets of one geometry remain together. Five overlapping partitions are
reported separately, then repeated observations are averaged within each shape.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from functools import partial
from typing import Any

import numpy as np

from .selection_experiment import SelectionCase, SelectionOracle, valid_measurement
from .selection_fixtures import FAMILIES, stock_volume_upper_bound
from .selection_policies import (
    BUDGETS,
    STEPS,
    TARGETS,
    action_order,
    best_observation,
    grid_policy,
    learned_order,
    refine_policy,
)


def partitions(cases: tuple[SelectionCase, ...]) -> list[dict[str, Any]]:
    """Five stratified partitions, with coverage of every input group.

    The default eight variants give five distinct train20/test12 partitions.
    Smaller smoke cohorts can repeat partitions and use
    at most n-1 held-out variants per family; they cannot establish the default
    cohort's performance. Geometry ordering is frozen by the fixture generator.
    """
    if len({case.fixture.group_id for case in cases}) != len(cases):
        raise ValueError("Each geometry must have one unique group identity")
    permuted = {}
    for index, family in enumerate(FAMILIES):
        ids = [case.fixture.group_id for case in cases if case.fixture.family == family]
        if len(ids) < 2:
            raise ValueError("At least two variants of every family are required")
        permutation = np.random.default_rng(82301 + index).permutation(len(ids))
        permuted[family] = [ids[int(j)] for j in permutation]
    result = []
    for split in range(5):
        test = [
            ids[((3 if len(ids) == 8 else 1) * split + j) % len(ids)]
            for ids in permuted.values()
            for j in range(min(3, len(ids) - 1))
        ]
        result.append(
            {
                "split": split,
                "test": test,
                "train": [
                    c.fixture.group_id for c in cases if c.fixture.group_id not in test
                ],
            }
        )
    return result


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(selections: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain invalid outcomes in success rates and permit empty conditional means."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[(row["method"], row["budget"])].append(row)
    result = {}
    for (method, budget), rows in grouped.items():
        by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_shape[row["group_id"]].append(row)
        means = [
            value
            for group in by_shape.values()
            if (
                value := _mean(
                    [
                        r["relative_error"]
                        for r in group
                        if r["relative_error"] is not None
                    ]
                )
            )
            is not None
        ]
        success = [
            float(np.mean([r["within_one_percent"] for r in group]))
            for group in by_shape.values()
        ]
        result[f"{method}_K{budget}"] = {
            "shape_count": len(by_shape),
            "valid_error_shape_count": len(means),
            "mean_relative_error": _mean(means),
            "unconditional_within_one_percent": _mean(success),
            "valid_edit_rows": sum(r["valid_edit"] for r in rows),
            "rows": len(rows),
            "five_split_mean_errors": [
                _mean(
                    [
                        r["relative_error"]
                        for r in rows
                        if r["split"] == i and r["relative_error"] is not None
                    ]
                )
                for i in range(5)
            ],
            "five_split_valid_rows": [
                sum(r["split"] == i and r["valid_edit"] for r in rows) for i in range(5)
            ],
            "median_replayed_query_plus_setup_seconds": float(
                np.median([r["probe_seconds_sum"] + r["setup_seconds"] for r in rows])
            ),
            "cost_scope": (
                "Replayed measurement times and preparation only. Excludes "
                "initial/final "
                "validation, baseline training and planning. JIT may be reused across "
                "shapes. This is not end-to-end timing or evidence "
                "of a speed advantage."
            ),
        }
    return result


def evaluate(
    cases: tuple[SelectionCase, ...], oracle: SelectionOracle
) -> dict[str, Any]:
    """Fit CAD-only ordering on training groups; select only observed endpoints."""
    case_map = {case.fixture.group_id: case for case in cases}
    splits = partitions(cases)
    selections = []
    for split in splits:
        training = np.stack(
            [
                case_map[group].volume_matrix() / case_map[group].cutter_volume
                for group in split["train"]
            ]
        )
        for group in split["test"]:
            case = case_map[group]
            actual = case.volume_matrix()
            stock_bound = float(stock_volume_upper_bound(case.fixture))
            query = partial(oracle.query, case)
            for fraction in TARGETS:
                target = fraction * case.cutter_volume
                rates = np.asarray(case.rates) if case.rates is not None else None
                orders = {"cad": learned_order(training, fraction)}
                if rates is not None:
                    orders["analytic"] = action_order(
                        np.maximum(rates, 0)[:, None] * STEPS, target
                    )
                for kind in ("cad", "analytic"):
                    for mode in ("grid", "refine"):
                        for budget in BUDGETS if mode == "grid" else (4, 8):
                            result: dict[str, Any] = {
                                "split": split["split"],
                                "group_id": group,
                                "family": case.fixture.family,
                                "target_fraction": fraction,
                                "target_volume": target,
                                "method": f"{kind}_{mode}",
                                "budget": budget,
                                "target_exceeds_stock_volume": target > stock_bound,
                                "stock_volume_upper_bound": stock_bound,
                                "relative_error": None,
                                "signed_excess_over_grid": None,
                                "valid_edit": False,
                                "within_one_percent": False,
                                "trace": [],
                                "probe_attempts": 0,
                                "probe_common_calls": 0,
                                "probe_fusion_calls": 0,
                                "probe_seconds_sum": 0.0,
                                "initial_common_calls": case.initial.common_calls,
                                "initial_fusion_calls": case.initial.fusion_calls,
                                "initial_seconds": case.initial.total_seconds,
                                "final_common_calls": 0,
                                "final_cut_calls": 0,
                                "final_fusion_calls": 0,
                                "final_seconds": 0.0,
                                "setup_seconds": case.preparation_seconds
                                + case.first_query_seconds
                                if kind == "analytic"
                                else 0.0,
                            }
                            grid_error = (
                                float(np.nanmin(np.abs(actual - target)) / target)
                                if np.isfinite(actual).any()
                                else None
                            )
                            result["grid_best_relative_error"] = grid_error
                            if not case.initial_valid:
                                result.update(
                                    error="Invalid initial CAD",
                                    outcome="invalid_initial",
                                )
                            elif kind not in orders:
                                result.update(
                                    error=case.preparation_error
                                    or "No direction rates",
                                    outcome="unavailable_model",
                                )
                            else:
                                trace = (
                                    grid_policy(
                                        orders[kind],
                                        target,
                                        budget,
                                        query,
                                        adaptive=kind == "cad",
                                    )
                                    if mode == "grid"
                                    else refine_policy(
                                        orders[kind],
                                        target,
                                        budget,
                                        query,
                                        rates=rates if kind == "analytic" else None,
                                    )
                                )
                                result.update(
                                    trace=[asdict(o) for o in trace],
                                    probe_attempts=len(trace),
                                    probe_common_calls=sum(
                                        o.common_calls for o in trace
                                    ),
                                    probe_fusion_calls=sum(
                                        o.fusion_calls for o in trace
                                    ),
                                    probe_seconds_sum=sum(o.seconds for o in trace),
                                )
                                chosen = best_observation(trace, target)
                                if chosen is None:
                                    result.update(
                                        error="No valid queried endpoint",
                                        outcome="no_valid_probe",
                                    )
                                else:
                                    edit = oracle.validate(case, chosen)
                                    ok = bool(
                                        valid_measurement(edit, case.cutter_volume)
                                        and edit.cut_valid
                                        and edit.working_balance_error is not None
                                        and edit.fixed_volume is not None
                                        and abs(edit.working_balance_error)
                                        <= 1e-8
                                        * max(case.cutter_volume, edit.fixed_volume)
                                    )
                                    error = (
                                        abs(edit.working_volume - target) / target
                                        if ok and edit.working_volume is not None
                                        else None
                                    )
                                    hit = bool(
                                        ok and error is not None and error <= 0.01
                                    )
                                    outcome = (
                                        "invalid_cut"
                                        if not ok
                                        else "hit_target"
                                        if hit
                                        else "target_exceeds_stock_volume"
                                        if target > stock_bound
                                        else "budget_or_motion_range_unresolved"
                                    )
                                    result.update(
                                        chosen=asdict(chosen),
                                        edit=asdict(edit),
                                        error=None
                                        if ok
                                        else edit.error or "Invalid Cut/balance",
                                        relative_error=error,
                                        signed_excess_over_grid=None
                                        if error is None or grid_error is None
                                        else error - grid_error,
                                        valid_edit=ok,
                                        within_one_percent=hit,
                                        outcome=outcome,
                                        final_common_calls=edit.common_calls,
                                        final_cut_calls=edit.cut_calls,
                                        final_fusion_calls=edit.fusion_calls,
                                        final_seconds=edit.total_seconds,
                                    )
                            result["total_common_calls"] = (
                                result["initial_common_calls"]
                                + result["probe_common_calls"]
                                + result["final_common_calls"]
                            )
                            result["total_cut_calls"] = result["final_cut_calls"]
                            selections.append(result)
    return {
        "splits": splits,
        "summary": summarize(selections),
        "selections": selections,
        "offline_training": [
            {
                "split": split["split"],
                "groups": split["train"],
                "common_calls_without_cross_partition_cache": sum(
                    case_map[g].initial.common_calls
                    + sum(e.measurement.common_calls for e in case_map[g].endpoints)
                    for g in split["train"]
                ),
            }
            for split in splits
        ],
    }
