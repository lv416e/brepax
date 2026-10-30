"""Score finite-edit outcomes without exposing references to the policies."""

from dataclasses import asdict
from fractions import Fraction
from typing import Any

import numpy as np

from brepax.experimental.polyhedral_targets import polyhedral_translation_targets

from .parametric_cad import CadMeasurement, valid_observation
from .parametric_fixtures import ParametricFixture
from .parametric_references import (
    reference_maximum,
    reference_solutions,
    reference_stock_volume,
    reference_volume,
)
from .parametric_selection import (
    Method,
    PolicyOutcome,
    PreparedPolicy,
    prepare_policy,
    select_target,
)

TARGET_FRACTIONS = (0.2, 0.6, 0.9, 1.1)
POLICIES: tuple[tuple[Method, int], ...] = (
    ("finite", 4),
    ("first_order", 4),
    ("cad", 4),
    ("first_order", 8),
    ("cad", 8),
)


def _within_cad_contract(
    fixture: ParametricFixture, record: CadMeasurement | None, *, cut: bool = False
) -> bool:
    if (
        record is None
        or not valid_observation(record, cut=cut)
        or record.working_volume is None
    ):
        return False
    stock = reference_stock_volume(fixture)
    tolerance = 1e-9 * stock
    if not -tolerance <= record.working_volume <= stock + tolerance:
        return False
    if abs(record.working_volume - reference_volume(fixture, record.step)) > tolerance:
        return False
    return bool(
        not cut
        or (
            record.working_balance_error is not None
            and abs(record.working_balance_error) <= tolerance
        )
    )


def score_outcome(fixture: ParametricFixture, outcome: PolicyOutcome) -> dict[str, Any]:
    """Keep target error, model/path infeasibility and operation failures separate."""
    maximum = reference_maximum(fixture)
    reachable = 0 <= outcome.target <= maximum
    cad_valid = _within_cad_contract(fixture, outcome.validation, cut=True)
    volume = (
        outcome.validation.working_volume if outcome.validation is not None else None
    )
    relative_error = (
        abs(volume - outcome.target) / outcome.target
        if cad_valid and volume is not None
        else None
    )
    target_success = relative_error is not None and relative_error <= 0.01
    correct_rejection = outcome.decision == "unreachable" and not reachable
    false_rejection = outcome.decision == "unreachable" and reachable
    probes_valid = all(
        _within_cad_contract(fixture, record) for record in outcome.probes
    )
    return {
        **asdict(outcome),
        "group": fixture.name,
        "family": fixture.family,
        "reference_path_reachable": reachable,
        "reference_path_maximum": maximum,
        "reference_stock_volume": reference_stock_volume(fixture),
        "relative_target_error": relative_error,
        "within_one_percent": target_success,
        "valid_edit": cad_valid,
        "probes_valid": probes_valid,
        "correct_unreachable": correct_rejection,
        "false_unreachable": false_rejection,
        "outcome_correct": bool(target_success or correct_rejection),
        "probe_attempts": len(outcome.probes),
        "common_calls": sum(record.common_calls for record in outcome.probes)
        + (outcome.validation.common_calls if outcome.validation is not None else 0),
        "cut_calls": outcome.validation.cut_calls
        if outcome.validation is not None
        else 0,
        "fusion_calls": outcome.validation.fusion_calls
        if outcome.validation is not None
        else 0,
    }


def inventory_check(policy: PreparedPolicy) -> dict[str, Any]:
    """Check all finite roots/ranges against independent analytic-family sections."""
    if policy.profile is None or policy.volume_range is None:
        return {"passed": False, "error": policy.error or "Missing finite profile"}
    fixture = policy.fixture
    maximum = reference_maximum(fixture)
    expected_maximum = Fraction(maximum)
    range_ok = (
        policy.volume_range.resolved
        and policy.volume_range.minimum[0] <= 0 <= policy.volume_range.minimum[1]
        and policy.volume_range.maximum[0]
        <= expected_maximum
        <= policy.volume_range.maximum[1]
    )
    rows = []
    for fraction in (0.0, *TARGET_FRACTIONS, 1.0):
        target = fraction * maximum
        actual = polyhedral_translation_targets(
            policy.profile,
            target,
            step_tolerance=1e-12 * fixture.max_step,
            volume_tolerance=1e-12 * fixture.a**3,
        )
        expected = reference_solutions(fixture, target)
        actual_steps = [
            root.step
            for root in actual.roots
            if root.resolved and root.step is not None
        ]
        expected_reachable = bool(expected.roots or expected.intervals)
        roots_ok = len(actual_steps) == len(expected.roots) and all(
            abs(a - b) <= 2e-11 * fixture.max_step
            for a, b in zip(sorted(actual_steps), expected.roots, strict=True)
        )
        intervals_ok = len(actual.intervals) == len(expected.intervals) and all(
            abs(float(a.lower) - b[0]) <= 2e-11 * fixture.max_step
            and abs(float(a.upper) - b[1]) <= 2e-11 * fixture.max_step
            for a, b in zip(actual.intervals, expected.intervals, strict=True)
        )
        rows.append(
            {
                "target": target,
                "fraction": fraction,
                "actual": asdict(actual),
                "reference": asdict(expected),
                "passed": bool(
                    actual.status != "unresolved"
                    and actual.exact_reachable == expected_reachable
                    and roots_ok
                    and intervals_ok
                ),
            }
        )
    return {
        "passed": bool(range_ok and all(row["passed"] for row in rows)),
        "range": asdict(policy.volume_range),
        "reference_maximum": maximum,
        "range_passed": range_ok,
        "targets": rows,
    }


def evaluate_case(
    fixture: ParametricFixture,
    policies: tuple[tuple[Method, int], ...] = POLICIES,
) -> dict[str, Any]:
    """Run independently prepared policies; cache no candidate observations."""
    bodies = fixture.bodies()
    results = []
    inventory = None
    for method, budget in policies:
        policy = prepare_policy(fixture, method)
        if method == "finite":
            try:
                inventory = inventory_check(policy)
            except Exception as error:
                inventory = {
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        rows = []
        for fraction in TARGET_FRACTIONS:
            target = fraction * reference_maximum(fixture)
            outcome = select_target(policy, target, budget)
            try:
                rows.append(score_outcome(fixture, outcome))
            except Exception as error:
                failed = score_outcome(
                    fixture,
                    PolicyOutcome(
                        target,
                        budget,
                        "failed",
                        None,
                        (),
                        None,
                        None,
                        outcome.selection_seconds,
                        outcome.total_seconds,
                        f"Scoring failed: {type(error).__name__}: {error}",
                    ),
                )
                failed["unscored_outcome"] = asdict(outcome)
                failed["probe_attempts"] = len(outcome.probes)
                records = (
                    *outcome.probes,
                    *((outcome.validation,) if outcome.validation is not None else ()),
                )
                for key in ("common_calls", "cut_calls", "fusion_calls"):
                    failed[key] = sum(getattr(record, key) for record in records)
                rows.append(failed)
        initial_valid = _within_cad_contract(fixture, policy.initial)
        completed = (
            policy.error is None
            and initial_valid
            and all(
                row["probes_valid"]
                and (
                    row["valid_edit"]
                    if row["decision"] == "selected"
                    else row["decision"] == "unreachable"
                )
                for row in rows
            )
        )
        results.append(
            {
                "method": method,
                "budget": budget,
                "preparation_error": policy.error,
                "setup_seconds": policy.setup_seconds,
                "geometry_seconds": policy.geometry_seconds,
                "model_seconds": policy.model_seconds,
                "first_jit_seconds": policy.first_jit_seconds,
                "initial": asdict(policy.initial)
                if policy.initial is not None
                else None,
                "initial_valid": initial_valid,
                "preparation_fusion_calls": policy.fusion_calls,
                "event_count": len(policy.profile.exact_knots)
                if policy.profile is not None
                else None,
                "initial_rate": policy.rate,
                "selected": rows,
                "passed": bool(completed),
                "accounting_complete": True,
                "first_target_seconds": policy.setup_seconds + rows[0]["total_seconds"],
                "amortized_target_seconds": (
                    policy.setup_seconds + sum(row["total_seconds"] for row in rows)
                )
                / len(rows),
            }
        )
    return {
        "fixture": asdict(fixture),
        "geometry_complexity": {
            "fixed_cells": len(bodies.fixed),
            "moving_cells": len(bodies.moving),
            "cell_pairs": len(bodies.fixed) * len(bodies.moving),
            "total_cell_facets": sum(
                len(prism.footprint) + 2 for prism in (*bodies.fixed, *bodies.moving)
            ),
        },
        "reference_path_maximum": reference_maximum(fixture),
        "reference_stock_volume": reference_stock_volume(fixture),
        "inventory": inventory,
        "policies": results,
        "passed": bool(
            all(result["passed"] for result in results)
            and (inventory is None or inventory["passed"])
        ),
    }


def _missing_policy(
    fixture: ParametricFixture, method: Method, budget: int, error: str
) -> dict[str, Any]:
    """Keep a whole-case failure in every requested target denominator."""
    rows = [
        score_outcome(
            fixture,
            PolicyOutcome(
                fraction * reference_maximum(fixture),
                budget,
                "failed",
                None,
                (),
                None,
                None,
                0.0,
                0.0,
                error,
            ),
        )
        for fraction in TARGET_FRACTIONS
    ]
    return {
        "method": method,
        "budget": budget,
        "selected": rows,
        "initial": None,
        "preparation_fusion_calls": 0,
        "first_target_seconds": None,
        "amortized_target_seconds": None,
        "accounting_complete": False,
        "error": error,
    }


def _median_known(values: list[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return float(np.median(observed)) if observed else None


def summarize(
    cases: list[dict[str, Any]], requested: tuple[tuple[Method, int], ...] = POLICIES
) -> dict[str, Any]:
    """Report this fixed panel; target rows are not independent shape samples."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        fixture = ParametricFixture(**case["fixture"])
        observed = {
            (policy["method"], policy["budget"]): policy
            for policy in case.get("policies", [])
        }
        for method, budget in requested:
            policy = observed.get((method, budget))
            if policy is None:
                policy = _missing_policy(
                    fixture,
                    method,
                    budget,
                    case.get("error", "Requested policy outcome is missing"),
                )
            key = f"{method}_K{budget}"
            groups.setdefault(key, []).append(policy)
    summaries = {}
    for key, policies in groups.items():
        rows = [row for policy in policies for row in policy["selected"]]
        reachable = [row for row in rows if row["reference_path_reachable"]]
        errors = [
            row["relative_target_error"]
            for row in rows
            if row["relative_target_error"] is not None
        ]
        reachable_errors = [
            row["relative_target_error"]
            for row in reachable
            if row["relative_target_error"] is not None
        ]
        summaries[key] = {
            "shape_groups": len(policies),
            "targets": len(rows),
            "reachable_targets": len(reachable),
            "mean_error_among_returned_edits": float(np.mean(errors))
            if errors
            else None,
            "returned_edit_count": len(errors),
            "reachable_mean_error_among_returned_edits": float(
                np.mean(reachable_errors)
            )
            if reachable_errors
            else None,
            "reachable_returned_edit_count": len(reachable_errors),
            "within_one_percent_count": sum(row["within_one_percent"] for row in rows),
            "unconditional_success_fraction": sum(
                row["within_one_percent"] for row in rows
            )
            / len(rows),
            "reachable_success_fraction": sum(
                row["within_one_percent"] for row in reachable
            )
            / len(reachable)
            if reachable
            else None,
            "correct_unreachable_count": sum(
                row["correct_unreachable"] for row in rows
            ),
            "false_unreachable_count": sum(row["false_unreachable"] for row in rows),
            "unsupported_or_failed_count": sum(
                row["decision"] in ("unsupported", "failed", "unresolved")
                for row in rows
            ),
            "probe_attempts": sum(row["probe_attempts"] for row in rows),
            "call_accounting_complete": all(
                policy["accounting_complete"] for policy in policies
            ),
            "call_counts_are_lower_bounds": not all(
                policy["accounting_complete"] for policy in policies
            ),
            "common_calls": sum(row["common_calls"] for row in rows)
            + sum(
                policy["initial"]["common_calls"]
                if policy["initial"] is not None
                else 0
                for policy in policies
            ),
            "cut_calls": sum(row["cut_calls"] for row in rows),
            "fusion_calls": sum(row["fusion_calls"] for row in rows)
            + sum(policy["preparation_fusion_calls"] for policy in policies),
            "median_first_target_seconds": _median_known(
                [p["first_target_seconds"] for p in policies]
            ),
            "median_amortized_target_seconds": _median_known(
                [p["amortized_target_seconds"] for p in policies]
            ),
        }
    return summaries
