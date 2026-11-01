"""Reproduce boundary editing with independent references and retained failures."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import jax

from brepax.experimental import polyhedral_boundary
from brepax.experimental.polyhedral_boundary import (
    prepare_polyhedral_event_boundary,
    specialize_polyhedral_event_boundary,
)
from brepax.experimental.polyhedral_targets import polyhedral_translation_targets
from brepax.experimental.polyhedral_translation import prepare_polyhedral_translation

from .event_boundary_cad import CadCallCounts, observe_scene, prepare_scene
from .event_boundary_fixtures import FAMILIES, Family, affine_cells, vertices
from .event_boundary_references import ReferenceInventory, reference, reference_volume
from .event_boundary_selection import BoundaryCache, EditAttempt, Method, evaluate

REPO = Path(__file__).parents[2]
METHODS = ("rebuild_profile", "cad_search", "reusable_boundary")


@dataclass(frozen=True)
class AuditResult:
    """Independent diagnostics that never determine the next policy seed."""

    reference: ReferenceInventory
    max_reference_volume_error: float | None
    reference_volume_matches: bool
    target_band_classification_disagreements: int
    final_reference_volume_error: float | None
    final_removed_reference_error: float | None
    earliest_exact_solution: float | None
    earliest_exact_displacement: float | None
    false_band_unreachable: bool
    correct_band_unreachable: bool


def audit(attempt: EditAttempt, family: Family, theta: Fraction) -> AuditResult:
    oracle = reference(family, theta)
    stock, target = float(oracle["stock_volume"]), float(family.target)
    records = [
        record
        for record in (attempt.initial, *attempt.probes, attempt.final)
        if record is not None
    ]
    errors, disagreements = [], 0
    for record in records:
        if record.working_volume is None:
            continue
        exact = float(reference_volume(family, theta, record.step))
        errors.append(abs(record.working_volume - exact))
        disagreements += (abs(record.working_volume - target) <= target / 1000) != (
            abs(exact - target) <= target / 1000
        )
    final = attempt.final
    final_reference = (
        None if final is None else float(reference_volume(family, theta, final.step))
    )
    final_error = (
        None
        if final is None or final.working_volume is None or final_reference is None
        else abs(final.working_volume - final_reference)
    )
    removed = (
        None
        if attempt.outcome is None
        else attempt.outcome.removed_stock_minus_cut_volume
    )
    removed_error = (
        None
        if removed is None or final_reference is None
        else abs(removed - final_reference)
    )
    earliest = min(
        [*oracle["roots"], *(lower for lower, _ in oracle["intervals"])], default=None
    )
    selected = attempt.selection.selected_step
    displacement = None if earliest is None or selected is None else selected - earliest
    rejected = (
        attempt.selection.decision == "profile_band_unreachable"
        and not attempt.timings.wall_cap_exceeded
    )
    return AuditResult(
        oracle,
        max(errors, default=None),
        all(error <= stock * 1e-9 for error in errors),
        disagreements,
        final_error,
        removed_error,
        earliest,
        displacement,
        rejected and oracle["tolerance_reachable"],
        rejected and not oracle["tolerance_reachable"],
    )


def render_attempt(
    attempt: EditAttempt, checked: AuditResult, mode: str
) -> dict[str, Any]:
    """Flatten typed records only at the artifact boundary, after seed selection."""
    result = asdict(attempt)
    selection = result.pop("selection")
    preparation = selection.pop("model")
    selection_error = selection.pop("error")
    result["error"] = result["error"] or selection_error
    result.update(selection)
    result.update(preparation)
    result.update(result.pop("timings"))
    result.update(result.pop("outcome"))
    result.update(asdict(checked))
    result.update(
        {
            "mode": mode,
            "calls": asdict(attempt.calls),
            "probe_attempts": len(attempt.probes),
            "target_binary64": float(attempt.target),
            "relative_tolerance": 0.001,
        }
    )
    return result


def serializable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(serializable(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def snapshot() -> dict[str, str]:
    paths = sorted((REPO / "src/brepax").rglob("*.py"))
    paths += sorted((REPO / "benchmarks").rglob("*.py"))
    paths += [REPO / "uv.lock", REPO / "pyproject.toml"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def reference_audit(family: Family, theta: Fraction) -> dict[str, Any]:
    started = perf_counter()
    calls = CadCallCounts()
    result: dict[str, Any] = {
        "family": family.name,
        "theta": theta,
        "passed": False,
        "error": None,
        "calls": {"common": 0, "cut": 0, "fuse": 0},
        "observations": [],
    }
    try:
        oracle = reference(family, theta)
        result["reference"] = oracle
        fixed, moving = vertices(family, theta)
        profile = prepare_polyhedral_translation(
            fixed, moving, family.velocity, max_step=family.extent
        )
        query = polyhedral_translation_targets(
            profile,
            family.target,
            step_tolerance=1e-9,
            volume_tolerance=1e-12,
            max_refinements=128,
        )
        result["query"] = asdict(query)
        actual_roots = [
            root.step for root in query.roots if root.resolved and root.step is not None
        ]
        actual_intervals = [
            (float(interval.lower), float(interval.upper))
            for interval in query.intervals
        ]
        root_match = len(actual_roots) == len(oracle["roots"]) and all(
            abs(a - b) <= 1e-8
            for a, b in zip(actual_roots, oracle["roots"], strict=True)
        )
        interval_match = actual_intervals == oracle["intervals"]
        result["inventory_matches"] = bool(
            root_match
            and interval_match
            and query.exact_reachable == oracle["exact_reachable"]
        )
        zero = polyhedral_translation_targets(
            profile, 0, step_tolerance=1e-9, volume_tolerance=1e-12
        )
        result["zero_target"] = asdict(zero)
        boundary_profile = None
        boundary_matches = False
        affine_fixed, affine_moving = affine_cells(family)
        try:
            boundary = prepare_polyhedral_event_boundary(
                affine_fixed,
                affine_moving,
                family.velocity,
                max_step=family.extent,
                parameter_radius=Fraction(1, 8),
            )
        except ValueError as error:
            result["boundary_rejection"] = f"{type(error).__name__}: {error}"
            boundary_matches = family.name == "ramp_birth"
        else:
            result["boundary_parameter_interval"] = boundary.parameter_interval
            boundary_profile = specialize_polyhedral_event_boundary(boundary, theta)
            boundary_query = polyhedral_translation_targets(
                boundary_profile,
                family.target,
                step_tolerance=1e-9,
                volume_tolerance=1e-12,
                max_refinements=128,
            )
            result["boundary_query"] = asdict(boundary_query)
            boundary_roots = [
                root.step
                for root in boundary_query.roots
                if root.resolved and root.step is not None
            ]
            boundary_intervals = [
                (float(interval.lower), float(interval.upper))
                for interval in boundary_query.intervals
            ]
            boundary_matches = (
                family.name != "ramp_birth"
                and boundary_query.exact_reachable == oracle["exact_reachable"]
                and len(boundary_roots) == len(oracle["roots"])
                and all(
                    abs(a - b) <= 1e-8
                    for a, b in zip(boundary_roots, oracle["roots"], strict=True)
                )
                and boundary_intervals == oracle["intervals"]
            )
        result["boundary_inventory_matches"] = boundary_matches
        scene, stock = prepare_scene(family, theta, calls)
        result["source_stock_volume"] = stock
        knots = oracle["knots"]
        positions = set(knots)
        for lower, upper in pairwise(knots):
            positions.add((lower + upper) / 2)
        for knot in knots[1:-1]:
            for offset in (-Fraction(1, 1024), Fraction(1, 1024)):
                if 0 <= knot + offset <= family.extent:
                    positions.add(knot + offset)
        errors, profile_errors, boundary_errors, valid = [], [], [], True
        for step in sorted(positions):
            cad = observe_scene(scene, float(step))
            calls.common += cad.common_calls
            item: dict[str, Any] = {
                "step": step,
                "reference_volume": reference_volume(family, theta, step),
                "cad": asdict(cad),
            }
            result["observations"].append(item)
            exact_piece = next(
                piece
                for piece in profile._exact_pieces
                if piece.lower <= step <= piece.upper
            )
            profile_errors.append(
                abs(exact_piece.value(step) - item["reference_volume"])
            )
            if boundary_profile is not None:
                boundary_piece = next(
                    piece
                    for piece in boundary_profile._exact_pieces
                    if piece.lower <= step <= piece.upper
                )
                boundary_errors.append(
                    abs(boundary_piece.value(step) - item["reference_volume"])
                )
            if cad.working_volume is not None:
                errors.append(abs(cad.working_volume - float(item["reference_volume"])))
            valid = (
                valid and cad.error is None and cad.source_valid and cad.common_valid
            )
        result["max_cad_reference_error"] = max(errors, default=None)
        result["max_exact_profile_error"] = max(profile_errors, default=Fraction(0))
        result["max_boundary_profile_error"] = max(boundary_errors, default=Fraction(0))
        result["passed"] = bool(
            result["inventory_matches"]
            and zero.exact_reachable
            and zero.intervals
            and valid
            and abs(stock - float(oracle["stock_volume"])) <= stock * 1e-9
            and all(error <= stock * 1e-9 for error in errors)
            and all(error == 0 for error in profile_errors)
            and boundary_matches
            and all(error == 0 for error in boundary_errors)
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["calls"] = asdict(calls)
    result["seconds"] = perf_counter() - started
    return result


def controls() -> list[dict[str, Any]]:
    results = []
    for family in (FAMILIES[0], FAMILIES[2]):
        for theta in (Fraction(-1), Fraction(-5, 4)):
            try:
                vertices(family, theta)
                results.append(
                    {
                        "family": family.name,
                        "theta": theta,
                        "passed": False,
                        "error": "Invalid width accepted",
                    }
                )
            except ValueError as error:
                results.append(
                    {
                        "family": family.name,
                        "theta": theta,
                        "passed": True,
                        "rejection": str(error),
                    }
                )
    tetra = FAMILIES[2]
    fixed, moving = vertices(tetra, Fraction(0))
    profile = prepare_polyhedral_translation(
        fixed, moving, tetra.velocity, max_step=tetra.extent
    )
    exact = polyhedral_translation_targets(
        profile, tetra.target, step_tolerance=1e-10, volume_tolerance=1e-14
    )
    rounded = polyhedral_translation_targets(
        profile, float(tetra.target), step_tolerance=1e-10, volume_tolerance=1e-14
    )
    results.append(
        {
            "control": "exact_vs_binary64_target",
            "exact_target": tetra.target,
            "binary64_target_as_rational": Fraction(float(tetra.target)),
            "exact_query": asdict(exact),
            "rounded_query": asdict(rounded),
            "passed": len(exact.roots) == 1 and len(rounded.roots) == 2,
        }
    )
    return results


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in ("cold", "sequential"):
        for method in METHODS:
            group = [
                row for row in rows if row["mode"] == mode and row["method"] == method
            ]
            if not group:
                continue
            classified = {}
            for label, predicate in (
                ("exact_reachable", lambda row: row["reference"]["exact_reachable"]),
                (
                    "exact_unreachable_band_reachable",
                    lambda row: (
                        not row["reference"]["exact_reachable"]
                        and row["reference"]["tolerance_reachable"]
                    ),
                ),
                (
                    "band_unreachable",
                    lambda row: not row["reference"]["tolerance_reachable"],
                ),
            ):
                members = [row for row in group if predicate(row)]
                classified[label] = {
                    "count": len(members),
                    "operational_success": sum(
                        row["operational_success"] for row in members
                    ),
                    "seconds": sum(row["seconds"] for row in members),
                    "common_calls": sum(row["calls"]["common"] for row in members),
                }
            result[f"{mode}:{method}"] = {
                "dimensions": len(group),
                "operational_success": sum(row["operational_success"] for row in group),
                "seed_hits": sum(row["seed_hit"] for row in group),
                "profile_attempts": sum(row["profile_attempted"] for row in group),
                "boundary_preparations": sum(
                    row["boundary_preparation_attempted"] for row in group
                ),
                "boundary_specializations": sum(
                    row["boundary_specialized"] for row in group
                ),
                "boundary_fallbacks": sum(row["boundary_fallback"] for row in group),
                "correct_band_unreachable": sum(
                    row["correct_band_unreachable"] for row in group
                ),
                "false_band_unreachable": sum(
                    row["false_band_unreachable"] for row in group
                ),
                "errors": sum(row["error"] is not None for row in group),
                "probes": sum(row["probe_attempts"] for row in group),
                "calls": {
                    name: sum(row["calls"][name] for row in group)
                    for name in ("common", "cut", "fuse")
                },
                "seconds": sum(row["seconds"] for row in group),
                "classes": classified,
            }
    return result


def run(
    output: Path,
    *,
    names: tuple[str, ...] = (),
    methods: tuple[str, ...] = METHODS,
    modes: tuple[str, ...] = ("cold", "sequential"),
    theta_index: int | None = None,
    audits: bool = True,
) -> dict[str, Any]:
    if any(name not in {family.name for family in FAMILIES} for name in names):
        raise ValueError("Unknown family selector")
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError("Unknown or empty method selection")
    if not modes or any(mode not in ("cold", "sequential") for mode in modes):
        raise ValueError("Unknown or empty mode selection")
    if theta_index is not None and "sequential" in modes:
        raise ValueError("Sequential paths cannot omit earlier dimension state")
    fixtures = [family for family in FAMILIES if not names or family.name in names]
    if theta_index is not None and any(
        not 0 <= theta_index < len(family.thetas) for family in fixtures
    ):
        raise ValueError("Theta index outside selected path")
    expected = {
        (mode, family.name, method, str(theta))
        for mode in modes
        for family in fixtures
        for method in methods
        for index, theta in enumerate(family.thetas)
        if theta_index is None or index == theta_index
    }
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    origins = {"boundary_module": str(Path(polyhedral_boundary.__file__).resolve())}
    if not Path(origins["boundary_module"]).is_relative_to(REPO / "src"):
        raise RuntimeError(
            "Imported BRepAX source does not belong to the hashed checkout"
        )
    before = snapshot()
    result: dict[str, Any] = {
        "passed": False,
        "passed_scope": (
            "Every attempted CAD observation matches the exact model "
            "diagnostic; negative near-contact observations remain "
            "failures of this strict diagnostic"
        ),
        "protocol": {
            "relative_target_tolerance": 0.001,
            "probe_cap": 128,
            "cooperative_stage_deadline_seconds": 60.0,
            "scope": (
                "Four predeclared synthetic one-parameter dimension paths; "
                "fixed normals and translation; no arbitrary CAD claim"
            ),
            "primary_success": (
                "Valid final Common/Cut with both Common and "
                "stock-minus-Cut removal within target tolerance"
            ),
            "exact_reachability": (
                "Separate from tolerance-qualified editing; rational targets retained"
            ),
            "sequential_reuse": (
                "Remeasure prior accepted position on fresh CAD before lazy "
                "model work; preserve last accepted position after a failed "
                "dimension"
            ),
            "boundary_policy": (
                "Lazy theta=0 preparation at exact center radius1/8; "
                "explicit rejection fallback to full fixed profile"
            ),
            "reference_separation": (
                "Typed online CAD outcome controls seeds before independent audit"
            ),
        },
        "sources": before,
        "import_origins": origins,
        "timing_scope": (
            "Correctness-run diagnostic times; no isolated cold-process claim"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "rows": [],
        "reference_audits": [],
        "controls": [],
    }
    fixtures = [family for family in FAMILIES if not names or family.name in names]
    for mode in modes:
        for family in fixtures:
            for method in methods:
                seed = None
                boundary_cache = BoundaryCache()
                for index, theta in enumerate(family.thetas):
                    if theta_index is not None and index != theta_index:
                        continue
                    attempt = evaluate(
                        family,
                        theta,
                        cast(Method, method),
                        seed=seed if mode == "sequential" else None,
                        boundary_cache=boundary_cache if mode == "sequential" else None,
                    )
                    if (
                        attempt.outcome is not None
                        and attempt.outcome.operational_success
                    ):
                        seed = attempt.selection.selected_step
                    row = render_attempt(attempt, audit(attempt, family, theta), mode)
                    result["rows"].append(row)
                    write(output, result)
    if audits:
        for family in fixtures:
            for theta in family.thetas:
                result["reference_audits"].append(reference_audit(family, theta))
                write(output, result)
        result["controls"] = controls()
    result["summary"] = summary(result["rows"])
    result["sources_unchanged"] = before == snapshot()
    actual = [
        (row["mode"], row["family"], row["method"], str(row["theta"]))
        for row in result["rows"]
    ]
    result["planned_row_count"] = len(expected)
    result["cohort_complete"] = len(actual) == len(expected) and set(actual) == expected
    result["completed"] = bool(
        result["cohort_complete"]
        and result["sources_unchanged"]
        and all(row["error"] is None for row in result["rows"])
    )
    result["all_probe_reference_matches"] = all(
        row["reference_volume_matches"] for row in result["rows"]
    )
    result["reference_audits_passed"] = all(
        item["passed"] for item in result["reference_audits"]
    )
    result["accepted_edits_reference_match"] = all(
        row["final_reference_volume_error"]
        <= float(row["reference"]["stock_volume"]) * 1e-9
        and row["final_removed_reference_error"]
        <= float(row["reference"]["stock_volume"]) * 1e-9
        for row in result["rows"]
        if row["operational_success"]
    )
    result["timing_fields"] = {
        "seconds": "Total per-dimension elapsed time; sum all rows for sequence cost",
        "selection_seconds": (
            "Includes profile, boundary preparation/specialization and query times"
        ),
        "profile_query_seconds": "Includes the actual selected CAD probe",
        "class_breakdowns": (
            "Marginal row costs; shared setup remains in full-sequence total"
        ),
    }
    result["passed"] = bool(
        result["completed"]
        and all(
            row["reference_volume_matches"]
            and not row["false_band_unreachable"]
            and row["error"] is None
            for row in result["rows"]
        )
        and all(item["passed"] for item in result["reference_audits"])
        and all(item["passed"] for item in result["controls"])
    )
    write(output, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "rows": len(result["rows"]),
                "audits": len(result["reference_audits"]),
                "summary": result["summary"],
            },
            indent=2,
        )
    )
    return result
