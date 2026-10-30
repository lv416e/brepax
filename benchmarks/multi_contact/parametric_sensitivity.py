"""Independent section, rebuilt-CAD and symmetry checks for design sensitivities."""

from dataclasses import asdict, replace
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.face_contact.cad import rotation_matrix
from brepax.experimental.polyhedral_parameter import (
    AffinePolyhedralCell,
    PreparedPolyhedralParameter,
    polyhedral_parameter_events,
    polyhedral_parameter_position_sensitivity,
    polyhedral_parameter_volume,
    prepare_polyhedral_parameter,
    specialize_polyhedral_parameter,
)
from brepax.experimental.polyhedral_targets import polyhedral_translation_targets
from brepax.experimental.polyhedral_translation import polyhedral_translation_volume

from .parametric_cad import CadMeasurement, rebuild_and_observe, valid_observation
from .parametric_fixtures import ParametricFixture, SecondParameter, parametric_fixtures
from .parametric_references import (
    clipped_volume,
    reference_derivatives,
    reference_maximum,
    reference_solutions,
    reference_stock_volume,
    reference_volume,
)

_VALUE = jax.jit(polyhedral_parameter_volume)
_DERIVATIVES = jax.jit(jax.grad(polyhedral_parameter_volume, argnums=(1, 2)))
_EVENTS = jax.jit(polyhedral_parameter_events)
_EVENT_DERIVATIVES = jax.jit(jax.jacfwd(polyhedral_parameter_events, argnums=1))
_POSITION = jax.jit(polyhedral_parameter_position_sensitivity)


def prepare_fixture_parameter(
    fixture: ParametricFixture,
    second: SecondParameter = "stock_height",
    *,
    scale: float = 1.0,
    rotation: np.ndarray | None = None,
) -> PreparedPolyhedralParameter:
    """Prepare the declared affine family; similarity transforms act on its basis."""
    matrix = np.eye(3) if rotation is None else np.asarray(rotation)
    fixed, moving = fixture.vertex_families(second=second)
    groups = [
        [
            AffinePolyhedralCell(
                scale * family.vertices @ matrix.T,
                scale * family.displacements @ matrix.T,
            )
            for family in families
        ]
        for families in (fixed, moving)
    ]
    return prepare_polyhedral_parameter(
        groups[0],
        groups[1],
        scale * matrix @ np.asarray(fixture.bodies().velocity),
        max_step=fixture.max_step,
    )


def _perturbed_fixture(
    fixture: ParametricFixture, theta: tuple[float, float], second: SecondParameter
) -> ParametricFixture:
    return replace(
        fixture,
        gap=fixture.gap + fixture.a * theta[0],
        height=fixture.height
        + (fixture.a * theta[1] if second == "stock_height" else 0),
        width=fixture.width + (fixture.a * theta[1] if second == "slab_width" else 0),
    )


def _root_controls(
    fixture: ParametricFixture,
    prepared: PreparedPolyhedralParameter,
    second: SecondParameter,
    target: float,
    cad_observations: list[CadMeasurement],
) -> list[dict[str, Any]]:
    theta0 = jnp.zeros(2)
    reference = reference_solutions(fixture, target)
    specialized = specialize_polyhedral_parameter(prepared, np.zeros(2))
    query = polyhedral_translation_targets(
        specialized,
        target,
        step_tolerance=1e-12 * fixture.max_step,
        volume_tolerance=1e-12 * fixture.a**3,
    )
    roots = sorted(
        root.step for root in query.roots if root.resolved and root.step is not None
    )
    if query.status != "reachable" or len(roots) != len(reference.roots):
        raise RuntimeError(
            "Parameter specialization did not preserve independent roots"
        )
    rows: list[dict[str, Any]] = []
    for root_index, step in enumerate(roots):
        expected_slope, expected_gradient = reference_derivatives(
            fixture, step, second=second
        )
        expected_position = -expected_gradient / expected_slope
        value = float(_VALUE(prepared.model, step, theta0))
        slope, gradient = _DERIVATIVES(prepared.model, step, theta0)
        gradient = np.asarray(gradient)
        position = np.asarray(_POSITION(prepared.model, step, theta0))
        base_cad = rebuild_and_observe(fixture.bodies(), step, cut=True)
        cad_observations.append(base_cad)
        finite_differences: list[dict[str, Any]] = []
        for epsilon in (1e-3, 3e-4, 1e-4):
            measured: list[dict[str, Any]] = []
            for parameter in range(2):
                values = []
                actual_roots = []
                audits: list[dict[str, Any]] = []
                for sign in (-1, 1):
                    theta = [0.0, 0.0]
                    theta[parameter] = sign * epsilon
                    params = (theta[0], theta[1])
                    bodies = fixture.bodies(params, second=second)
                    sample = rebuild_and_observe(bodies, step)
                    cad_observations.append(sample)
                    independent = reference_volume(fixture, step, params, second=second)
                    clipped = clipped_volume(bodies, step)
                    if not valid_observation(sample) or sample.working_volume is None:
                        raise RuntimeError(
                            f"Parameter-rebuilt CAD observation failed: {sample.error}"
                        )
                    values.append(sample.working_volume)
                    changed = _perturbed_fixture(fixture, params, second)
                    independent_roots = reference_solutions(changed, target).roots
                    if len(independent_roots) != len(roots):
                        raise RuntimeError(
                            "Independent root branch changed inside FD sweep"
                        )
                    changed_profile = specialize_polyhedral_parameter(
                        prepared, np.asarray(theta)
                    )
                    changed_query = polyhedral_translation_targets(
                        changed_profile,
                        target,
                        step_tolerance=1e-12 * fixture.max_step,
                        volume_tolerance=1e-12 * fixture.a**3,
                    )
                    changed_roots = sorted(
                        r.step
                        for r in changed_query.roots
                        if r.resolved and r.step is not None
                    )
                    if changed_query.status != "reachable" or len(changed_roots) != len(
                        roots
                    ):
                        raise RuntimeError(
                            "Specialized parameter root became unresolved"
                        )
                    changed_step = changed_roots[root_index]
                    validation = rebuild_and_observe(bodies, changed_step, cut=True)
                    cad_observations.append(validation)
                    actual_roots.append(changed_step)
                    tolerance = 1e-9 * reference_stock_volume(changed)
                    passed = bool(
                        abs(sample.working_volume - independent) <= tolerance
                        and abs(clipped - independent) <= tolerance
                        and abs(changed_step - independent_roots[root_index])
                        <= 2e-11 * fixture.max_step
                        and valid_observation(validation, cut=True)
                        and validation.working_volume is not None
                        and abs(validation.working_volume - target) <= tolerance
                        and validation.working_balance_error is not None
                        and abs(validation.working_balance_error) <= tolerance
                    )
                    audits.append(
                        {
                            "parameters": theta,
                            "cad_at_fixed_step": asdict(sample),
                            "reference_volume": independent,
                            "clipped_volume": clipped,
                            "changed_step": changed_step,
                            "reference_changed_step": independent_roots[root_index],
                            "cad_at_changed_root": asdict(validation),
                            "passed": passed,
                        }
                    )
                measured.append(
                    {
                        "parameter": parameter,
                        "volume_derivative": (values[1] - values[0]) / (2 * epsilon),
                        "root_derivative": (actual_roots[1] - actual_roots[0])
                        / (2 * epsilon),
                        "audits": audits,
                    }
                )
            finite_differences.append({"epsilon": epsilon, "parameters": measured})
        final_fd = finite_differences[-1]["parameters"]
        volume_fd = np.array([row["volume_derivative"] for row in final_fd])
        position_fd = np.array([row["root_derivative"] for row in final_fd])
        volume_scale = reference_stock_volume(fixture)
        tolerance = 1e-9 * volume_scale
        passed = bool(
            abs(value - target) <= tolerance
            and abs(float(slope) - expected_slope) <= tolerance
            and np.allclose(gradient, expected_gradient, rtol=1e-10, atol=tolerance)
            and np.allclose(position, expected_position, rtol=1e-10, atol=1e-10)
            and np.allclose(
                volume_fd, expected_gradient, rtol=1e-5, atol=1e-7 * volume_scale
            )
            and np.allclose(position_fd, expected_position, rtol=1e-5, atol=1e-7)
            and valid_observation(base_cad, cut=True)
            and base_cad.working_volume is not None
            and abs(base_cad.working_volume - target) <= tolerance
            and base_cad.working_balance_error is not None
            and abs(base_cad.working_balance_error) <= tolerance
            and all(
                audit["passed"]
                for fd in finite_differences
                for parameter in fd["parameters"]
                for audit in parameter["audits"]
            )
        )
        rows.append(
            {
                "step": step,
                "value": value,
                "target": target,
                "step_derivative": float(slope),
                "parameter_gradient": gradient.tolist(),
                "expected_step_derivative": expected_slope,
                "expected_parameter_gradient": expected_gradient.tolist(),
                "position_gradient": position.tolist(),
                "expected_position_gradient": expected_position.tolist(),
                "base_cad": asdict(base_cad),
                "finite_differences": finite_differences,
                "passed": passed,
            }
        )
    return rows


def sensitivity_case(
    fixture: ParametricFixture,
    *,
    second: SecondParameter = "stock_height",
    target: float | None = None,
) -> dict[str, Any]:
    """Measure live derivatives and constant-target inverse-position sensitivity."""
    started = perf_counter()
    cad_observations: list[CadMeasurement] = []
    try:
        prepared = prepare_fixture_parameter(fixture, second)
        preparation_seconds = perf_counter() - started
        target = 0.6 * reference_maximum(fixture) if target is None else target
        roots = _root_controls(fixture, prepared, second, target, cad_observations)
        parameters = jnp.zeros(2)
        events = np.asarray(_EVENTS(prepared.model, parameters))
        event_gradient = np.asarray(_EVENT_DERIVATIVES(prepared.model, parameters))
        exact_events = np.array(
            [
                0.0,
                *(float(event.reference_step) for event in prepared.events),
                fixture.max_step,
            ]
        )
        exact_gradients = np.array(
            [
                [0.0, 0.0],
                *(
                    [float(x) for x in event.parameter_gradient]
                    for event in prepared.events
                ),
                [0.0, 0.0],
            ]
        )
        # Internal physical boundaries have gap derivative a; endpoints are
        # fixed path bounds. Width shifts only the trailing slab boundary.
        analytic_gradients = []
        physical = (
            (0.0, fixture.a)
            if fixture.family == "ramp"
            else (0.0, fixture.a, 2 * fixture.a)
        )
        expected_positions = {0.0: (0.0, 0.0), fixture.max_step: (0.0, 0.0)}
        for boundary in physical:
            expected_positions[fixture.gap + boundary] = (fixture.a, 0.0)
            expected_positions[fixture.gap + fixture.width + boundary] = (
                fixture.a,
                fixture.a if second == "slab_width" else 0.0,
            )
        physical_rows = []
        for event, gradient in zip(exact_events, event_gradient, strict=True):
            expected = expected_positions.get(event)
            if expected is not None:
                analytic_gradients.append(expected)
                physical_rows.append(
                    {"step": event, "actual": gradient.tolist(), "expected": expected}
                )
        event_passed = bool(
            np.array_equal(events, exact_events)
            and np.array_equal(event_gradient, exact_gradients)
            and len(physical_rows) == len(expected_positions)
            and all(
                np.array_equal(row["actual"], row["expected"]) for row in physical_rows
            )
        )
        return {
            "name": fixture.name,
            "second_parameter": second,
            "parameter_units": "dimensionless; physical dimension change is a*theta",
            "target_objective": "constant physical volume at reference theta",
            "finite_difference_sources": {
                "volume": (
                    "Central differences of independently rebuilt CAD Common volumes"
                ),
                "position": (
                    "Central differences of specialized model roots, checked against "
                    "independent analytic roots and rebuilt CAD residuals"
                ),
            },
            "target": target,
            "preparation_seconds": preparation_seconds,
            "event_controls": physical_rows,
            "event_passed": event_passed,
            "roots": roots,
            "passed": bool(event_passed and all(row["passed"] for row in roots)),
            "total_seconds": perf_counter() - started,
            "physical_calls": {
                "common": sum(row.common_calls for row in cad_observations),
                "cut": sum(row.cut_calls for row in cad_observations),
                "fuse": sum(row.fusion_calls for row in cad_observations),
            },
        }
    except Exception as error:
        return {
            "name": fixture.name,
            "second_parameter": second,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "total_seconds": perf_counter() - started,
            "attempted_cad_observations": [asdict(row) for row in cad_observations],
            "physical_calls": {
                "common": sum(row.common_calls for row in cad_observations),
                "cut": sum(row.cut_calls for row in cad_observations),
                "fuse": sum(row.fusion_calls for row in cad_observations),
            },
        }


def _symmetry_and_degeneracy(cad_observations: list[CadMeasurement]) -> dict[str, Any]:
    """Coordinate/unit covariance and explicitly undefined local inverse derivatives."""
    fixtures = {fixture.name: fixture for fixture in parametric_fixtures()}
    fixture = fixtures["ramp_0"]
    prepared = prepare_fixture_parameter(fixture)
    theta = jnp.array([1 / 64, 1 / 32])
    step = 0.75
    original = float(_VALUE(prepared.model, step, theta))
    slope, gradient = _DERIVATIVES(prepared.model, step, theta)
    position = np.asarray(_POSITION(prepared.model, step, theta))
    rows: list[dict[str, Any]] = []
    exact_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for scale in (0.125, 1.0, 8.0):
        transformed = prepare_fixture_parameter(
            fixture, scale=scale, rotation=exact_rotation
        )
        value = float(_VALUE(transformed.model, step, theta))
        actual_slope, actual_gradient = _DERIVATIVES(transformed.model, step, theta)
        actual_position = np.asarray(_POSITION(transformed.model, step, theta))
        rows.append(
            {
                "scale": scale,
                "normalized_volume_error": abs(value / scale**3 - original),
                "normalized_gradient_error": float(
                    np.max(
                        np.abs(
                            np.asarray(actual_gradient) / scale**3
                            - np.asarray(gradient)
                        )
                    )
                ),
                "passed": bool(
                    np.isclose(value / scale**3, original, rtol=1e-12, atol=1e-13)
                    and np.isclose(
                        float(actual_slope) / scale**3,
                        float(slope),
                        rtol=1e-12,
                        atol=1e-13,
                    )
                    and np.allclose(
                        np.asarray(actual_gradient) / scale**3,
                        gradient,
                        rtol=1e-12,
                        atol=1e-13,
                    )
                    and np.allclose(actual_position, position, rtol=1e-12, atol=1e-13)
                ),
            }
        )
    placement: list[dict[str, Any]] = []
    rotation = rotation_matrix(np.array([0.2, -0.35, 0.1]))
    for scale in (0.125, 1.0, 8.0):
        record = rebuild_and_observe(
            fixture.bodies((float(theta[0]), float(theta[1]))),
            step,
            cut=True,
            scale=scale,
            rotation=rotation,
            origin=np.array([2.0, -3.0, 4.0]),
        )
        cad_observations.append(record)
        placement.append(
            {
                "scale": scale,
                "cad": asdict(record),
                "passed": bool(
                    valid_observation(record, cut=True)
                    and record.working_volume is not None
                    and abs(record.working_volume - original) <= 1e-9
                    and record.working_balance_error is not None
                    and abs(record.working_balance_error) <= 1e-9
                ),
            }
        )
    tent = fixtures["tent_0"]
    tent_model = prepare_fixture_parameter(tent)
    maximum_step = tent.gap + tent.a + tent.width / 2
    traced_maximum = float(_VALUE(tent_model.model, maximum_step, jnp.zeros(2)))
    specialized_tent = specialize_polyhedral_parameter(tent_model, np.zeros(2))
    maximum_value = float(
        polyhedral_translation_volume(specialized_tent.profile, maximum_step)
    )
    zero_slope = np.asarray(_POSITION(tent_model.model, maximum_step, jnp.zeros(2)))
    width_model = prepare_fixture_parameter(fixture, "slab_width")
    tie_theta = jnp.array([0.0, (fixture.a - fixture.width) / fixture.a])
    rejected = np.asarray(_EVENTS(width_model.model, tie_theta))
    at_boundary = np.asarray(
        _POSITION(prepared.model, fixture.gap + fixture.a, jnp.zeros(2))
    )
    controls = {
        "smooth_maximum_value": maximum_value,
        "traced_maximum_value": traced_maximum,
        "smooth_maximum_matches_reference": abs(maximum_value - reference_maximum(tent))
        <= 1e-12,
        "numerical_maximum_derivative_unresolved": bool(np.isnan(traced_maximum)),
        "maximum_scope": (
            "Valid represented model inside an event chamber, but exact "
            "derivative cancellation is unresolved by the floating chart. "
            "Host specialization retains the finite volume."
        ),
        "zero_slope_position_rejected": bool(np.isnan(zero_slope).all()),
        "event_order_tie_rejected": bool(np.isnan(rejected).all()),
        "event_boundary_position_rejected": bool(np.isnan(at_boundary).all()),
        "tie_scope": (
            "Prepared chart is unsupported; this is not a claim that volume "
            "is nondifferentiable everywhere."
        ),
    }
    return {
        "model_similarity": rows,
        "actual_transformed_cad": placement,
        "degeneracy": controls,
        "passed": bool(
            all(row["passed"] for row in (*rows, *placement))
            and controls["smooth_maximum_matches_reference"]
            and controls["numerical_maximum_derivative_unresolved"]
            and controls["zero_slope_position_rejected"]
            and controls["event_order_tie_rejected"]
            and controls["event_boundary_position_rejected"]
        ),
    }


def symmetry_and_degeneracy() -> dict[str, Any]:
    """Retain completed CAD attempts if a later invariant/degeneracy check fails."""
    observations: list[CadMeasurement] = []
    try:
        result = _symmetry_and_degeneracy(observations)
    except Exception as error:
        result = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "attempted_cad_observations": [asdict(row) for row in observations],
        }
    result["physical_calls"] = {
        "common": sum(row.common_calls for row in observations),
        "cut": sum(row.cut_calls for row in observations),
        "fuse": sum(row.fusion_calls for row in observations),
    }
    return result
