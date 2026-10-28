"""Measure sampled methods on the declared convex-prism population."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.contact_transition.measure import compile_timed
from brepax.experimental.planar_contact import (
    planar_contact_volume_direction,
    prepare_planar_contact_patch,
)

from . import cad
from .baselines import estimators, polygon_moments
from .fixtures import (
    CONTROL_RESOLUTIONS,
    MAIN_RESOLUTION,
    SAME_FORWARD_FD_STEPS,
    TEMPERATURES,
    WARM_REPETITIONS,
    audit_directions,
    fixtures,
    placements,
    selection_directions,
)
from .references import directional_reference


def collect_sampled(resolution: int = MAIN_RESOLUTION) -> dict[str, Any]:
    result: dict[str, Any] = {
        "resolution": resolution,
        "records": [],
        "runtime": [],
        "own_forward_generic_pose_checks": [],
        "cost_contract": {
            "setup": (
                "Grid and all compiled forward/backward programs "
                "recorded per static vertex count; reused across "
                "polygon variants and placements"
            ),
            "gradient_proxy": (
                "One six-coordinate gradient call plus dot products "
                "per fixture/temperature; C's internal dispatch "
                "includes its own quadrature work"
            ),
            "same_forward_scan": (
                "One shared primal and K scalar positive-step forward "
                "calls per finite candidate panel; controls reuse "
                "stored evaluations"
            ),
            "units": (
                "All sampled inputs normalized to working units; "
                "placement rotates geometry relative to a translated "
                "grid. Physical volume multiplies by scale cubed. Unit "
                "covariance is constructed here, tested independently "
                "by fresh physical CAD elsewhere."
            ),
        },
    }
    started = time.perf_counter()
    family = estimators(resolution)
    result["grid_setup_seconds"] = time.perf_counter() - started
    cache: dict[int, dict[str, Any]] = {}
    for fixture in fixtures():
        vertex_count = len(fixture.vertices)
        for placement in placements():
            try:
                args = (
                    jnp.zeros(6),
                    jnp.asarray(fixture.vertices),
                    jnp.asarray(fixture.gap),
                    jnp.zeros(3),
                    jnp.asarray(placement.rotation),
                    jnp.asarray(np.asarray(placement.origin) / placement.scale),
                    jnp.asarray(1.0),
                    jnp.asarray(1.0),
                )
                if vertex_count not in cache:
                    compiled: dict[str, Any] = {}
                    for method in ("stratum", "sigmoid", "smoothing"):
                        forward, forward_runtime = compile_timed(
                            getattr(family, method), args
                        )
                        backward, backward_runtime = compile_timed(
                            jax.grad(getattr(family, method)), args
                        )
                        compiled[method] = (forward, backward)
                        result["runtime"].append(
                            {
                                "vertices": vertex_count,
                                "method": method,
                                "forward": asdict(forward_runtime),
                                "backward": asdict(backward_runtime),
                            }
                        )
                    classify, runtime = compile_timed(family.classify, args[:6])
                    compiled["classify"] = classify
                    result["runtime"].append(
                        {
                            "vertices": vertex_count,
                            "method": "classify",
                            "runtime": asdict(runtime),
                        }
                    )
                    cache[vertex_count] = compiled
                compiled = cache[vertex_count]
                label = float(compiled["classify"](*args[:6]))
                for method in ("stratum", "sigmoid", "smoothing"):
                    forward, backward = compiled[method]
                    temperatures = (
                        ((1.0, 1.0),) if method == "stratum" else TEMPERATURES
                    )
                    for temperature in temperatures:
                        arguments = (*args[:6], *map(jnp.asarray, temperature))
                        row: dict[str, Any] = {
                            "group_id": fixture.group_id,
                            "family": fixture.family,
                            "placement": placement.name,
                            "method": method,
                            "temperature": list(temperature),
                            "label": label,
                            "scale": placement.scale,
                        }
                        try:
                            start = time.perf_counter()
                            primal = float(forward(*arguments))
                            gradient = np.asarray(backward(*arguments))
                            row["forward_gradient_seconds"] = (
                                time.perf_counter() - start
                            )
                            if not np.isfinite(primal) or not np.all(
                                np.isfinite(gradient)
                            ):
                                raise FloatingPointError(
                                    "Nonfinite sampled value or gradient"
                                )
                            row.update(
                                value=primal,
                                gradient=gradient.tolist(),
                                selection_forward={},
                            )
                            for direction in selection_directions():
                                vector = np.asarray(
                                    (*direction.translation, *direction.angular)
                                )
                                values = {}
                                for step in (0.02, 0.005):
                                    before = time.perf_counter()
                                    value = float(
                                        forward(
                                            jnp.asarray(step * vector), *arguments[1:]
                                        )
                                    )
                                    if not np.isfinite(value):
                                        raise FloatingPointError(
                                            "Nonfinite finite-step forward"
                                        )
                                    values[str(step)] = {
                                        "value": value,
                                        "positive_slope": (value - primal) / step,
                                        "seconds": time.perf_counter() - before,
                                    }
                                row["selection_forward"][direction.name] = values
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                        result["records"].append(row)
            except Exception as error:
                result["records"].append(
                    {
                        "group_id": fixture.group_id,
                        "placement": placement.name,
                        "stage": "compile_or_classify",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    # Exact-contact lattice/SDF ties do not supply a classical differentiability
    # control. Use one generic posed prism with actual sampled partial overlap.
    vertices = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    q = jnp.array([0.013, 0.027, -0.31, 0.011, -0.007, 0.019])
    arguments = (
        q,
        vertices,
        jnp.array(0.0),
        jnp.zeros(3),
        jnp.eye(3),
        jnp.zeros(3),
        jnp.array(1.0),
        jnp.array(1.0),
    )
    gradients: dict[str, Any] = {}
    for method in ("stratum", "sigmoid", "smoothing"):
        try:
            forward, backward = cache[4][method]
            gradient = np.asarray(backward(*arguments))
            if not np.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite own-forward gradient")
            gradients[method] = gradient
            check: dict[str, Any] = {
                "method": method,
                "label": float(cache[4]["classify"](*arguments[:6])),
                "gradient": gradient.tolist(),
                "central_differences": [],
            }
            for step in (1e-4, 1e-5, 1e-6):
                fd = []
                for axis_direction in np.eye(6):
                    plus = float(forward(q + step * axis_direction, *arguments[1:]))
                    minus = float(forward(q - step * axis_direction, *arguments[1:]))
                    if not np.isfinite([plus, minus]).all():
                        raise FloatingPointError("Nonfinite own-forward probe")
                    fd.append((plus - minus) / (2 * step))
                check["central_differences"].append(
                    {
                        "step": step,
                        "gradient": fd,
                        "max_abs_difference_to_ad": float(
                            np.max(np.abs(np.asarray(fd) - gradient))
                        ),
                    }
                )
            result["own_forward_generic_pose_checks"].append(check)
        except Exception as error:
            result["own_forward_generic_pose_checks"].append(
                {"method": method, "error": f"{type(error).__name__}: {error}"}
            )
    result["intersecting_C_same_sigmoid_max_abs_difference"] = (
        float(np.max(np.abs(gradients["stratum"] - gradients["sigmoid"])))
        if {"stratum", "sigmoid"} <= gradients.keys()
        else None
    )
    return result


def collect_analytic() -> dict[str, Any]:
    """Measure explicit directional evaluation against independent strip integration."""
    rows, runtime = [], []
    directions = (*audit_directions(), *selection_directions())
    velocities = jnp.asarray([d.translation for d in directions])
    angular = jnp.asarray([d.angular for d in directions])
    pivots = jnp.asarray([d.pivot for d in directions])
    cache: dict[int, Any] = {}
    for fixture in fixtures():
        try:
            vertices = np.asarray(fixture.vertices)
            before = time.perf_counter()
            patch = prepare_planar_contact_patch(
                vertices, origin=np.zeros(3), basis=np.eye(3)[:, :2]
            )
            prepare_seconds = time.perf_counter() - before
            arguments = (patch, velocities, angular, pivots)
            if len(vertices) not in cache:
                executable, timing = compile_timed(
                    jax.vmap(planar_contact_volume_direction, in_axes=(None, 0, 0, 0)),
                    arguments,
                )
                cache[len(vertices)] = executable
                runtime.append(
                    {
                        "vertices": len(vertices),
                        "directions_per_batch": len(directions),
                        "runtime": asdict(timing),
                    }
                )
            values = np.asarray(cache[len(vertices)](*arguments))
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite analytic directional response")
            area, centroid = polygon_moments(vertices)
            for direction, value in zip(directions, values, strict=True):
                velocity, omega, pivot = map(
                    np.asarray,
                    (direction.translation, direction.angular, direction.pivot),
                )
                vertex_velocity = velocity + np.cross(
                    omega, np.column_stack((vertices, np.zeros(len(vertices)))) - pivot
                )
                normal_speed = float(np.max(np.abs(vertex_velocity[:, 2])))
                full_speed = float(np.max(np.linalg.norm(vertex_velocity, axis=1)))
                normalizer = area * (normal_speed if normal_speed else full_speed)
                speed_centroid = -float(
                    (velocity + np.cross(omega, np.r_[centroid, 0.0] - pivot))[2]
                )
                reference = directional_reference(
                    vertices, direction.translation, direction.angular, direction.pivot
                )
                if not np.isfinite([reference, normalizer]).all() or normalizer <= 0:
                    raise FloatingPointError(
                        "Nonfinite reference or invalid normalization"
                    )
                rows.append(
                    {
                        "group_id": fixture.group_id,
                        "family": fixture.family,
                        "direction": direction.name,
                        "value": float(value),
                        "reference": reference,
                        "absolute_error": abs(float(value) - reference),
                        "normalization": normalizer,
                        "area": area,
                        "prepare_seconds": prepare_seconds,
                        "ablations": {
                            "translation_only_area": area
                            * max(-float(velocity[2]), 0.0),
                            "centroid_velocity_area": area * max(speed_centroid, 0.0),
                            "signed_linear_flux": area * speed_centroid,
                        },
                    }
                )
        except Exception as error:
            rows.extend(
                {
                    "group_id": fixture.group_id,
                    "direction": direction.name,
                    "error": f"{type(error).__name__}: {error}",
                }
                for direction in directions
            )
    return {
        "rows": rows,
        "runtime": runtime,
        "input_contract": (
            "Known complete exact contact patch; host patch "
            "preparation is timed, general contact detection is not "
            "performed"
        ),
    }


def collect_cad() -> dict[str, Any]:
    """Rebuild every endpoint for independent finite-step outcome and cost evidence."""
    started = time.perf_counter()
    baselines, rows = [], []
    steps = (0.02, 0.01, 0.005, 0.0025)
    for source in fixtures():
        fixture = cad.Fixture(
            source.vertices,
            source.gap,
            source.height,
            source.stock_bounds,
            source.stock_depth,
        )
        for source_placement in placements():
            placement = cad.placement_from_matrix(
                source_placement.scale,
                source_placement.origin,
                source_placement.rotation,
            )
            before = time.perf_counter()
            baseline = cad.volume_sample(fixture, cad.Motion(), 0.0, placement)
            baselines.append(
                {
                    "group_id": source.group_id,
                    "placement": source_placement.name,
                    **asdict(baseline),
                    "seconds": time.perf_counter() - before,
                }
            )
            for direction in selection_directions():
                motion = cad.Motion(
                    direction.translation, direction.angular, direction.pivot
                )
                for step in steps:
                    before = time.perf_counter()
                    sample = cad.volume_sample(fixture, motion, step, placement)
                    if (
                        sample.working_volume is not None
                        and baseline.working_volume is not None
                    ):
                        sample.working_difference = (
                            sample.working_volume - baseline.working_volume
                        ) / step
                    rows.append(
                        {
                            "group_id": source.group_id,
                            "family": source.family,
                            "placement": source_placement.name,
                            "direction": direction.name,
                            **asdict(sample),
                            "seconds": time.perf_counter() - before,
                            "baseline_working_volume": baseline.working_volume,
                            "baseline_error": baseline.error,
                        }
                    )
    return {
        "steps": steps,
        "common_call_count": len(baselines) + len(rows),
        "baseline_call_count": len(baselines),
        "endpoint_call_count": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "baselines": baselines,
        "rows": rows,
    }


def collect_grid_controls() -> dict[str, Any]:
    """Retain C's lattice-dependent contact value and same-forward differences.

    These are limitation measurements, not an assertion that a hard sampled
    forward or its custom gradient equals exact one-sided CAD volume response.
    """
    vertices = jnp.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    arguments = (
        jnp.zeros(6),
        vertices,
        jnp.array(0.0),
        jnp.zeros(3),
        jnp.eye(3),
        jnp.zeros(3),
        jnp.array(1.0),
        jnp.array(1.0),
    )
    directions = (
        ("approach", np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]), 4.0),
        ("separate", np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 0.0),
        ("tilt_positive", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 1.0),
        ("tilt_negative", np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]), 1.0),
    )
    rows = []
    for resolution in (MAIN_RESOLUTION, *CONTROL_RESOLUTIONS):
        row: dict[str, Any] = {"resolution": resolution, "exact_volume": 0.0}
        try:
            family = estimators(resolution)
            forward, runtime = compile_timed(
                family.stratum, arguments, WARM_REPETITIONS
            )
            gradient = np.asarray(jax.jit(jax.grad(family.stratum))(*arguments))
            value = float(forward(*arguments))
            if not np.isfinite(np.r_[value, gradient]).all():
                raise FloatingPointError("Nonfinite grid contact control")
            row.update(
                value=value,
                gradient=gradient.tolist(),
                runtime=asdict(runtime),
                label=float(family.classify(*arguments[:6])),
                directions=[],
            )
            for name, direction, reference in directions:
                differences = []
                for step in SAME_FORWARD_FD_STEPS:
                    plus = float(forward(jnp.asarray(step * direction), *arguments[1:]))
                    slope = (plus - value) / step
                    if not np.isfinite(slope):
                        raise FloatingPointError("Nonfinite grid positive difference")
                    differences.append({"step": step, "positive_slope": slope})
                row["directions"].append(
                    {
                        "name": name,
                        "exact_directional_response": reference,
                        "gradient_dot_direction": float(gradient @ direction),
                        "same_forward_differences": differences,
                    }
                )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return {
        "passed": not any(row.get("error") for row in rows),
        "passed_scope": (
            "All attempted grid limitation measurements completed with finite "
            "outputs; continuum accuracy is not a pass criterion"
        ),
        "rows": rows,
    }
