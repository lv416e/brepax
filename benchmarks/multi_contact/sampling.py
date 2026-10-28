"""Measure sampled multi-contact baselines without fitting on heldout shapes."""

from dataclasses import asdict
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from benchmarks.contact_transition.baselines import TEMPERATURES, Temperature
from benchmarks.contact_transition.measure import compile_timed

from .baselines import (
    Measure,
    Representation,
    SampledCase,
    estimators,
    prepare_case,
    sampling_capacity,
)
from .fixtures import candidate_motions, fixtures, placements

GAIN_CANDIDATES = (0.5, 1.0, 2.0, 4.0)
REPRESENTATIONS: tuple[Representation, ...] = ("exterior", "cell_min")
METHODS = ("stratum", "sigmoid", "smoothing")
DIRECTIONAL_FD_STEP = 0.005


def _temperatures(method: str) -> tuple[Temperature, ...]:
    if method == "stratum":
        return (Temperature(1.0, 1.0),)
    if method == "smoothing":
        return TEMPERATURES
    return tuple(
        Temperature(1.0, beta) for beta in sorted({t.beta_over_h for t in TEMPERATURES})
    )


def measure_directional_differences(
    forward: Measure,
    geometry: SampledCase,
    directions: NDArray[np.float64],
    *,
    base_value: float,
    temperature: Temperature,
) -> dict[str, Any]:
    """Probe the same sampled forward along each supplied positive direction.

    Reuse the already measured f(0). Each probe uses q=h*(v, omega), with
    h=0.005, and returns (f(q)-f(0))/h. The compiled forward is called in a
    synchronized loop to bound grid memory. Counts include attempted calls;
    a failed probe preserves earlier rates and leaves unattempted rates null.
    This finite-step nonlinear proxy is distinct from the AD linear rates.
    """
    rates: list[float | None] = [None] * len(directions)
    calls = 0
    error_message = None
    started = perf_counter()
    try:
        if (
            directions.ndim != 2
            or directions.shape[1] != 6
            or not np.isfinite(directions).all()
            or not np.isfinite(base_value)
        ):
            raise ValueError("Expected finite rigid directions and sampled f(0)")
        for index, direction in enumerate(directions):
            calls += 1
            value = float(
                forward(
                    jnp.asarray(DIRECTIONAL_FD_STEP * direction),
                    geometry,
                    temperature.k_over_h,
                    temperature.beta_over_h,
                )
            )
            rate = (value - base_value) / DIRECTIONAL_FD_STEP
            if not np.isfinite(value) or not np.isfinite(rate):
                raise FloatingPointError(
                    f"Nonfinite directional probe at index {index}"
                )
            rates[index] = rate
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
    return {
        "directional_fd_rates": rates,
        "directional_fd_seconds": perf_counter() - started,
        "directional_fd_step": DIRECTIONAL_FD_STEP,
        "directional_fd_forward_calls": calls,
        "directional_fd_error": error_message,
    }


def collect_sampled(variants: int = 5, resolution: int = 48) -> dict[str, Any]:
    """Retain gradients, timings and failures for every group and placement.

    Geometry preparation, compilation, first execution and warmed execution
    are separate. A row's gradient is of its fixed sampled integrand (or C's
    dispatched surrogate). Linear rates dot this gradient with each declared
    local twist. Exterior sigmoid/smoothing also use positive finite differences
    of the same forward at h=0.005. Selection and train-only gain/temperature
    calibration belong to evaluation; no exact response or endpoint truth is
    read here.
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Sampled comparison requires JAX x64")
    population = fixtures(variants)
    started = perf_counter()
    capacity = sampling_capacity(population)
    capacity_seconds = perf_counter() - started
    methods = estimators(resolution)
    q = jnp.zeros(6)
    rows = []
    runtimes = {}
    compile_errors = {}
    for representation in REPRESENTATIONS:
        example = prepare_case(
            population[0], representation=representation, capacity=capacity
        )
        compiled = {}
        for name in METHODS:
            key = f"{representation}/{name}"
            function = getattr(methods, name)
            arguments = (q, example, 1.0, 1.0)
            try:
                forward, forward_runtime = compile_timed(function, arguments)
                backward, backward_runtime = compile_timed(
                    jax.grad(function), arguments
                )
                compiled[name] = (forward, backward)
                runtimes[key] = {
                    "forward": asdict(forward_runtime),
                    "gradient": asdict(backward_runtime),
                }
            except Exception as error:
                compile_errors[key] = f"{type(error).__name__}: {error}"
        try:
            classify, classification_runtime = compile_timed(
                methods.classify, (q, example)
            )
            runtimes[f"{representation}/classification"] = asdict(
                classification_runtime
            )
        except Exception as error:
            compile_errors[f"{representation}/classification"] = (
                f"{type(error).__name__}: {error}"
            )
            classify = None
        for case in population:
            motions = candidate_motions(case)
            directions = np.asarray(
                [
                    (*candidate.motion.velocity, *candidate.motion.angular_velocity)
                    for candidate in motions
                ]
            )
            for placement in placements():
                row: dict[str, Any] = {
                    "group_id": case.group_id,
                    "family": case.family,
                    "placement_name": placement.name,
                    "representation": representation,
                    "candidate_names": [candidate.name for candidate in motions],
                    "records": [],
                    "error": None,
                }
                try:
                    started = perf_counter()
                    geometry = prepare_case(
                        case,
                        placement.transform,
                        representation=representation,
                        capacity=capacity,
                    )
                    row["prepare_seconds"] = perf_counter() - started
                    lower = np.asarray(geometry.lower_domain)
                    upper = np.asarray(geometry.upper_domain)
                    row["domain"] = (lower.tolist(), upper.tolist())
                    row["cell_width"] = float(
                        np.prod((upper - lower) / (resolution - 1)) ** (1 / 3)
                    )
                    row["stratum_label"] = (
                        None if classify is None else float(classify(q, geometry))
                    )
                    for name in METHODS:
                        for temperature in _temperatures(name):
                            record: dict[str, Any] = {
                                "method": name,
                                "k_over_h": temperature.k_over_h,
                                "beta_over_h": temperature.beta_over_h,
                                "error": None,
                            }
                            key = f"{representation}/{name}"
                            if name not in compiled:
                                record["error"] = compile_errors[key]
                                row["records"].append(record)
                                continue
                            forward, backward = compiled[name]
                            arguments = (
                                q,
                                geometry,
                                temperature.k_over_h,
                                temperature.beta_over_h,
                            )
                            try:
                                started = perf_counter()
                                value = float(forward(*arguments))
                                forward_seconds = perf_counter() - started
                                started = perf_counter()
                                gradient = np.asarray(backward(*arguments), dtype=float)
                                gradient_seconds = perf_counter() - started
                                rates = directions @ gradient
                                if (
                                    not np.isfinite(value)
                                    or not np.isfinite(gradient).all()
                                    or not np.isfinite(rates).all()
                                ):
                                    raise FloatingPointError(
                                        "Nonfinite sampled value, gradient or rate"
                                    )
                                record.update(
                                    value=value,
                                    gradient=gradient.tolist(),
                                    linear_rates=rates.tolist(),
                                    forward_seconds=forward_seconds,
                                    gradient_seconds=gradient_seconds,
                                )
                            except Exception as error:
                                record["error"] = f"{type(error).__name__}: {error}"
                            if representation == "exterior" and name != "stratum":
                                if record["error"] is None:
                                    record.update(
                                        measure_directional_differences(
                                            forward,
                                            geometry,
                                            directions,
                                            base_value=value,
                                            temperature=temperature,
                                        )
                                    )
                                else:
                                    record.update(
                                        directional_fd_rates=[None] * len(directions),
                                        directional_fd_seconds=0.0,
                                        directional_fd_step=DIRECTIONAL_FD_STEP,
                                        directional_fd_forward_calls=0,
                                        directional_fd_error=(
                                            "Base sampled evaluation failed"
                                        ),
                                    )
                            row["records"].append(record)
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
    failed_rows = sum(row["error"] is not None for row in rows)
    failed_records = sum(
        record["error"] is not None for row in rows for record in row["records"]
    )
    failed_directional_records = sum(
        record.get("directional_fd_error") is not None
        for row in rows
        for record in row["records"]
    )
    directional_forward_calls = sum(
        record.get("directional_fd_forward_calls", 0)
        for row in rows
        for record in row["records"]
    )
    return {
        "passed": (
            not compile_errors
            and failed_rows == 0
            and failed_records == 0
            and failed_directional_records == 0
        ),
        "passed_scope": (
            "All declared sampled evaluations completed with finite values and "
            "gradients; no accuracy or superiority gate."
        ),
        "resolution": resolution,
        "capacity": asdict(capacity),
        "capacity_seconds": capacity_seconds,
        "gain_candidates": GAIN_CANDIDATES,
        "temperature_candidates": [asdict(value) for value in TEMPERATURES],
        "compile_errors": compile_errors,
        "runtime": runtimes,
        "row_count": len(rows),
        "record_count": sum(len(row["records"]) for row in rows),
        "failed_row_count": failed_rows,
        "failed_record_count": failed_records,
        "failed_directional_record_count": failed_directional_records,
        "directional_fd_step": DIRECTIONAL_FD_STEP,
        "directional_fd_forward_calls": directional_forward_calls,
        "rows": rows,
        "contract": (
            "Dynamic geometry is freshly placed on a common frozen grid during "
            "pose AD. Exterior distance uses the union boundary; cell_min "
            "retains internal partition seams as an ablation. All gradients are "
            "linear proxies. Exterior sigmoid/smoothing also probe their own "
            "forward at h=0.005 along every candidate; these finite-step rates "
            "can be nonlinear in direction and cost one extra forward per probe. "
            "Method C's forward and stratum dispatch are unchanged. "
            "Gain/temperature selection must use calibration groups only. "
            "The synthetic shapes informed method development and do not establish "
            "generalization to external CAD."
        ),
    }
