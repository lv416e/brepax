"""Synchronized numerical measurements; output rendering belongs to the CLI."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from numpy.typing import NDArray

from .baselines import Measure, Temperature


@dataclass(frozen=True)
class Runtime:
    """Compilation, first execution, and warmed execution are separate costs."""

    lower_compile_seconds: float
    first_execution_seconds: float
    warm_median_seconds: float
    warm_min_seconds: float
    repetitions: int


@dataclass(frozen=True)
class CompiledMeasure:
    forward: Measure
    backward: Measure
    forward_runtime: Runtime
    backward_runtime: Runtime


def compile_timed(
    function: Callable[..., Any], example: tuple[Any, ...], repetitions: int = 5
) -> tuple[Callable[..., Any], Runtime]:
    """Compile once, synchronize every measurement, retain setup overhead."""
    if repetitions < 1:
        raise ValueError("At least one timing repetition is required")
    started = time.perf_counter()
    executable = jax.jit(function).lower(*example).compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jax.block_until_ready(executable(*example))  # type: ignore[no-untyped-call]
    first_seconds = time.perf_counter() - started
    elapsed = []
    for _ in range(repetitions):
        started = time.perf_counter()
        jax.block_until_ready(executable(*example))  # type: ignore[no-untyped-call]
        elapsed.append(time.perf_counter() - started)
    return executable, Runtime(
        compile_seconds,
        first_seconds,
        statistics.median(elapsed),
        min(elapsed),
        repetitions,
    )


def compile_measure(function: Measure, example: Array) -> CompiledMeasure:
    """Use dynamic temperatures so sweeps reuse each compiled computation."""
    arguments = (example, jnp.asarray(1.0), jnp.asarray(1.0))
    forward, forward_runtime = compile_timed(function, arguments)
    backward, backward_runtime = compile_timed(jax.grad(function), arguments)
    return CompiledMeasure(forward, backward, forward_runtime, backward_runtime)


def evaluate(
    compiled: CompiledMeasure,
    parameters: Array,
    temperature: Temperature,
    reference_value: float,
    reference_gradient: NDArray[np.float64] | None,
) -> dict[str, Any]:
    """Keep failures visible and JSON-finite without substituting fake values."""
    try:
        arguments = (parameters, temperature.k_over_h, temperature.beta_over_h)
        value = float(compiled.forward(*arguments))
        gradient = np.atleast_1d(np.asarray(compiled.backward(*arguments), dtype=float))
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return {
                "finite": False,
                "value": value if np.isfinite(value) else None,
                "gradient": [float(g) if np.isfinite(g) else None for g in gradient],
                "error": "Nonfinite sampled value or gradient",
            }
        result: dict[str, Any] = {
            "finite": True,
            "value": value,
            "gradient": gradient.tolist(),
            "reference_value": reference_value,
            "value_absolute_error": abs(value - reference_value),
        }
        if reference_gradient is None:
            result["reference_gradient"] = None
            result["gradient_error"] = None
            result["gradient_contract"] = (
                "No classical gradient; compare one-sided slopes"
            )
        else:
            difference = gradient - reference_gradient
            result.update(
                reference_gradient=reference_gradient.tolist(),
                gradient_absolute_error=np.abs(difference).tolist(),
                gradient_l2_error=float(np.linalg.norm(difference)),
            )
        return result
    except Exception as error:
        return {"finite": False, "error": f"{type(error).__name__}: {error}"}


def forward_differences(
    compiled: CompiledMeasure,
    parameters: Array,
    temperature: Temperature,
    steps: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6),
) -> list[dict[str, Any]]:
    """Differentiate the same forward; no silent comparison to a different CAD."""
    points = np.asarray(parameters, dtype=float)
    flat = np.atleast_1d(points)
    directions = np.eye(flat.size).reshape((flat.size, *points.shape))
    k, beta = temperature.k_over_h, temperature.beta_over_h
    primal = float(compiled.forward(parameters, k, beta))
    if not np.isfinite(primal):
        raise FloatingPointError("Nonfinite finite-difference primal")
    rows = []
    for step in steps:
        central, left, right = [], [], []
        for direction in directions:
            plus = float(
                compiled.forward(jnp.asarray(points + step * direction), k, beta)
            )
            minus = float(
                compiled.forward(jnp.asarray(points - step * direction), k, beta)
            )
            if not np.isfinite(plus) or not np.isfinite(minus):
                raise FloatingPointError(
                    f"Nonfinite finite-difference probe at step {step}"
                )
            central.append((plus - minus) / (2 * step))
            left.append((primal - minus) / step)
            right.append((plus - primal) / step)
        if not np.isfinite(np.asarray([central, left, right])).all():
            raise FloatingPointError(
                f"Nonfinite finite-difference slope at step {step}"
            )
        rows.append({"step": step, "central": central, "left": left, "right": right})
    return rows


def radius_trajectory(
    function: Measure,
    temperature: Temperature,
    target_volume: float,
    reference: Callable[[float], float],
    *,
    initial_radius: float = 1.0,
    target_radius: float = 1.7,
    center: tuple[float, float] = (1.5, 0.2),
    height: float = 2.0,
    steps: int = 100,
    learning_rate: float = 0.1,
    warm_budgets: tuple[float, ...] = (0.02, 0.05),
) -> dict[str, Any]:
    """Fixed-step inverse volume experiment with identical solver settings.

    Optimize radius/initial_radius; loss is normalized by the initial removed
    volume pi*r0**2*height. Bounds [0.2,2.5]*r0 do not reveal the target. No
    early stop uses surrogate loss. Warm budgets include synchronized updates
    and host scalar transfer, exclude compilation and independent validation;
    reaching the iteration cap before a budget is reported explicitly.
    """
    volume_scale = np.pi * initial_radius**2 * height

    def update(q: Array, k: float, beta: float) -> tuple[Array, Array, Array]:
        def loss(relative_radius: Array) -> tuple[Array, Array]:
            radius = relative_radius * initial_radius
            parameters = jnp.array([center[0], center[1], radius])
            volume = function(parameters, k, beta)
            return ((volume - target_volume) / volume_scale) ** 2 / 2, volume

        (loss_value, volume), gradient = jax.value_and_grad(loss, has_aux=True)(q)
        next_q = jnp.clip(q - learning_rate * gradient, 0.2, 2.5)
        return next_q, loss_value, volume

    arguments = (jnp.asarray(1.0), temperature.k_over_h, temperature.beta_over_h)
    executable, timing = compile_timed(update, arguments)
    q = jnp.asarray(1.0)
    trajectory: list[dict[str, Any]] = []
    warm_seconds = 0.0
    for iteration in range(steps):
        radius = float(q) * initial_radius
        started = time.perf_counter()
        next_q, loss, estimated_volume = jax.block_until_ready(  # type: ignore[no-untyped-call]
            executable(q, temperature.k_over_h, temperature.beta_over_h)
        )
        next_radius = float(next_q) * initial_radius
        loss_value, estimated = float(loss), float(estimated_volume)
        warm_seconds += time.perf_counter() - started
        if not all(np.isfinite(x) for x in (next_radius, loss_value, estimated)):
            return {
                "finite": False,
                "error": "Nonfinite optimizer state",
                "completed_updates": iteration,
                "trajectory": trajectory,
                "update_runtime": asdict(timing),
            }
        trajectory.append(
            {
                "iteration": iteration,
                "radius": radius,
                "next_radius": next_radius,
                "surrogate_loss": loss_value,
                "estimated_volume": estimated,
                "warm_seconds": warm_seconds,
            }
        )
        q = next_q
    validation_started = time.perf_counter()
    for point in trajectory:
        next_radius = float(point["next_radius"])
        point["true_next_volume_error"] = abs(reference(next_radius) - target_volume)
        point["next_radius_error"] = abs(next_radius - target_radius)
    final_radius = float(q) * initial_radius
    final_volume = reference(final_radius)
    snapshots = []
    for budget in warm_budgets:
        eligible = [r for r in trajectory if r["warm_seconds"] <= budget]
        selected_radius = (
            float(eligible[-1]["next_radius"]) if eligible else initial_radius
        )
        snapshots.append(
            {
                "budget_seconds": budget,
                "completed_updates": len(eligible),
                "iteration_cap_reached_before_budget": warm_seconds < budget,
                "radius": selected_radius,
                "radius_error": abs(selected_radius - target_radius),
                "true_volume_error": abs(reference(selected_radius) - target_volume),
            }
        )
    return {
        "finite": True,
        "initial_radius": initial_radius,
        "target_radius": target_radius,
        "target_volume": target_volume,
        "learning_rate": learning_rate,
        "completed_updates": steps,
        "final_radius": final_radius,
        "radius_error": abs(final_radius - target_radius),
        "true_volume_error": abs(final_volume - target_volume),
        "warm_seconds": warm_seconds,
        "independent_validation_seconds": time.perf_counter() - validation_started,
        "update_runtime": asdict(timing),
        "warm_budget_snapshots": snapshots,
        "trajectory": trajectory,
    }
