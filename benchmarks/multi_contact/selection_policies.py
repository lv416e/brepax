"""Budgeted experimental selection policies.

Observations are available only through the query.

The grid comparator is a learned target-conditioned ordering followed by
alternating measured one-direction interpolation and exploration. Continuous
refinement uses observed brackets where present; unbracketed scaling is only
a bounded heuristic. Every returned action was queried. No monotonicity or
global optimum is promised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

STEPS = np.array([0.005, 0.01, 0.02, 0.04])
STEPS.setflags(write=False)
BUDGETS = (1, 2, 4, 8)
TARGETS = (0.0025, 0.01, 0.03)


@dataclass(frozen=True)
class Observation:
    direction: int
    step: float
    volume: float | None
    error: str | None = None
    seconds: float = 0.0
    common_calls: int = 0
    fusion_calls: int = 0


Query = Callable[[int, float], Observation]


def action_order(
    predictions: NDArray[np.float64], target: float
) -> list[tuple[int, int]]:
    losses = np.abs(predictions - target)
    indices = np.argsort(losses.ravel(), kind="stable")
    return [(int(i // len(STEPS)), int(i % len(STEPS))) for i in indices]


def learned_order(
    training_volumes: NDArray[np.float64], fraction: float
) -> list[tuple[int, int]]:
    """Minimize expected normalized target loss, using training groups only."""
    losses = np.abs(training_volumes - fraction) / fraction
    # An attempted failed endpoint contributes a bounded worst loss to training.
    scores = np.mean(np.where(np.isfinite(losses), np.minimum(losses, 10), 10), axis=0)
    return action_order(scores, 0.0)


def _loss(observation: Observation, target: float) -> float:
    if observation.error is not None or observation.volume is None:
        return float("inf")
    return abs(observation.volume - target)


def best_observation(
    observations: list[Observation], target: float
) -> Observation | None:
    finite = [o for o in observations if np.isfinite(_loss(o, target))]
    return min(finite, key=lambda o: _loss(o, target)) if finite else None


def grid_policy(
    order: list[tuple[int, int]],
    target: float,
    budget: int,
    query: Query,
    *,
    adaptive: bool,
) -> list[Observation]:
    observations: list[Observation] = []
    visited: set[tuple[int, int]] = set()
    for index in range(budget):
        candidate = None
        best = best_observation(observations, target)
        if adaptive and index % 2 == 1 and best is not None:
            remaining = [
                j for j in range(len(STEPS)) if (best.direction, j) not in visited
            ]
            if remaining:
                if best.volume is not None and best.volume > 0:
                    predicted = best.volume * STEPS / best.step
                    step_index = min(
                        remaining, key=lambda j: abs(predicted[j] - target)
                    )
                else:
                    step_index = max(remaining)
                candidate = (best.direction, step_index)
        if candidate is None:
            candidate = next(action for action in order if action not in visited)
        visited.add(candidate)
        observations.append(query(candidate[0], float(STEPS[candidate[1]])))
    return observations


def _next_amount(
    direction: int, observations: list[Observation], target: float
) -> float | None:
    points = [(0.0, 0.0)] + [
        (o.step, float(o.volume))
        for o in observations
        if o.direction == direction and o.volume is not None and o.error is None
    ]
    points = sorted(set(points))
    queried = [o.step for o in observations if o.direction == direction]
    brackets = [
        (a, b) for a, b in pairwise(points) if (a[1] - target) * (b[1] - target) < 0
    ]
    if brackets:
        left, right = min(brackets, key=lambda pair: pair[1][0] - pair[0][0])
        value = left[0] + (target - left[1]) * (right[0] - left[0]) / (
            right[1] - left[1]
        )
        # End-point stalls are prevented without claiming monotone volume.
        width = right[0] - left[0]
        value = float(np.clip(value, left[0] + 0.05 * width, right[0] - 0.05 * width))
    else:
        useful = [p for p in points if p[0] > 0 and p[1] > 0]
        if useful:
            step, volume = min(useful, key=lambda p: abs(p[1] - target))
            value = float(np.clip(step * target / volume, 1e-6, STEPS[-1]))
        else:
            value = float(STEPS[-1])
    if any(abs(value - step) <= 1e-12 for step in queried):
        return None
    return value


def refine_policy(
    order: list[tuple[int, int]],
    target: float,
    budget: int,
    query: Query,
    *,
    rates: NDArray[np.float64] | None,
) -> list[Observation]:
    """Two-direction exploration followed by measured best-direction refinement.

    Both methods use the identical observation update. Analytic rates only select
    and initialize candidate directions. CAD-only starts from learned grid actions.
    Unbracketed proposals and exhausted intervals remain explicit trial actions.
    """
    seeds: list[tuple[int, int]] = []
    for direction, step_index in order:
        if direction not in {d for d, _ in seeds}:
            seeds.append((direction, step_index))
    observations: list[Observation] = []
    explored: set[int] = set()
    for index in range(budget):
        candidate: tuple[int, float] | None = None
        best = best_observation(observations, target)
        if index >= 2 and best is not None:
            amount = _next_amount(best.direction, observations, target)
            if amount is not None:
                candidate = (best.direction, amount)
        if candidate is None:
            direction, step_index = next((d, s) for d, s in seeds if d not in explored)
            explored.add(direction)
            if rates is not None and rates[direction] > 0:
                amount = float(np.clip(target / rates[direction], 1e-6, STEPS[-1]))
            else:
                amount = float(STEPS[step_index])
            candidate = direction, amount
        observations.append(query(*candidate))
    return observations
