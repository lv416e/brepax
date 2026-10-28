"""Hybrid optimization combining smoothing and stratum-aware gradients.

Defines an experimental interface for a proposed coarse-to-fine schedule.
The optimizer is not implemented. See ADR-0011 for the design rationale.

Smoothing may supply exploration signals; the intersecting Method (C)
gradient remains a grid estimate, not an exact refinement guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float, PyTree


@dataclass(frozen=True)
class HybridSchedule:
    """Configuration for when to switch from exploration to refinement.

    Attributes:
        switch_criterion: How to decide when to switch methods.
            - "steps": Switch after a fixed number of exploration steps.
            - "boundary_distance": Switch when boundary distance drops below threshold.
            - "loss_plateau": Switch when exploration loss improvement stalls.
        explore_steps: Number of exploration steps (for "steps" criterion).
        boundary_threshold: Distance threshold (for "boundary_distance" criterion).
        plateau_window: Window size for plateau detection
            (for "loss_plateau" criterion).
    """

    switch_criterion: Literal["steps", "boundary_distance", "loss_plateau"] = "steps"
    explore_steps: int = 100
    boundary_threshold: float = 0.1
    plateau_window: int = 20


@dataclass
class HybridResult:
    """Result of hybrid optimization.

    Attributes:
        params: Final optimized parameters.
        trajectory: Parameter values at each step.
        losses: Loss values at each step.
        method_log: Which method was active at each step ("smoothing" or "stratum").
        stratum_transitions: Steps where stratum label changed.
        converged: Whether the optimization converged within tolerance.
    """

    params: PyTree
    trajectory: list[PyTree]
    losses: list[float]
    method_log: list[str]
    stratum_transitions: list[int]
    converged: bool


def hybrid_optimize(
    objective: Callable[[PyTree], Float[Array, ""]],
    params_init: PyTree,
    *,
    explore_method: Literal["smoothing"] = "smoothing",
    refine_method: Literal["stratum"] = "stratum",
    schedule: HybridSchedule | None = None,
    lr: float = 0.01,
    max_steps: int = 1000,
    tol: float = 1e-6,
) -> HybridResult:
    """Describe the proposed hybrid strategy; always raise NotImplementedError.

    The proposed exploration stage uses smoothing; refinement switches to
    relation-dispatched gradients. Neither cross-boundary progress nor exact
    convergence is established by this interface.

    Args:
        objective: Scalar objective function to minimize.
        params_init: Initial parameter values (any PyTree).
        explore_method: Method for cross-stratum exploration.
        refine_method: Method for within-stratum refinement.
        schedule: When to switch from exploration to refinement.
        lr: Learning rate for gradient descent.
        max_steps: Maximum total optimization steps.
        tol: Convergence tolerance on loss value.

    Returns:
        HybridResult with optimization trajectory and diagnostics.

    Raises:
        NotImplementedError: Always (not yet implemented).
    """
    raise NotImplementedError(
        "Hybrid optimizer is not yet implemented. "
        "See ADR-0011 and docs/explanation/hybrid_optimization_strategy.md "
        "for design rationale. Use manual method switching in the meantime."
    )
