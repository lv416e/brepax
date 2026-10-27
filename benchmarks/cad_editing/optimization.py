"""Bounded inverse design on the supplied roof/hole mesh, in working units."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from brepax.brep.triangulate import divergence_volume, mesh_surface_area

from .geometry import cad_measures, in_edit_box, make_shape
from .mesh import RoofMesh, check_mesh

RESIDUAL_TOLERANCE = 1e-10
DEFAULT_MAX_UPDATES = 8

EditStatus = Literal[
    "converged", "invalid_input", "iteration_limit", "no_descent", "evaluation_failed"
]


@dataclass(frozen=True)
class EditStep:
    """One evaluated design, including CAD evidence independent of the loss."""

    dimensions: list[float]
    mesh_residual: list[float]
    cad_residual: list[float]


@dataclass
class EditResult:
    """Solver termination is distinct from acceptance of the rebuilt geometry."""

    status: EditStatus
    steps: list[EditStep] = field(default_factory=list)
    error: str | None = None
    failed_dimensions: list[float] | None = None
    failed_mesh_residual: list[float] | None = None


class EditSolver:
    """Reuse compiled residuals/Jacobians across starts and positive [V, A] targets.

    The target is in working volume/area units, and dimensions are working
    lengths. Only this fixed-topology family and the declared edit box apply.
    A converged surrogate still needs independent final CAD verification.
    """

    def __init__(self, model: RoofMesh) -> None:
        self.model = model

        def residual(dimensions: jax.Array, target: jax.Array) -> jax.Array:
            triangles = model.triangles(dimensions)
            measures = jnp.stack(
                (divergence_volume(triangles), mesh_surface_area(triangles))
            )
            return measures / target - 1

        self.residual = jax.jit(residual)
        self.jacobian = jax.jit(jax.jacrev(residual, argnums=0))

    def solve(
        self,
        initial: NDArray[np.float64],
        target: NDArray[np.float64],
        *,
        max_updates: int = DEFAULT_MAX_UPDATES,
    ) -> EditResult:
        """Damped Newton with bounded candidates and an evaluated final iterate.

        Retain checked iterates and failed evaluation points, with explicit status.
        Rejected line-search trials are not part of the accepted trajectory.
        The 1e-10 residual stop is a solver tolerance, not a CAD accuracy claim.
        """
        result = EditResult("invalid_input")
        if (
            not in_edit_box(initial)
            or target.shape != (2,)
            or not np.isfinite(target).all()
            or np.any(target <= 0)
            or max_updates < 0
        ):
            result.error = (
                "Invalid initial dimensions, target measures, or update budget"
            )
            return result
        dimensions = initial.copy()
        measured_residual: list[float] | None = None
        try:
            for update in range(max_updates + 1):
                measured_residual = None
                residual = np.asarray(self.residual(dimensions, target))
                if not np.isfinite(residual).all():
                    raise FloatingPointError("Nonfinite mesh objective")
                measured_residual = residual.tolist()
                cad_residual = cad_measures(make_shape(dimensions)) / target - 1
                if not np.isfinite(cad_residual).all():
                    raise FloatingPointError("Nonfinite CAD objective")
                if not check_mesh(self.model, dimensions):
                    raise ValueError("Open, degenerate, or locally flipped edited mesh")
                result.steps.append(
                    EditStep(
                        dimensions.tolist(), residual.tolist(), cad_residual.tolist()
                    )
                )
                if np.max(abs(residual)) <= RESIDUAL_TOLERANCE:
                    result.status = "converged"
                    return result
                if update == max_updates:
                    result.status = "iteration_limit"
                    return result
                jacobian = np.asarray(self.jacobian(dimensions, target))
                if not np.isfinite(jacobian).all():
                    raise FloatingPointError("Nonfinite mesh Jacobian")
                delta = np.linalg.solve(jacobian, -residual)
                if not np.isfinite(delta).all():
                    raise FloatingPointError("Nonfinite Newton update")
                for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    candidate = dimensions + fraction * delta
                    if not in_edit_box(candidate):
                        continue
                    trial = np.asarray(self.residual(candidate, target))
                    if np.isfinite(trial).all() and np.linalg.norm(
                        trial
                    ) < np.linalg.norm(residual):
                        dimensions = candidate
                        break
                else:
                    result.status = "no_descent"
                    return result
        except Exception as error:
            # Preserve the trajectory when a numerical or CAD evaluation fails.
            result.status = "evaluation_failed"
            result.error = f"{type(error).__name__}: {error}"
            result.failed_dimensions = dimensions.tolist()
            result.failed_mesh_residual = measured_residual
        return result
