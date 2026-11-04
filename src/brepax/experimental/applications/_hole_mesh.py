"""Coupled hole motion fields on a closed shared-vertex discretization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from brepax.experimental.applications._hole_geometry import ThroughHole


@dataclass(frozen=True)
class HoleFields:
    radial: NDArray[np.float64]
    translation: NDArray[np.float64]


def displacement_fields(
    vertices: NDArray[np.float64], holes: tuple[ThroughHole, ...], tolerance: float
) -> HoleFields:
    fields = []
    weights = []
    for hole in holes:
        axis = np.asarray(hole.axis)
        delta = vertices - hole.base
        axial = delta @ axis
        radial = delta - axial[:, None] * axis
        distance = np.linalg.norm(radial, axis=-1)
        # Disjoint support (less than half the clearance) moves the planar
        # annulus without reaching any other face or another hole's support.
        support = 0.4 * hole.clearance
        weight = np.clip((hole.radius + support - distance) / support, 0, 1)
        weight *= (axial >= -tolerance) & (axial <= hole.height + tolerance)
        direction = radial / np.maximum(distance[:, None], hole.radius * 1e-12)
        fields.append(weight[:, None] * direction)
        weights.append(weight)
    return HoleFields(np.asarray(fields), np.asarray(weights))
