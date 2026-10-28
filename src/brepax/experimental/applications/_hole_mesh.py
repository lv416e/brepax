"""Prepare a closed shared-vertex discretization and coupled hole motion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from brepax.experimental.applications._hole_geometry import ThroughHole


def closed_mesh(
    triangles: NDArray[np.float64], tolerance: float
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Weld only coincident samples; reject missing or inconsistent boundaries.

    Neighbor buckets avoid rounding-boundary cracks without a spatial-tree
    dependency. Lexicographic traversal is deterministic for fixed input.
    This cannot repair mismatched edge tessellations or fill missing faces.
    """
    if not triangles.size or not np.isfinite(triangles).all():
        raise ValueError("Expected a finite nonempty mesh")
    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    buckets: dict[tuple[int, ...], list[int]] = {}
    vertices: list[NDArray[np.float64]] = []
    ids = np.empty(len(points), dtype=np.int64)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for i, point in enumerate(points):
        cell = tuple(np.floor(point / tolerance).astype(np.int64))
        candidates = [
            index
            for offset in offsets
            for index in buckets.get(
                tuple(a + b for a, b in zip(cell, offset, strict=True)), ()
            )
            if np.linalg.norm(point - vertices[index]) <= tolerance
        ]
        if candidates:
            ids[i] = min(candidates)
        else:
            ids[i] = len(vertices)
            buckets.setdefault(cell, []).append(len(vertices))
            vertices.append(point)
    faces = ids[inverse].reshape(-1, 3)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, labels, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    orientation = np.bincount(labels, weights=np.sign(edges[:, 1] - edges[:, 0]))
    if (
        np.any(counts != 2)
        or np.any(orientation != 0)
        or np.any(edges[:, 0] == edges[:, 1])
    ):
        raise ValueError(
            "Mesh is not closed and consistently oriented at the weld tolerance"
        )
    return np.asarray(vertices), faces


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
