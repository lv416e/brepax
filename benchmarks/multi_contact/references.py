"""Independent interval oracle for translating unions of axis-aligned boxes.

Each body's boxes must have pairwise disjoint interiors. Shared faces, edges,
and vertices are allowed. The bodies can overlap for finite-volume queries,
but the contact-direction query requires zero initial overlap. Inputs use one
length unit; time is dimensionless and velocity has length units.

There is no geometric tolerance or snapping: classification is for the supplied
float64 coordinates. This is independent of OCCT and the planar-contact kernels.
It does not certify an intended gap lost while constructing those coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from benchmarks.face_contact.references import positive_affine_reference

if TYPE_CHECKING:
    from benchmarks.face_contact.cad import Motion

    from .fixtures import ContactFixture, ContactPatchReference

RelationKind = Literal["gap", "face", "edge", "vertex", "overlap"]


@dataclass(frozen=True)
class BoxRelation:
    """One component pair, including lower-dimensional and separated cases.

    Signed interval widths are positive in overlap, zero at contact, and
    negative across a gap. A face normal points from fixed toward moving.
    Face area and normal are absent for all other relation kinds.
    """

    fixed_index: int
    moving_index: int
    kind: RelationKind
    signed_widths: tuple[float, float, float]
    normal: tuple[float, float, float] | None = None
    face_area: float | None = None


def _boxes(
    lower: ArrayLike, upper: ArrayLike, name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.ndim != 2 or lo.shape[1] != 3 or hi.shape != lo.shape:
        raise ValueError(f"{name} bounds require matching (N, 3) arrays")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError(f"{name} bounds must be finite")
    with np.errstate(over="ignore"):
        widths = hi - lo
    if not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError(f"{name} boxes need finite positive represented widths")
    for i, j in combinations(range(len(lo)), 2):
        if np.all(np.maximum(lo[i], lo[j]) < np.minimum(hi[i], hi[j])):
            raise ValueError(f"{name} boxes {i} and {j} have overlapping interiors")
    return lo, hi


def _vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires a finite three-vector")
    return result


def box_relations(
    fixed_lower: ArrayLike,
    fixed_upper: ArrayLike,
    moving_lower: ArrayLike,
    moving_upper: ArrayLike,
) -> tuple[BoxRelation, ...]:
    """Classify every source component pair, without discarding gap/control rows."""
    flo, fhi = _boxes(fixed_lower, fixed_upper, "Fixed")
    mlo, mhi = _boxes(moving_lower, moving_upper, "Moving")
    rows = []
    for i in range(len(flo)):
        for j in range(len(mlo)):
            with np.errstate(over="ignore"):
                width = np.minimum(fhi[i], mhi[j]) - np.maximum(flo[i], mlo[j])
            if not np.isfinite(width).all():
                raise ValueError("Pairwise interval separation exceeds float64 range")
            values = (float(width[0]), float(width[1]), float(width[2]))
            if np.any(width < 0):
                rows.append(BoxRelation(i, j, "gap", values))
                continue
            zeros = np.flatnonzero(width == 0)
            if len(zeros) != 1:
                kind: RelationKind
                if len(zeros) == 0:
                    kind = "overlap"
                elif len(zeros) == 2:
                    kind = "edge"
                else:
                    kind = "vertex"
                rows.append(BoxRelation(i, j, kind, values))
                continue
            axis = int(zeros[0])
            normal = np.zeros(3)
            normal[axis] = 1.0 if fhi[i, axis] == mlo[j, axis] else -1.0
            area = math.prod(float(width[k]) for k in range(3) if k != axis)
            if not math.isfinite(area):
                raise ValueError("Contact area exceeds float64 range")
            rows.append(
                BoxRelation(
                    i,
                    j,
                    "face",
                    values,
                    (float(normal[0]), float(normal[1]), float(normal[2])),
                    area,
                )
            )
    return tuple(rows)


def boxes_overlap_volume(
    fixed_lower: ArrayLike,
    fixed_upper: ArrayLike,
    moving_lower: ArrayLike,
    moving_upper: ArrayLike,
    translation: ArrayLike,
) -> float:
    """Volume of fixed union intersected with the translated moving union.

    Source-cell interior disjointness makes pairwise intersection interiors
    disjoint too, so summation needs no inclusion-exclusion correction. A rigid
    translation preserves that disjointness. Roundoff/underflow remain possible;
    nonfinite outputs and collapsed translated boxes raise ``ValueError``.
    """
    flo, fhi = _boxes(fixed_lower, fixed_upper, "Fixed")
    mlo, mhi = _boxes(moving_lower, moving_upper, "Moving")
    shift = _vector(translation, "Translation")
    with np.errstate(over="ignore"):
        moved_lower, moved_upper = mlo + shift, mhi + shift
    mlo, mhi = _boxes(moved_lower, moved_upper, "Translated moving")
    pieces = []
    for i in range(len(flo)):
        for j in range(len(mlo)):
            lo = np.maximum(flo[i], mlo[j])
            hi = np.minimum(fhi[i], mhi[j])
            if np.any(hi <= lo):
                continue
            pieces.append(math.prod(float(x) for x in hi - lo))
    try:
        result = math.fsum(pieces)
    except OverflowError as error:
        raise ValueError("Intersection volume exceeds float64 range") from error
    if not math.isfinite(result):
        raise ValueError("Intersection volume exceeds float64 range")
    return result


def contact_direction(
    fixed_lower: ArrayLike,
    fixed_upper: ArrayLike,
    moving_lower: ArrayLike,
    moving_upper: ArrayLike,
    velocity: ArrayLike,
) -> float:
    """Positive one-sided translational volume rate at initially disjoint interiors.

    A face contributes area times the positive velocity into the fixed body.
    Positive gaps contribute zero for sufficiently small positive time. Edge
    and vertex contacts produce at most quadratic/cubic new overlap, hence
    zero first-order response. Pre-existing cross-body overlap is rejected.
    """
    v = _vector(velocity, "Velocity")
    rows = box_relations(fixed_lower, fixed_upper, moving_lower, moving_upper)
    if any(row.kind == "overlap" for row in rows):
        raise ValueError("Contact direction requires zero initial interior overlap")
    terms = []
    for row in rows:
        if row.kind != "face":
            continue
        assert row.normal is not None and row.face_area is not None
        # Normals are signed coordinate axes, so this avoids unnecessary dot
        # products and makes the independently derived interval limit explicit.
        axis = next(k for k, component in enumerate(row.normal) if component)
        inward = -row.normal[axis] * float(v[axis])
        terms.append(row.face_area * max(inward, 0.0))
    try:
        result = math.fsum(terms)
    except OverflowError as error:
        raise ValueError("Directional response exceeds float64 range") from error
    if not math.isfinite(result):
        raise ValueError("Directional response exceeds float64 range")
    return result


@dataclass(frozen=True)
class ReferencePatch:
    """Prescribed rectangle in a local plane; lengths are working units."""

    vertices: NDArray[np.float64]
    origin: NDArray[np.float64]
    basis: NDArray[np.float64]


def patch_geometry(patch: ContactPatchReference) -> ReferencePatch:
    """Convert a prescribed axis-aligned patch to an oriented plane frame.

    The normal points from stock to cutter. This constructs design-reference
    geometry and does not inspect or certify the corresponding CAD faces.
    """
    lower = np.asarray(patch.lower, dtype=np.float64)
    upper = np.asarray(patch.upper, dtype=np.float64)
    origin = (lower + upper) / 2
    indices = ((patch.axis + 1) % 3, (patch.axis + 2) % 3)
    basis = np.eye(3)[:, indices].copy()
    basis[:, 1] *= patch.sign
    half = (upper - lower)[list(indices)] / 2
    vertices = np.array(
        (
            (-half[0], -half[1]),
            (half[0], -half[1]),
            (half[0], half[1]),
            (-half[0], half[1]),
        )
    )
    return ReferencePatch(vertices, origin, basis)


def directional_reference(
    fixture: ContactFixture, motion: Motion
) -> NDArray[np.float64]:
    """Independent section integrals for each prescribed contact rectangle.

    This uses neither CAD-extracted patches nor BRepAX's triangle kernel.
    The initial fixed/moving interiors are disjoint and every positive-area
    contact must appear in the fixture. The result has volume units per
    dimensionless time. Sum the returned patch values for the total response.
    """
    velocity = np.asarray(motion.velocity)
    angular = np.asarray(motion.angular_velocity)
    pivot = np.asarray(motion.pivot)
    answers = []
    for specification in fixture.patches:
        patch = patch_geometry(specification)
        normal = np.cross(patch.basis[:, 0], patch.basis[:, 1])
        constant = -normal @ (velocity + np.cross(angular, patch.origin - pivot))
        slope = patch.basis.T @ np.cross(angular, normal)
        answers.append(positive_affine_reference(patch.vertices, (constant, *slope)))
    return np.asarray(answers, dtype=np.float64)
