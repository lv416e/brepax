"""Finite polyhedral sockets and supplied cutter motions in working units.

Each stock is partitioned into convex boxes with disjoint interiors. The
partition permits complete component-pair contact checks; the independently
rebuilt stock is fused before volume and Cut observations. Every base design
is one evaluation group, including all its placements, motions, and amounts.
"""

from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

import numpy as np

from benchmarks.face_contact.cad import Motion, Placement

Vector3 = tuple[float, float, float]
Family = Literal["corner2", "corner3", "slot2", "supports2"]
FAMILIES: tuple[Family, ...] = ("corner2", "corner3", "slot2", "supports2")
FIXTURE_SEED = 20261029
ACTION_STEPS = (0.005, 0.01, 0.02, 0.04)
TARGET_VOLUME_FRACTION = 0.01
DESIGN_GRID = 1.0 / 1024


@dataclass(frozen=True)
class Box:
    """Axis-aligned convex cell with strictly positive working dimensions."""

    lower: Vector3
    upper: Vector3


@dataclass(frozen=True)
class ContactPatchReference:
    """Prescribed common rectangle, independently checked by interval geometry.

    ``axis`` has zero extent and ``sign`` orients its normal from fixed stock
    toward the cutter. These design records score CAD preparation; they do
    not substitute for extracted contacts or certify supplied B-Rep geometry.
    """

    fixed_index: int
    moving_index: int
    axis: int
    sign: int
    lower: Vector3
    upper: Vector3


@dataclass(frozen=True)
class ContactFixture:
    """Finite stock/cutter unions with no initial volume overlap.

    ``length`` is the cutter's circumscribed radius about ``pivot``. It sets
    the initial maximum vertex-speed budget, not a guaranteed finite-path
    displacement bound. Lengths, translations, and pivot use working units.
    """

    group_id: str
    family: str
    fixed_cells: tuple[Box, ...]
    moving_cells: tuple[Box, ...]
    patches: tuple[ContactPatchReference, ...]
    pivot: Vector3
    length: float
    design_grid: float | None = None


@dataclass(frozen=True)
class NamedMotion:
    """A deterministic candidate identity and its working-coordinate twist."""

    name: str
    motion: Motion


@dataclass(frozen=True)
class CadPlacement:
    """A group-preserving actual B-Rep similarity, including its unit scale."""

    name: str
    transform: Placement


def _grid_point(values: Vector3) -> Vector3:
    rounded = np.rint(np.asarray(values) / DESIGN_GRID) * DESIGN_GRID
    return float(rounded[0]), float(rounded[1]), float(rounded[2])


def fixtures(
    variants: int = 5, *, grid_design: bool = True
) -> tuple[ContactFixture, ...]:
    """Four families with deterministic dimensions, independent of batch size.

    Corner sockets have two or three orthogonal patches. A slot has opposing
    wall contacts. Two supports have disconnected coplanar contact patches.
    Connector cells lie below the cutter and have positive clearance from it.
    All fixed cells have disjoint interiors, including at socket corners.
    The core design rounds all shared coordinates coherently to a 1/1024
    working-unit grid before constructing CAD. This defines representable
    intended contacts; it does not repair supplied source geometry. Set
    ``grid_design=False`` only for retained construction-roundoff controls.
    """
    if variants < 1:
        raise ValueError("At least one variant per family is required")
    result = []
    for family_index, family in enumerate(FAMILIES):
        random = np.random.default_rng(
            np.random.SeedSequence((FIXTURE_SEED, family_index))
        )
        for variant in range(variants):
            cells: tuple[Box, ...]
            patches: tuple[ContactPatchReference, ...]
            a, b, height = map(float, random.uniform(0.6, 1.5, 3))
            margin = 0.4
            tool = Box((-a, -b, 0.0), (a, b, height))
            floor = Box(
                (-a - margin, -b - margin, -0.75), (a + margin, b + margin, 0.0)
            )
            right = Box(
                (a, -b - margin, 0.0), (a + margin, b + margin, height + margin)
            )
            floor_patch = ContactPatchReference(0, 0, 2, 1, (-a, -b, 0.0), (a, b, 0.0))
            right_patch = ContactPatchReference(
                1, 0, 0, -1, (a, -b, 0.0), (a, b, height)
            )
            if family == "corner2":
                cells = (floor, right)
                patches = (floor_patch, right_patch)
            elif family == "corner3":
                # Partition the wall corner at x=a; do not double-count stock.
                back = Box((-a - margin, b, 0.0), (a, b + margin, height + margin))
                cells = (floor, right, back)
                patches = (
                    floor_patch,
                    right_patch,
                    ContactPatchReference(2, 0, 1, -1, (-a, b, 0.0), (a, b, height)),
                )
            else:
                base = Box(
                    (-a - margin, -b - margin, -1.25),
                    (a + margin, b + margin, -0.75),
                )
                if family == "slot2":
                    right = Box(
                        (a, -b - margin, -0.75),
                        (a + margin, b + margin, height + margin),
                    )
                    left = Box(
                        (-a - margin, -b - margin, -0.75),
                        (-a, b + margin, height + margin),
                    )
                    cells = (base, right, left)
                    patches = (
                        right_patch,
                        ContactPatchReference(
                            2, 0, 0, 1, (-a, -b, 0.0), (-a, b, height)
                        ),
                    )
                else:
                    left_width, right_width = map(
                        float, random.uniform(0.2, 0.35, 2) * a
                    )
                    left = Box(
                        (-0.6 * a - left_width, -b - margin, -0.75),
                        (-0.6 * a + left_width, b + margin, 0.0),
                    )
                    right = Box(
                        (0.6 * a - right_width, -b - margin, -0.75),
                        (0.6 * a + right_width, b + margin, 0.0),
                    )
                    cells = (base, left, right)
                    patches = (
                        ContactPatchReference(
                            1,
                            0,
                            2,
                            1,
                            (left.lower[0], -b, 0.0),
                            (left.upper[0], b, 0.0),
                        ),
                        ContactPatchReference(
                            2,
                            0,
                            2,
                            1,
                            (right.lower[0], -b, 0.0),
                            (right.upper[0], b, 0.0),
                        ),
                    )
            fixture = ContactFixture(
                f"{family}_{variant:02d}",
                family,
                cells,
                (tool,),
                patches,
                (0.0, 0.0, height / 2),
                float(np.sqrt(a * a + b * b + (height / 2) ** 2)),
            )
            if grid_design:
                fixed_cells = tuple(
                    Box(_grid_point(box.lower), _grid_point(box.upper))
                    for box in fixture.fixed_cells
                )
                moving_cells = tuple(
                    Box(_grid_point(box.lower), _grid_point(box.upper))
                    for box in fixture.moving_cells
                )
                tool_lower = np.asarray(moving_cells[0].lower)
                tool_upper = np.asarray(moving_cells[0].upper)
                center = (tool_lower + tool_upper) / 2
                fixture = replace(
                    fixture,
                    fixed_cells=fixed_cells,
                    moving_cells=moving_cells,
                    patches=tuple(
                        replace(
                            patch,
                            lower=_grid_point(patch.lower),
                            upper=_grid_point(patch.upper),
                        )
                        for patch in fixture.patches
                    ),
                    pivot=(float(center[0]), float(center[1]), float(center[2])),
                    length=float(np.linalg.norm((tool_upper - tool_lower) / 2)),
                    design_grid=DESIGN_GRID,
                )
            result.append(fixture)
    return tuple(result)


def candidate_motions(fixture: ContactFixture) -> tuple[NamedMotion, ...]:
    """Twenty-four fixed directions with the same initial vertex-speed budget.

    Twelve signed coordinate twists, eight mixed twists, and four XZ diagonal
    translations are retained in this order. Scale each entire twist so its
    maximum initial cutter-vertex speed equals ``fixture.length``. Angles are
    radians per dimensionless time. Every method sees the same candidates.
    """
    raw = []
    for axis in range(6):
        for sign in (-1, 1):
            components = np.zeros(6)
            components[axis] = sign
            raw.append((f"axis_{axis}_{sign:+d}", components))
    for index, angle in enumerate(np.arange(8) * np.pi / 4):
        raw.append(
            (
                f"mixed_{index}",
                np.array(
                    (0.4, -0.2, -0.3, 0.6 * np.cos(angle), 0.6 * np.sin(angle), 0.15)
                ),
            )
        )
    for index, (x, z) in enumerate(product((-1, 1), repeat=2)):
        raw.append((f"diagonal_{index}", np.array((x, 0, z, 0, 0, 0))))
    corners = np.array(
        [
            point
            for box in fixture.moving_cells
            for point in product(*zip(box.lower, box.upper, strict=True))
        ]
    )
    result = []
    for name, components in raw:
        velocity, angular = components[:3], components[3:]
        speeds = velocity + np.cross(angular, corners - np.asarray(fixture.pivot))
        factor = fixture.length / np.max(np.linalg.norm(speeds, axis=1))
        v = factor * velocity
        w = factor * angular
        scaled_velocity = (float(v[0]), float(v[1]), float(v[2]))
        scaled_angular = (float(w[0]), float(w[1]), float(w[2]))
        result.append(
            NamedMotion(name, Motion(scaled_velocity, scaled_angular, fixture.pivot))
        )
    return tuple(result)


def placements() -> tuple[CadPlacement, ...]:
    """Actual translated/rotated CAD at three moderate working-to-world scales."""
    return (
        CadPlacement("identity", Placement()),
        CadPlacement("small_rigid", Placement(0.1, (0.3, -0.2, 0.4), (0.2, 0.4, -0.3))),
        CadPlacement(
            "large_rigid", Placement(10.0, (30.0, -20.0, 40.0), (-0.3, 0.1, 0.4))
        ),
    )
