"""Deterministic finite-contact designs for budgeted edit selection.

All source coordinates are integers times 1/1024. The test and pilot cohorts
use independent deterministic streams. Each design, including every motion,
amount and target, is one indivisible evaluation group.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Literal

import numpy as np

from .fixtures import (
    Box,
    ContactFixture,
    ContactPatchReference,
)
from .references import box_relations

FAMILIES: tuple[str, ...] = ("partial_offset", "multicutter", "gap_event", "thin_stock")
GRID = 1.0 / 1024
SEED = 20261030


def _box(lower: tuple[int, int, int], upper: tuple[int, int, int]) -> Box:
    return Box(
        (float(lower[0] * GRID), float(lower[1] * GRID), float(lower[2] * GRID)),
        (float(upper[0] * GRID), float(upper[1] * GRID), float(upper[2] * GRID)),
    )


def _assemble(
    name: str,
    family: str,
    fixed: tuple[Box, ...],
    moving: tuple[Box, ...],
    pivot: tuple[int, int, int],
) -> ContactFixture:
    flo = np.asarray([x.lower for x in fixed])
    fhi = np.asarray([x.upper for x in fixed])
    mlo = np.asarray([x.lower for x in moving])
    mhi = np.asarray([x.upper for x in moving])
    relations = box_relations(flo, fhi, mlo, mhi)
    if any(row.kind == "overlap" for row in relations):
        raise ValueError(f"Initial overlap in generated design {name}")
    patches: list[ContactPatchReference] = []
    for row in relations:
        if row.kind != "face":
            continue
        assert row.normal is not None
        axis = next(i for i, value in enumerate(row.normal) if value != 0)
        lower = np.maximum(flo[row.fixed_index], mlo[row.moving_index])
        upper = np.minimum(fhi[row.fixed_index], mhi[row.moving_index])
        patches.append(
            ContactPatchReference(
                row.fixed_index,
                row.moving_index,
                axis,
                int(row.normal[axis]),
                (float(lower[0]), float(lower[1]), float(lower[2])),
                (float(upper[0]), float(upper[1]), float(upper[2])),
            )
        )
    origin = np.asarray(pivot, dtype=float) * GRID
    corners = np.asarray(
        [
            corner
            for cell in moving
            for corner in product(*zip(cell.lower, cell.upper, strict=True))
        ]
    )
    length = float(np.max(np.linalg.norm(corners - origin, axis=1)))
    return ContactFixture(
        name,
        family,
        fixed,
        moving,
        tuple(patches),
        (float(origin[0]), float(origin[1]), float(origin[2])),
        length,
        GRID,
    )


def fixtures(
    variants: int = 8, *, cohort: Literal["test", "pilot"] = "test"
) -> tuple[ContactFixture, ...]:
    """Generate independent designs without reading any finite CAD outcomes.

    partial_offset: partial asymmetric floor/wall support and offset rotation pivot.
    multicutter: two separated moving cells with floor and end-wall contacts.
    gap_event: two initial faces plus a deliberately gapped third wall.
    thin_stock: narrow floor support and thin wall, causing finite clipping changes.
    """
    if variants < 1 or cohort not in ("test", "pilot"):
        raise ValueError("Expected positive variants and test or pilot cohort")
    out: list[ContactFixture] = []
    for fi, family in enumerate(FAMILIES):
        rng = np.random.default_rng(
            np.random.SeedSequence((SEED, 0 if cohort == "test" else 1, fi))
        )
        for vi in range(variants):
            fixed: tuple[Box, ...]
            moving: tuple[Box, ...]
            a, b, h = map(int, rng.integers((650, 500, 650), (1350, 1250, 1500)))
            margin = int(rng.integers(180, 420))
            tool = _box((-a, -b, 0), (a, b, h))
            pivot = (
                int(rng.integers(-a // 3, a // 3)),
                int(rng.integers(-b // 3, b // 3)),
                h // 2,
            )
            if family == "partial_offset":
                x_end = int(rng.integers(-a // 3, a // 2))
                y_end = int(rng.integers(-b // 4, 3 * b // 4))
                fixed = (
                    _box((-a - margin, -b - margin, -512), (x_end, b + margin, 0)),
                    _box((a, -b, 0), (a + margin, y_end, h + margin)),
                )
                moving = (tool,)
            elif family == "multicutter":
                gap = int(rng.integers(128, 384))
                split = gap // 2
                moving = (
                    _box((-a, -b, 0), (-split, b, h)),
                    _box((split, -b, 0), (a, b, h)),
                )
                fixed = (
                    _box((-a - margin, -b - margin, -512), (a + margin, b + margin, 0)),
                    _box(
                        (-a - margin, b, 0), (a + margin, b + margin, int(3 * h // 5))
                    ),
                )
            elif family == "gap_event":
                gap = int(rng.integers(3, 26))
                moving = (tool,)
                fixed = (
                    _box((-a - margin, -b - margin, -512), (a + margin, b + margin, 0)),
                    _box((-a - margin, b, 0), (a, b + margin, h + margin)),
                    _box(
                        (a + gap, -b - margin, 0),
                        (a + gap + margin, b + margin, h + margin),
                    ),
                )
            else:
                thickness = int(rng.integers(4, 23))
                sliver = int(rng.integers(6, 29))
                moving = (tool,)
                fixed = (
                    _box(
                        (a - sliver, -b - margin, -thickness),
                        (a + margin, b + margin, 0),
                    ),
                    _box((a, -b - margin, 0), (a + thickness, b + margin, h)),
                )
            out.append(
                _assemble(f"{cohort}_{family}_{vi:02d}", family, fixed, moving, pivot)
            )
    return tuple(out)


def stock_volume_upper_bound(fixture: ContactFixture) -> Fraction:
    """Return an exact upper bound on material removable from the stock.

    The bound sums represented box volumes in cubic working units. It equals
    the stock volume when cell interiors are disjoint, as in these fixtures.
    Overlapping cells would make it conservative, not a union-volume oracle.
    Float coordinates are converted to their exact dyadic rational values
    before subtraction, so this bound does not use CAD tolerance or rounding.
    """
    total = Fraction(0)
    for cell in fixture.fixed_cells:
        volume = Fraction(1)
        for lower, upper in zip(cell.lower, cell.upper, strict=True):
            width = Fraction.from_float(upper) - Fraction.from_float(lower)
            if width <= 0:
                raise ValueError("Stock cells require positive represented dimensions")
            volume *= width
        total += volume
    return total
