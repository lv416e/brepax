"""Actual source construction and counted CAD operations with a cooperative deadline.

Checks occur at named composite-stage boundaries. Preparation or an observation
can overrun inside one stage; the deadline does not interrupt an OCCT operation.
"""

from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from benchmarks.face_contact.cad import (
    boundary_tolerance,
    cad_volume,
    make_prism,
    transform_shape,
)
from brepax._occt.backend import (
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepBuilderAPI_Copy,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepCheck_Analyzer,
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakePrism,
    gp_Pnt,
    gp_Vec,
)
from brepax._occt.types import TopoDS_Shape

from .event_boundary_fixtures import Family, vertices
from .parametric_cad import CadMeasurement, CadScene, valid_observation

WALL_CAP = 60.0


class ResourceCapError(Exception):
    """A cooperative deadline or observation cap has been reached."""


@dataclass
class CadCallCounts:
    """Attempted source and validation operations, including failed operations."""

    common: int = 0
    cut: int = 0
    fuse: int = 0


def _boolean(
    operation_type: type[BRepAlgoAPI_Common]
    | type[BRepAlgoAPI_Cut]
    | type[BRepAlgoAPI_Fuse],
    fixed: TopoDS_Shape,
    moving: TopoDS_Shape,
) -> TopoDS_Shape:
    """Run the default Boolean on disposable deep copies, including geometry."""
    # OCCT's default mode may update input tolerances. Its non-destructive mode
    # returned an incorrect empty Common in a near-contact fixture, so isolate
    # the inputs instead of changing the Boolean algorithm's mode.
    fixed_copy = BRepBuilderAPI_Copy(fixed, True, False).Shape()
    moving_copy = BRepBuilderAPI_Copy(moving, True, False).Shape()
    operation = operation_type(fixed_copy, moving_copy)
    if not operation.IsDone() or operation.Shape().IsNull():
        raise RuntimeError("OCCT Boolean did not produce a shape")
    return operation.Shape()


def _box(points: NDArray[np.float64]) -> TopoDS_Shape:
    lower, upper = points.min(axis=0), points.max(axis=0)
    return BRepPrimAPI_MakeBox(
        gp_Pnt(*map(float, lower)), *map(float, upper - lower)
    ).Shape()


def _tetra(counts: CadCallCounts) -> TopoDS_Shape:
    # This oblique prism plus the unit box has exactly the four tetra supports.
    polygon = BRepBuilderAPI_MakePolygon()
    for point in ((0, 0, 0), (1, 0, 0), (0, 0, 1)):
        polygon.Add(gp_Pnt(*point))
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(-1, 1, 0)).Shape()
    unit = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    counts.common += 1
    return _boolean(BRepAlgoAPI_Common, unit, prism)


def _fuse(cells: tuple[TopoDS_Shape, ...], counts: CadCallCounts) -> TopoDS_Shape:
    result = cells[0]
    for cell in cells[1:]:
        counts.fuse += 1
        result = _boolean(BRepAlgoAPI_Fuse, result, cell)
    return result


def prepare_scene(
    family: Family, theta: Fraction, counts: CadCallCounts
) -> tuple[CadScene, float]:
    started = perf_counter()
    fixed_points, moving_points = vertices(family, theta)
    fixed: tuple[TopoDS_Shape, ...]
    if family.name.startswith("ramp"):
        fixed = (make_prism(((0, 0), (1, 0), (1, 1)), 0, 1),)
    elif family.name == "tetra_full":
        fixed = (_tetra(counts),)
    else:
        fixed = tuple(_box(points) for points in fixed_points)
    moving = tuple(_box(points) for points in moving_points)
    a, b = _fuse(fixed, counts), _fuse(moving, counts)
    valid = bool(BRepCheck_Analyzer(a).IsValid() and BRepCheck_Analyzer(b).IsValid())
    tolerances = [
        value
        for cell in (*fixed, *moving)
        if (value := boundary_tolerance(cell)) is not None
    ]
    scene = CadScene(
        fixed,
        moving,
        a,
        b,
        np.asarray(family.velocity),
        1.0,
        perf_counter() - started,
        counts.fuse,
        valid,
        max(tolerances, default=None),
    )
    return scene, cad_volume(a)


def check_deadline(started: float, stage: str) -> None:
    if perf_counter() - started >= WALL_CAP:
        raise ResourceCapError(f"Cooperative deadline reached before {stage}")


def observe_scene(
    scene: CadScene,
    step: float,
    *,
    cut: bool = False,
    deadline_started: float | None = None,
) -> CadMeasurement:
    """Observe fresh Boolean results while preserving the scene for later steps."""
    if deadline_started is not None:
        check_deadline(deadline_started, "observation")
    started = perf_counter()
    record = CadMeasurement(
        step, source_valid=scene.source_valid, source_tolerance=scene.source_tolerance
    )
    try:
        if not np.isfinite(step) or step < 0:
            raise ValueError("Expected finite nonnegative step")
        if not scene.source_valid:
            raise RuntimeError("Source solid is invalid")
        if deadline_started is not None:
            check_deadline(deadline_started, "observation transform")
        moving = transform_shape(scene.moving, np.eye(3), step * scene.velocity)
        if deadline_started is not None:
            check_deadline(deadline_started, "Common")
        common_started = perf_counter()
        record.common_calls += 1
        common = _boolean(BRepAlgoAPI_Common, scene.fixed, moving)
        record.common_seconds = perf_counter() - common_started
        record.common_valid = bool(BRepCheck_Analyzer(common).IsValid())
        record.volume = cad_volume(common)
        record.working_volume = record.volume / scene.scale**3
        record.boundary_tolerance = boundary_tolerance(common)
        if cut and valid_observation(record):
            if deadline_started is not None:
                check_deadline(deadline_started, "Cut")
            cut_started = perf_counter()
            record.cut_calls += 1
            removed = _boolean(BRepAlgoAPI_Cut, scene.fixed, moving)
            record.cut_seconds = perf_counter() - cut_started
            record.cut_valid = bool(BRepCheck_Analyzer(removed).IsValid())
            record.cut_volume = cad_volume(removed)
            record.working_balance_error = (
                cad_volume(scene.fixed) - record.cut_volume - record.volume
            ) / scene.scale**3
    except ResourceCapError as error:
        if record.common_calls == 0:
            raise
        record.error = f"{type(error).__name__}: {error}"
    except Exception as error:
        record.error = f"{type(error).__name__}: {error}"
    record.total_seconds = perf_counter() - started
    return record
