"""Actual OCCT preparation, observations and independent finite-edit checks."""

from dataclasses import dataclass
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
    BRepCheck_Analyzer,
)
from brepax._occt.types import TopoDS_Shape

from .parametric_fixtures import PrismBodies


@dataclass(frozen=True)
class CadScene:
    """Prepared solid geometry only; no Boolean observations are cached."""

    fixed_cells: tuple[TopoDS_Shape, ...]
    moving_cells: tuple[TopoDS_Shape, ...]
    fixed: TopoDS_Shape
    moving: TopoDS_Shape
    velocity: NDArray[np.float64]
    scale: float
    construction_seconds: float
    fusion_calls: int
    source_valid: bool
    source_tolerance: float | None


@dataclass
class CadConstructionCounts:
    """Attempted source fusions, including attempts preceding preparation failure."""

    fusion_calls: int = 0


@dataclass
class CadMeasurement:
    """Every attempted Common/Cut remains visible, including failed operations."""

    step: float
    volume: float | None = None
    working_volume: float | None = None
    cut_volume: float | None = None
    working_balance_error: float | None = None
    source_valid: bool = False
    common_valid: bool = False
    cut_valid: bool | None = None
    source_tolerance: float | None = None
    boundary_tolerance: float | None = None
    construction_seconds: float = 0.0
    common_seconds: float = 0.0
    cut_seconds: float = 0.0
    total_seconds: float = 0.0
    fusion_calls: int = 0
    common_calls: int = 0
    cut_calls: int = 0
    error: str | None = None


def _union(
    cells: tuple[TopoDS_Shape, ...], counts: CadConstructionCounts
) -> TopoDS_Shape:
    combined = cells[0]
    for cell in cells[1:]:
        counts.fusion_calls += 1
        operation = BRepAlgoAPI_Fuse(combined, cell)
        if not operation.IsDone() or operation.Shape().IsNull():
            raise RuntimeError("OCCT fusion did not produce a shape")
        combined = operation.Shape()
    return combined


def prepare_scene(
    bodies: PrismBodies,
    *,
    scale: float = 1.0,
    rotation: NDArray[np.float64] | None = None,
    origin: NDArray[np.float64] | None = None,
    counts: CadConstructionCounts | None = None,
) -> CadScene:
    """Build geometry afresh, then apply a shared physical similarity.

    Transformed CAD is used as independent finite-volume evidence, never as an
    implicit replacement for a rejected preserved-frame contact model.
    """
    started = perf_counter()
    attempted = CadConstructionCounts() if counts is None else counts
    matrix = np.eye(3) if rotation is None else np.asarray(rotation)
    offset = np.zeros(3) if origin is None else np.asarray(origin)
    if (
        not np.isfinite(scale)
        or scale <= 0
        or matrix.shape != (3, 3)
        or offset.shape != (3,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(offset).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-12)
        or not np.isclose(np.linalg.det(matrix), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("Expected a positive scale and finite proper rigid frame")
    groups = []
    for prisms in (bodies.fixed, bodies.moving):
        cells = tuple(
            make_prism(prism.footprint, prism.lower_z, prism.upper_z - prism.lower_z)
            for prism in prisms
        )
        if scale != 1 or np.any(matrix != np.eye(3)) or np.any(offset != 0):
            cells = tuple(
                transform_shape(cell, scale * matrix, offset) for cell in cells
            )
        groups.append(cells)
    fixed, moving = groups
    fused_fixed, fused_moving = _union(fixed, attempted), _union(moving, attempted)
    tolerances = [
        value
        for cell in (*fixed, *moving)
        if (value := boundary_tolerance(cell)) is not None
    ]
    return CadScene(
        fixed,
        moving,
        fused_fixed,
        fused_moving,
        scale * matrix @ np.asarray(bodies.velocity),
        scale,
        perf_counter() - started,
        attempted.fusion_calls,
        bool(
            BRepCheck_Analyzer(fused_fixed).IsValid()
            and BRepCheck_Analyzer(fused_moving).IsValid()
        ),
        max(tolerances) if tolerances else None,
    )


def observe_scene(scene: CadScene, step: float, *, cut: bool = False) -> CadMeasurement:
    """Perform an uncached Boolean observation; construction was paid separately."""
    record = CadMeasurement(
        step, source_valid=scene.source_valid, source_tolerance=scene.source_tolerance
    )
    started = perf_counter()
    try:
        if not np.isfinite(step) or step < 0:
            raise ValueError("Expected finite nonnegative step")
        if not scene.source_valid:
            raise RuntimeError("Source solid is invalid")
        moving = transform_shape(scene.moving, np.eye(3), step * scene.velocity)
        common_start = perf_counter()
        record.common_calls += 1
        common = BRepAlgoAPI_Common(scene.fixed, moving)
        record.common_seconds = perf_counter() - common_start
        if not common.IsDone() or common.Shape().IsNull():
            raise RuntimeError("Common did not produce a shape")
        common_shape = common.Shape()
        record.common_valid = bool(BRepCheck_Analyzer(common_shape).IsValid())
        record.volume = cad_volume(common_shape)
        record.working_volume = record.volume / scene.scale**3
        record.boundary_tolerance = boundary_tolerance(common_shape)
        if cut:
            cut_start = perf_counter()
            record.cut_calls += 1
            operation = BRepAlgoAPI_Cut(scene.fixed, moving)
            record.cut_seconds = perf_counter() - cut_start
            if not operation.IsDone() or operation.Shape().IsNull():
                raise RuntimeError("Cut did not produce a shape")
            record.cut_valid = bool(BRepCheck_Analyzer(operation.Shape()).IsValid())
            record.cut_volume = cad_volume(operation.Shape())
            record.working_balance_error = (
                cad_volume(scene.fixed) - record.cut_volume - record.volume
            ) / scene.scale**3
    except Exception as error:
        record.error = f"{type(error).__name__}: {error}"
    record.total_seconds = perf_counter() - started
    return record


def rebuild_and_observe(
    bodies: PrismBodies,
    step: float,
    *,
    cut: bool = False,
    scale: float = 1.0,
    rotation: NDArray[np.float64] | None = None,
    origin: NDArray[np.float64] | None = None,
) -> CadMeasurement:
    """Independent final validation or parameter FD with freshly rebuilt solids."""
    started = perf_counter()
    counts = CadConstructionCounts()
    try:
        scene = prepare_scene(
            bodies, scale=scale, rotation=rotation, origin=origin, counts=counts
        )
    except Exception as error:
        return CadMeasurement(
            step,
            error=f"{type(error).__name__}: {error}",
            fusion_calls=counts.fusion_calls,
            construction_seconds=perf_counter() - started,
            total_seconds=perf_counter() - started,
        )
    record = observe_scene(scene, step, cut=cut)
    record.construction_seconds = scene.construction_seconds
    record.fusion_calls = scene.fusion_calls
    record.total_seconds = perf_counter() - started
    return record


def valid_observation(record: CadMeasurement, *, cut: bool = False) -> bool:
    return bool(
        record.error is None
        and record.source_valid
        and record.common_valid
        and record.working_volume is not None
        and np.isfinite(record.working_volume)
        and (not cut or record.cut_valid is True)
    )
