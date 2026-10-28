"""Fresh finite CAD and actual Boolean edits for multi-contact experiments.

Stock components are fused only for these independent finite observations.
Contact preparation receives the original convex components. Every attempted
Boolean keeps its failure/validity and separately measured construction cost.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from benchmarks.face_contact.cad import (
    IDENTITY_PLACEMENT,
    STATIONARY_MOTION,
    Motion,
    Placement,
    boundary_tolerance,
    cad_volume,
    rotation_matrix,
    topology_count,
    transform_shape,
)
from brepax._occt.backend import (
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepCheck_Analyzer,
    BRepPrimAPI_MakeBox,
    TopAbs_SOLID,
    gp_Pnt,
)
from brepax._occt.types import TopoDS_Shape

from .fixtures import Box, ContactFixture


@dataclass(frozen=True)
class ComponentPair:
    """Convex components before fusion; both tuples use one common world frame."""

    fixed: tuple[TopoDS_Shape, ...]
    moving: tuple[TopoDS_Shape, ...]


@dataclass(frozen=True)
class BodyPair:
    """Stock and cutter after component union, with finite motion applied."""

    fixed: TopoDS_Shape
    moving: TopoDS_Shape


@dataclass
class VolumeMeasurement:
    """One attempted intersection observation with external and working units.

    Time is dimensionless. Common/Cut call counts include attempted operations
    even when they fail; component-fusion counts are separate setup work.
    A valid Boolean shape alone is not an accuracy certificate.
    """

    step: float
    volume: float | None = None
    working_volume: float | None = None
    fixed_volume: float | None = None
    moving_volume: float | None = None
    source_valid: bool | None = None
    valid: bool | None = None
    solid_count: int | None = None
    source_tolerance: float | None = None
    boundary_tolerance: float | None = None
    construction_seconds: float = 0.0
    common_seconds: float = 0.0
    total_seconds: float = 0.0
    fusion_calls: int = 0
    common_calls: int = 0
    error: str | None = None


@dataclass
class EditMeasurement(VolumeMeasurement):
    """Intersection plus actual stock-minus-cutter edit and its volume balance."""

    cut_volume: float | None = None
    working_cut_volume: float | None = None
    working_balance_error: float | None = None
    cut_valid: bool | None = None
    cut_solid_count: int | None = None
    cut_seconds: float = 0.0
    cut_calls: int = 0


@dataclass
class EditResult:
    """Retain the real Cut shape for export alongside serializable evidence.

    ``shape`` is None on failure before a shape exists. Always inspect
    ``measurement.error`` and ``measurement.cut_valid`` before using a shape.
    """

    shape: TopoDS_Shape | None
    measurement: EditMeasurement


def _box_shape(box: Box) -> TopoDS_Shape:
    lower = np.asarray(box.lower, dtype=np.float64)
    upper = np.asarray(box.upper, dtype=np.float64)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite([lower, upper]).all()
        or np.any(upper <= lower)
    ):
        raise ValueError("Expected finite positive box dimensions")
    return BRepPrimAPI_MakeBox(
        gp_Pnt(*map(float, lower)), *map(float, upper - lower)
    ).Shape()


def build_components(
    fixture: ContactFixture,
    motion: Motion = STATIONARY_MOTION,
    step: float = 0.0,
    placement: Placement = IDENTITY_PLACEMENT,
) -> ComponentPair:
    """Construct fresh convex cells, apply the cutter motion, then similarity.

    ``step`` is finite and nonnegative. The path and placement follow the
    existing face-contact CAD convention: p + R(t omega)(x-p) + t v, then
    world = origin + scale * R_global * working. Source tolerances arise from
    these actual OCCT operations; no contacts are snapped or silently healed.
    """
    values = np.r_[
        step,
        motion.velocity,
        motion.angular_velocity,
        motion.pivot,
        placement.scale,
        placement.origin,
    ]
    if not np.isfinite(values).all() or step < 0 or placement.scale <= 0:
        raise ValueError("Expected finite motion, nonnegative step and positive scale")
    if not fixture.fixed_cells or not fixture.moving_cells:
        raise ValueError("Both bodies require at least one convex component")
    fixed = tuple(_box_shape(box) for box in fixture.fixed_cells)
    moving = tuple(_box_shape(box) for box in fixture.moving_cells)
    rotation = rotation_matrix(step * np.asarray(motion.angular_velocity))
    pivot = np.asarray(motion.pivot)
    translation = pivot - rotation @ pivot + step * np.asarray(motion.velocity)
    moving = tuple(transform_shape(shape, rotation, translation) for shape in moving)
    global_linear = placement.scale * rotation_matrix(placement.rotation_vector)
    return ComponentPair(
        tuple(
            transform_shape(shape, global_linear, placement.origin) for shape in fixed
        ),
        tuple(
            transform_shape(shape, global_linear, placement.origin) for shape in moving
        ),
    )


def _fuse(
    shapes: tuple[TopoDS_Shape, ...],
    measurement: VolumeMeasurement | None = None,
) -> TopoDS_Shape:
    combined = shapes[0]
    for shape in shapes[1:]:
        if measurement is not None:
            measurement.fusion_calls += 1
        operation = BRepAlgoAPI_Fuse(combined, shape)
        if not operation.IsDone():
            raise RuntimeError("OCCT component fusion did not finish")
        combined = operation.Shape()
    return combined


def build_pair(
    fixture: ContactFixture,
    motion: Motion = STATIONARY_MOTION,
    step: float = 0.0,
    placement: Placement = IDENTITY_PLACEMENT,
) -> BodyPair:
    """Fresh fused finite solids, independently of contact-patch preparation."""
    components = build_components(fixture, motion, step, placement)
    return BodyPair(_fuse(components.fixed), _fuse(components.moving))


def _observe(
    fixture: ContactFixture,
    motion: Motion,
    placement: Placement,
    measurement: VolumeMeasurement,
) -> BodyPair:
    started = perf_counter()
    try:
        components = build_components(fixture, motion, measurement.step, placement)
        pair = BodyPair(
            _fuse(components.fixed, measurement),
            _fuse(components.moving, measurement),
        )
    finally:
        measurement.construction_seconds = perf_counter() - started
    measurement.source_valid = bool(
        BRepCheck_Analyzer(pair.fixed).IsValid()
        and BRepCheck_Analyzer(pair.moving).IsValid()
    )
    measurement.fixed_volume = cad_volume(pair.fixed)
    measurement.moving_volume = cad_volume(pair.moving)
    tolerances = (boundary_tolerance(pair.fixed), boundary_tolerance(pair.moving))
    measurement.source_tolerance = max(t for t in tolerances if t is not None)
    started = perf_counter()
    measurement.common_calls += 1
    try:
        operation = BRepAlgoAPI_Common(pair.fixed, pair.moving)
        if not operation.IsDone():
            raise RuntimeError("OCCT Common did not finish")
        common = operation.Shape()
        if common.IsNull():
            raise RuntimeError("OCCT Common returned a null shape")
        measurement.volume = cad_volume(common)
        measurement.working_volume = measurement.volume / placement.scale**3
        measurement.valid = bool(BRepCheck_Analyzer(common).IsValid())
        measurement.solid_count = topology_count(common, TopAbs_SOLID)
        measurement.boundary_tolerance = boundary_tolerance(common)
    finally:
        measurement.common_seconds = perf_counter() - started
    return pair


def measure_volume(
    fixture: ContactFixture,
    motion: Motion,
    step: float,
    placement: Placement = IDENTITY_PLACEMENT,
) -> VolumeMeasurement:
    """Keep every attempted Common, including unsupported or failed geometry."""
    measurement = VolumeMeasurement(step)
    started = perf_counter()
    try:
        _observe(fixture, motion, placement, measurement)
    except Exception as error:
        measurement.error = f"{type(error).__name__}: {error}"
    finally:
        measurement.total_seconds = perf_counter() - started
    return measurement


def apply_cut(
    fixture: ContactFixture,
    motion: Motion,
    step: float,
    placement: Placement = IDENTITY_PLACEMENT,
) -> EditResult:
    """Measure overlap and apply one real stock-minus-cutter Boolean edit.

    Cut and Common are separate OCCT operations, not independent geometric
    kernels. Compare their volume balance and shape validity, then use the
    independent interval/integral references for accuracy evidence. Returning
    a valid shape does not imply that a target removal amount was achieved.
    """
    measurement = EditMeasurement(step)
    result = EditResult(None, measurement)
    total_started = perf_counter()
    try:
        pair = _observe(fixture, motion, placement, measurement)
        started = perf_counter()
        measurement.cut_calls += 1
        try:
            operation = BRepAlgoAPI_Cut(pair.fixed, pair.moving)
            if not operation.IsDone():
                raise RuntimeError("OCCT Cut did not finish")
            result.shape = operation.Shape()
            if result.shape.IsNull():
                raise RuntimeError("OCCT Cut returned a null shape")
            measurement.cut_volume = cad_volume(result.shape)
            measurement.working_cut_volume = measurement.cut_volume / placement.scale**3
            measurement.cut_valid = bool(BRepCheck_Analyzer(result.shape).IsValid())
            measurement.cut_solid_count = topology_count(result.shape, TopAbs_SOLID)
            assert measurement.fixed_volume is not None
            assert measurement.working_volume is not None
            measurement.working_balance_error = (
                measurement.fixed_volume / placement.scale**3
                - measurement.working_cut_volume
                - measurement.working_volume
            )
        finally:
            measurement.cut_seconds = perf_counter() - started
    except Exception as error:
        measurement.error = f"{type(error).__name__}: {error}"
    finally:
        measurement.total_seconds = perf_counter() - total_started
    return result
