"""Finite-solid OCCT evidence for bounded planar-contact experiments.

The parameter t is dimensionless. Working lengths and linear velocity are in
the same units; angular velocity is radians per unit t. Placement changes actual
B-Reps, not a mesh. Every step constructs the solids and Boolean afresh.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from brepax._occt.backend import (
    BRep_Tool,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Fuse,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_Transform,
    BRepCheck_Analyzer,
    BRepGProp,
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakePrism,
    GProp_GProps,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopExp_Explorer,
    TopoDS,
    gp_Pnt,
    gp_Trsf,
    gp_Vec,
)
from brepax._occt.types import TopoDS_Shape


@dataclass(frozen=True)
class Placement:
    """Actual B-Rep similarity; world lengths are working lengths times scale."""

    scale: float = 1.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_vector: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Motion:
    """Working-coordinate rigid twist; ``step`` multiplies both velocities."""

    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)


IDENTITY_PLACEMENT = Placement()
STATIONARY_MOTION = Motion()


def placement_from_matrix(
    scale: float, origin: ArrayLike, rotation: ArrayLike
) -> Placement:
    """Convert a finite proper rotation and positive unit scale to CAD placement.

    Orthogonality and determinant use an absolute dimensionless tolerance of
    1e-12. SciPy is already used by CAD benchmarks; its rotation conversion
    avoids axis/angle branch problems near zero and half turns.
    """
    offset = np.asarray(origin, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    if (
        offset.shape != (3,)
        or matrix.shape != (3, 3)
        or not np.isfinite(offset).all()
        or not np.isfinite(matrix).all()
        or not np.isfinite(scale)
        or scale <= 0
        or not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-12)
        or not np.isclose(np.linalg.det(matrix), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Expected a finite proper rotation, origin, and positive scale"
        )
    vector = Rotation.from_matrix(matrix).as_rotvec()
    return Placement(
        float(scale),
        (float(offset[0]), float(offset[1]), float(offset[2])),
        (float(vector[0]), float(vector[1]), float(vector[2])),
    )


@dataclass(frozen=True)
class Fixture:
    """A convex tool prism above finite rectangular stock, optionally a side wall.

    The footprint is CCW in the XY plane. A positive gap separates the solids;
    a negative gap starts in overlap. Only zero gap prescribes nominal contact.
    The optional side wall is a deliberate multiple-contact counterexample.
    """

    vertices: tuple[tuple[float, float], ...]
    gap: float = 0.0
    height: float = 1.0
    stock_bounds: tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0)
    stock_depth: float = 1.0
    # Right wall adds a second planar contact to the default square fixture.
    right_wall: bool = False


@dataclass
class VolumeSample:
    """One attempted OCCT evaluation, including validity and resolution evidence.

    Volumes/tolerances are in external units except the explicitly ``working``
    fields. A valid OCCT shape does not certify an accurate small-step slope.
    Failed samples retain their error and all diagnostics available before it.
    """

    step: float
    volume: float | None = None
    working_volume: float | None = None
    fixed_volume: float | None = None
    moving_volume: float | None = None
    valid: bool | None = None
    fixed_valid: bool | None = None
    moving_valid: bool | None = None
    solid_count: int | None = None
    face_count: int | None = None
    boundary_tolerance: float | None = None
    source_tolerance: float | None = None
    working_difference: float | None = None
    error: str | None = None


def rotation_matrix(rotation_vector: ArrayLike) -> NDArray[np.float64]:
    vector = np.asarray(rotation_vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected finite three-component rotation vector")
    angle = float(np.linalg.norm(vector))
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.asarray(
        np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew,
        dtype=np.float64,
    )


def transform_shape(
    shape: TopoDS_Shape, linear: ArrayLike, translation: ArrayLike
) -> TopoDS_Shape:
    transform = gp_Trsf()
    coefficients = np.column_stack((linear, translation)).reshape(-1)
    transform.SetValues(*map(float, coefficients))
    operation = BRepBuilderAPI_Transform(shape, transform, True, False)
    if not operation.IsDone():
        raise RuntimeError("OCCT transform did not finish")
    return operation.Shape()


def make_prism(vertices: ArrayLike, base: float, height: float) -> TopoDS_Shape:
    """Construct one finite prism from a strictly convex CCW footprint."""
    points = np.asarray(vertices, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
        or not np.isfinite([base, height]).all()
        or height <= 0
    ):
        raise ValueError("Expected finite polygon/base and positive prism height")
    edges = np.roll(points, -1, axis=0) - points
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
        edges[:, 0], -1
    )
    offsets = points[None, :, :] - points[:, None, :]
    supports = (
        edges[:, None, 0] * offsets[:, :, 1] - edges[:, None, 1] * offsets[:, :, 0]
    )
    # Positive local turns alone would also admit a self-intersecting star.
    if np.any(turns <= 0) or np.any(supports < 0):
        raise ValueError("Fixture footprint must be strictly convex and CCW")
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in points:
        polygon.Add(gp_Pnt(float(x), float(y), float(base)))
    polygon.Close()
    if not polygon.IsDone():
        raise RuntimeError("OCCT polygon did not finish")
    face = BRepBuilderAPI_MakeFace(polygon.Wire())
    if not face.IsDone():
        raise RuntimeError("OCCT polygon face did not finish")
    prism = BRepPrimAPI_MakePrism(face.Face(), gp_Vec(0, 0, height), True)
    if not prism.IsDone():
        raise RuntimeError("OCCT prism did not finish")
    return prism.Shape()


def build_pair(
    fixture: Fixture,
    motion: Motion = STATIONARY_MOTION,
    step: float = 0.0,
    placement: Placement = IDENTITY_PLACEMENT,
) -> tuple[TopoDS_Shape, TopoDS_Shape]:
    """Fresh finite stock/tool, physical motion, then actual global similarity."""
    xmin, xmax, ymin, ymax = fixture.stock_bounds
    values = np.r_[
        fixture.stock_bounds,
        fixture.stock_depth,
        fixture.height,
        fixture.gap,
        step,
        motion.velocity,
        motion.angular_velocity,
        motion.pivot,
        placement.origin,
        placement.scale,
    ]
    if (
        not np.isfinite(values).all()
        or xmin >= xmax
        or ymin >= ymax
        or fixture.stock_depth <= 0
        or placement.scale <= 0
        or step < 0
    ):
        raise ValueError("Invalid finite-solid fixture, placement, or step")
    fixed = BRepPrimAPI_MakeBox(
        gp_Pnt(xmin, ymin, -fixture.stock_depth),
        xmax - xmin,
        ymax - ymin,
        fixture.stock_depth,
    ).Shape()
    if fixture.right_wall:
        if fixture.stock_bounds != (-2.0, 2.0, -2.0, 2.0):
            raise ValueError("Side-contact fixture requires default stock bounds")
        wall = BRepPrimAPI_MakeBox(gp_Pnt(1, -2, -0.5), 1, 4, 1.5).Shape()
        union = BRepAlgoAPI_Fuse(fixed, wall)
        if not union.IsDone():
            raise RuntimeError("OCCT side-wall union did not finish")
        fixed = union.Shape()
    moving = make_prism(fixture.vertices, fixture.gap, fixture.height)
    rotation = rotation_matrix(step * np.asarray(motion.angular_velocity))
    pivot = np.asarray(motion.pivot)
    moving = transform_shape(
        moving, rotation, pivot - rotation @ pivot + step * np.asarray(motion.velocity)
    )
    global_linear = placement.scale * rotation_matrix(placement.rotation_vector)
    return (
        transform_shape(fixed, global_linear, placement.origin),
        transform_shape(moving, global_linear, placement.origin),
    )


def cad_volume(shape: TopoDS_Shape) -> float:
    """Adaptive OCCT solid mass in external units; independent of meshing."""
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties, 1e-12)
    value = float(properties.Mass())
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite OCCT volume")
    return value


def topology_count(shape: TopoDS_Shape, kind: object) -> int:
    explorer = TopExp_Explorer(shape, kind)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def boundary_tolerance(shape: TopoDS_Shape) -> float | None:
    values = []
    for kind, cast in ((TopAbs_EDGE, TopoDS.Edge_s), (TopAbs_VERTEX, TopoDS.Vertex_s)):
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            values.append(float(BRep_Tool.Tolerance_s(cast(explorer.Current()))))
            explorer.Next()
    return max(values) if values else None


def volume_sample(
    fixture: Fixture,
    motion: Motion,
    step: float,
    placement: Placement = IDENTITY_PLACEMENT,
) -> VolumeSample:
    """Preserve invalid/null/failed Boolean results as experiment evidence."""
    sample = VolumeSample(step)
    try:
        fixed, moving = build_pair(fixture, motion, step, placement)
        sample.fixed_valid = BRepCheck_Analyzer(fixed).IsValid()
        sample.moving_valid = BRepCheck_Analyzer(moving).IsValid()
        sample.fixed_volume = cad_volume(fixed)
        sample.moving_volume = cad_volume(moving)
        tolerances = (boundary_tolerance(fixed), boundary_tolerance(moving))
        sample.source_tolerance = max(t for t in tolerances if t is not None)
        operation = BRepAlgoAPI_Common(fixed, moving)
        if not operation.IsDone():
            raise RuntimeError("OCCT common did not finish")
        common = operation.Shape()
        if common.IsNull():
            raise RuntimeError("OCCT common returned null shape")
        sample.volume = cad_volume(common)
        sample.working_volume = sample.volume / placement.scale**3
        sample.valid = BRepCheck_Analyzer(common).IsValid()
        sample.solid_count = topology_count(common, TopAbs_SOLID)
        sample.face_count = topology_count(common, TopAbs_FACE)
        sample.boundary_tolerance = boundary_tolerance(common)
    except Exception as error:
        sample.error = f"{type(error).__name__}: {error}"
    return sample


def difference_sweep(
    fixture: Fixture,
    motion: Motion,
    steps: ArrayLike,
    placement: Placement = IDENTITY_PLACEMENT,
) -> list[VolumeSample]:
    """One-sided [I(t)-I(0)]/t in working volume units, retaining every step."""
    samples = [volume_sample(fixture, motion, 0.0, placement)]
    for step in np.asarray(steps, dtype=float):
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Difference steps must be finite and positive")
        sample = volume_sample(fixture, motion, float(step), placement)
        if sample.working_volume is not None and samples[0].working_volume is not None:
            sample.working_difference = (
                sample.working_volume - samples[0].working_volume
            ) / step
        samples.append(sample)
    return samples
