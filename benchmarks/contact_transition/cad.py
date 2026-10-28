"""Independent OCCT rebuilds for a cylindrical hole crossing a plate edge.

CAD construction, adaptive B-Rep integration and fresh meshing are host-side
oracles. None of them calls the analytical area/derivative candidate. Lengths
are supplied in working coordinates; a placement changes the actual B-Rep
before meshing. Results retain failures at degenerate contact configurations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from brepax._occt.backend import (
    BRep_Tool,
    BRepAlgoAPI_Cut,
    BRepBuilderAPI_Transform,
    BRepCheck_Analyzer,
    BRepGProp,
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    GProp_GProps,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopExp_Explorer,
    TopoDS,
    gp_Ax2,
    gp_Dir,
    gp_Pnt,
    gp_Trsf,
)
from brepax._occt.types import TopoDS_Shape
from brepax.brep.triangulate import triangulate_shape


@dataclass(frozen=True)
class CadPlacement:
    """Proper rotation about (1,2,3), positive unit scale, and translation."""

    name: str
    scale: float = 1.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angle: float = 0.0


PLACEMENTS = (
    CadPlacement("identity"),
    CadPlacement("rotated_translated", origin=(3.0, -2.0, 4.0), angle=0.7),
    CadPlacement("small_far", 1e-3, (1.0, -2.0, 3.0), 0.7),
    CadPlacement("large", 1e3, (3e3, -2e3, 4e3), 0.7),
)


@dataclass(frozen=True)
class CadDifference:
    """Rebuilt one-sided and central derivatives in working area units."""

    working_step: float
    left: float | None = None
    right: float | None = None
    central: float | None = None
    absolute_error: float | None = None
    sample_validity: tuple[bool, bool, bool] | None = None
    seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class CadCenterDifference:
    """Two rebuilt local-axis differences and their derived world-area vector.

    The physical world gradient is s**2 R[:, :2] @ local_gradient. Its three
    components are a basis conversion of two measured in-plane derivatives,
    not three separately sampled world-axis perturbations. Through-hole volume
    is unchanged by translating the cylinder along its extrusion axis.
    """

    working_step: float
    local_gradient: tuple[float, float] | None = None
    derived_world_gradient: tuple[float, float, float] | None = None
    absolute_error: float | None = None
    sample_validity: tuple[bool, ...] | None = None
    seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class MeshCheck:
    """Fresh mesh diagnostics after tolerance-bounded coordinate welding.

    Closure means exactly two opposite directed incidences per welded edge.
    Edge incidences do not establish vertex-manifoldness or absence of arbitrary
    self-intersection. Positive signed volume is checked separately.
    """

    triangles: int
    working_volume: float
    cad_relative_error: float
    closed_oriented: bool
    positive_volume: bool
    degenerate_triangles: int
    euler_characteristic: int
    working_deflection: float
    working_weld_tolerance: float
    maximum_weld_displacement: float
    seconds: float


@dataclass
class CadCheck:
    """Completed measurements survive a later meshing or derivative failure."""

    radius: float
    placement: CadPlacement
    working_volume: float | None = None
    volume_absolute_error: float | None = None
    valid: bool | None = None
    solid_count: int | None = None
    face_count: int | None = None
    working_boundary_tolerance: float | None = None
    construction_seconds: float | None = None
    integration_seconds: float | None = None
    mesh: MeshCheck | None = None
    differences: list[CadDifference] | None = None
    error: str | None = None


def _placement_transform(placement: CadPlacement) -> gp_Trsf:
    values = np.asarray((*placement.origin, placement.scale, placement.angle))
    if not np.isfinite(values).all() or placement.scale <= 0:
        raise ValueError("Placement must be finite with a positive scale")
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = (
        np.eye(3)
        + np.sin(placement.angle) * skew
        + (1 - np.cos(placement.angle)) * skew @ skew
    )
    coefficients = np.column_stack(
        (placement.scale * rotation, np.asarray(placement.origin))
    ).reshape(-1)
    transform = gp_Trsf()
    transform.SetValues(*map(float, coefficients))
    return transform


def build_plate_cut(
    center: ArrayLike,
    radius: float,
    half_extents: ArrayLike,
    height: float,
    placement: CadPlacement = PLACEMENTS[0],
) -> TopoDS_Shape:
    """Freshly construct/cut a plate, then place its B-Rep without copying a mesh.

    The stock is [-bx,bx] x [-by,by] x [-height/2,height/2]. A cylinder
    extends past both end faces. Inputs are finite working lengths with positive
    radius, half extents and height. Invalid OCCT results are returned so callers
    can record contact degeneracies; an unfinished Boolean raises RuntimeError.
    """
    c, bounds = np.asarray(center, dtype=float), np.asarray(half_extents, dtype=float)
    if (
        c.shape != (2,)
        or bounds.shape != (2,)
        or not np.isfinite(np.r_[c, bounds, radius, height]).all()
        or np.any(bounds <= 0)
        or radius <= 0
        or height <= 0
    ):
        raise ValueError("Expected finite center and positive radius/extents/height")
    bx, by = map(float, bounds)
    stock = BRepPrimAPI_MakeBox(
        gp_Pnt(-bx, -by, -height / 2), 2 * bx, 2 * by, height
    ).Shape()
    tool = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(float(c[0]), float(c[1]), -height), gp_Dir(0, 0, 1)),
        radius,
        2 * height,
    ).Shape()
    operation = BRepAlgoAPI_Cut(stock, tool)
    if not operation.IsDone():
        raise RuntimeError("OCCT Boolean cut did not finish")
    shape = operation.Shape()
    if shape.IsNull():
        raise RuntimeError("OCCT Boolean cut returned a null shape")
    return BRepBuilderAPI_Transform(
        shape, _placement_transform(placement), True, False
    ).Shape()


def cad_volume(shape: TopoDS_Shape) -> float:
    """Adaptive B-Rep volume in external units, independent of triangulation."""
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties, 1e-12)
    value = float(properties.Mass())
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite OCCT volume")
    return value


def _boundary_tolerance(shape: TopoDS_Shape) -> float:
    values = []
    for kind, cast in ((TopAbs_EDGE, TopoDS.Edge_s), (TopAbs_VERTEX, TopoDS.Vertex_s)):
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            values.append(BRep_Tool.Tolerance_s(cast(explorer.Current())))
            explorer.Next()
    if not values:
        raise ValueError("No edges or vertices in rebuilt CAD")
    return float(max(values))


def _mesh_check(
    shape: TopoDS_Shape,
    placement: CadPlacement,
    working_volume: float,
    deflection: float,
    working_tolerance: float,
) -> MeshCheck:
    start = perf_counter()
    triangles, _ = triangulate_shape(shape, deflection=deflection * placement.scale)
    if np.asarray(triangles).dtype != np.float64:
        raise ValueError("Fresh-mesh verification requires JAX x64")
    # Normalize after fresh external meshing. Rotation remains in the coordinates.
    points = (np.asarray(triangles).reshape(-1, 3) - placement.origin) / placement.scale
    if not np.isfinite(points).all() or not len(points):
        raise ValueError("Fresh mesh has no finite triangles")
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    parents = np.arange(len(unique))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    weld_tolerance = 2 * working_tolerance
    for left, right in sorted(cKDTree(unique).query_pairs(weld_tolerance)):
        a, b = root(left), root(right)
        parents[max(a, b)] = min(a, b)
    selected, ids = np.unique(
        [root(i) for i in range(len(unique))], return_inverse=True
    )
    vertices, faces = unique[selected], ids[inverse].reshape(-1, 3)
    displacement = float(
        np.linalg.norm(points - vertices[faces.reshape(-1)], axis=1).max()
    )
    if displacement > weld_tolerance:
        raise ValueError("Transitive vertex weld exceeds measured CAD tolerance")
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    unique_edges, edge_ids, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    closed = bool(np.all((counts == 2) & (np.bincount(edge_ids, weights=signs) == 0)))
    original = points.reshape(-1, 3, 3)
    shifted = original - original.mean(axis=(0, 1))
    normals = np.cross(shifted[:, 1] - shifted[:, 0], shifted[:, 2] - shifted[:, 0])
    degenerate = int(np.count_nonzero(np.linalg.norm(normals, axis=1) == 0))
    volume = float(
        np.einsum(
            "ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])
        ).sum()
        / 6
    )
    return MeshCheck(
        triangles=len(original),
        working_volume=volume,
        cad_relative_error=abs(volume / working_volume - 1),
        closed_oriented=closed,
        positive_volume=bool(volume > 0),
        degenerate_triangles=degenerate,
        euler_characteristic=len(vertices) - len(unique_edges) + len(faces),
        working_deflection=deflection,
        working_weld_tolerance=weld_tolerance,
        maximum_weld_displacement=displacement,
        seconds=perf_counter() - start,
    )


def radius_difference_sweep(
    center: ArrayLike,
    radius: float,
    half_extents: ArrayLike,
    height: float,
    *,
    placement: CadPlacement = PLACEMENTS[0],
    steps: Sequence[float] = (1e-3, 1e-4, 1e-5),
    expected_gradient: float | None = None,
) -> list[CadDifference]:
    """Rebuild at r +/- h; no differentiation through OCCT or a frozen mesh.

    Steps and returned derivatives are in working units. In external units the
    step is s*h and the derivative is s**2 times the reported value. Each step
    preserves its own failure. At contact, the one-sided limits matter; central
    differences need not converge quadratically even for a C1 volume.
    """
    base_shape = build_plate_cut(center, radius, half_extents, height, placement)
    base = cad_volume(base_shape)
    base_valid = bool(BRepCheck_Analyzer(base_shape).IsValid())
    results = []
    for step in steps:
        start = perf_counter()
        try:
            if not np.isfinite(step) or step <= 0 or step >= radius:
                raise ValueError(
                    "FD step must be positive, finite and smaller than radius"
                )
            left_shape = build_plate_cut(
                center, radius - step, half_extents, height, placement
            )
            right_shape = build_plate_cut(
                center, radius + step, half_extents, height, placement
            )
            left, right = cad_volume(left_shape), cad_volume(right_shape)
            denominator = step * placement.scale**3
            central = (right - left) / (2 * denominator)
            results.append(
                CadDifference(
                    working_step=step,
                    left=(base - left) / denominator,
                    right=(right - base) / denominator,
                    central=central,
                    absolute_error=None
                    if expected_gradient is None
                    else abs(central - expected_gradient),
                    sample_validity=(
                        bool(BRepCheck_Analyzer(left_shape).IsValid()),
                        base_valid,
                        bool(BRepCheck_Analyzer(right_shape).IsValid()),
                    ),
                    seconds=perf_counter() - start,
                )
            )
        except Exception as error:
            results.append(
                CadDifference(
                    step,
                    seconds=perf_counter() - start,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return results


def center_difference_sweep(
    center: ArrayLike,
    radius: float,
    half_extents: ArrayLike,
    height: float,
    *,
    placement: CadPlacement = PLACEMENTS[0],
    steps: Sequence[float] = (1e-3, 1e-4, 1e-5),
    expected_gradient: Sequence[float] | None = None,
) -> list[CadCenterDifference]:
    """Rebuild after each local center perturbation, then transform the CAD.

    Rotated placements perturb the tool along the rotated plate axes relative
    to the stock. Local results have working area units. The derived world
    vector has external area units and obeys g_world = s**2 R @ g_local.
    """
    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (2,) or not np.isfinite(center_array).all():
        raise ValueError("Center must contain two finite coordinates")
    transform = _placement_transform(placement)
    basis = (
        np.array([[transform.Value(i, j) for j in (1, 2)] for i in (1, 2, 3)])
        / placement.scale
    )
    results = []
    for step in steps:
        start = perf_counter()
        try:
            if not np.isfinite(step) or step <= 0:
                raise ValueError("Center FD step must be finite and positive")
            gradient, validity = [], []
            for direction in np.eye(2):
                volumes = []
                for sign in (-1, 1):
                    shape = build_plate_cut(
                        center_array + sign * step * direction,
                        radius,
                        half_extents,
                        height,
                        placement,
                    )
                    validity.append(bool(BRepCheck_Analyzer(shape).IsValid()))
                    volumes.append(cad_volume(shape))
                gradient.append(
                    (volumes[1] - volumes[0]) / (2 * step * placement.scale**3)
                )
            world = placement.scale**2 * basis @ gradient
            results.append(
                CadCenterDifference(
                    working_step=step,
                    local_gradient=(float(gradient[0]), float(gradient[1])),
                    derived_world_gradient=(
                        float(world[0]),
                        float(world[1]),
                        float(world[2]),
                    ),
                    absolute_error=None
                    if expected_gradient is None
                    else float(
                        np.linalg.norm(np.asarray(gradient) - expected_gradient)
                    ),
                    sample_validity=tuple(validity),
                    seconds=perf_counter() - start,
                )
            )
        except Exception as error:
            results.append(
                CadCenterDifference(
                    step,
                    seconds=perf_counter() - start,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return results


def verify_cad_transition(
    radii: Sequence[float],
    *,
    center: ArrayLike = (1.5, 0.2),
    half_extents: ArrayLike = (3.0, 2.0),
    height: float = 2.0,
    expected_volumes: Sequence[float] | None = None,
    expected_gradients: Sequence[float] | None = None,
    placements: Sequence[CadPlacement] = PLACEMENTS,
    deflection: float = 1e-4,
    difference_steps: Sequence[float] = (1e-3, 1e-4, 1e-5),
) -> list[CadCheck]:
    """Rebuild, transform, remesh and finite-difference every requested case.

    Values and gradients supplied for comparison are in working units. Returned
    dataclasses can be serialized with dataclasses.asdict. There is no aggregate
    pass flag: CAD validity, mesh quality, FD convergence and candidate errors
    answer different questions. Exact tangency can be a nonmanifold CAD solid
    even though the continuum volume has a unique first derivative there.
    """
    if not np.isfinite(deflection) or deflection <= 0:
        raise ValueError("Mesh deflection must be finite and positive")
    for expected in (expected_volumes, expected_gradients):
        if expected is not None and len(expected) != len(radii):
            raise ValueError("Expected measurements must align with radii")
    results = []
    for index, radius in enumerate(radii):
        for placement in placements:
            check = CadCheck(float(radius), placement)
            try:
                start = perf_counter()
                shape = build_plate_cut(center, radius, half_extents, height, placement)
                check.construction_seconds = perf_counter() - start
                check.valid = bool(BRepCheck_Analyzer(shape).IsValid())
                counts = []
                for kind in (TopAbs_SOLID, TopAbs_FACE):
                    explorer, count = TopExp_Explorer(shape, kind), 0
                    while explorer.More():
                        count += 1
                        explorer.Next()
                    counts.append(count)
                check.solid_count, check.face_count = counts
                start = perf_counter()
                check.working_volume = cad_volume(shape) / placement.scale**3
                check.integration_seconds = perf_counter() - start
                if expected_volumes is not None:
                    check.volume_absolute_error = abs(
                        check.working_volume - expected_volumes[index]
                    )
                check.working_boundary_tolerance = (
                    _boundary_tolerance(shape) / placement.scale
                )
                check.differences = radius_difference_sweep(
                    center,
                    radius,
                    half_extents,
                    height,
                    placement=placement,
                    steps=difference_steps,
                    expected_gradient=None
                    if expected_gradients is None
                    else expected_gradients[index],
                )
                check.mesh = _mesh_check(
                    shape,
                    placement,
                    check.working_volume,
                    deflection,
                    check.working_boundary_tolerance,
                )
            except Exception as error:
                check.error = f"{type(error).__name__}: {error}"
            results.append(check)
    return results
