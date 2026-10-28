"""Exhaustive contact preparation for two explicit convex-cell collections.

Cell-pair predicates use convex hulls of represented CAD vertex coordinates.
The returned face patches use the selected-face nominal reference contract;
neither reference certifies exact contact of the original CAD surfaces.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, product
from typing import Literal

import jax.numpy as jnp
import numpy as np

from brepax._occt.types import TopoDS_Shape
from brepax.experimental._polyhedral_contact import (
    ContactSupport,
    HullRelation,
    VertexHull,
    hull_relation,
    vertex_hull,
)
from brepax.experimental.planar_contact import (
    PlanarContactPatch,
    PlanarContactSet,
    prepare_planar_contact_set,
)
from brepax.experimental.planar_contact_cad import (
    _ROUND_OFF,
    PlanarContactDiagnostics,
    PreparedPlanarContact,
    _convex_solid,
    _ConvexSolid,
    _FacePolygon,
    _Mode,
    _reference_from_supports,
    _select_support,
)

_Scope = Literal["fixed", "moving", "cross"]
_PairKind = Literal[
    "separated", "face_contact", "edge_contact", "point_contact", "overlap", "uncertain"
]


@dataclass(frozen=True)
class CellPairDiagnostics:
    """One exhaustive cell-pair check, with input-unit length diagnostics.

    Indices address the fixed/moving sequences for ``scope='cross'`` and the
    same-body sequence otherwise. ``axis_gap`` is the largest normalized SAT
    projection separation, not Euclidean distance or penetration depth. Its
    rounded value can underflow; ``relation`` uses exact represented-coordinate
    signs. OCCT tolerances and ``roundoff_length_bound`` are evidence only:
    they never promote a represented positive gap or overlap to contact.
    """

    scope: _Scope
    first_cell: int
    second_cell: int
    relation: _PairKind
    axis_gap: float
    axis: tuple[float, float, float]
    tested_axis_count: int
    roundoff_length_bound: float
    face_tolerances: tuple[float, float]
    max_wire_geometry_residual: float
    max_support_residual: float


@dataclass(frozen=True)
class ContactPatchSource:
    """Trace one patch to source faces and the exact represented support.

    ``support_normal_residuals`` are dimensionless outward-normal differences
    from the oriented represented support. ``support_plane_residuals`` are
    signed distances of the two source plane origins from that support, in
    input length units and using the fixed-to-moving support orientation.
    """

    patch_index: int
    fixed_cell_id: int
    moving_cell_id: int
    diagnostics: PlanarContactDiagnostics
    support_normal_residuals: tuple[float, float]
    support_plane_residuals: tuple[float, float]


@dataclass(frozen=True)
class PreparedPlanarContactSet:
    """Complete pair inventory and projected patches for represented cells.

    Completeness covers the supplied cells only. It does not establish that
    they form a decomposition of some separately supplied CAD object. Face
    projection residuals remain visible in ``patch_sources``. Distinct patches
    may share edges; same-body represented interiors have been checked not to
    overlap. Empty and lower-dimensional inventories legitimately have zero
    first-order response, but still require finite-edit CAD verification.
    """

    contact_set: PlanarContactSet
    patches: tuple[PlanarContactPatch, ...]
    patch_sources: tuple[ContactPatchSource, ...]
    cell_pairs: tuple[CellPairDiagnostics, ...]
    partition_pairs: tuple[CellPairDiagnostics, ...]
    reference_kind: Literal["vertex_hulls_with_projected_face_patches"] = field(
        default="vertex_hulls_with_projected_face_patches", init=False
    )
    original_cad_contact_certified: Literal[False] = field(default=False, init=False)


class ContactPreparationError(ValueError):
    """Rejected pair with retained evidence; no partial contact set is returned."""

    pair: CellPairDiagnostics

    def __init__(self, message: str, pair: CellPairDiagnostics) -> None:
        super().__init__(message)
        self.pair = pair


@dataclass(frozen=True)
class _Cell:
    solid: _ConvexSolid
    hull: VertexHull
    extent: float
    tolerance: float


@dataclass(frozen=True)
class _MappedContact:
    prepared: PreparedPlanarContact
    normal_residuals: tuple[float, float]
    plane_residuals: tuple[float, float]


def _cells(shapes: Sequence[TopoDS_Shape]) -> tuple[_Cell, ...]:
    cells = []
    for shape in shapes:
        solid = _convex_solid(shape)
        points = solid.topological_vertices
        extent = float(np.max(np.linalg.norm(points - points[0], axis=1)))
        cells.append(
            _Cell(
                solid,
                vertex_hull(points),
                extent,
                max(face.tolerance for face in solid.faces),
            )
        )
    return tuple(cells)


def _pair_diagnostics(
    first: _Cell,
    second: _Cell,
    relation: HullRelation,
    scope: _Scope,
    first_index: int,
    second_index: int,
    *,
    uncertain: bool = False,
) -> CellPairDiagnostics:
    return CellPairDiagnostics(
        scope,
        first_index,
        second_index,
        "uncertain" if uncertain else relation.relation,
        relation.axis_gap,
        relation.axis,
        relation.tested_axis_count,
        float(_ROUND_OFF * max(first.extent, second.extent)),
        (first.tolerance, second.tolerance),
        max(first.solid.wire_residual, second.solid.wire_residual),
        max(first.solid.support_residual, second.solid.support_residual),
    )


def _support_face(cell: _Cell, exposed_indices: tuple[int, ...]) -> int:
    exposed = {
        tuple(
            float(Fraction(value, cell.hull.denominator))
            for value in cell.hull.vertices[index]
        )
        for index in exposed_indices
    }
    candidates = [
        index
        for index, face in enumerate(cell.solid.faces)
        if {tuple(map(float, vertex)) for vertex in face.topological_vertices}
        == exposed
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Exact represented contact support needs one source face with the same "
            "exposed topological vertices; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _support_residuals(
    face: _FacePolygon, support: ContactSupport, outward_sign: int
) -> tuple[float, float]:
    largest = max(abs(value) for value in support.axis)
    scaled = np.asarray([value / largest for value in support.axis])
    norm = float(np.linalg.norm(scaled))
    normal_residual = float(np.linalg.norm(face.normal - outward_sign * scaled / norm))
    plane_offset = sum(
        axis * Fraction(float(coordinate))
        for axis, coordinate in zip(support.axis, face.origin, strict=True)
    ) - Fraction(support.offset, support.denominator)
    plane_residual = float(plane_offset / largest) / norm
    bound = _ROUND_OFF * np.sqrt(face.area)
    if normal_residual > _ROUND_OFF or abs(plane_residual) > bound:
        raise ValueError(
            "Source face does not resolve the exact represented contact support: "
            f"normal residual={normal_residual:.17g}, "
            f"plane residual={plane_residual:.17g}"
        )
    return normal_residual, plane_residual


def _touching_patch(
    first: _Cell, second: _Cell, relation: HullRelation, mode: _Mode
) -> _MappedContact:
    support = relation.support
    if support is None:
        raise ValueError("Represented contact has no exact support provenance")
    first_id = _support_face(first, support.first_vertices)
    second_id = _support_face(second, support.second_vertices)
    first_residuals = _support_residuals(first.solid.faces[first_id], support, 1)
    second_residuals = _support_residuals(second.solid.faces[second_id], support, -1)
    prepared = _reference_from_supports(
        _select_support(first.solid, first_id),
        _select_support(second.solid, second_id),
        fixed_face_id=first_id,
        moving_face_id=second_id,
        mode=mode,
    )
    return _MappedContact(
        prepared,
        (first_residuals[0], second_residuals[0]),
        (first_residuals[1], second_residuals[1]),
    )


def prepare_planar_contact_set_reference(
    fixed_cells: Sequence[TopoDS_Shape],
    moving_cells: Sequence[TopoDS_Shape],
    *,
    mode: _Mode,
) -> PreparedPlanarContactSet:
    """Prepare all planar contacts between two explicit convex-cell bodies.

    Each input cell must satisfy the selected-face preparation's finite convex
    polyhedral contract. Empty sequences are allowed. The caller supplies the
    decomposition: this function neither decomposes nonconvex STEP solids nor
    proves equality to another whole-body shape.

    Extraction runs once per cell. Exact integer predicates classify the
    convex hulls of the binary64 topological vertex coordinates, testing face
    normals and every nonparallel edge-cross-edge axis without an angular
    cutoff. Same-body and cross-body interior overlap is rejected even when
    smaller than OCCT tolerances. Every positive represented gap remains
    separated, including sub-tolerance gaps. Edge and point contacts remain
    in the inventory with zero first-order contribution.

    Required ``mode`` controls face preparation only *after* represented
    face contact is established. Each exposed topological vertex set must map
    uniquely to one source face; its normal and plane must resolve the exact
    represented support within the stated roundoff bounds. An unresolved
    small patch cannot fall back to some other nearby source-face pair.
    ``zero_residual`` requires computed-zero
    plane/opposition residuals. ``projected_nominal`` permits roundoff-sized
    face residuals using the existing projected-support-polygon reference.
    Unlike the explicitly selected single-patch API, this collection API
    never projects a represented positive gap or overlap into contact.

    Neither mode certifies contact of the source CAD surfaces. Their trimmed
    planes may differ from the vertex hulls within extraction roundoff; face
    diagnostics preserve this distinction. A common CAD transform can turn a
    nominally touching partition into slightly overlapping represented hulls,
    which is conservatively rejected. Preparing known design-frame cells and
    transforming the numerical model is a separate operation, not an automatic
    fallback or proof about transformed source CAD.

    Lengths and tolerances use input units. JAX x64 is required. This host-side
    exhaustive procedure targets small cell collections: hull construction
    enumerates vertex triples and has O(n**4) worst-case cost per cell. The
    returned packed set supports the numerical evaluator's jit/vmap contract.
    No finite collision-free step or Boolean result follows from a zero rate;
    independently verify every finite edit, including edge/point contacts.

    Raises ``ValueError`` for unsupported cells, mode or precision, and
    ``ContactPreparationError`` with pair evidence for overlap or unresolved
    source-face preparation. No partial set or modified CAD is returned.

    Example, for a cutter touching a floor and a side wall::

        prepared = prepare_planar_contact_set_reference(
            (floor, wall), (cutter,), mode="zero_residual",
        )
        rate = planar_contact_set_volume_direction(
            prepared.contact_set, velocity, angular_velocity, pivot,
        )
        # Inspect prepared.cell_pairs and independently verify the finite Cut.
    """
    if mode not in ("zero_residual", "projected_nominal"):
        raise ValueError("Mode must be zero_residual or projected_nominal")
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("CAD contact preparation requires JAX x64")
    fixed, moving = _cells(fixed_cells), _cells(moving_cells)
    partitions = []
    bodies: tuple[tuple[_Scope, tuple[_Cell, ...]], ...] = (
        ("fixed", fixed),
        ("moving", moving),
    )
    for scope, cells in bodies:
        for first_index, second_index in combinations(range(len(cells)), 2):
            first, second = cells[first_index], cells[second_index]
            relation = hull_relation(first.hull, second.hull)
            record = _pair_diagnostics(
                first, second, relation, scope, first_index, second_index
            )
            if relation.relation == "overlap":
                raise ContactPreparationError(
                    f"{scope} cells {first_index}, {second_index} have overlapping "
                    f"represented interiors (axis gap {record.axis_gap:.17g}); "
                    "no roundoff or CAD-tolerance overlap is accepted",
                    record,
                )
            partitions.append(record)
    pairs = []
    patches: list[PlanarContactPatch] = []
    sources = []
    for fixed_index, moving_index in product(range(len(fixed)), range(len(moving))):
        first, second = fixed[fixed_index], moving[moving_index]
        relation = hull_relation(first.hull, second.hull)
        record = _pair_diagnostics(
            first, second, relation, "cross", fixed_index, moving_index
        )
        if relation.relation == "overlap":
            raise ContactPreparationError(
                f"Cross cells {fixed_index}, {moving_index} have existing represented "
                f"overlap (axis gap {record.axis_gap:.17g})",
                record,
            )
        if relation.relation == "face_contact":
            try:
                mapped = _touching_patch(first, second, relation, mode)
            except ValueError as error:
                unresolved = _pair_diagnostics(
                    first,
                    second,
                    relation,
                    "cross",
                    fixed_index,
                    moving_index,
                    uncertain=True,
                )
                raise ContactPreparationError(
                    f"Cross cells {fixed_index}, {moving_index} source-face reference "
                    f"is unresolved: {error}",
                    unresolved,
                ) from error
            sources.append(
                ContactPatchSource(
                    len(patches),
                    fixed_index,
                    moving_index,
                    mapped.prepared.diagnostics,
                    mapped.normal_residuals,
                    mapped.plane_residuals,
                )
            )
            patches.append(mapped.prepared.patch)
        pairs.append(record)
    return PreparedPlanarContactSet(
        prepare_planar_contact_set(patches),
        tuple(patches),
        tuple(sources),
        tuple(pairs),
        tuple(partitions),
    )
