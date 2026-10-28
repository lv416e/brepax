"""Fresh B-Rep preparation and physical-frame directional measurements."""

import time
from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.contact_transition.measure import compile_timed
from brepax._occt.backend import (
    BRepAdaptor_Surface,
    BRepCheck_Analyzer,
    GeomAbs_Plane,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopExp_Explorer,
    TopoDS,
)
from brepax._occt.types import TopoDS_Shape
from brepax.experimental.planar_contact import (
    planar_contact_volume_direction,
    prepare_planar_contact_patch,
)
from brepax.experimental.planar_contact_cad import prepare_planar_contact_reference

from . import cad
from .fixtures import (
    audit_directions,
    fixtures,
    placements,
    polygon_area_centroid,
    selection_directions,
    transform_direction,
)
from .references import directional_reference


def known_support_face_id(
    shape: TopoDS_Shape, normal: tuple[float, float, float], offset: float | None = None
) -> int:
    """Select a prescribed fixture plane, not detect arbitrary CAD contact.

    This helper is used before any fixture placement. Exact comparison is
    intentional: fixture construction specifies axis-aligned supporting planes.
    """
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    index = 0
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        if adaptor.GetType() == GeomAbs_Plane:
            plane = adaptor.Plane()
            axis, location = plane.Axis().Direction(), plane.Location()
            direction = np.array([axis.X(), axis.Y(), axis.Z()])
            if face.Orientation() == TopAbs_REVERSED:
                direction = -direction
            position = np.array([location.X(), location.Y(), location.Z()])
            if np.array_equal(direction, normal) and (
                offset is None or np.dot(position, direction) == offset
            ):
                return index
        index += 1
        explorer.Next()
    raise ValueError("Prescribed fixture support face was not found")


def collect_prepared_contact() -> dict[str, Any]:
    """Prepare all actual placed solids, then measure every transformed twist.

    Candidate values are independently evaluated for each placement; identity
    scores are never copied. A failed preparation retains all attempted
    directions with errors so it cannot silently reduce the denominator.
    """
    directions = (*audit_directions(), *selection_directions())
    result: dict[str, Any] = {
        "rows": [],
        "preparations": [],
        "runtime": [],
        "input_contract": (
            "Fresh convex-prism B-Reps with a construction-prescribed zero gap; "
            "explicit selected faces; projected support-polygon reference. "
            "Source gaps and residuals are retained, not certified exact."
        ),
        "cost_contract": (
            "Base fixture construction/selection and each physical B-Rep build "
            "are timed separately from host extraction, direct polygon setup, "
            "compile/first/warm execution, and each 28-direction evaluation."
        ),
    }
    cache: dict[tuple[int, int], Any] = {}
    batch = jax.vmap(planar_contact_volume_direction, in_axes=(None, 0, 0, 0))
    for fixture in fixtures():
        vertices = np.asarray(fixture.vertices)
        references = {}
        normalizations = {}
        reference_error = None
        try:
            area, _ = polygon_area_centroid(vertices)
            for direction in directions:
                velocity, angular, pivot = map(
                    np.asarray,
                    (direction.translation, direction.angular, direction.pivot),
                )
                field = velocity + np.cross(
                    angular,
                    np.column_stack((vertices, np.zeros(len(vertices)))) - pivot,
                )
                normal_speed = float(np.max(np.abs(field[:, 2])))
                full_speed = float(np.max(np.linalg.norm(field, axis=1)))
                normalizations[direction.name] = area * (normal_speed or full_speed)
                references[direction.name] = directional_reference(
                    vertices, direction.translation, direction.angular, direction.pivot
                )
        except Exception as error:
            reference_error = f"Reference failed: {type(error).__name__}: {error}"
        source = cad.Fixture(
            fixture.vertices,
            fixture.gap,
            fixture.height,
            fixture.stock_bounds,
            fixture.stock_depth,
        )
        for placement in placements():
            preparation: dict[str, Any] = {
                "group_id": fixture.group_id,
                "family": fixture.family,
                "placement": placement.name,
                "scale": placement.scale,
                "nominal_construction_gap": fixture.gap,
            }
            rows = [
                {
                    "group_id": fixture.group_id,
                    "family": fixture.family,
                    "placement": placement.name,
                    "scale": placement.scale,
                    "direction": direction.name,
                    "reference": references.get(direction.name),
                    "normalization": normalizations.get(direction.name),
                }
                for direction in directions
            ]
            try:
                if reference_error is not None:
                    raise RuntimeError(reference_error)
                before = time.perf_counter()
                base_fixed, base_moving = cad.build_pair(source)
                preparation["selection_fixture_build_seconds"] = (
                    time.perf_counter() - before
                )
                before = time.perf_counter()
                fixed_id = known_support_face_id(base_fixed, (0, 0, 1), 0)
                moving_id = known_support_face_id(base_moving, (0, 0, -1), 0)
                preparation["face_selection_seconds"] = time.perf_counter() - before
                world = cad.placement_from_matrix(
                    placement.scale, placement.origin, placement.rotation
                )
                before = time.perf_counter()
                fixed, moving = cad.build_pair(source, placement=world)
                preparation["build_seconds"] = time.perf_counter() - before
                preparation["source_validity"] = [
                    bool(BRepCheck_Analyzer(shape).IsValid())
                    for shape in (fixed, moving)
                ]
                before = time.perf_counter()
                prepared = prepare_planar_contact_reference(
                    fixed,
                    moving,
                    fixed_face_id=fixed_id,
                    moving_face_id=moving_id,
                    mode="projected_nominal",
                )
                preparation["prepare_seconds"] = time.perf_counter() - before
                preparation["diagnostics"] = asdict(prepared.diagnostics)
                transformed = [transform_direction(d, placement) for d in directions]
                motion = (
                    jnp.asarray([d.translation for d in transformed]),
                    jnp.asarray([d.angular for d in transformed]),
                    jnp.asarray([d.pivot for d in transformed]),
                )
                rotation = np.asarray(placement.rotation)
                observed_normal = np.cross(
                    np.asarray(prepared.patch.basis)[:, 0],
                    np.asarray(prepared.patch.basis)[:, 1],
                )
                if not np.allclose(observed_normal, rotation[:, 2], rtol=0, atol=1e-14):
                    raise ValueError(
                        "Selected transformed face lost its fixture normal"
                    )
                before = time.perf_counter()
                direct = prepare_planar_contact_patch(
                    placement.scale * vertices,
                    origin=np.asarray(placement.origin),
                    basis=rotation[:, :2],
                )
                preparation["direct_patch_prepare_seconds"] = (
                    time.perf_counter() - before
                )
                values = []
                for label, patch in (("prepared", prepared.patch), ("direct", direct)):
                    size = len(patch.vertices)
                    arguments = (patch, *motion)
                    cache_key = (size, len(directions))
                    if cache_key not in cache:
                        executable, runtime = compile_timed(batch, arguments)
                        cache[cache_key] = executable
                        result["runtime"].append(
                            {
                                "vertices": size,
                                "directions_per_batch": len(directions),
                                "runtime": asdict(runtime),
                            }
                        )
                    before = time.perf_counter()
                    measured = np.asarray(cache[cache_key](*arguments))
                    preparation[f"{label}_batch_seconds"] = time.perf_counter() - before
                    if not np.isfinite(measured).all():
                        raise FloatingPointError("Nonfinite prepared directional value")
                    values.append(measured)
                preparation["candidate_panels"] = {}
                start = len(audit_directions())
                for panel, count in (
                    ("bounded_tilt_choice", 8),
                    ("unconstrained_controls", 12),
                ):
                    panel_motion = tuple(item[start : start + count] for item in motion)
                    panel_arguments: tuple[Any, ...] = (prepared.patch, *panel_motion)
                    cache_key = (len(prepared.patch.vertices), count)
                    if cache_key not in cache:
                        executable, runtime = compile_timed(batch, panel_arguments)
                        cache[cache_key] = executable
                        result["runtime"].append(
                            {
                                "vertices": cache_key[0],
                                "directions_per_batch": count,
                                "runtime": asdict(runtime),
                            }
                        )
                    before = time.perf_counter()
                    panel_values = np.asarray(cache[cache_key](*panel_arguments))
                    seconds = time.perf_counter() - before
                    if not np.isfinite(panel_values).all():
                        raise FloatingPointError("Nonfinite candidate panel values")
                    preparation["candidate_panels"][panel] = {
                        "candidate_count": count,
                        "evaluation_seconds": seconds,
                        "working_values": (panel_values / placement.scale**3).tolist(),
                        "max_difference_from_full_batch": float(
                            np.max(
                                np.abs(panel_values - values[0][start : start + count])
                            )
                            / placement.scale**3
                        ),
                    }
                for row, value, direct_value in zip(rows, *values, strict=True):
                    working = float(value / placement.scale**3)
                    row.update(
                        value=float(value),
                        working_value=working,
                        direct_working_value=float(direct_value / placement.scale**3),
                        normalized_error=abs(working - row["reference"])
                        / row["normalization"],
                    )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                preparation["error"] = message
                for row in rows:
                    row["error"] = message
            result["preparations"].append(preparation)
            result["rows"].extend(rows)
    return result
