"""Construction-frame evidence, separate from unresolved source-CAD limits.

Explicit design dimensions can be rebuilt in suitable working units before
Boolean evaluation. Changing coordinates of a completed B-Rep also scales its
tolerances and does not recover missing source precision. Both routes and raw
external evaluation remain visible, with the same physical geometry and steps.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Literal

import numpy as np

from brepax._occt.backend import (
    BRepAlgoAPI_Common,
    BRepCheck_Analyzer,
    TopAbs_FACE,
    TopAbs_SOLID,
)
from brepax.experimental.applications.cad_frame import CadFrame

from .cad import (
    Fixture,
    Motion,
    Placement,
    VolumeSample,
    boundary_tolerance,
    build_pair,
    cad_volume,
    topology_count,
    transform_shape,
)
from .fixtures import polygon_area_centroid
from .references import directional_reference
from .stress import SQUARE

ResolutionRoute = Literal["raw_external", "reconstructed_design", "transformed_source"]
_ROUTES: tuple[ResolutionRoute, ...] = (
    "raw_external",
    "reconstructed_design",
    "transformed_source",
)
_ROTATIONS = (
    (0.0, 0.0, 0.0),
    (0.2, 0.4, -0.3),
    (-0.4, 0.2, 0.7),
    (0.8, -0.6, 0.2),
    (0.0, float(np.pi), 0.0),
)
_GAP = 1e-8
_GAP_STEPS = tuple(ratio * _GAP for ratio in (0.5, 0.9, 1.0, 1.1, 2.0, 10.0))
_MIXED_STEPS = (1e-3, 1e-4, 1e-5)
_MIXED = Motion((0.2, -0.1, -0.25), (0.5, 0.7, 0.3))


def _working_design(
    fixture: Fixture, motion: Motion, factor: float
) -> tuple[Fixture, Motion]:
    if fixture.right_wall:
        raise ValueError("Resolution reconstruction supports the convex prism fixture")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Construction scale ratio must be finite and positive")
    vertices = tuple(
        (float(x * factor), float(y * factor)) for x, y in fixture.vertices
    )
    xmin, xmax, ymin, ymax = fixture.stock_bounds
    vx, vy, vz = motion.velocity
    px, py, pz = motion.pivot
    return (
        Fixture(
            vertices,
            gap=fixture.gap * factor,
            height=fixture.height * factor,
            stock_bounds=(xmin * factor, xmax * factor, ymin * factor, ymax * factor),
            stock_depth=fixture.stock_depth * factor,
        ),
        Motion(
            (vx * factor, vy * factor, vz * factor),
            motion.angular_velocity,
            (px * factor, py * factor, pz * factor),
        ),
    )


def _resolution_sample(
    fixture: Fixture,
    motion: Motion,
    step: float,
    placement: Placement,
    frame: CadFrame,
    route: ResolutionRoute,
) -> dict[str, Any]:
    started = time.perf_counter()
    sample = VolumeSample(step)
    common_attempted = False
    original_source_tolerance = None
    measure_scale = 1.0 if route == "raw_external" else frame.length_scale
    try:
        if route == "reconstructed_design":
            design, twist = _working_design(
                fixture, motion, placement.scale / frame.length_scale
            )
            offset = (
                np.asarray(placement.origin) - np.asarray(frame.origin)
            ) / frame.length_scale
            working_placement = Placement(
                1.0,
                (float(offset[0]), float(offset[1]), float(offset[2])),
                placement.rotation_vector,
            )
            fixed, moving = build_pair(design, twist, step, working_placement)
        else:
            fixed, moving = build_pair(fixture, motion, step, placement)
            source_tolerances = (boundary_tolerance(fixed), boundary_tolerance(moving))
            original_source_tolerance = max(
                value for value in source_tolerances if value is not None
            )
            if route == "transformed_source":
                linear = np.eye(3) / frame.length_scale
                offset = -np.asarray(frame.origin) / frame.length_scale
                fixed = transform_shape(fixed, linear, offset)
                moving = transform_shape(moving, linear, offset)
        sample.fixed_valid = BRepCheck_Analyzer(fixed).IsValid()
        sample.moving_valid = BRepCheck_Analyzer(moving).IsValid()
        sample.fixed_volume = cad_volume(fixed) * measure_scale**3
        sample.moving_volume = cad_volume(moving) * measure_scale**3
        tolerances = (boundary_tolerance(fixed), boundary_tolerance(moving))
        sample.source_tolerance = (
            max(value for value in tolerances if value is not None) * measure_scale
        )
        common_attempted = True
        operation = BRepAlgoAPI_Common(fixed, moving)
        if not operation.IsDone():
            raise RuntimeError("OCCT Common did not finish")
        common = operation.Shape()
        if common.IsNull():
            raise RuntimeError("OCCT Common returned null shape")
        sample.volume = cad_volume(common) * measure_scale**3
        sample.working_volume = sample.volume / placement.scale**3
        sample.valid = BRepCheck_Analyzer(common).IsValid()
        sample.solid_count = topology_count(common, TopAbs_SOLID)
        sample.face_count = topology_count(common, TopAbs_FACE)
        tolerance = boundary_tolerance(common)
        sample.boundary_tolerance = (
            None if tolerance is None else tolerance * measure_scale
        )
    except Exception as error:
        sample.error = f"{type(error).__name__}: {error}"
    return {
        **asdict(sample),
        "original_source_tolerance": original_source_tolerance,
        "common_attempted": common_attempted,
        "seconds": time.perf_counter() - started,
    }


def resolution_sweep(
    fixture: Fixture,
    motion: Motion,
    steps: tuple[float, ...],
    *,
    placement: Placement,
    frame: CadFrame,
    route: ResolutionRoute,
) -> dict[str, Any]:
    """Measure one explicit design under a fixed, caller-specified CAD frame.

    The physical path is ``placement.origin + placement.scale * R * x(t)``,
    with x(t) given by the fixture/motion in design units. ``frame`` expresses
    external length units per construction unit and the construction origin
    in external coordinates. It is fixed for the entire sweep, never inferred
    from a step or from uncertain source topology. Rotations remain physical.

    ``reconstructed_design`` builds from rescaled dimensions before Common;
    ``transformed_source`` first builds external B-Reps and transforms those,
    retaining their tolerance semantics. Neither certifies imported CAD.
    ``raw_external`` evaluates the original coordinate route unchanged.

    Output volumes and tolerances use external units. ``working_volume`` and
    ``working_difference`` retain the existing CAD benchmark's design-volume
    units, so all routes compare directly. t is dimensionless. Every failed
    step remains with an error, including failures before a Boolean call.
    The reconstruction route supports convex-prism fixtures without the wall.
    """
    if route not in _ROUTES:
        raise ValueError("Unknown CAD resolution route")
    if any(not np.isfinite(step) or step <= 0 for step in steps):
        raise ValueError("Difference steps must be finite and positive")
    samples = [
        _resolution_sample(fixture, motion, step, placement, frame, route)
        for step in (0.0, *steps)
    ]
    initial = samples[0]["working_volume"]
    for sample in samples[1:]:
        if initial is not None and sample["working_volume"] is not None:
            sample["working_difference"] = (
                sample["working_volume"] - initial
            ) / sample["step"]
    return {
        "route": route,
        "geometry_source": (
            "Fresh CAD from trusted explicit design dimensions; no source repair"
            if route == "reconstructed_design"
            else "External B-Reps with their original stored tolerances"
        ),
        "fixture": asdict(fixture),
        "motion": asdict(motion),
        "physical_placement": asdict(placement),
        "construction_frame": asdict(frame),
        "source_cad_uncertainty_resolved": False,
        "common_call_count": sum(sample["common_attempted"] for sample in samples),
        "seconds": sum(sample["seconds"] for sample in samples),
        "samples": samples,
    }


def _footprints() -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    pentagon = tuple(
        (float(np.cos(angle)), float(np.sin(angle)))
        for angle in np.linspace(0, 2 * np.pi, 5, endpoint=False)
    )
    return (
        ("square", SQUARE),
        ("rectangle", ((-0.5, -0.8), (1.2, -0.8), (1.2, 0.8), (-0.5, 0.8))),
        ("triangle", ((-1.0, -0.75), (1.1, -0.5), (0.2, 1.1))),
        ("trapezoid", ((-1.1, -0.7), (1.2, -0.7), (0.8, 1.0), (-0.4, 1.0))),
        ("pentagon", pentagon),
    )


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for family in ("milliscale_mixed", "positive_gap"):
        for route in _ROUTES:
            selected = [
                row for row in rows if row["family"] == family and row["route"] == route
            ]
            samples = [sample for row in selected for sample in row["samples"]]
            assessed = [sample for sample in samples if sample.get("assessed")]
            errors = [
                sample["normalized_error"]
                for sample in assessed
                if sample["normalized_error"] is not None
            ]
            result[f"{family}/{route}"] = {
                "case_count": len(selected),
                "attempt_count": len(samples),
                "common_call_count": sum(
                    sample["common_attempted"] for sample in samples
                ),
                "failed_attempt_count": sum(
                    sample["error"] is not None for sample in samples
                ),
                "invalid_attempt_count": sum(
                    not (
                        sample["fixed_valid"]
                        and sample["moving_valid"]
                        and sample["valid"]
                    )
                    for sample in samples
                ),
                "assessed_count": len(assessed),
                "resolved_assessed_count": len(errors),
                "passed_count": sum(error <= 5e-4 for error in errors),
                "max_normalized_error": max(errors) if errors else None,
                "mean_normalized_error": float(np.mean(errors)) if errors else None,
                "seconds": sum(row["seconds"] for row in selected),
            }
    return result


def collect_resolution_controls() -> dict[str, Any]:
    """Compare fixed construction frames on five footprints and five rotations.

    Two original failed controls are retained within this expanded panel. The
    milliscale mixed case uses external scale 1e-3 and that fixed frame scale;
    its assessed t=1e-4 matches the previous control criterion. The positive
    design gap 1e-8 uses the explicit construction scale 1e-4, selected in the
    preceding experiment, with six steps before/at/after the event. Scaling an
    existing B-Rep is a measured alternative, not a substitute for rebuilding.

    Mixed-motion errors divide by the independent planar slice integral. Gap
    errors divide by footprint area and compare to the exact finite slope,
    area*max(t-gap,0)/t. The gate applies only to fresh design reconstruction,
    at normalized error <=5e-4; the original source-CAD limits remain unresolved.
    This deterministic panel is not learned generalization or imported-STEP
    healing. All attempts and costs remain, including raw numerical failures.
    """
    rows = []
    for shape, vertices in _footprints():
        area, _ = polygon_area_centroid(np.asarray(vertices))
        mixed_reference = directional_reference(
            vertices, _MIXED.velocity, _MIXED.angular_velocity, _MIXED.pivot
        )
        for rotation_id, rotation in enumerate(_ROTATIONS):
            for family in ("milliscale_mixed", "positive_gap"):
                mixed = family == "milliscale_mixed"
                fixture = Fixture(vertices, gap=0.0 if mixed else _GAP)
                motion = _MIXED if mixed else Motion((0.0, 0.0, -1.0))
                placement = Placement(
                    1e-3 if mixed else 1.0,
                    (0.013, -0.007, 0.004) if mixed else (0.0, 0.0, 0.0),
                    rotation,
                )
                frame = CadFrame(1e-3 if mixed else 1e-4, placement.origin)
                for route in _ROUTES:
                    row = resolution_sweep(
                        fixture,
                        motion,
                        _MIXED_STEPS if mixed else _GAP_STEPS,
                        placement=placement,
                        frame=frame,
                        route=route,
                    )
                    row.update(
                        family=family,
                        shape=shape,
                        rotation_id=rotation_id,
                        original_control=(
                            shape == "square" and rotation_id == (1 if mixed else 0)
                        ),
                        expected_initial_direction=mixed_reference if mixed else 0.0,
                    )
                    for sample in row["samples"]:
                        step = sample["step"]
                        if step == 0:
                            continue
                        expected = (
                            mixed_reference
                            if mixed
                            else area * max(step - _GAP, 0.0) / step
                        )
                        normalization = mixed_reference if mixed else area
                        observed = sample["working_difference"]
                        sample.update(
                            assessed=(not mixed or step == 1e-4),
                            reference_kind="directional_limit"
                            if mixed
                            else "exact_finite_difference",
                            expected_difference=expected,
                            normalized_error=(
                                None
                                if observed is None
                                else abs(observed - expected) / normalization
                            ),
                        )
                    rows.append(row)
    summaries = _summaries(rows)
    design = [
        summary
        for key, summary in summaries.items()
        if key.endswith("/reconstructed_design")
    ]
    passed = all(
        summary["passed_count"] == summary["assessed_count"]
        and summary["failed_attempt_count"] == 0
        and summary["invalid_attempt_count"] == 0
        for summary in design
    )
    return {
        "passed": passed,
        "passed_scope": (
            "Fresh explicit-design reconstruction in fixed working frames only; "
            "source-CAD uncertainty is not resolved."
        ),
        "source_cad_unresolved_cases": ["gap_below_occt_tolerance", "milliscale_mixed"],
        "shape_count": 5,
        "rotations_per_shape": len(_ROTATIONS),
        "physical_case_count": len(rows) // len(_ROUTES),
        "common_call_count": sum(row["common_call_count"] for row in rows),
        "unit_contract": (
            "Volumes/tolerances are external; working_volume/difference use "
            "original design volume units; t is dimensionless."
        ),
        "summaries": summaries,
        "rows": rows,
    }
