"""Compose contact, plate-breakout, and inverse-design measurements."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from brepax.experimental.contact import (
    _intersection_geometry,
    circle_rectangle_intersection_area,
    plate_with_hole_volume,
)

from .baselines import (
    TEMPERATURES,
    ContactKind,
    Measure,
    SampledFamily,
    Temperature,
    analytic_contact,
    contact_family,
    plate_family,
)
from .measure import (
    compile_measure,
    compile_timed,
    evaluate,
    forward_differences,
    radius_trajectory,
)
from .references import (
    breakout_volume_reference,
    circle_rectangle_area,
    contact_reference,
)


@dataclass(frozen=True)
class Case:
    name: str
    parameters: float | tuple[float, ...]
    value: float
    gradient: tuple[float, ...] | None
    relation: str
    reference: str
    event: str | None = None


DEFAULT_TEMPERATURE = Temperature(1.0, 1.0)
CONTACT_DELTAS = (-0.05, 0.0, 0.001, 0.01, 0.05, 0.2)
BREAKOUT_OFFSETS = (-0.2, -0.01, 0.0, 1e-6, 1e-3, 0.01, 0.2, 0.3)
BREAKOUT_CENTER = (1.5, 0.2)
BREAKOUT_EXTENTS = (3.0, 2.0)
BREAKOUT_HEIGHT = 2.0
TARGET_RADIUS = 1.7


def sampled_methods(
    family: SampledFamily,
) -> tuple[tuple[str, Measure, tuple[Temperature, ...]], ...]:
    """Retain every declared tuning; no test-set winner is selected."""
    return (
        ("C", family.stratum, (DEFAULT_TEMPERATURE,)),
        (
            "same_sigmoid_AD",
            family.sigmoid,
            tuple(Temperature(1.0, b) for b in (0.25, 0.5, 1.0)),
        ),
        ("A_intersection_extension", family.smoothing, TEMPERATURES),
    )


def boundary_methods(
    half_extents: tuple[float, float], height: float, *, remaining: bool
) -> tuple[tuple[str, Measure, tuple[Temperature, ...]], ...]:
    """Compare custom flux against ordinary AD of the same safe Green primal."""
    extents, thickness = jnp.asarray(half_extents), jnp.asarray(height)

    def flux(parameters: Array, unused_k: float, unused_beta: float) -> Array:
        center, radius = parameters[:2], parameters[2]
        if remaining:
            return plate_with_hole_volume(center, radius, extents, thickness)
        return thickness * circle_rectangle_intersection_area(center, radius, extents)

    def ordinary(parameters: Array, unused_k: float, unused_beta: float) -> Array:
        area = _intersection_geometry(parameters[:2], parameters[2], extents).area
        return thickness * (4 * jnp.prod(extents) - area if remaining else area)

    return (
        ("boundary_flux", flux, (DEFAULT_TEMPERATURE,)),
        ("same_Green_primal_AD", ordinary, (DEFAULT_TEMPERATURE,)),
    )


def measure_cases(
    methods: tuple[tuple[str, Measure, tuple[Temperature, ...]], ...],
    cases: tuple[Case, ...],
    *,
    suite: str,
    resolution: int | None,
    family: SampledFamily | None = None,
    differences_at: int = -1,
    classifier: Callable[[Array], Array] | None = None,
) -> dict[str, Any]:
    """Each attempted method/case survives compilation or evaluation failure."""
    rows, timings, differences = [], [], []
    classify = (
        jax.jit(classifier)
        if classifier is not None
        else None
        if family is None
        else jax.jit(family.classify)
    )
    for name, function, temperatures in methods:
        try:
            compiled = compile_measure(
                function, jnp.asarray(cases[differences_at].parameters)
            )
            timings.append(
                {
                    "method": name,
                    "resolution": resolution,
                    "forward": asdict(compiled.forward_runtime),
                    "backward": asdict(compiled.backward_runtime),
                }
            )
        except Exception as error:
            for temperature in temperatures:
                for case in cases:
                    rows.append(
                        {
                            "case": case.name,
                            "method": name,
                            "resolution": resolution,
                            **asdict(temperature),
                            "finite": False,
                            "error": f"Compilation: {type(error).__name__}: {error}",
                        }
                    )
            continue
        for temperature in temperatures:
            for case in cases:
                parameters = jnp.asarray(case.parameters)
                row = {
                    "case": case.name,
                    "method": name,
                    "resolution": resolution,
                    "parameters": case.parameters,
                    "geometric_relation": case.relation,
                    "reference": case.reference,
                    "event": case.event,
                    **asdict(temperature),
                    **evaluate(
                        compiled,
                        parameters,
                        temperature,
                        case.value,
                        None if case.gradient is None else np.asarray(case.gradient),
                    ),
                }
                if family is not None and classify is not None:
                    try:
                        label = int(classify(parameters))
                        row.update(
                            sampled_label=label,
                            false_disjoint=case.relation
                            in ("intersecting", "a_contained", "b_contained")
                            and label == 0,
                            classification_mismatch=label
                            != {
                                "disjoint": 0,
                                "contact": 0,
                                "intersecting": 1,
                                "a_contained": 2,
                                "b_contained": 3,
                            }.get(case.relation, label),
                            grid_points=family.grid_points,
                            cell_width=family.cell_width,
                            domain=family.domain,
                        )
                    except Exception as error:
                        row.update(finite=False, error=f"Classification: {error}")
                rows.append(row)
        try:
            case = cases[differences_at]
            differences.append(
                {
                    "method": name,
                    "parameters": case.parameters,
                    "temperature": asdict(DEFAULT_TEMPERATURE),
                    "steps": forward_differences(
                        compiled, jnp.asarray(case.parameters), DEFAULT_TEMPERATURE
                    ),
                }
            )
        except Exception as error:
            differences.append({"method": name, "error": str(error)})
    return {
        "suite": suite,
        "rows": rows,
        "timings": timings,
        "same_forward_fd": differences,
    }


def static_contacts(resolutions: tuple[int, ...]) -> list[dict[str, Any]]:
    """Point, parallel-line, and planar-face volume contact in fixed unit geometry."""
    reports = []
    kinds: tuple[ContactKind, ...] = ("sphere", "cylinder", "box")
    for kind in kinds:
        cases = []
        for delta in CONTACT_DELTAS:
            ref = contact_reference(kind, delta)
            cases.append(
                Case(
                    f"{kind}_delta_{delta:g}",
                    delta,
                    ref.value,
                    None if ref.derivative is None else (ref.derivative,),
                    "intersecting"
                    if delta > 0
                    else "contact"
                    if delta == 0
                    else "disjoint",
                    (
                        "Independent analytic/cross-section volume; "
                        "box at contact has slopes 0,4"
                    ),
                )
            )
        fixture = tuple(cases)
        for resolution in resolutions:
            family = contact_family(kind, resolution)
            reports.append(
                measure_cases(
                    sampled_methods(family),
                    fixture,
                    suite=kind,
                    resolution=resolution,
                    family=family,
                )
            )
        reports.append(
            measure_cases(
                (("analytic_AD", analytic_contact(kind), (DEFAULT_TEMPERATURE,)),),
                fixture,
                suite=kind,
                resolution=None,
            )
        )
    return reports


def plate_cases() -> tuple[Case, ...]:
    """Independent chord quadrature; endpoint gradients use the analytic limits."""
    fixtures = (
        ("contained", (0.2, 0.1, 1.0)),
        ("inside_tangent", (1.0, 0.0, 1.0)),
        ("opened_edge", (1.2, 0.0, 1.0)),
        ("center_on_edge", (2.0, 0.0, 1.0)),
        ("outside_tangent", (3.0, 0.0, 1.0)),
        ("corner_partial", (2.2, 3.1, 1.0)),
        ("center_on_corner", (2.0, 3.0, 1.0)),
        ("two_edge_tangency", (1.0, 2.0, 1.0)),
        ("rectangle_contained", (0.0, 0.0, 5.0)),
        ("corner_tangent", (2.6, 3.8, 1.0)),
    )
    result = []
    for name, parameters in fixtures:
        cx, cy, radius = parameters
        value = circle_rectangle_area((cx, cy), radius, (2.0, 3.0)).value
        gradient: tuple[float, ...]
        if name in ("contained", "inside_tangent", "two_edge_tangency"):
            gradient = (0.0, 0.0, 2 * math.pi * radius)
            source = "Chord quadrature value, full-disk analytic gradient limit"
        elif name in ("outside_tangent", "corner_tangent", "rectangle_contained"):
            gradient = (0.0, 0.0, 0.0)
            source = "Chord quadrature value, zero analytic gradient limit"
        elif name == "center_on_corner":
            gradient = (-radius, -radius, math.pi * radius / 2)
            source = "Chord quadrature value, exact quarter-circle boundary flux"
        else:
            p = np.asarray(parameters)
            step = 1e-5
            derivative = []
            for direction in np.eye(3):
                plus, minus = p + step * direction, p - step * direction
                vp = circle_rectangle_area(
                    tuple(plus[:2]), float(plus[2]), (2.0, 3.0)
                ).value
                vm = circle_rectangle_area(
                    tuple(minus[:2]), float(minus[2]), (2.0, 3.0)
                ).value
                derivative.append((vp - vm) / (2 * step))
            gradient = tuple(derivative)
            source = (
                "Independent chord quadrature and central FD h=1e-5 away from contact"
            )
        relation = (
            "b_contained"
            if name in ("contained", "inside_tangent", "two_edge_tangency")
            else "a_contained"
            if name == "rectangle_contained"
            else "contact"
            if name in ("outside_tangent", "corner_tangent")
            else "intersecting"
        )
        result.append(
            Case(
                name,
                parameters,
                value,
                gradient,
                relation,
                source,
                name if "tangent" in name else None,
            )
        )
    return tuple(result)


def plate_comparison(resolutions: tuple[int, ...]) -> list[dict[str, Any]]:
    cases = plate_cases()
    reports = []
    for resolution in resolutions:
        family = plate_family((2.0, 3.0), 1.0, resolution, ((-3.0, -4.0), (3.0, 4.0)))
        reports.append(
            measure_cases(
                sampled_methods(family),
                cases,
                suite="plate",
                resolution=resolution,
                family=family,
                differences_at=5,
            )
        )
    reports.append(
        measure_cases(
            boundary_methods((2.0, 3.0), 1.0, remaining=False),
            cases,
            suite="plate",
            resolution=None,
            differences_at=5,
        )
    )
    return reports


def breakout_cases() -> tuple[Case, ...]:
    result = []
    for offset in BREAKOUT_OFFSETS:
        radius = 1.5 + offset
        ref = breakout_volume_reference(radius)
        # This suite evaluates radius sensitivity only; the caller projects
        # the full [cx,cy,r] measure onto its last parameter.
        result.append(
            Case(
                f"radius_{radius:.8g}",
                radius,
                ref.value,
                (ref.derivative,),
                "b_contained" if offset <= 0 else "intersecting",
                (
                    "Independent single-edge formula; "
                    "second contact at r=1.8 has a C1 limit"
                ),
            )
        )
    return tuple(result)


def radius_projection(function: Measure) -> Measure:
    """Fix the center so both estimated and reference derivatives are dV/dr."""

    def measure(radius: Array, k: float, beta: float) -> Array:
        return function(jnp.array([*BREAKOUT_CENTER, radius]), k, beta)

    return measure


def breakout_comparison(resolutions: tuple[int, ...], steps: int) -> dict[str, Any]:
    families: list[
        tuple[
            int | None,
            tuple[tuple[str, Measure, tuple[Temperature, ...]], ...],
            SampledFamily | None,
        ]
    ] = []
    for resolution in resolutions:
        family = plate_family(
            BREAKOUT_EXTENTS,
            BREAKOUT_HEIGHT,
            resolution,
            ((-4.0, -3.0), (4.0, 3.0)),
            remaining_volume=True,
        )
        families.append((resolution, sampled_methods(family), family))
    families.append(
        (
            None,
            boundary_methods(BREAKOUT_EXTENTS, BREAKOUT_HEIGHT, remaining=True),
            None,
        )
    )
    target = breakout_volume_reference(TARGET_RADIUS).value

    def independent_volume(radius: float) -> float:
        area = circle_rectangle_area(BREAKOUT_CENTER, radius, BREAKOUT_EXTENTS).value
        return BREAKOUT_HEIGHT * (4 * math.prod(BREAKOUT_EXTENTS) - area)

    reports, trajectories = [], []
    for selected_resolution, methods, selected_family in families:
        projected = tuple(
            (name, radius_projection(function), temps)
            for name, function, temps in methods
        )
        classifier = None
        if selected_family is not None:
            original = selected_family.classify

            def classifier(
                radius: Array, classify: Callable[[Array], Array] = original
            ) -> Array:
                return classify(jnp.array([*BREAKOUT_CENTER, radius]))

        reports.append(
            measure_cases(
                projected,
                breakout_cases(),
                suite="breakout",
                resolution=selected_resolution,
                differences_at=3,
                family=selected_family,
                classifier=classifier,
            )
        )
        for name, function, temperatures in methods:
            for temperature in temperatures:
                row = {
                    "method": name,
                    "resolution": selected_resolution,
                    **asdict(temperature),
                }
                try:
                    row.update(
                        radius_trajectory(
                            function,
                            temperature,
                            target,
                            independent_volume,
                            steps=steps,
                            target_radius=TARGET_RADIUS,
                        )
                    )
                except Exception as error:
                    row.update(finite=False, error=f"{type(error).__name__}: {error}")
                trajectories.append(row)
    return {
        "measurements": reports,
        "trajectories": trajectories,
        "target_radius": TARGET_RADIUS,
        "target_volume": target,
        "budget_contract": (
            "Same iteration cap and solver. Warm snapshots exclude "
            "compilation and independent validation; capped runs "
            "explicitly marked. All temperatures retained; "
            "no test-set tuning winner."
        ),
    }


def run(
    contact_resolutions: tuple[int, ...],
    plate_resolutions: tuple[int, ...],
    suites: tuple[str, ...],
    optimization_steps: int,
    *,
    include_cad: bool = False,
) -> dict[str, Any]:
    """Run bounded experiments; failure to execute never becomes a silent pass."""
    report: dict[str, Any] = {
        "settings": {
            "contact_resolutions": contact_resolutions,
            "plate_resolutions": plate_resolutions,
            "suites": suites,
            "optimization_steps": optimization_steps,
            "x64": (jnp.asarray(0.0).dtype == jnp.float64),
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "temperatures": [asdict(t) for t in TEMPERATURES],
            "scope": (
                "Static continuous volumes only; C backward is surrogate; "
                "no arbitrary CAD kernel, surface-area, or general "
                "topology guarantee."
            ),
        },
        "measurements": [],
        "trajectories": [],
        "failures": [],
    }
    if not (jnp.asarray(0.0).dtype == jnp.float64):
        raise ValueError("Contact comparisons require JAX x64")
    started = time.perf_counter()
    operations: dict[str, Callable[[], Any]] = {
        "contact": lambda: static_contacts(contact_resolutions),
        "plate": lambda: plate_comparison(plate_resolutions),
        "breakout": lambda: breakout_comparison(plate_resolutions, optimization_steps),
    }
    for suite in suites:
        try:
            result = operations[suite]()
            if suite == "breakout":
                report["measurements"].extend(result.pop("measurements"))
                report["trajectories"].extend(result.pop("trajectories"))
                report["breakout"] = result
            else:
                report["measurements"].extend(result)
        except Exception as error:
            report["failures"].append(
                {"suite": suite, "error": f"{type(error).__name__}: {error}"}
            )
    if "plate" in suites:
        report["batched_boundary_timings"] = batched_boundary_timings()
    report["cad_assessed"] = include_cad
    if include_cad:
        try:
            report["cad"] = cad_comparison(report["trajectories"])
        except Exception as error:
            report["failures"].append({"suite": "cad", "error": str(error)})
    report["cad_validation_passed"] = (
        None
        if not include_cad
        else report.get("cad", {}).get("validation_passed", False)
    )
    report["wall_seconds"] = time.perf_counter() - started
    all_rows = [row for block in report["measurements"] for row in block["rows"]]
    finite_differences_ok = all(
        "error" not in item
        for block in report["measurements"]
        for item in block["same_forward_fd"]
    )
    batch_ok = all(
        row["finite"]
        for row in report.get("batched_boundary_timings", {}).get("rows", [])
    )
    cad_ok = not include_cad or report.get("cad", {}).get("execution_success", False)
    report["execution_success"] = (
        not report["failures"]
        and finite_differences_ok
        and batch_ok
        and cad_ok
        and all(row["finite"] for row in all_rows + report["trajectories"])
    )
    candidate_rows = [row for row in all_rows if row["method"] == "boundary_flux"]
    report["candidate_accuracy"] = {
        "assessed": bool(candidate_rows),
        "value_absolute_tolerance": 1e-10,
        "gradient_l2_tolerance": 3e-8,
        "passed": None
        if not candidate_rows
        else all(
            row["finite"]
            and row["value_absolute_error"] <= 1e-10
            and row["gradient_l2_error"] <= 3e-8
            for row in candidate_rows
        ),
        "scope": (
            "Unit-sized fixed fixtures; independent chord/analytic "
            "references, not a general proof"
        ),
    }
    return report


def batched_boundary_timings(batch_size: int = 256) -> dict[str, Any]:
    """Match batch sizes to reduce host-dispatch domination of cheap formulas."""
    seed = 20261028
    rng = np.random.default_rng(seed)
    parameters = np.column_stack(
        (
            rng.uniform(-2.5, 2.5, batch_size),
            rng.uniform(-3.5, 3.5, batch_size),
            rng.uniform(0.2, 1.8, batch_size),
        )
    )
    points = jnp.asarray(parameters)
    rows, gradients = [], []
    for name, function, _ in boundary_methods((2.0, 3.0), 1.0, remaining=False):
        try:
            arguments = (points, jnp.asarray(1.0), jnp.asarray(1.0))
            forward, ft = compile_timed(
                jax.vmap(function, in_axes=(0, None, None)), arguments, repetitions=9
            )
            backward, bt = compile_timed(
                jax.vmap(jax.grad(function), in_axes=(0, None, None)),
                arguments,
                repetitions=9,
            )
            values, derivative = (
                np.asarray(forward(*arguments)),
                np.asarray(backward(*arguments)),
            )
            finite = bool(
                np.all(np.isfinite(values)) and np.all(np.isfinite(derivative))
            )
            gradients.append(derivative)
            rows.append(
                {
                    "method": name,
                    "finite": finite,
                    "forward": asdict(ft),
                    "backward": asdict(bt),
                }
            )
        except Exception as error:
            rows.append({"method": name, "finite": False, "error": str(error)})
    agreement = None
    if len(gradients) == 2 and all(np.all(np.isfinite(g)) for g in gradients):
        agreement = float(np.max(np.abs(gradients[0] - gradients[1])))
    return {
        "batch_size": batch_size,
        "seed": seed,
        "rows": rows,
        "gradient_max_absolute_difference": agreement,
        "scope": (
            "Identical sampled parameters and batch sizes; shared-machine "
            "exploratory timings, no universal speed claim"
        ),
    }


def cad_comparison(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild geometry and actual inverse outcomes, outside optimization timing."""
    # CAD setup is optional and substantially heavier than analytic/grid sweeps.
    from brepax._occt.backend import BRepCheck_Analyzer

    from .cad import (
        PLACEMENTS,
        build_plate_cut,
        cad_volume,
        center_difference_sweep,
        verify_cad_transition,
    )

    radii = (1.0, 1.49, 1.5, 1.500001, 1.501, 1.51, TARGET_RADIUS, 1.8)
    refs = [breakout_volume_reference(r) for r in radii]
    checks = verify_cad_transition(
        radii,
        expected_volumes=[r.value for r in refs],
        expected_gradients=[r.derivative for r in refs],
    )
    center_checks: list[dict[str, Any]] = []
    for placement in PLACEMENTS:
        differences = center_difference_sweep(
            BREAKOUT_CENTER,
            TARGET_RADIUS,
            BREAKOUT_EXTENTS,
            BREAKOUT_HEIGHT,
            placement=placement,
            expected_gradient=(3.2, 0.0),
        )
        center_checks.append(
            {
                "placement": asdict(placement),
                "differences": [asdict(d) for d in differences],
            }
        )
    target_volume = breakout_volume_reference(TARGET_RADIUS).value
    # Identical actual radii share only their independent CAD rebuild, not an
    # optimizer result; each method/tuning still retains a separate outcome.
    rebuilt: dict[float, dict[str, Any]] = {}
    final_outcomes: list[dict[str, Any]] = []
    flux_checks = []
    for trajectory in trajectories:
        row = {
            key: trajectory[key]
            for key in ("method", "resolution", "k_over_h", "beta_over_h")
        }
        if not trajectory["finite"]:
            row.update(
                assessed=False, error="Optimizer did not produce a finite solution"
            )
            final_outcomes.append(row)
            continue
        radius = float(trajectory["final_radius"])
        if radius not in rebuilt:
            start = time.perf_counter()
            result: dict[str, Any] = {"assessed": True, "radius": radius}
            try:
                shape = build_plate_cut(
                    BREAKOUT_CENTER, radius, BREAKOUT_EXTENTS, BREAKOUT_HEIGHT
                )
                volume = cad_volume(shape)
                reference = BREAKOUT_HEIGHT * (
                    4 * math.prod(BREAKOUT_EXTENTS)
                    - circle_rectangle_area(
                        BREAKOUT_CENTER, radius, BREAKOUT_EXTENTS
                    ).value
                )
                result.update(
                    valid=bool(BRepCheck_Analyzer(shape).IsValid()),
                    cad_volume=volume,
                    target_residual=abs(volume - target_volume),
                    cad_vs_quadrature_error=abs(volume - reference),
                )
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
            result["seconds"] = time.perf_counter() - start
            rebuilt[radius] = result
        row.update(rebuilt[radius])
        final_outcomes.append(row)
        if trajectory["method"] == "boundary_flux":
            flux_checks.extend(
                verify_cad_transition((radius,), expected_volumes=(target_volume,))
            )

    execution_success = (
        all(
            check.error is None
            and all(d.error is None for d in check.differences or [])
            for check in checks + flux_checks
        )
        and all(d["error"] is None for row in center_checks for d in row["differences"])
        and all("error" not in row for row in final_outcomes)
    )

    def geometry_passes(selected: list[Any]) -> bool:
        return bool(selected) and all(
            check.error is None
            and check.valid
            and check.mesh is not None
            and check.mesh.closed_oriented
            and check.mesh.positive_volume
            and check.volume_absolute_error is not None
            and check.volume_absolute_error <= 1e-7
            for check in selected
        )

    target_geometry_passed = geometry_passes(
        [c for c in checks if c.radius == TARGET_RADIUS]
    )
    flux_final_passed = geometry_passes(flux_checks) if flux_checks else None
    return {
        "radius_checks": [asdict(check) for check in checks],
        "center_checks": center_checks,
        "final_inverse_outcomes": final_outcomes,
        "flux_final_geometry_checks": [asdict(check) for check in flux_checks],
        "execution_success": execution_success,
        "target_geometry_passed": target_geometry_passed,
        "flux_final_geometry_passed": flux_final_passed,
        "validation_passed": target_geometry_passed and flux_final_passed is not False,
        "target_volume_absolute_tolerance": 1e-7,
        "event_contract": (
            "r=1.5 and 1.8 are event diagnostics; a valid scalar volume does not "
            "guarantee a manifold contact mesh. The inverse target r=1.7 is "
            "strictly between contacts. Baseline target residuals are results, "
            "not execution failures. Actual flux final geometry is required "
            "when an inverse trajectory was run."
        ),
    }
