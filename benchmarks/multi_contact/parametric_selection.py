"""Frozen single-path policies using only their counted CAD observations."""

from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from brepax.experimental.planar_contact import planar_contact_set_volume_direction
from brepax.experimental.planar_contact_collection_cad import (
    prepare_planar_contact_set_reference,
)
from brepax.experimental.polyhedral_targets import (
    TranslationTargetResult,
    TranslationVolumeRange,
    polyhedral_translation_range,
    polyhedral_translation_targets,
)
from brepax.experimental.polyhedral_translation import (
    PreparedPolyhedralTranslation,
    prepare_polyhedral_translation,
)

from .parametric_cad import (
    CadConstructionCounts,
    CadMeasurement,
    CadScene,
    observe_scene,
    prepare_scene,
    rebuild_and_observe,
    valid_observation,
)
from .parametric_fixtures import ParametricFixture

Method = Literal["finite", "first_order", "cad"]
_RATE = jax.jit(planar_contact_set_volume_direction)


@dataclass(frozen=True)
class PreparedPolicy:
    """One method's paid preparation; observations are never shared across methods."""

    fixture: ParametricFixture
    method: Method
    scene: CadScene | None
    initial: CadMeasurement | None
    profile: PreparedPolyhedralTranslation | None
    volume_range: TranslationVolumeRange | None
    rate: float | None
    geometry_seconds: float
    model_seconds: float
    first_jit_seconds: float
    setup_seconds: float
    fusion_calls: int
    error: str | None


@dataclass(frozen=True)
class PolicyOutcome:
    """Selected edit and complete attempted-query trace for one target."""

    target: float
    budget: int
    decision: Literal["selected", "unreachable", "unresolved", "unsupported", "failed"]
    selected_step: float | None
    probes: tuple[CadMeasurement, ...]
    validation: CadMeasurement | None
    target_result: TranslationTargetResult | None
    selection_seconds: float
    total_seconds: float
    error: str | None = None


def prepare_policy(fixture: ParametricFixture, method: Method) -> PreparedPolicy:
    """Include source construction, initial Common and the method's real setup."""
    started = perf_counter()
    counts = CadConstructionCounts()
    scene = None
    initial = None
    profile = None
    volume_range = None
    rate = None
    geometry_seconds = model_seconds = jit_seconds = 0.0
    error = None
    try:
        bodies = fixture.bodies()
        geometry_started = perf_counter()
        try:
            scene = prepare_scene(bodies, counts=counts)
        finally:
            geometry_seconds = perf_counter() - geometry_started
        initial = observe_scene(scene, 0.0)
        if not valid_observation(initial) or initial.working_volume is None:
            raise RuntimeError("Initial CAD observation failed")
        if abs(initial.working_volume) > 1e-10 * fixture.a**3:
            raise RuntimeError("Initial CAD bodies have nonzero overlap")
        model_started = perf_counter()
        try:
            if method == "finite":
                profile = prepare_polyhedral_translation(
                    [prism.vertices() for prism in bodies.fixed],
                    [prism.vertices() for prism in bodies.moving],
                    bodies.velocity,
                    max_step=bodies.max_step,
                )
                volume_range = polyhedral_translation_range(
                    profile,
                    step_tolerance=1e-12 * bodies.max_step,
                    volume_tolerance=1e-12 * fixture.a**3,
                )
            elif method == "first_order":
                contact = prepare_planar_contact_set_reference(
                    scene.fixed_cells, scene.moving_cells, mode="zero_residual"
                )
                jit_started = perf_counter()
                value = _RATE(
                    contact.contact_set,
                    jnp.asarray(bodies.velocity),
                    jnp.zeros(3),
                    jnp.zeros(3),
                )
                value.block_until_ready()
                jit_seconds = perf_counter() - jit_started
                rate = float(value)
            elif method != "cad":
                raise ValueError("Unknown finite-edit method")
        finally:
            model_seconds = perf_counter() - model_started - jit_seconds
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    return PreparedPolicy(
        fixture,
        method,
        scene,
        initial,
        profile,
        volume_range,
        rate,
        geometry_seconds,
        model_seconds,
        jit_seconds,
        perf_counter() - started,
        counts.fusion_calls,
        error,
    )


def _next_cad_step(
    observations: list[CadMeasurement], target: float, extent: float
) -> float | None:
    """Safeguarded observed-bracket interpolation or largest-gap subdivision."""
    used = {record.step for record in observations}
    if extent not in used:
        return extent
    good = sorted(
        (record.step, float(record.working_volume))
        for record in observations
        if valid_observation(record) and record.working_volume is not None
    )
    brackets = [
        (left, right)
        for left, right in pairwise(good)
        if (left[1] - target) * (right[1] - target) < 0
    ]
    if brackets:
        left, right = min(
            brackets, key=lambda pair: (pair[1][0] - pair[0][0], pair[0][0])
        )
        width = right[0] - left[0]
        guess = left[0] + (target - left[1]) * width / (right[1] - left[1])
        proposed = float(np.clip(guess, left[0] + width / 20, right[0] - width / 20))
        if proposed not in used:
            return proposed
    partitions = sorted(used | {0.0, extent})
    for lower, upper in sorted(
        pairwise(partitions), key=lambda pair: (pair[0] - pair[1], pair[0])
    ):
        midpoint = (lower + upper) / 2
        if midpoint not in used:
            return midpoint
    return None


def select_target(policy: PreparedPolicy, target: float, budget: int) -> PolicyOutcome:
    """Attempt at most ``budget`` CAD probes, followed by independent final Cut.

    No reference formula, global endpoint scan or another policy's observations
    is available here. Absence of a sampled bracket never returns unreachable.
    The finite method's exact-model unreachable decision is a separate outcome.
    """
    if not np.isfinite(target) or target <= 0 or budget < 1:
        raise ValueError("Expected positive target volume and probe budget")
    started = perf_counter()
    if policy.error or policy.scene is None or policy.initial is None:
        return PolicyOutcome(
            target,
            budget,
            "unsupported",
            None,
            (),
            None,
            None,
            0.0,
            perf_counter() - started,
            policy.error,
        )
    probes: list[CadMeasurement] = []
    target_result = None
    try:
        if policy.method == "finite":
            if policy.profile is None:
                raise RuntimeError("Missing prepared finite profile")
            target_result = polyhedral_translation_targets(
                policy.profile,
                target,
                step_tolerance=1e-12 * policy.fixture.max_step,
                volume_tolerance=1e-12 * policy.fixture.a**3,
            )
            if target_result.status != "reachable":
                return PolicyOutcome(
                    target,
                    budget,
                    target_result.status,
                    None,
                    (),
                    None,
                    target_result,
                    perf_counter() - started,
                    perf_counter() - started,
                )
            steps = [
                root.step
                for root in target_result.roots
                if root.resolved and root.step is not None
            ]
            steps += [
                interval.step
                for interval in target_result.intervals
                if interval.step is not None
            ]
            if not steps:
                raise RuntimeError("Reachable target had no resolved action")
            probes.append(observe_scene(policy.scene, min(steps)))
        else:
            extent = policy.fixture.max_step
            step = (
                float(np.clip(target / policy.rate, np.nextafter(0.0, 1.0), extent))
                if policy.rate is not None and policy.rate > 0
                else extent / 2
            )
            for _ in range(budget):
                probes.append(observe_scene(policy.scene, step))
                latest = probes[-1]
                if (
                    valid_observation(latest)
                    and latest.working_volume is not None
                    and abs(latest.working_volume - target) <= 1e-10 * target
                ):
                    break
                next_step = _next_cad_step([policy.initial, *probes], target, extent)
                if next_step is None:
                    break
                step = next_step
        selected = min(
            (
                record
                for record in probes
                if valid_observation(record) and record.working_volume is not None
            ),
            key=lambda record: (
                abs(cast(float, record.working_volume) - target),
                record.step,
            ),
            default=None,
        )
        selection_seconds = perf_counter() - started
        if selected is None:
            return PolicyOutcome(
                target,
                budget,
                "failed",
                None,
                tuple(probes),
                None,
                target_result,
                selection_seconds,
                perf_counter() - started,
                "No valid CAD probe",
            )
        validation = rebuild_and_observe(
            policy.fixture.bodies(), selected.step, cut=True
        )
        return PolicyOutcome(
            target,
            budget,
            "selected",
            selected.step,
            tuple(probes),
            validation,
            target_result,
            selection_seconds,
            perf_counter() - started,
        )
    except Exception as error:
        return PolicyOutcome(
            target,
            budget,
            "failed",
            None,
            tuple(probes),
            None,
            target_result,
            perf_counter() - started,
            perf_counter() - started,
            f"{type(error).__name__}: {error}",
        )
