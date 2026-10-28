"""Finite-CAD controls separating contact assumptions from numerical resolution."""

from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from brepax.experimental.planar_contact import (
    planar_contact_volume_direction,
    prepare_planar_contact_patch,
)

from .cad import (
    IDENTITY_PLACEMENT,
    Fixture,
    Motion,
    Placement,
    difference_sweep,
)

SQUARE = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
STRESS_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


@dataclass(frozen=True)
class ContactControl:
    """A prescribed physical condition with independently known volume limit."""

    name: str
    fixture: Fixture
    motion: Motion
    expected_direction: float
    contact_law_applicable: bool
    interpretation: str
    placement: Placement = IDENTITY_PLACEMENT
    patch_vertices: tuple[tuple[float, float], ...] = SQUARE
    # None means this sweep cannot independently resolve the limit with OCCT.
    assessed_step: float | None = 1e-4


def contact_controls() -> tuple[ContactControl, ...]:
    """Bounded controls, including deliberate violations and resolution stress."""
    fixture = Fixture(SQUARE)
    approach = Motion((0.0, 0.0, -1.0))
    mixed = Motion((0.2, -0.1, -0.25), (0.5, 0.7, 0.3))
    return (
        ContactControl(
            "normal_approach",
            fixture,
            approach,
            4.0,
            True,
            "Isolated face contact with exact linear overlap growth.",
        ),
        ContactControl(
            "positive_tilt",
            fixture,
            Motion(angular_velocity=(1.0, 0.0, 0.0)),
            1.0,
            True,
            "Positive tilt creates a wedge; finite slope is tan(t)/t.",
        ),
        ContactControl(
            "negative_tilt",
            fixture,
            Motion(angular_velocity=(-1.0, 0.0, 0.0)),
            1.0,
            True,
            "Opposite tilt also creates volume; one gradient cannot represent both.",
        ),
        ContactControl(
            "mixed_motion",
            fixture,
            mixed,
            1.4082738095238094,
            True,
            "Tiny mixed-speed wedges lose OCCT accuracy before complete collapse.",
        ),
        ContactControl(
            "tangential",
            fixture,
            Motion((1.0, -0.4, 0.0)),
            0.0,
            True,
            "Tangential motion remains on the contact plane with zero volume.",
        ),
        ContactControl(
            "normal_spin",
            fixture,
            Motion(angular_velocity=(0.0, 0.0, 1.0)),
            0.0,
            True,
            "Rotation about the plane normal creates no overlap.",
        ),
        ContactControl(
            "positive_gap",
            Fixture(SQUARE, gap=1e-3),
            approach,
            0.0,
            False,
            "Finite motion can close a gap; the initial derivative is zero.",
        ),
        ContactControl(
            "gap_below_occt_tolerance",
            Fixture(SQUARE, gap=1e-8),
            approach,
            0.0,
            False,
            "The gap is positive; the OCCT sweep cannot resolve its zero limit.",
            assessed_step=None,
        ),
        ContactControl(
            "initial_overlap",
            Fixture(SQUARE, gap=-1e-3),
            Motion((0.0, 0.0, 1.0)),
            -4.0,
            False,
            "Existing overlap can shrink; the positive contact law does not apply.",
        ),
        ContactControl(
            "second_contact_patch",
            Fixture(SQUARE, right_wall=True),
            Motion((1.0, 0.0, 0.0)),
            2.0,
            False,
            "Selecting only the bottom patch misses the simultaneous side contact.",
        ),
        ContactControl(
            "two_patch_approach",
            Fixture(SQUARE, right_wall=True),
            Motion((1.0, 0.0, -1.0)),
            6.0,
            False,
            "Bottom and side contribute; finite overlap is 6*t - 2*t**2.",
        ),
        ContactControl(
            "clipped_common_patch",
            Fixture(SQUARE, stock_bounds=(-0.5, 2.0, -2.0, 2.0)),
            approach,
            3.0,
            True,
            "The true common patch has area three, not the tool footprint area four.",
            patch_vertices=((-0.5, -1.0), (1.0, -1.0), (1.0, 1.0), (-0.5, 1.0)),
        ),
        ContactControl(
            "tiny_corner_cap",
            fixture,
            Motion((0.0, 0.0, 1.99), (-1.0, 1.0, 0.0)),
            (2.0 - 1.99) ** 3 / 6,
            True,
            "The epsilon**3/6 cap integral is lost below CAD resolution.",
        ),
        ContactControl(
            "milliscale_mixed",
            fixture,
            mixed,
            1.4082738095238094,
            True,
            "The OCCT tolerance prevents a resolved small-step derivative.",
            placement=Placement(1e-3, (0.013, -0.007, 0.004), (0.2, 0.4, -0.3)),
            assessed_step=None,
        ),
        ContactControl(
            "thin_finite_prism",
            Fixture(SQUARE, height=1e-3),
            approach,
            4.0,
            True,
            "Steps beyond prism thickness do not establish the local derivative.",
        ),
    )


def collect_cad_controls() -> dict[str, Any]:
    """Record every positive step, including valid but inaccurate OCCT results.

    JAX x64 must be enabled by the caller. Expected limits come from elementary
    box/wedge formulas; the mixed-speed constant was checked by independent
    high-precision slice integration. The fixed assessment step is 1e-4, with
    tolerance 5e-4*abs(reference)+1e-11. This checks a bounded approximation at
    that step, not all steps or arbitrary CAD. Two unresolved cases are reported
    without a CAD-limit pass; all their samples remain visible.
    """
    rows = []
    for control in contact_controls():
        patch = prepare_planar_contact_patch(
            np.asarray(control.patch_vertices),
            origin=np.zeros(3),
            basis=np.array(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        )
        patch_prediction = float(
            planar_contact_volume_direction(
                patch,
                jnp.asarray(control.motion.velocity),
                jnp.asarray(control.motion.angular_velocity),
                jnp.asarray(control.motion.pivot),
            )
        )
        samples = difference_sweep(
            control.fixture, control.motion, STRESS_STEPS, control.placement
        )
        threshold = 5e-4 * abs(control.expected_direction) + 1e-11
        departures = []
        for sample in samples[1:]:
            if sample.working_difference is not None:
                error = abs(sample.working_difference - control.expected_direction)
                if error > threshold:
                    departures.append({"step": sample.step, "absolute_error": error})
        assessed = next(
            (sample for sample in samples if sample.step == control.assessed_step), None
        )
        cad_check = (
            None
            if assessed is None
            else bool(
                assessed.error is None
                and assessed.working_difference is not None
                and abs(assessed.working_difference - control.expected_direction)
                <= threshold
            )
        )
        kernel_check = (
            None
            if not control.contact_law_applicable
            else bool(
                np.isclose(
                    patch_prediction, control.expected_direction, rtol=2e-12, atol=1e-14
                )
            )
        )
        rows.append(
            {
                **asdict(control),
                "selected_patch_prediction": patch_prediction,
                "kernel_reference_agreement": kernel_check,
                "cad_assessed_step_agreement": cad_check,
                "cad_absolute_error_threshold": threshold,
                "finite_difference_reference_departures": departures,
                "samples": [asdict(sample) for sample in samples],
            }
        )
    recorded_samples = [record for row in rows for record in row["samples"]]
    all_completed = all(
        record["error"] is None
        and record["valid"]
        and record["fixed_valid"]
        and record["moving_valid"]
        for record in recorded_samples
    )
    kernel_checks_pass = all(
        row["kernel_reference_agreement"] is not False for row in rows
    )
    cad_checks_pass = all(
        row["cad_assessed_step_agreement"] is not False for row in rows
    )
    return {
        "passed": all_completed and kernel_checks_pass and cad_checks_pass,
        "passed_scope": (
            "Every CAD attempt completed with valid shapes, applicable kernel "
            "references agreed, and designated CAD step checks agreed. "
            "Listed unresolved CAD limits are excluded from this bounded gate."
        ),
        "case_count": len(rows),
        "common_call_count": len(recorded_samples),
        "steps": STRESS_STEPS,
        "all_attempts_completed_with_valid_shapes": all_completed,
        "attempt_error_count": sum(
            record["error"] is not None for record in recorded_samples
        ),
        "invalid_shape_attempt_count": sum(
            record["valid"] is False
            or record["fixed_valid"] is False
            or record["moving_valid"] is False
            for record in recorded_samples
        ),
        "assessed_cad_limit_case_count": sum(
            row["cad_assessed_step_agreement"] is not None for row in rows
        ),
        "applicable_kernel_checks_pass": kernel_checks_pass,
        "assessed_cad_limit_checks_pass": cad_checks_pass,
        "unresolved_cad_limit_cases": [
            row["name"] for row in rows if row["cad_assessed_step_agreement"] is None
        ],
        "contact_law_inapplicable_cases": [
            row["name"] for row in rows if not row["contact_law_applicable"]
        ],
        "finite_difference_reference_departure_count": sum(
            len(row["finite_difference_reference_departures"]) for row in rows
        ),
        "interpretation": (
            "Shape validity does not certify derivative accuracy. Selected-patch "
            "predictions use working units. For inapplicable cases they are "
            "counterfactual diagnostics, not supported outputs. Reference "
            "departures include finite-step truncation and OCCT resolution loss; "
            "they are retained, without automatic attribution to the kernel."
        ),
        "rows": rows,
    }
