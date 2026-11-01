"""Independent volume, directional, and inverse-set checks at an event exchange."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from brepax.experimental.polyhedral_boundary import (
    polyhedral_event_boundary_response,
    prepare_polyhedral_event_boundary,
    specialize_polyhedral_event_boundary,
)
from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell
from brepax.experimental.polyhedral_targets import polyhedral_translation_targets
from brepax.experimental.polyhedral_translation import (
    polyhedral_translation_volume,
    prepare_polyhedral_translation,
)


def _box(lower, upper):
    return np.array(list(product(*zip(lower, upper, strict=True))), dtype=float)


def _slab(kind):
    if kind == "ramp":
        fixed = np.array(
            [(x, y, z) for x, y in ((0, 0), (1, 0), (1, 1)) for z in (0, 1)],
            dtype=float,
        )
    else:
        fixed = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
    moving = _box((-1.125, -1, -1), (-0.125, 2, 2))
    displacement = np.zeros((1, 8, 3))
    displacement[0, moving[:, 0] == -1.125, 0] = -1
    velocity, maximum = (1, 0, 0), Fraction(5, 2)
    if kind == "tetra_reverse":
        moving = _box((2, -1, -1), (3, 2, 2))
        displacement[:] = 0
        displacement[0, moving[:, 0] == 3, 0] = 1
        velocity, maximum = (-1, 0, 0), Fraction(4)
    return (
        [AffinePolyhedralCell(fixed, np.zeros((1, len(fixed), 3)))],
        [AffinePolyhedralCell(moving, displacement)],
        velocity,
        maximum,
    )


def _prepare(family, radius=Fraction(1, 8)):
    fixed, moving, velocity, maximum = family
    return prepare_polyhedral_event_boundary(
        fixed, moving, velocity, max_step=maximum, parameter_radius=radius
    )


def _reference(kind, step, theta):
    def primitive(x):
        x = max(Fraction(0), min(Fraction(1), x))
        return x**2 / 2 if kind == "ramp" else (1 - (1 - x) ** 3) / 6

    if kind == "tetra_reverse":
        return primitive(3 + theta - step) - primitive(2 - step)
    return primitive(step - Fraction(1, 8)) - primitive(
        step - Fraction(1, 8) - 1 - theta
    )


@pytest.mark.parametrize("kind", ["ramp", "tetra", "tetra_reverse"])
def test_solution_set_changes_from_empty_to_point_to_interval(kind):
    family = _slab(kind)
    prepared = _prepare(family)
    target = Fraction(1, 2) if kind == "ramp" else Fraction(1, 6)
    for theta in (Fraction(-1, 8), Fraction(0), Fraction(1, 10)):
        specialized = specialize_polyhedral_event_boundary(prepared, theta)
        query = polyhedral_translation_targets(
            specialized, target, step_tolerance=1e-10, volume_tolerance=1e-12
        )
        assert query.status == ("unreachable" if theta < 0 else "reachable")
        if theta == 0:
            assert len(query.roots) == 1 and not query.intervals
            assert (
                query.roots[0].lower == query.roots[0].upper == prepared.tied_event_step
            )
        elif theta > 0:
            assert not query.roots and len(query.intervals) == 1
            assert query.intervals[0].lower == prepared.tied_event_step
            assert query.intervals[0].upper == prepared.tied_event_step + theta
        for step in (
            Fraction(1, 2),
            Fraction(1),
            Fraction(9, 8),
            Fraction(3, 2),
            Fraction(2),
        ):
            actual = polyhedral_translation_volume(specialized.profile, float(step))
            assert float(actual) == pytest.approx(
                float(_reference(kind, step, theta)), abs=2e-15
            )
    if kind != "ramp":
        # Float1/6 is a distinct target below the exact full stock volume.
        rounded = polyhedral_translation_targets(
            specialize_polyhedral_event_boundary(prepared, 0),
            float(target),
            step_tolerance=1e-10,
            volume_tolerance=1e-12,
        )
        assert len(rounded.roots) == 2 and not rounded.intervals


@pytest.mark.parametrize("kind", ["ramp", "tetra", "tetra_reverse"])
def test_same_exchange_can_have_different_directional_regularity(kind):
    prepared = _prepare(_slab(kind))
    for dt, dp in product(map(Fraction, (-1, 0, 1)), repeat=2):
        response = polyhedral_event_boundary_response(
            prepared,
            prepared.tied_event_step,
            step_direction=dt,
            parameter_direction=dp,
        )
        if kind == "ramp":
            expected = min(dt, 0)
        elif kind == "tetra":
            expected = -max(dt - dp, 0) / 2
        else:
            expected = min(dt, 0) / 2
        assert response.directional_derivative == expected
        assert response.left_parameter_derivative == (
            Fraction(1, 2) if kind == "tetra" else 0
        )
        assert response.right_parameter_derivative == 0
        assert response.ordinary_parameter_derivative == (
            None if kind == "tetra" else 0
        )
    # Equal scalar theta derivatives do not imply a linear joint direction map.
    forward = polyhedral_event_boundary_response(
        prepared, prepared.tied_event_step, step_direction=1, parameter_direction=0
    )
    backward = polyhedral_event_boundary_response(
        prepared, prepared.tied_event_step, step_direction=-1, parameter_direction=0
    )
    assert forward.directional_derivative + backward.directional_derivative != 0


def _contacts(equal=False):
    if equal:
        fixed = [_box((0, 0, 0), (1, 3, 1))]
        moving = [_box((-2, 0, 0), (-1, 1, 1)), _box((-2, 2, 0), (-1, 3, 1))]
    else:
        fixed = [_box((0, 0, 0), (4, 1, 1)), _box((0, 3, 0), (5, 5, 1))]
        moving = [
            _box((-3, -0.25, -0.25), (-1, 1.25, 1.25)),
            _box((-4, 2.75, -0.25), (-1, 5.25, 1.25)),
        ]
    d = np.zeros((1, 8, 3))
    d[0, :, 0] = 1
    return (
        [AffinePolyhedralCell(v, np.zeros_like(d)) for v in fixed],
        [AffinePolyhedralCell(moving[0], -d), AffinePolyhedralCell(moving[1], d)],
        (1, 0, 0),
        Fraction(4) if equal else Fraction(10),
    )


def test_independent_contacts_have_nonlinear_response_but_later_smooth_root():
    prepared = _prepare(_contacts())
    for dt, dp in product(map(Fraction, (-1, 0, 1)), repeat=2):
        response = polyhedral_event_boundary_response(
            prepared, 1, step_direction=dt, parameter_direction=dp
        )
        assert response.directional_derivative == max(dt - dp, 0) + 2 * max(dt + dp, 0)
        assert response.left_parameter_derivative == -1
        assert response.right_parameter_derivative == 2
        assert response.ordinary_parameter_derivative is None
    for theta in (Fraction(-1, 16), Fraction(0), Fraction(1, 16)):
        query = polyhedral_translation_targets(
            specialize_polyhedral_event_boundary(prepared, theta),
            Fraction(3, 8),
            step_tolerance=1e-10,
            volume_tolerance=1e-12,
        )
        root = Fraction(9, 8) - theta / 3
        assert any(r.lower <= root <= r.upper for r in query.roots)
    response = polyhedral_event_boundary_response(prepared, Fraction(9, 8))
    assert response.ordinary_parameter_derivative == 1


def test_rejects_other_degeneracies_and_open_domain_endpoints():
    with pytest.raises(ValueError, match="isolated pair"):
        _prepare(_contacts(equal=True))
    fixed, moving, velocity, maximum = _slab("ramp")
    with pytest.raises(ValueError, match=r"overlap|separation"):
        _prepare((fixed + fixed, moving, velocity, maximum))
    transverse = np.array(moving[0].vertices)
    controls = np.array(moving[0].parameter_displacements)
    face = transverse[:, 1] == -1
    transverse[face, 1], controls[0, face, 1] = 0, 1
    with pytest.raises(ValueError, match=r"inventories|non-event"):
        _prepare(
            (fixed, [AffinePolyhedralCell(transverse, controls)], velocity, maximum)
        )
    prepared = _prepare(_slab("ramp"))
    for endpoint in prepared.parameter_interval:
        assert endpoint is not None
        with pytest.raises(ValueError, match="outside"):
            specialize_polyhedral_event_boundary(prepared, endpoint)
    for step, direction in ((0, -1), (maximum, 1)):
        with pytest.raises(ValueError, match="exits"):
            polyhedral_event_boundary_response(prepared, step, step_direction=direction)


def test_exact_frame_unit_laws_and_fixed_parameter_rebuild():
    base = _slab("ramp")
    transform = np.array(((0, -1, 0), (0, 0, 1), (-1, 0, 0)))
    theta = Fraction(-1, 8)
    for scale in (Fraction(1, 16), Fraction(1), Fraction(16)):
        bodies = []
        for body in base[:2]:
            bodies.append(
                [
                    AffinePolyhedralCell(
                        float(scale) * (np.asarray(c.vertices) @ transform.T)
                        + np.array((8, -4, 16)),
                        float(scale)
                        * (np.asarray(c.parameter_displacements) @ transform.T),
                    )
                    for c in body
                ]
            )
        velocity = float(scale) * (np.asarray(base[2]) @ transform.T)
        prepared = _prepare((*bodies, velocity, base[3]))
        response = polyhedral_event_boundary_response(
            prepared, Fraction(9, 8), step_direction=-1
        )
        assert response.volume == scale**3 / 2
        assert response.directional_derivative == -(scale**3)
        targets = polyhedral_translation_targets(
            specialize_polyhedral_event_boundary(prepared, Fraction(1, 8)),
            scale**3 / 2,
            step_tolerance=1e-10,
            volume_tolerance=1e-12 * float(scale**3),
        )
        assert targets.status == "reachable" and not targets.roots
        assert len(targets.intervals) == 1
        assert (targets.intervals[0].lower, targets.intervals[0].upper) == (
            Fraction(9, 8),
            Fraction(5, 4),
        )
        rebuilt = prepare_polyhedral_translation(
            [
                np.asarray(c.vertices)
                + float(theta) * np.asarray(c.parameter_displacements)[0]
                for c in bodies[0]
            ],
            [
                np.asarray(c.vertices)
                + float(theta) * np.asarray(c.parameter_displacements)[0]
                for c in bodies[1]
            ],
            velocity,
            max_step=float(base[3]),
        )
        specialized = specialize_polyhedral_event_boundary(prepared, theta)
        for t in (1.0, 1.125, 1.5):
            expected = float(scale**3 * _reference("ramp", Fraction(t), theta))
            assert float(
                polyhedral_translation_volume(specialized.profile, t)
            ) == pytest.approx(expected, rel=1e-13)
            assert float(
                polyhedral_translation_volume(rebuilt.profile, t)
            ) == pytest.approx(expected, rel=1e-13)


def test_rational_centers_preserve_the_represented_decimal_family():
    stock = _box((0, 0, 0), (4, 3, 1))
    first = _box((-3, 0, 0), (-0.1, 1, 1))
    second = _box((-3, 2, 0), (-0.1, 3, 1))
    first_control, second_control = np.zeros((1, 8, 3)), np.zeros((1, 8, 3))
    first_control[0, :, 0], second_control[0, :, 0] = 0.1, -0.3
    fixed = [AffinePolyhedralCell(stock, np.zeros((1, 8, 3)))]
    moving = [
        AffinePolyhedralCell(first, first_control),
        AffinePolyhedralCell(second, second_control),
    ]
    prepared = prepare_polyhedral_event_boundary(
        fixed, moving, (1, 0, 0), max_step=1.5, parameter_radius=Fraction(1, 10)
    )
    theta, event = Fraction(1, 70), Fraction(0.1)
    # At the original onset only the first cutter enters. Its overlap equals
    # the represented displacement rate times the exact rational parameter.
    target = Fraction(0.1) * theta
    query = polyhedral_translation_targets(
        specialize_polyhedral_event_boundary(prepared, theta),
        target,
        step_tolerance=1e-10,
        volume_tolerance=0.0,
    )
    assert len(query.roots) == 1
    assert query.roots[0].step == float(event)
    assert query.roots[0].volume_residual == 0
    rounded = prepare_polyhedral_translation(
        [stock],
        [
            first + float(theta) * first_control[0],
            second + float(theta) * second_control[0],
        ],
        (1, 0, 0),
        max_step=1.5,
    )
    rounded_query = polyhedral_translation_targets(
        rounded,
        target,
        step_tolerance=1e-10,
        volume_tolerance=0.0,
    )
    assert len(rounded_query.roots) == 1
    assert rounded_query.roots[0].volume_residual != 0
