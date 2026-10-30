"""Changing geometry, event validity and independent inverse-position evidence."""

from fractions import Fraction
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brepax.experimental.polyhedral_parameter import (
    AffinePolyhedralCell,
    polyhedral_parameter_events,
    polyhedral_parameter_position_sensitivity,
    polyhedral_parameter_volume,
    prepare_polyhedral_parameter,
    specialize_polyhedral_parameter,
)
from brepax.experimental.polyhedral_targets import polyhedral_translation_targets
from brepax.experimental.polyhedral_translation import (
    polyhedral_translation_slope,
    polyhedral_translation_volume,
)


def _box(
    lower: tuple[float, float, float], upper: tuple[float, float, float]
) -> np.ndarray:
    return np.array(list(product(*zip(lower, upper, strict=True))), dtype=float)


def _ramp_cells(kind: str = "height", width: float = 0.25):
    fixed = np.array(
        [(x, y, z) for x, y in ((0, 0), (1, 0), (1, 1)) for z in (0, 1)], dtype=float
    )
    moving = _box((-0.125 - width, -1, -1), (-0.125, 2, 2))
    fixed_displacements = np.zeros((2, len(fixed), 3))
    moving_displacements = np.zeros((2, len(moving), 3))
    moving_displacements[0, :, 0] = -1
    if kind == "height":
        fixed_displacements[1, :, 2] = fixed[:, 2]
    else:
        moving_displacements[1, moving[:, 0] == -0.125 - width, 0] = -1
    return AffinePolyhedralCell(fixed, fixed_displacements), AffinePolyhedralCell(
        moving, moving_displacements
    )


def _ramp(kind: str = "height", width: float = 0.25):
    fixed, moving = _ramp_cells(kind, width)
    return prepare_polyhedral_parameter([fixed], [moving], (1, 0, 0), max_step=1.75)


def _ramp_reference(step: float, theta: np.ndarray, kind: str):
    gap = 0.125 + theta[0]
    height = 1 + theta[1] if kind == "height" else 1.0
    width = 0.25 + theta[1] if kind == "width" else 0.25
    front, back = step - gap, step - gap - width

    def integral(x: float) -> float:
        return float(np.clip(x, 0, 1) ** 2 / 2)

    def section(x: float) -> float:
        return x if 0 < x < 1 else 0.0

    area = integral(front) - integral(back)
    slope = height * (section(front) - section(back))
    parameter_gradient = np.array(
        [-slope, area if kind == "height" else height * section(back)]
    )
    return height * area, slope, parameter_gradient


@pytest.mark.parametrize("kind", ["height", "width"])
def test_live_shape_derivatives_and_moving_event_positions(kind: str) -> None:
    prepared = _ramp(kind)
    evaluate = jax.jit(polyhedral_parameter_volume)
    derivative = jax.jit(jax.grad(polyhedral_parameter_volume, argnums=(1, 2)))
    for theta in (
        np.zeros(2),
        np.array([1 / 32, 1 / 16]),
        np.array([-1 / 32, -1 / 16]),
    ):
        for step in (0.0625, 0.25, 0.625, 0.875, 1.25, 1.625):
            expected, slope, shape_gradient = _ramp_reference(step, theta, kind)
            np.testing.assert_allclose(
                evaluate(prepared.model, step, theta), expected, atol=2e-15
            )
            actual_slope, actual_gradient = derivative(prepared.model, step, theta)
            np.testing.assert_allclose(actual_slope, slope, atol=2e-15)
            np.testing.assert_allclose(actual_gradient, shape_gradient, atol=2e-15)
    events = jax.jit(polyhedral_parameter_events)(prepared.model, jnp.zeros(2))
    np.testing.assert_array_equal(events, [0, 0.125, 0.375, 1.125, 1.375, 1.75])
    event_gradient = jax.jacfwd(polyhedral_parameter_events, argnums=1)(
        prepared.model, jnp.zeros(2)
    )
    expected_gradient = np.zeros((6, 2))
    expected_gradient[1:5, 0] = 1
    if kind == "width":
        expected_gradient[[2, 4], 1] = 1
    np.testing.assert_array_equal(event_gradient, expected_gradient)


@pytest.mark.parametrize("kind", ["height", "width"])
def test_exact_specialization_connects_target_roots_and_parameter_sensitivity(
    kind: str,
) -> None:
    prepared = _ramp(kind)
    target = Fraction(15, 128)
    specialized = specialize_polyhedral_parameter(prepared, np.zeros(2))
    result = polyhedral_translation_targets(
        specialized, target, step_tolerance=1e-12, volume_tolerance=1e-14
    )
    assert result.status == "reachable"
    np.testing.assert_allclose(
        [root.step for root in result.roots], [23 / 32, 5 / 4], atol=1e-12
    )
    expected = (
        ([1, -15 / 32], [1, 15 / 112]) if kind == "height" else ([1, -11 / 8], [1, 1])
    )
    sensitivity = jax.jit(polyhedral_parameter_position_sensitivity)
    for root, derivative in zip(result.roots, expected, strict=True):
        assert root.resolved and root.step is not None
        np.testing.assert_allclose(
            polyhedral_translation_volume(specialized.profile, root.step),
            float(target),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            sensitivity(prepared.model, root.step, jnp.zeros(2)), derivative, atol=1e-12
        )
    if kind == "height":
        np.testing.assert_allclose(
            sensitivity(
                prepared.model,
                23 / 32,
                jnp.zeros(2),
                target_parameter_gradient=jnp.array([0.0, float(target)]),
            ),
            [1, 0],
            atol=1e-14,
        )
        # A public position argument also accepts an integer-valued step.
        np.testing.assert_allclose(
            sensitivity(prepared.model, 1, np.zeros(2)), [1, -0.75], atol=1e-14
        )


def test_specialization_exactly_substitutes_nonzero_parameters() -> None:
    prepared = _ramp("width")
    theta = np.array([1 / 32, -1 / 16])
    specialized = specialize_polyhedral_parameter(prepared, theta)
    assert specialized.exact_knots == (
        Fraction(0),
        Fraction(5, 32),
        Fraction(11, 32),
        Fraction(37, 32),
        Fraction(43, 32),
        Fraction(7, 4),
    )
    steps = jnp.array([0.0625, 0.25, 0.625, 1.25, 1.625])
    actual = jax.jit(jax.vmap(polyhedral_parameter_volume, in_axes=(None, 0, None)))(
        prepared.model, steps, theta
    )
    reference = jax.vmap(polyhedral_translation_volume, in_axes=(None, 0))(
        specialized.profile, steps
    )
    np.testing.assert_allclose(actual, reference, atol=2e-15)


def test_combined_direction_jvp_matches_the_same_physical_volume() -> None:
    prepared = _ramp()
    theta = jnp.array([1 / 32, 1 / 16])
    step = jnp.asarray(0.625)
    step_tangent = jnp.asarray(0.3)
    parameter_tangent = jnp.array([-0.2, 0.4])
    value, derivative = jax.jvp(
        lambda t, p: polyhedral_parameter_volume(prepared.model, t, p),
        (step, theta),
        (step_tangent, parameter_tangent),
    )
    expected, slope, gradient = _ramp_reference(
        float(step), np.asarray(theta), "height"
    )
    np.testing.assert_allclose(value, expected, atol=1e-15)
    np.testing.assert_allclose(
        derivative, slope * step_tangent + gradient @ parameter_tangent, atol=1e-15
    )


def test_custom_jvp_preserves_structural_zero_with_opposing_parameter_bases() -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    static = np.zeros((1, 8, 3))
    basis = static.copy()
    basis[:, :, 0] = 1e12
    prepared = prepare_polyhedral_parameter(
        [
            AffinePolyhedralCell(box, static),
            AffinePolyhedralCell(box + np.array([0, 2, 0]), static),
        ],
        [
            AffinePolyhedralCell(box + np.array([2, 0, 0]), basis),
            AffinePolyhedralCell(box + np.array([2.5, 2, 0]), -basis),
        ],
        (-0.7, 0, 0),
        max_step=5 / 0.7,
    )
    step, theta = 1.7 / 0.7, jnp.zeros(1)
    # Both entered cells have constant cross-section. Their physical theta
    # changes cancel exactly, despite distinct moving normalization origins.
    np.testing.assert_allclose(
        polyhedral_parameter_volume(prepared.model, step, theta), 0.9, atol=1e-14
    )
    gradient = jax.jit(jax.grad(polyhedral_parameter_volume, argnums=2))(
        prepared.model, step, theta
    )
    np.testing.assert_array_equal(gradient, [0.0])
    _, tangent = jax.jvp(
        lambda p: polyhedral_parameter_volume(prepared.model, step, p),
        (theta,),
        (jnp.array([1e-12]),),
    )
    assert float(tangent) == 0


def test_event_uncertainty_is_distinct_from_exact_host_event_limits() -> None:
    prepared = _ramp()
    theta = np.array([1 / 32, 0.0])
    event = 5 / 32
    for step in (event, np.nextafter(event, -np.inf), np.nextafter(event, np.inf)):
        assert np.isnan(polyhedral_parameter_volume(prepared.model, step, theta))
        assert np.isnan(
            jax.grad(polyhedral_parameter_volume, argnums=1)(
                prepared.model, step, theta
            )
        )
    fixed = specialize_polyhedral_parameter(prepared, theta)
    assert float(polyhedral_translation_volume(fixed.profile, event)) == 0
    assert float(polyhedral_translation_slope(fixed.profile, event)) == 0
    # A moving lower origin retains the small overlap instead of cancelling
    # O(gap**2) terms from a fixed midpoint expansion.
    step = event + 1e-10
    expected = (step - event) ** 2 / 2
    np.testing.assert_allclose(
        polyhedral_parameter_volume(prepared.model, step, theta),
        expected,
        rtol=1e-14,
        atol=0,
    )

    box = _box((0, 0, 0), (1, 1, 1))
    diagonal = prepare_polyhedral_parameter(
        [AffinePolyhedralCell(box, np.zeros((1, 8, 3)))],
        [AffinePolyhedralCell(box + np.array([2, 2, 2]), np.ones((1, 8, 3)))],
        (-1, -1, -1),
        max_step=5,
    )
    # A nearer upper-event expansion avoids cancellation of (1-u)**3.
    end = 3.125
    step = end - 1e-9
    np.testing.assert_allclose(
        polyhedral_parameter_volume(diagonal.model, step, jnp.array([0.125])),
        (end - step) ** 3,
        rtol=1e-14,
        atol=0,
    )


@pytest.mark.parametrize("theta", [(-0.25, 0.0), (0.0, -1.0), (0.0, 2.0)])
def test_invalid_chart_parameters_reject_and_do_not_return_zero_derivatives(
    theta: tuple[float, float],
) -> None:
    prepared = _ramp()
    parameters = np.array(theta)
    with pytest.raises(ValueError, match="validity region"):
        specialize_polyhedral_parameter(prepared, parameters)
    assert np.isnan(polyhedral_parameter_volume(prepared.model, 0.5, parameters))
    assert np.isnan(
        jax.grad(polyhedral_parameter_volume, argnums=2)(
            prepared.model, 0.5, parameters
        )
    ).all()
    assert np.isnan(polyhedral_parameter_events(prepared.model, parameters)).all()


def test_conflicting_duplicate_vertices_and_rotating_facets_reject() -> None:
    fixed, moving = _ramp_cells()
    vertices = np.asarray(moving.vertices)
    controls = np.asarray(moving.parameter_displacements)
    duplicate_vertices = np.concatenate((vertices, vertices[:1]))
    duplicate_controls = np.concatenate((controls, controls[:, :1]), axis=1)
    duplicate_controls[0, -1, 0] += 1
    with pytest.raises(ValueError, match="conflicting"):
        prepare_polyhedral_parameter(
            [fixed],
            [AffinePolyhedralCell(duplicate_vertices, duplicate_controls)],
            (1, 0, 0),
            max_step=1.75,
        )
    changed = controls.copy()
    changed[0, 0, 1] = 1
    with pytest.raises(ValueError, match="facet normal"):
        prepare_polyhedral_parameter(
            [fixed], [AffinePolyhedralCell(vertices, changed)], (1, 0, 0), max_step=1.75
        )


def test_event_functions_that_split_at_reference_are_not_merged() -> None:
    stock = _box((0, 0, 0), (1, 3, 1))
    a = _box((-2, 0, 0), (-1, 1, 1))
    b = _box((-2, 2, 0), (-1, 3, 1))
    static = np.zeros((1, 8, 3))
    forward = static.copy()
    forward[:, :, 0] = 1
    with pytest.raises(ValueError, match=r"nonregular|events tie"):
        prepare_polyhedral_parameter(
            [AffinePolyhedralCell(stock, static)],
            [AffinePolyhedralCell(a, forward), AffinePolyhedralCell(b, -forward)],
            (1, 0, 0),
            max_step=4,
        )
    # Equal stock/cutter width splits the plateau events under width changes.
    with pytest.raises(ValueError, match=r"nonregular|events tie"):
        _ramp("width", width=1.0)


def test_shared_face_motion_requires_an_open_partition_certificate() -> None:
    first = _box((0, 0, 0), (1, 1, 1))
    second = first + np.array([1, 0, 0])
    static = np.zeros((1, 8, 3))
    moving = static.copy()
    moving[:, :, 0] = -1
    with pytest.raises(ValueError, match="separation certificate"):
        prepare_polyhedral_parameter(
            [AffinePolyhedralCell(first, static), AffinePolyhedralCell(second, moving)],
            [AffinePolyhedralCell(first + np.array([4, 0, 0]), static)],
            (-1, 0, 0),
            max_step=5,
        )


def test_persistent_lower_dimensional_intersection_has_zero_volume() -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    static = np.zeros((1, 8, 3))
    prepared = prepare_polyhedral_parameter(
        [AffinePolyhedralCell(box, static)],
        [AffinePolyhedralCell(box + np.array([1, 0, 0]), static)],
        (0, 1, 0),
        max_step=2,
    )
    specialized = specialize_polyhedral_parameter(prepared, [0.0])
    for step in (0.25, 0.5, 0.75):
        assert (
            float(polyhedral_parameter_volume(prepared.model, step, jnp.zeros(1))) == 0
        )
        assert float(polyhedral_translation_volume(specialized.profile, step)) == 0
        assert np.isnan(
            polyhedral_parameter_position_sensitivity(
                prepared.model, step, jnp.zeros(1)
            )
        ).all()


def test_stationary_target_root_has_no_ordinary_position_sensitivity() -> None:
    vertices = np.array(
        [(x, y, z) for x, y in ((0, 0), (2, 0), (1, 1)) for z in (0, 1)], dtype=float
    )
    controls = np.zeros((2, len(vertices), 3))
    controls[1, :, 2] = vertices[:, 2]
    moving = _box((-0.625, -1, -1), (-0.125, 2, 2))
    displacement = np.zeros((2, 8, 3))
    displacement[0, :, 0] = -1
    prepared = prepare_polyhedral_parameter(
        [AffinePolyhedralCell(vertices, controls)],
        [AffinePolyhedralCell(moving, displacement)],
        (1, 0, 0),
        max_step=3,
    )
    maximum_step = 1.375
    specialized = specialize_polyhedral_parameter(prepared, np.zeros(2))
    np.testing.assert_allclose(
        polyhedral_translation_volume(specialized.profile, maximum_step),
        7 / 16,
        atol=1e-15,
    )
    # The host model remains valid; cancellation of the interior numerical
    # derivative is conservatively unresolved rather than a false zero.
    assert np.isnan(
        polyhedral_parameter_volume(prepared.model, maximum_step, jnp.zeros(2))
    )
    assert np.isnan(
        polyhedral_parameter_position_sensitivity(
            prepared.model, maximum_step, jnp.zeros(2)
        )
    ).all()


def test_frame_and_unit_covariance_includes_parameter_displacements() -> None:
    fixed, moving = _ramp_cells()
    original = _ramp()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([3.0, -2.0, 4.0])
    theta = jnp.array([1 / 32, 1 / 16])
    derivative = jax.grad(polyhedral_parameter_volume, argnums=(1, 2))

    def transform(cell: AffinePolyhedralCell, scale: float) -> AffinePolyhedralCell:
        return AffinePolyhedralCell(
            scale * np.asarray(cell.vertices) @ rotation.T + shift,
            scale * np.asarray(cell.parameter_displacements) @ rotation.T,
        )

    for scale in (0.125, 8.0):
        prepared = prepare_polyhedral_parameter(
            [transform(fixed, scale)],
            [transform(moving, scale)],
            scale * rotation @ np.array([1.0, 0.0, 0.0]),
            max_step=1.75,
        )
        np.testing.assert_allclose(
            polyhedral_parameter_volume(prepared.model, 0.75, theta),
            scale**3 * polyhedral_parameter_volume(original.model, 0.75, theta),
            rtol=1e-14,
        )
        for actual, reference in zip(
            derivative(prepared.model, 0.75, theta),
            derivative(original.model, 0.75, theta),
            strict=True,
        ):
            np.testing.assert_allclose(actual, scale**3 * reference, rtol=1e-14)
        np.testing.assert_array_equal(
            polyhedral_parameter_events(prepared.model, theta),
            polyhedral_parameter_events(original.model, theta),
        )


def test_subnormal_parameters_and_target_gradients_cannot_bypass_guards() -> None:
    box = _box((0, 0, 0), (1, 1, 1))
    static = np.zeros((1, 8, 3))
    displacement = static.copy()
    displacement[:, :, 0] = 1e308
    prepared = prepare_polyhedral_parameter(
        [AffinePolyhedralCell(box, static)],
        [AffinePolyhedralCell(box + np.array([2, 0, 0]), displacement)],
        (-1, 0, 0),
        max_step=4,
    )
    evaluate = jax.jit(polyhedral_parameter_volume)
    for parameter in (-1e-309, -2e-308):
        assert np.isnan(evaluate(prepared.model, 1.5, jnp.array([parameter])))
        assert np.isnan(
            jax.jit(polyhedral_parameter_events)(prepared.model, jnp.array([parameter]))
        ).all()
    with pytest.raises(ValueError, match="validity region"):
        specialize_polyhedral_parameter(prepared, [-2e-308])
    # Host exact substitution distinguishes a valid subnormal from an overlap.
    exact = specialize_polyhedral_parameter(prepared, [-1e-309])
    assert float(polyhedral_translation_volume(exact.profile, 1.5)) > 0.59
    scale = 1e-100
    tiny = prepare_polyhedral_parameter(
        [AffinePolyhedralCell(scale * box, static)],
        [AffinePolyhedralCell(scale * (box + np.array([2, 0, 0])), static)],
        (-scale, 0, 0),
        max_step=4,
    )
    assert np.isnan(
        jax.jit(polyhedral_parameter_position_sensitivity)(
            tiny.model, 1.5, jnp.zeros(1), target_parameter_gradient=jnp.array([1e-309])
        )
    ).all()

    scale = 2.0**-332
    displacement = static.copy()
    displacement[:, :, 0] = scale
    scaled_gap = prepare_polyhedral_parameter(
        [AffinePolyhedralCell(scale * box, static)],
        [AffinePolyhedralCell(scale * (box + np.array([2, 0, 0])), displacement)],
        (-scale, 0, 0),
        max_step=4,
    )
    volume_scale = scale**3
    sensitivity = jax.jit(polyhedral_parameter_position_sensitivity)
    assert np.isnan(
        sensitivity(
            scaled_gap.model,
            1.5,
            jnp.zeros(1),
            target_parameter_gradient=jnp.array([-volume_scale + volume_scale / 2**30]),
        )
    ).all()
    np.testing.assert_array_equal(
        sensitivity(
            scaled_gap.model,
            1.5,
            jnp.zeros(1),
            target_parameter_gradient=jnp.array([-volume_scale]),
        ),
        [0.0],
    )


def test_normal_inputs_do_not_hide_a_subnormal_shape_derivative() -> None:
    scale = 2.0**-332
    stock, cutters = [], []
    for y, start, height_rate in ((0, -1, 1), (3, -2, -2)):
        vertices = _box((0, y, 0), (4, y + 1, 1))
        controls = np.zeros((1, 8, 3))
        controls[0, :, 2] = height_rate * vertices[:, 2]
        stock.append(AffinePolyhedralCell(scale * vertices, scale * controls))
        cutter = _box((start - 4, y - 1, -1), (start, y + 2, 2))
        cutters.append(AffinePolyhedralCell(scale * cutter, np.zeros((1, 8, 3))))
    prepared = prepare_polyhedral_parameter(stock, cutters, (scale, 0, 0), max_step=11)
    step = 3 + 2.0**-30
    # V_theta=-s**3*2**-30 is flushed by XLA although the inverse sensitivity
    # 2**-31 is normal. This unsupported numerical query must not report zero.
    assert np.isnan(
        jax.jit(polyhedral_parameter_volume)(prepared.model, step, jnp.zeros(1))
    )
    assert np.isnan(
        jax.jit(jax.grad(polyhedral_parameter_volume, argnums=2))(
            prepared.model, step, jnp.zeros(1)
        )
    ).all()
    assert np.isnan(
        jax.jit(polyhedral_parameter_position_sensitivity)(
            prepared.model, step, jnp.zeros(1)
        )
    ).all()


def test_narrow_chamber_may_be_exactly_valid_but_numerically_unresolved() -> None:
    width = 2.0**-48
    prepared = _ramp(width=width)
    step = 0.125 + width / 2
    assert np.isnan(polyhedral_parameter_volume(prepared.model, step, jnp.zeros(2)))
    specialized = specialize_polyhedral_parameter(prepared, np.zeros(2))
    expected = (step - 0.125) ** 2 / 2
    np.testing.assert_allclose(
        polyhedral_translation_volume(specialized.profile, step),
        expected,
        rtol=1e-14,
        atol=0,
    )
