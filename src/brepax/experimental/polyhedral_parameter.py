"""Local design sensitivities for fixed-normal polyhedral translation paths.

Host preparation certifies an open parameter region for represented vertex
families. Within that region, JAX differentiates the changing volume rather
than a frozen set of polynomial coefficients. Events and their ordering remain
explicit; neither this chart nor its derivatives certify original CAD input.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from numpy.typing import ArrayLike

from brepax.experimental._polyhedral_parameter import (
    ExactParameterGeometry,
    parameter_derivative_polynomials,
    prepare_geometry,
    shift_parameter_polynomial,
)
from brepax.experimental.polyhedral_translation import (
    PreparedPolyhedralTranslation,
    _prepared_from_pieces,
)

_ROUNDOFF = 64 * np.finfo(np.float64).eps
_UNDERFLOW = 64 * np.finfo(np.float64).tiny


@dataclass(frozen=True)
class AffinePolyhedralCell:
    """Bounded convex cell with affine vertex displacements.

    ``vertices`` has shape (N, 3); ``parameter_displacements`` has shape
    (P, N, 3), with P equal to one or two. At dimensionless parameters theta,
    vertex i is ``vertices[i] + sum(theta[p] * displacements[p, i])``.
    Vertices and displacements use the same length unit. Preparation checks
    fixed facet normals and derives a conservative open validity region.
    """

    vertices: ArrayLike
    parameter_displacements: ArrayLike


class PolyhedralParameterModel(eqx.Module):
    """Packed changing-volume chart, evaluated only inside resolved chambers.

    Prepare on the host with :func:`prepare_polyhedral_parameter`. Floating
    event positions are rounded values, not exact contact classifications.
    Strict guards with roundoff bands reject uncertain parameter/chamber
    boundaries; use host specialization for exact model event diagnostics.
    """

    event_coefficients: Float[Array, "K Q"]
    parameter_guards: Float[Array, "G Q"]
    widths: Float[Array, " C"]
    coefficients: Float[Array, "C 2 M"]
    derivative_coefficients: Float[Array, "C 2 Q M"]
    terms: Int[Array, "M 3"]


@dataclass(frozen=True)
class ParameterTranslationEvent:
    """Affine combinatorial event; not necessarily a physical contact event.

    Its exact step is ``reference_step + parameter_gradient dot theta``.
    The derivative is valid only in the prepared open parameter region.
    """

    reference_step: Fraction
    parameter_gradient: tuple[Fraction, ...]


@dataclass(frozen=True)
class PreparedPolyhedralParameter:
    """Exact parameter certificates and their conservative numerical evaluator.

    Each ``parameter_constraints`` row is (constant, parameter coefficients)
    and must be strictly positive. These sufficient conditions need not be the
    largest valid region. The reference is theta=0. Event-order ties with
    different affine expressions are unsupported, even if volume is smooth.
    """

    model: PolyhedralParameterModel
    parameter_constraints: tuple[tuple[Fraction, ...], ...]
    events: tuple[ParameterTranslationEvent, ...]
    parameter_count: int
    pair_count: int
    _geometry: ExactParameterGeometry = field(repr=False)
    reference_kind: Literal["affine_represented_vertex_hulls"] = field(
        default="affine_represented_vertex_hulls", init=False
    )
    original_cad_contact_certified: Literal[False] = field(default=False, init=False)


def _packed(values: Sequence[Sequence[Fraction]], columns: int) -> Array:
    array = np.array(values, dtype=np.float64).reshape(-1, columns)
    if not np.isfinite(array).all():
        raise ValueError("Parameter chart coefficients overflow float64")
    if any(
        exact != 0 and abs(rounded) < np.finfo(np.float64).tiny
        for exact_row, row in zip(values, array, strict=True)
        for exact, rounded in zip(exact_row, row, strict=True)
    ):
        raise ValueError("Parameter chart coefficients underflow normal float64")
    return jnp.asarray(array)


def prepare_polyhedral_parameter(
    fixed_cells: Sequence[AffinePolyhedralCell],
    moving_cells: Sequence[AffinePolyhedralCell],
    velocity: ArrayLike | Sequence[float | int],
    *,
    max_step: float | int,
) -> PreparedPolyhedralParameter:
    """Prepare a local design chart for a specified finite translation.

    Each cell is a bounded, positive-volume vertex hull. Parameters are
    dimensionless and reference theta=0; there must be one or two shared
    parameters. Vertex displacements must preserve facet normals exactly for
    the represented inputs. Same-body interiors and initial cross-body
    interiors remain disjoint under the returned linear validity conditions.
    A nonregular reference, including an event tie that splits with theta,
    raises ``ValueError``. This rejects some otherwise valid volume families.

    The moving body follows ``vertex(theta) + step * velocity`` for
    0 <= step <= max_step. Step is dimensionless; velocity uses the vertex
    length unit. Geometry and velocity are host inputs; t and theta are the
    differentiation variables. Preparation enumerates plane triples and is
    intended for small cells. No CAD observations are fitted.

    First JAX derivatives are supported in strict, numerically resolved open chambers.
    Values at or near a changing event are NaN in that evaluator; specialize
    on the host for a complete fixed-parameter profile and one-sided slopes.
    A custom JVP uses exact-derived physical derivative polynomials to avoid
    cancellation in the moving normalization's floating chain rule. Prepared
    model arrays are explicitly frozen; only step and theta are differentiated.
    This separates discrete chart selection from changing-geometry evaluation.
    Neither an uncertain floating guard nor a source-CAD contact is certified.

    Exact global rigid frame changes preserve values and step/parameter
    derivatives. Scaling vertices, displacements and velocity by s scales all
    those quantities by s**3, keeping dimensionless theta and step fixed.
    Independently rounded transformed inputs can change validity or reject.
    A derivative with respect to a physical length parameter instead has the
    corresponding chain-rule units.

    Example::

        from itertools import product
        import jax
        import numpy as np

        jax.config.update("jax_enable_x64", True)
        box = np.array(list(product((0., 1.), repeat=3)))
        fixed = AffinePolyhedralCell(box, np.zeros((1, 8, 3)))
        shift = np.tile(np.array([1., 0., 0.]), (1, 8, 1))
        moving = AffinePolyhedralCell(box + (2., 0., 0.), shift)
        prepared = prepare_polyhedral_parameter(
            [fixed], [moving], (-1., 0., 0.), max_step=4.
        )
        # Initial gap grows by theta; the interior overlap is step - 1 - theta.
        volume = polyhedral_parameter_volume(prepared.model, 1.5, np.array([0.1]))
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Polyhedral parameters require JAX x64")
    if not fixed_cells or not moving_cells:
        raise ValueError("Both bodies require at least one convex cell")
    vector = np.asarray(velocity, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected finite velocity with shape (3,)")
    maximum = float(max_step)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Expected finite positive max_step")
    bodies = []
    count = None
    for body in (fixed_cells, moving_cells):
        arrays = []
        for cell in body:
            vertices = np.array(cell.vertices, dtype=np.float64, copy=True)
            displacements = np.array(
                cell.parameter_displacements, dtype=np.float64, copy=True
            )
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or displacements.ndim != 3
                or displacements.shape[1:] != vertices.shape
                or displacements.shape[0] not in (1, 2)
                or not np.isfinite(vertices).all()
                or not np.isfinite(displacements).all()
            ):
                raise ValueError(
                    "Expected finite vertices (N,3) and displacements (P,N,3), P=1 or 2"
                )
            if count is None:
                count = displacements.shape[0]
            elif displacements.shape[0] != count:
                raise ValueError("All cells must share the same parameter count")
            arrays.append((vertices, displacements))
        bodies.append(arrays)
    assert count is not None
    try:
        geometry = prepare_geometry(bodies[0], bodies[1], vector, maximum)
        terms = np.array(geometry.terms, dtype=np.int32)
        polynomials: list[tuple[Fraction, ...]] = []
        derivatives: list[tuple[Fraction, ...]] = []
        for i, lower_polynomial in enumerate(geometry.coefficients):
            lower_origin = geometry.origins[i]
            upper_origin = tuple(Fraction(float(q)) for q in geometry.knots[i + 1])
            displacement = tuple(
                (b - a) / geometry.widths[i]
                for a, b in zip(lower_origin, upper_origin, strict=True)
            )
            upper_polynomial = shift_parameter_polynomial(
                lower_polynomial, geometry.terms, displacement
            )
            for polynomial, origin in (
                (lower_polynomial, lower_origin),
                (upper_polynomial, upper_origin),
            ):
                polynomials.append(polynomial)
                derivatives.extend(
                    parameter_derivative_polynomials(
                        polynomial, geometry.terms, origin, geometry.widths[i]
                    )
                )
        model = PolyhedralParameterModel(
            _packed(geometry.knots, count + 1),
            _packed(geometry.guards, count + 1),
            _packed([geometry.widths], len(geometry.widths)).ravel(),
            _packed(polynomials, len(geometry.terms)).reshape(
                -1, 2, len(geometry.terms)
            ),
            _packed(derivatives, len(geometry.terms)).reshape(
                -1, 2, count + 1, len(geometry.terms)
            ),
            jnp.asarray(terms),
        )
    except (OverflowError, FloatingPointError) as error:
        raise ValueError("Parameter chart is not representable in float64") from error
    return PreparedPolyhedralParameter(
        model,
        geometry.guards,
        tuple(
            ParameterTranslationEvent(row[0], row[1:]) for row in geometry.knots[1:-1]
        ),
        count,
        geometry.pair_count,
        geometry,
    )


def specialize_polyhedral_parameter(
    prepared: PreparedPolyhedralParameter, parameters: ArrayLike | Sequence[float | int]
) -> PreparedPolyhedralTranslation:
    """Substitute theta exactly, yielding an ordinary finite translation profile.

    Supplied finite float64 theta values are interpreted as exact rationals.
    Strict validity guards are checked on the host; invalid parameters raise
    ``ValueError``. This uses the prepared affine family, not newly rounded
    vertex coordinates. Existing profile packing may reject unresolved events
    or underflow. The result supports the usual value, slope and target queries.
    """
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (prepared.parameter_count,) or not np.isfinite(values).all():
        raise ValueError("Expected one finite value per parameter")
    exact = tuple(Fraction(float(value)) for value in values)
    return _prepared_from_pieces(
        prepared._geometry.specialize(exact), prepared.pair_count
    )


def _query_array(
    values: ArrayLike | Sequence[float | int] | float | int, dtype: np.dtype | jnp.dtype
) -> tuple[Array, Array]:
    original = jnp.asarray(values)
    if jnp.issubdtype(original.dtype, jnp.complexfloating):
        raise ValueError("Queries must be real-valued")
    valid = jnp.all(jnp.isfinite(original))
    # XLA may flush subnormals before arithmetic comparisons. Inspect their
    # bits before converting a narrower input or evaluating any affine guard.
    if original.dtype == jnp.float64:
        bits = jax.lax.bitcast_convert_type(original, jnp.uint64)
        exponent, significand = 0x7FF0000000000000, 0x000FFFFFFFFFFFFF
    elif original.dtype == jnp.float32:
        bits = jax.lax.bitcast_convert_type(original, jnp.uint32)
        exponent, significand = 0x7F800000, 0x007FFFFF
    elif original.dtype == jnp.float16:
        bits = jax.lax.bitcast_convert_type(original, jnp.uint16)
        exponent, significand = 0x7C00, 0x03FF
    elif original.dtype == jnp.bfloat16:
        bits = jax.lax.bitcast_convert_type(original, jnp.uint16)
        exponent, significand = 0x7F80, 0x007F
    else:
        return original.astype(dtype), valid
    subnormal = ((bits & exponent) == 0) & ((bits & significand) != 0)
    return original.astype(dtype), valid & ~jnp.any(subnormal)


def _parameter_data(
    model: PolyhedralParameterModel, parameters: Float[Array, " P"] | ArrayLike
) -> tuple[Array, Array]:
    theta, valid = _query_array(parameters, model.coefficients.dtype)
    if theta.shape != (model.event_coefficients.shape[1] - 1,):
        raise ValueError("Expected one value per parameter")
    extended = jnp.concatenate((jnp.ones(1, dtype=theta.dtype), theta))
    margins = model.parameter_guards @ extended
    magnitudes = jnp.abs(model.parameter_guards) @ jnp.abs(extended)
    valid &= jnp.all(jnp.isfinite(theta)) & jnp.all(
        margins > _ROUNDOFF * magnitudes + _UNDERFLOW
    )
    return extended, valid


def polyhedral_parameter_events(
    model: PolyhedralParameterModel, parameters: Float[Array, " P"] | ArrayLike
) -> Float[Array, " K"]:
    """Return rounded affine knots, including zero and max_step.

    Supports jit/vmap and parameter AD inside the resolved parameter region.
    These rounded positions are not exact event classifications. Invalid or
    roundoff-ambiguous parameters return NaNs; host specialization retains the
    exact rational event positions. Some knots may be combinatorial only.
    """
    extended, valid = _parameter_data(model, parameters)
    values = model.event_coefficients @ extended
    return values * jnp.where(valid & jnp.all(jnp.isfinite(values)), 1.0, jnp.nan)


def _evaluate(
    model: PolyhedralParameterModel,
    step: Float[Array, ""] | Int[Array, ""] | float | int,
    parameters: Float[Array, " P"] | ArrayLike,
) -> tuple[Array, Array, Array]:
    extended, valid = _parameter_data(model, parameters)
    step_value, step_valid = _query_array(step, model.coefficients.dtype)
    step = step_value
    valid &= step_valid
    events = model.event_coefficients @ extended
    event_magnitudes = jnp.abs(model.event_coefficients) @ jnp.abs(extended)
    index = jnp.clip(
        jnp.searchsorted(events, step, side="right") - 1, 0, len(model.widths) - 1
    )
    bounds = jnp.array([index, index + 1])
    error = _ROUNDOFF * (jnp.abs(step) + event_magnitudes[bounds]) + _UNDERFLOW
    valid &= jnp.isfinite(step) & jnp.all(jnp.isfinite(events))
    valid &= (step - events[index] > error[0]) & (events[index + 1] - step > error[1])
    side = jnp.asarray(step - events[index] > events[index + 1] - step, dtype=jnp.int32)
    u = (step - events[index + side]) / model.widths[index]
    variables = jnp.concatenate(
        (jnp.ones(1, dtype=extended.dtype), jnp.atleast_1d(u), extended[1:])
    )
    coefficients = model.coefficients[index, side]
    factors = jnp.where(coefficients[:, None] != 0, variables[model.terms], 1.0)
    contributions = coefficients * jnp.prod(factors, axis=1)
    nonzero = (coefficients != 0) & jnp.all(factors != 0, axis=1)
    valid &= jnp.all(jnp.isfinite(contributions))
    valid &= jnp.all(~nonzero | (jnp.abs(contributions) >= np.finfo(np.float64).tiny))
    value = jnp.sum(contributions)
    value_magnitude = jnp.sum(jnp.abs(contributions))
    valid &= jnp.isfinite(value) & (
        (value_magnitude == 0) | (value > _ROUNDOFF * value_magnitude + _UNDERFLOW)
    )
    derivative_coefficients = model.derivative_coefficients[index, side]
    derivative_factors = jnp.where(
        derivative_coefficients[:, :, None] != 0,
        variables[model.terms][None, :, :],
        1.0,
    )
    derivative_terms = derivative_coefficients * jnp.prod(derivative_factors, axis=2)
    derivative_nonzero = (derivative_coefficients != 0) & jnp.all(
        derivative_factors != 0, axis=2
    )
    valid &= jnp.all(jnp.isfinite(derivative_terms))
    valid &= jnp.all(
        ~derivative_nonzero | (jnp.abs(derivative_terms) >= np.finfo(np.float64).tiny)
    )
    gradient = jnp.sum(derivative_terms, axis=1)
    gradient_magnitudes = jnp.sum(jnp.abs(derivative_terms), axis=1)
    valid &= jnp.all(
        (gradient_magnitudes == 0)
        | (jnp.abs(gradient) > _ROUNDOFF * gradient_magnitudes + _UNDERFLOW)
    )
    # Even a zero-volume chamber must not return a plausible zero derivative
    # outside its validity region. Keep each intended variable on the invalid
    # branch's derivative path; multiply before summing to avoid overflow.
    invalid = jnp.where(valid, 0.0, jnp.nan)
    value = value + invalid * step + jnp.sum(invalid * extended[1:])
    return value, gradient, gradient_magnitudes


@jax.custom_jvp
def _parameter_volume(
    model: PolyhedralParameterModel,
    step: Float[Array, ""],
    parameters: Float[Array, " P"],
) -> Float[Array, ""]:
    return _evaluate(model, step, parameters)[0]


@_parameter_volume.defjvp
def _parameter_volume_jvp(
    primals: tuple[PolyhedralParameterModel, Float[Array, ""], Float[Array, " P"]],
    tangents: tuple[object, Float[Array, ""], Float[Array, " P"]],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    model, step, parameters = primals
    _, step_tangent, parameter_tangent = tangents
    value, derivatives, _ = _evaluate(model, step, parameters)
    derivatives = derivatives * jnp.where(jnp.isfinite(value), 1.0, jnp.nan)
    tangent = derivatives[0] * step_tangent + jnp.dot(
        derivatives[1:], parameter_tangent
    )
    return value, tangent


def polyhedral_parameter_volume(
    model: PolyhedralParameterModel,
    step: Float[Array, ""] | Int[Array, ""] | float | int,
    parameters: Float[Array, " P"] | ArrayLike,
) -> Float[Array, ""]:
    """Evaluate volume with live step and design parameters in an open chamber.

    Supports jit, vmap and first JVP/reverse derivatives in step and theta.
    A custom JVP evaluates exact-derived derivatives of the same volume;
    it avoids cancellation in a floating moving-origin chain rule. Model
    arrays are frozen, and higher derivatives are outside the contract.
    Strict linear guards and conservative roundoff bands reject uncertain
    classification.
    At/near events, outside the parameter region or step interval, or when a
    required nonzero value/derivative monomial underflows, or cancellation
    prevents resolving a value/derivative sum, value and intended derivatives
    are NaN. This includes some valid interior stationary points. Subnormal
    query inputs also reject because XLA may flush them before arithmetic.
    Packing/evaluation is floating arithmetic, not an exact volume oracle.
    Use :func:`specialize_polyhedral_parameter` for boundary values and slopes.
    """
    theta, parameters_valid = _query_array(parameters, model.coefficients.dtype)
    position, step_valid = _query_array(step, model.coefficients.dtype)
    frozen_model = jax.tree.map(jax.lax.stop_gradient, model)
    value = _parameter_volume(frozen_model, position, theta)
    invalid = jnp.where(parameters_valid & step_valid, 0.0, jnp.nan)
    return value + invalid * position + jnp.sum(invalid * theta)


def polyhedral_parameter_position_sensitivity(
    model: PolyhedralParameterModel,
    step: Float[Array, ""] | Int[Array, ""] | float | int,
    parameters: Float[Array, " P"] | ArrayLike,
    *,
    target_parameter_gradient: Float[Array, " P"] | ArrayLike | None = None,
) -> Float[Array, " P"]:
    """Differentiate a local volume level set through the supplied position.

    Returns (target_parameter_gradient - dV/dtheta) / (dV/dstep). The default
    holds the physical volume target fixed; an explicit target gradient uses
    volume per dimensionless parameter. This is not a root finder or a check
    that step attains a requested target. Verify the root residual separately.
    The selected root branch must be simple and inside a resolved chamber.
    Invalid queries, unresolved/zero slopes, and underflow of a nonzero target
    difference or resulting ratio return NaNs. Equal represented target and
    volume derivatives retain zero sensitivity. No derivative at extrema,
    event boundaries or switching between root branches is promised.
    """
    theta, parameters_valid = _query_array(parameters, model.coefficients.dtype)
    position, step_valid = _query_array(step, model.coefficients.dtype)
    target, target_valid = _query_array(
        jnp.zeros_like(theta)
        if target_parameter_gradient is None
        else target_parameter_gradient,
        theta.dtype,
    )
    if target.shape != theta.shape:
        raise ValueError("Target gradient must match the parameter shape")
    value, derivatives, derivative_magnitudes = _evaluate(model, position, theta)
    slope, gradient = derivatives[0], derivatives[1:]
    slope_scale = derivative_magnitudes[0]
    valid = (
        jnp.isfinite(value) & jnp.all(jnp.isfinite(target)) & jnp.isfinite(slope_scale)
    )
    valid &= parameters_valid & step_valid & target_valid
    valid &= jnp.abs(slope) > _ROUNDOFF * slope_scale + _UNDERFLOW
    denominator = jnp.where(valid, slope, 1.0)
    numerator = target - gradient
    # Distinct normal inputs can have a subnormal difference which XLA flushes
    # to zero. Equal represented derivatives legitimately give zero motion.
    target_bits = jax.lax.bitcast_convert_type(target, jnp.uint64)
    gradient_bits = jax.lax.bitcast_convert_type(gradient, jnp.uint64)
    magnitude_mask = 0x7FFFFFFFFFFFFFFF
    equal = (target_bits == gradient_bits) | (
        ((target_bits & magnitude_mask) == 0) & ((gradient_bits & magnitude_mask) == 0)
    )
    valid &= jnp.all(equal | (jnp.abs(numerator) >= np.finfo(np.float64).tiny))
    result = numerator / denominator
    valid &= jnp.all(equal | (jnp.abs(result) >= np.finfo(np.float64).tiny))
    return result * jnp.where(valid & jnp.all(jnp.isfinite(result)), 1.0, jnp.nan)
