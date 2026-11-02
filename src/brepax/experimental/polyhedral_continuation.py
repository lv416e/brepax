"""Bounded automatic continuation for one affine polyhedral design parameter."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from brepax.experimental._polyhedral_continuation import (
    ContinuationAttempt,
    ContinuationGap,
    ExactContinuation,
    prepare_continuation,
)
from brepax.experimental.polyhedral_boundary import (
    _rational,
    _validated_body,
    _validated_velocity,
)
from brepax.experimental.polyhedral_parameter import AffinePolyhedralCell
from brepax.experimental.polyhedral_translation import (
    PreparedPolyhedralTranslation,
    _prepared_from_pieces,
)

_Scalar = float | int | Fraction


@dataclass(frozen=True)
class PreparedPolyhedralContinuation:
    """Covered charts and explicit uncertainty on a closed design window.

    A gap is unconfirmed coverage, not invalid geometry or target infeasibility.
    Boundary parameters are included only after both adjoining chart closures
    agree with an independently assembled exact fixed-parameter profile.
    """

    parameter_bounds: tuple[Fraction, Fraction]
    seed_parameter: Fraction
    _geometry: ExactContinuation = field(repr=False)

    @property
    def chart_count(self) -> int:
        """Number of accepted regular charts, including the seed when available."""
        return len(self._geometry.charts)

    @property
    def boundary_parameters(self) -> tuple[Fraction, ...]:
        """Automatically discovered and certified isolated exchange parameters."""
        return tuple(boundary.parameter for boundary in self._geometry.boundaries)

    @property
    def uncovered(self) -> tuple[ContinuationGap, ...]:
        """Unconfirmed portions of the query window, retaining endpoint membership."""
        return self._geometry.uncovered

    @property
    def attempts(self) -> tuple[ContinuationAttempt, ...]:
        """Every seed, trial center and unresolved frontier observation."""
        return self._geometry.attempts


def prepare_polyhedral_continuation(
    fixed_cells: Sequence[AffinePolyhedralCell],
    moving_cells: Sequence[AffinePolyhedralCell],
    velocity: ArrayLike | Sequence[float | int],
    *,
    max_step: _Scalar,
    parameter_bounds: tuple[_Scalar, _Scalar],
    seed_parameter: _Scalar,
    max_charts: int = 8,
    max_center_trials: int = 8,
) -> PreparedPolyhedralContinuation:
    """Discover isolated event exchanges from a regular seed, within a budget.

    Supply original float64 vertex cells with one affine, dimensionless design
    parameter, fixed facet normals and fixed translation velocity. The seed is
    an ordinary design parameter, not a supplied event or a finite-difference
    radius. Original cells at theta=0 must admit fixed-normal cell preparation;
    a nonzero seed does not bypass a nonregular original cell reference.
    Exact affine guard roots propose frontiers; each accepted transition
    has one isolated event pair, unchanged event inventory, valid non-event
    guards and equal exact closures. No rounded rebased vertices are substituted.

    The finite closed window must contain the seed. ``max_charts`` caps accepted
    charts including the seed; ``max_center_trials`` caps exact rational side
    centers per frontier. Failed trials remain in ``attempts``. Malformed arrays,
    nonfinite scalars and invalid bounds/budgets raise ``ValueError``. A family
    whose seed or next chart cannot be certified instead leaves explicit gaps.
    No seed search, simultaneous-event handling, or alternative separator search
    is performed. An event at a window endpoint may remain an uncovered point.

    Cell coordinates, displacements and velocity have length units; translation
    step and parameter are dimensionless. Exact frame changes preserve volume
    on jointly certified parameters; scaling all lengths by s scales it by
    s**3. The sufficient separator chosen by existing preparation can depend on
    the frame: coverage and resolution status are not promised equivariant.
    Rounded input, source CAD and geometric invalidity are not certified.
    This host preparation has no JAX transformation or completeness guarantee;
    specialization retains existing JAX x64 and numerical packing requirements.

    Example::

        prepared = prepare_polyhedral_continuation(
            fixed_cells, moving_cells, (1, 0, 0), max_step=4,
            parameter_bounds=(-0.05, 0.6), seed_parameter=0,
        )
        if not prepared.uncovered:
            path = specialize_polyhedral_continuation(prepared, 0.5)
            # Existing exact target/range queries apply to this model path.
    """
    fixed, moving = _validated_body(fixed_cells), _validated_body(moving_cells)
    vector = _validated_velocity(velocity)
    maximum = _rational(max_step, "max_step")
    lower, upper = (_rational(x, "parameter bound") for x in parameter_bounds)
    seed = _rational(seed_parameter, "seed_parameter")
    if maximum <= 0 or not lower <= seed <= upper:
        raise ValueError(
            "Require positive max_step and an ordered window containing the seed"
        )
    for name, limit in (
        ("max_charts", max_charts),
        ("max_center_trials", max_center_trials),
    ):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    geometry = prepare_continuation(
        fixed,
        moving,
        vector,
        maximum,
        (lower, upper),
        seed,
        max_charts,
        max_center_trials,
    )
    return PreparedPolyhedralContinuation((lower, upper), seed, geometry)


def specialize_polyhedral_continuation(
    prepared: PreparedPolyhedralContinuation, parameter: _Scalar
) -> PreparedPolyhedralTranslation:
    """Specialize only a certified parameter; unresolved coverage raises ValueError.

    Fraction values retain the original affine family exactly. Existing
    translation-profile packing may reject unresolved float64 event chambers
    or coefficient scales even when the host parameter is certified.
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Polyhedral translation specialization requires JAX x64")
    pieces = prepared._geometry.specialize(_rational(parameter, "parameter"))
    pair_count = prepared._geometry.charts[0].side.geometry.pair_count
    return _prepared_from_pieces(pieces, pair_count)
