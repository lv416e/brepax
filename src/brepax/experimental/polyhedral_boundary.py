"""Host queries across one supplied polyhedral event-order boundary."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite
from typing import cast

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from numpy.typing import NDArray

from brepax.experimental._polyhedral_boundary import (
    ExactEventBoundary,
    boundary_directional,
    prepare_boundary,
)
from brepax.experimental._polyhedral_parameter import Affine, ParameterGuard, _cell
from brepax.experimental._polyhedral_translation import Vector
from brepax.experimental.polyhedral_parameter import (
    AffinePolyhedralCell,
    ParameterTranslationEvent,
)
from brepax.experimental.polyhedral_translation import (
    PreparedPolyhedralTranslation,
    _prepared_from_pieces,
)

_Scalar = float | int | Fraction
_CellArrays = tuple[NDArray[np.float64], NDArray[np.float64]]


def _rational(value: _Scalar, name: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if np.iscomplexobj(value):
        raise ValueError(f"Expected a finite real {name}")
    try:
        number = float(value)
        if isfinite(number):
            return Fraction(number)
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError(f"Expected a finite real {name}")


def _validated_velocity(velocity: ArrayLike | Sequence[float | int]) -> Vector:
    if np.iscomplexobj(velocity):
        raise ValueError("Expected finite real velocity with shape (3,)")
    vector = np.asarray(velocity, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected finite real velocity with shape (3,)")
    return cast(Vector, tuple(Fraction(float(v)) for v in vector))


def _validated_body(body: Sequence[AffinePolyhedralCell]) -> tuple[_CellArrays, ...]:
    if not body:
        raise ValueError("Both bodies require at least one convex cell")
    result = []
    for cell in body:
        if np.iscomplexobj(cell.vertices) or np.iscomplexobj(
            cell.parameter_displacements
        ):
            raise ValueError("Expected real affine cell coordinates")
        vertices = np.asarray(cell.vertices, dtype=np.float64)
        displacements = np.asarray(cell.parameter_displacements, dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.shape[0] < 4
            or displacements.shape != (1, *vertices.shape)
            or not np.isfinite(vertices).all()
            or not np.isfinite(displacements).all()
        ):
            raise ValueError("Expected finite vertices (N,3) and displacements (1,N,3)")
        result.append((vertices, displacements))
    return tuple(result)


@dataclass(frozen=True)
class PreparedPolyhedralEventBoundary:
    """Exact host certificate for an exchange supplied at theta=0.

    ``parameter_interval`` is open at each finite outer endpoint and includes
    zero. None denotes an unbounded endpoint. ``events`` labels the two affine
    event functions; their tuple order is not chronological on both sides.
    No original-CAD contact or rounded transformed input is certified.
    """

    parameter_interval: tuple[Fraction | None, Fraction | None]
    events: tuple[ParameterTranslationEvent, ParameterTranslationEvent]
    _geometry: ExactEventBoundary = field(repr=False)

    @property
    def tied_event_step(self) -> Fraction:
        return self.events[0].reference_step


@dataclass(frozen=True)
class PolyhedralBoundaryResponse:
    """Exact volume and first responses at fixed reference parameter theta=0.

    ``directional_derivative`` is the right derivative along the supplied
    (step_direction, parameter_direction), generally nonlinear in direction.
    The left parameter derivative is the quotient as theta approaches zero
    from below: it is the negative of the response to direction (0,-1).
    All fields use volume units per dimensionless step/parameter as applicable.
    """

    volume: Fraction
    directional_derivative: Fraction
    left_parameter_derivative: Fraction
    right_parameter_derivative: Fraction

    @property
    def ordinary_parameter_derivative(self) -> Fraction | None:
        """Scalar theta derivative at fixed step, not joint differentiability."""
        if self.left_parameter_derivative == self.right_parameter_derivative:
            return self.right_parameter_derivative
        return None


def prepare_polyhedral_event_boundary(
    fixed_cells: Sequence[AffinePolyhedralCell],
    moving_cells: Sequence[AffinePolyhedralCell],
    velocity: ArrayLike | Sequence[float | int],
    *,
    max_step: _Scalar,
    parameter_radius: _Scalar,
) -> PreparedPolyhedralEventBoundary:
    """Certify two adjacent charts at one supplied event exchange at theta=0.

    Cells use one dimensionless affine design parameter, fixed facet normals
    and fixed translation velocity. Geometry, displacements and velocity share
    length units; step is dimensionless on [0,max_step]. Positive cell volume,
    disjoint same-body interiors and initially disjoint cross-body interiors
    must have strict validity certificates across the returned parameter region.

    ``parameter_radius`` is a positive exact reference-center choice: prepare
    regular charts at its negative and positive values. It is not a tolerance,
    finite-difference increment or claimed domain radius. This function does
    not discover a boundary or search for suitable centers. Their certified
    open regions must adjoin at zero, with the same affine event inventory and
    exactly one isolated pair of distinct event functions meeting inside the
    path. Other event ties, static feasibility changes, or invalid geometry
    raise ValueError, even for some physically valid families.

    Vertices and displacements denote represented float64 coordinates; centers
    and queries preserve supplied Fraction values. Rebased cells/planes are
    exact, without rebuilding rounded center coordinates. Polynomial closures
    must agree with the fixed-reference geometry. No observations are fitted.
    This host preparation and its response query are not JAX transformations.
    Global exact rigid transforms preserve values and responses; scaling all
    length inputs by s multiplies them by s**3. Independently rounded inputs
    can change the certificate. The implementation is intended for small cells.

    Example::

        from fractions import Fraction
        from itertools import product
        import numpy as np

        stock = np.array(list(product((0., 1.), repeat=3)))
        cutter = np.array(list(product((-1.5, -.5), (-1., 2.), (-1., 2.))))
        shift = np.zeros((1, 8, 3))
        shift[0, cutter[:, 0] == -1.5, 0] = -1.
        prepared = prepare_polyhedral_event_boundary(
            [AffinePolyhedralCell(stock, np.zeros((1, 8, 3)))],
            [AffinePolyhedralCell(cutter, shift)], (1., 0., 0.),
            max_step=3., parameter_radius=Fraction(1, 8),
        )
        response = polyhedral_event_boundary_response(prepared, Fraction(3, 2))
        assert response.left_parameter_derivative == 1
        assert response.right_parameter_derivative == 0
    """
    vector = _validated_velocity(velocity)
    maximum, radius = (
        _rational(max_step, "max_step"),
        _rational(parameter_radius, "parameter_radius"),
    )
    if maximum <= 0 or radius <= 0:
        raise ValueError("max_step and parameter_radius must be positive")
    guards: set[Affine] = set()
    records: set[ParameterGuard] = set()
    bodies = []
    for body in (fixed_cells, moving_cells):
        bodies.append(
            tuple(_cell(v, d, guards, records) for v, d in _validated_body(body))
        )
    geometry = prepare_boundary(
        bodies[0],
        bodies[1],
        vector,
        maximum,
        records,
        radius,
    )
    events = tuple(
        ParameterTranslationEvent(event[0], (event[1],)) for event in geometry.collision
    )
    return PreparedPolyhedralEventBoundary(
        geometry.interval,
        cast(tuple[ParameterTranslationEvent, ParameterTranslationEvent], events),
        geometry,
    )


def specialize_polyhedral_event_boundary(
    prepared: PreparedPolyhedralEventBoundary, parameter: _Scalar
) -> PreparedPolyhedralTranslation:
    """Build the ordinary finite profile at any certified theta, including zero.

    Fraction parameters retain their exact values; floats denote their exact
    binary64 values. Outer domain endpoints reject. The existing profile uses
    JAX x64 and may reject unresolved float event packing or coefficient range.
    Its usual target/range queries apply, including empty, singleton and
    interval target solution sets. No derivative of target selection is added.
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Polyhedral translation specialization requires JAX x64")
    pieces = prepared._geometry.specialize(_rational(parameter, "parameter"))
    return _prepared_from_pieces(pieces, prepared._geometry.left.geometry.pair_count)


def polyhedral_event_boundary_response(
    prepared: PreparedPolyhedralEventBoundary,
    step: _Scalar,
    *,
    step_direction: _Scalar = 0,
    parameter_direction: _Scalar = 1,
) -> PolyhedralBoundaryResponse:
    """Return exact one-sided volume responses at (step,theta=0).

    Direction means (step,theta)=(step,0)+epsilon*(step_direction,
    parameter_direction) for epsilon approaching zero from above. Tangent
    event rays require agreement of all incident polynomial limits; directions
    exiting the translation interval raise ValueError. The direction map can
    be nonlinear; this is not a JVP, JAX gradient, or derivative of a root set.
    No higher derivative or general boundary discovery is provided.
    """
    t = _rational(step, "step")
    dt, dtheta = (
        _rational(step_direction, "step_direction"),
        _rational(parameter_direction, "parameter_direction"),
    )
    geometry = prepared._geometry
    directional = boundary_directional(geometry, t, dt, dtheta)
    left = -boundary_directional(geometry, t, Fraction(0), Fraction(-1))
    right = boundary_directional(geometry, t, Fraction(0), Fraction(1))
    volume = next(
        piece.value(t) for piece in geometry.pieces if piece.lower <= t <= piece.upper
    )
    return PolyhedralBoundaryResponse(volume, directional, left, right)
