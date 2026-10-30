"""Finite translation-volume profiles for explicit convex-cell bodies.

Host preparation uses exact rational geometry of represented binary64 vertices.
The numerical profile supports JAX step queries. This is neither an importer
nor a general Boolean JVP: cells and velocity remain fixed after preparation.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, pairwise, product
from typing import Literal, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float
from numpy.typing import ArrayLike

from brepax.experimental._polyhedral_contact import hull_relation, vertex_hull
from brepax.experimental._polyhedral_translation import (
    ExactPiece,
    Vector,
    collection_pieces,
)


class PolyhedralTranslationProfile(eqx.Module):
    """Static piece arrays for a fixed pure-translation path.

    Create with :func:`prepare_polyhedral_translation` outside tracing.
    ``thresholds`` round exact event times upward for correct binary64 query
    classification; they are not generally the actual event times. The mask
    identifies exactly representable knots. Coefficients use the represented
    origins and widths, not the unrounded rational event coordinates.

    Only the scalar step is an intended differentiation variable. Ordinary AD
    is supported inside open event chambers, not at a derivative jump. Use
    the explicit slope query or host event limits at chamber boundaries.
    """

    thresholds: Float[Array, " K"]
    exact_knot_mask: Bool[Array, " K"]
    origins: Float[Array, " P"]
    widths: Float[Array, " P"]
    coefficients: Float[Array, "P 4"]


@dataclass(frozen=True)
class TranslationEvent:
    """Exact step and one-sided volume/step limits at an internal chamber knot."""

    step: Fraction
    left_derivative: Fraction
    right_derivative: Fraction


@dataclass(frozen=True)
class PreparedPolyhedralTranslation:
    """Exact model diagnostics plus the rounded numerical evaluator.

    This certifies only the supplied vertex-hull model. It does not establish
    correspondence to separately supplied CAD or identify every knot as a
    physical contact event: redundant combinatorial knots can remain.
    """

    profile: PolyhedralTranslationProfile
    exact_knots: tuple[Fraction, ...]
    events: tuple[TranslationEvent, ...]
    pair_count: int
    reference_kind: Literal["represented_vertex_hulls"] = field(
        default="represented_vertex_hulls", init=False
    )
    original_cad_contact_certified: Literal[False] = field(default=False, init=False)


def _ceil_float(value: Fraction) -> float:
    rounded = float(value)
    if Fraction(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    return rounded


def _pack(pieces: tuple[ExactPiece, ...]) -> PolyhedralTranslationProfile:
    exact_knots = (*(p.lower for p in pieces), pieces[-1].upper)
    thresholds = np.array([_ceil_float(t) for t in exact_knots], dtype=np.float64)
    widths = np.diff(thresholds)
    if not np.isfinite(thresholds).all() or not np.isfinite(widths).all():
        raise ValueError("Translation interval is not representable in float64")
    if np.any(widths <= 0) or np.any(
        np.nextafter(thresholds[:-1], np.inf) >= thresholds[1:]
    ):
        raise ValueError(
            "Distinct translation events have no resolvable float64 chamber"
        )
    if np.any(widths < np.finfo(np.float64).tiny):
        raise ValueError(
            "Translation chamber widths underflow normal float64 arithmetic"
        )
    exact_coefficients = tuple(
        p.reparameterize(Fraction(float(lo)), Fraction(float(w)))
        for p, lo, w in zip(pieces, thresholds[:-1], widths, strict=True)
    )
    coefficients = np.array(
        [[float(c) for c in row] for row in exact_coefficients], dtype=np.float64
    )
    if not np.isfinite(coefficients).all():
        raise ValueError("Translation coefficients overflow float64")
    # XLA backends may flush subnormals. An unsupported physical scale must
    # not silently turn a positive-volume chamber into the zero polynomial.
    if any(
        exact != 0 and abs(rounded) < np.finfo(np.float64).tiny
        for exact_row, row in zip(exact_coefficients, coefficients, strict=True)
        for exact, rounded in zip(exact_row, row, strict=True)
    ):
        raise ValueError("Translation coefficients underflow normal float64 arithmetic")
    return PolyhedralTranslationProfile(
        jnp.asarray(thresholds),
        jnp.asarray(
            [
                Fraction(float(t)) == q
                for t, q in zip(thresholds, exact_knots, strict=True)
            ]
        ),
        jnp.asarray(thresholds[:-1]),
        jnp.asarray(widths),
        jnp.asarray(coefficients),
    )


def prepare_polyhedral_translation(
    fixed_cells: Sequence[ArrayLike],
    moving_cells: Sequence[ArrayLike],
    velocity: ArrayLike | Sequence[float | int],
    *,
    max_step: float,
) -> PreparedPolyhedralTranslation:
    """Prepare exact finite overlap along ``moving + step * velocity``.

    Each nonempty body is a list of bounded convex cells, each supplied as
    finite float64 vertices of shape ``(N, 3)``. Every hull must have positive
    volume; same-body interiors and initial cross-body interiors must be
    disjoint. Face, edge, point contact and positive gaps are retained.
    ``max_step`` is finite and positive. Step is dimensionless; vertices and
    velocity share a length unit, and output has its cube as the volume unit.

    Preparation enumerates exact hull facets and triple-plane intersections;
    it is intended for small cells, not large meshes. No volume measurements
    are fitted. No new contacts are omitted within the prepared interval.
    Rational event times and one-sided limits remain in the host record.
    Numerically unresolvable intervals or overflowing/underflowing coefficients
    raise ``ValueError`` instead of merging events or snapping geometry.

    Exact global rigid coordinate changes preserve volume; scaling geometry
    and velocity by s scales volume and step derivatives by s**3. Independently
    rounding transformed vertices can change the represented contact state and
    cause preparation to reject; this law does not certify such rounded inputs
    or CAD import tolerances. Host preparation does not support shape/velocity AD.

    Example::

        from itertools import product
        import jax
        import numpy as np

        jax.config.update("jax_enable_x64", True)
        box = np.array(list(product((0., 1.), repeat=3)))
        prepared = prepare_polyhedral_translation(
            [box], [box + (1., 1., 0.)], (-1., -1., 0.), max_step=2.
        )
        # Edge contact has zero initial slope but positive finite overlap.
        volume = polyhedral_translation_volume(prepared.profile, 0.25)  # 0.0625
    """
    if jnp.asarray(0.0).dtype != np.dtype("float64"):
        raise ValueError("Polyhedral translation requires JAX x64")
    step = float(max_step)
    vector = np.asarray(velocity, dtype=np.float64)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Expected finite positive max_step")
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected finite velocity with shape (3,)")
    if not fixed_cells or not moving_cells:
        raise ValueError("Both bodies require at least one convex cell")
    fixed = tuple(vertex_hull(np.asarray(v, dtype=np.float64)) for v in fixed_cells)
    moving = tuple(vertex_hull(np.asarray(v, dtype=np.float64)) for v in moving_cells)
    for name, body in (("fixed", fixed), ("moving", moving)):
        for i, j in combinations(range(len(body)), 2):
            if hull_relation(body[i], body[j]).relation == "overlap":
                raise ValueError(
                    f"The {name} cells {i} and {j} have overlapping interiors"
                )
    for i, j in product(range(len(fixed)), range(len(moving))):
        if hull_relation(fixed[i], moving[j]).relation == "overlap":
            raise ValueError(f"Initial fixed/moving cells {i} and {j} overlap")
    exact_velocity = cast(Vector, tuple(Fraction(float(v)) for v in vector))
    pieces = collection_pieces(fixed, moving, exact_velocity, Fraction(step))
    try:
        profile = _pack(pieces)
    except (OverflowError, FloatingPointError) as error:
        raise ValueError(
            "Translation profile is not representable in float64"
        ) from error
    events = tuple(
        TranslationEvent(right.lower, left.slope(right.lower), right.slope(right.lower))
        for left, right in pairwise(pieces)
    )
    return PreparedPolyhedralTranslation(
        profile,
        (*(p.lower for p in pieces), pieces[-1].upper),
        events,
        len(fixed) * len(moving),
    )


def _piece_index(
    profile: PolyhedralTranslationProfile,
    step: Float[Array, ""] | float,
    side: Literal["left", "right"],
) -> Array:
    index = jnp.clip(
        jnp.searchsorted(profile.thresholds, step, side="right") - 1,
        0,
        len(profile.origins) - 1,
    )
    if side == "left":
        at_exact_event = (step == profile.thresholds[index]) & profile.exact_knot_mask[
            index
        ]
        index = jnp.where(at_exact_event & (index > 0), index - 1, index)
    return index


def polyhedral_translation_volume(
    profile: PolyhedralTranslationProfile, step: Float[Array, ""] | float
) -> Float[Array, ""]:
    """Evaluate finite overlap, returning NaN outside the prepared interval.

    Supports jit/vmap and ordinary step differentiation inside open chambers.
    The value is continuous at events; its derivative need not be. This
    evaluates the fixed represented model, not a rebuilt CAD solid.
    """
    index = _piece_index(profile, step, "right")
    u = (step - profile.origins[index]) / profile.widths[index]
    c = profile.coefficients[index]
    value = ((c[3] * u + c[2]) * u + c[1]) * u + c[0]
    valid = (
        jnp.isfinite(step)
        & (step >= profile.thresholds[0])
        & (step <= profile.thresholds[-1])
    )
    return jnp.where(valid, value, jnp.nan)


def polyhedral_translation_slope(
    profile: PolyhedralTranslationProfile,
    step: Float[Array, ""] | float,
    *,
    side: Literal["left", "right"] = "right",
) -> Float[Array, ""]:
    """Return the specified one-sided volume/step derivative.

    ``side`` is static under jit. At a nonrepresentable rational event, a
    rounded-up threshold lies strictly after it: both sides there use the
    following chamber. Exact event limits are in the host ``events`` record.
    Missing interval sides (left at zero, right at max_step), nonfinite steps
    and out-of-range steps return NaN. Derivatives of this slope are not part
    of the contract.
    """
    if side not in ("left", "right"):
        raise ValueError("Expected side='left' or side='right'")
    index = _piece_index(profile, step, side)
    u = (step - profile.origins[index]) / profile.widths[index]
    c = profile.coefficients[index]
    value = ((3 * c[3] * u + 2 * c[2]) * u + c[1]) / profile.widths[index]
    valid = (
        jnp.isfinite(step)
        & (step >= profile.thresholds[0])
        & (step <= profile.thresholds[-1])
    )
    valid &= (
        step > profile.thresholds[0]
        if side == "left"
        else step < profile.thresholds[-1]
    )
    return jnp.where(valid, value, jnp.nan)
