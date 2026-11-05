# ADR-0023: Validated dimension handles and certified inverse queries

## Status

Accepted

## Context

Imported STEP solids carry no modeling history. ADR-0021 turns recognized
through-hole radii and centers into differentiable parameters with validated
OCCT rebuilds. ADR-0022 establishes exact event-delimited volume polynomials,
model-scoped unreachability proofs and certified minimal-change search, but
only for explicitly supplied convex cells. Its inputs do not come from
imported parts: in the local fixture census, no imported production part was
bounded by planar faces only, and cylindrical faces were the largest surface
class.

The two lines answer complementary halves of one question: which dimensions
of an imported part can change, and by how little, to meet a target, with
evidence that the edited CAD matches. The following measurements, made on
local NIST, FreeCAD and miscellaneous STEP fixtures, motivate a common layer:

- After split cylindrical faces are unified, every recognized through-hole
  group rebuilds and passes validation (183 of 183). Preparing the
  mesh-based model, however, fails on 8 of 39 solids, or 14 after
  unification, because the reference mesh does not close at the weld
  tolerance. A discretized layer therefore needs a robust closed reference
  mesh first.
- A planar face whose neighbors are perpendicular planes, or cylinders with
  axes along its normal, can be pushed or pulled by fusing or cutting the
  swept face prism. On the NIST and FreeCAD fixtures, such faces made up 22%
  of planar faces (279 of 1261).
  - In 164 of 198 sampled moves, whole-solid volume changed exactly by area
    times distance.
  - None of the remaining 34 moves met other geometry. Fifteen faces had inner
    loops whose bounding surfaces vary along the move. Sixteen showed a
    distance-independent volume error introduced by the Boolean result
    simplification. Three Booleans were invalid even for a tiny move.
  - Fillet neighbors were the most common reason a planar face was not of this
    kind (599 faces had no other reason).
- OCCT's per-face offset (`BRepOffset_MakeOffset` with `SetOffsetOnFace`)
  produced a valid first-order result in 1 of 96 trials and is not a usable
  rebuild path for this purpose.
- Some manufacturability distances are exact functions of these handles,
  checked against OCCT distances on rebuilt parts:
  - wall thickness between opposing parallel planar faces;
  - the ligament from a through-hole to a planar wall parallel to its axis,
    `l0 - (r - r0) - s . n`;
  - the ligament between two parallel holes.

  Each law holds only while the pair remains the closest features: 79% of
  naively enumerated hole-to-wall pairs were separated by other geometry. The
  smallest wall thickness of the whole part came from such a pair in 6 of
  36 solids and in none of the 16 NIST parts. There, it came from planes
  meeting at an angle or from tangencies.

Direct-modeling research expresses push/pull as a Boolean with the swept
volume so that the result remains a valid solid, and notes that the motions
of neighboring faces have no explicit expressions
([Zou and Feng](https://arxiv.org/abs/1906.08455)). Algorithmic
differentiation has been applied to OCCT for parametrized CAD models
([Banović 2019](https://digital.ub.uni-paderborn.de/hs/download/pdf/3296051)).
The sources reviewed for this decision did not differentiate dimensions of
history-free imported B-Rep. The DFM tools reviewed check rules and
thresholds rather than computing sensitivities or minimal changes. This
reflects the reviewed sources, not a novelty claim.

## Decision

Introduce dimension handles: typed host records describing one editable
dimension of a prepared, history-free solid. A handle names its kind, its
faces in the prepared working solid, a reference value, and an admissible
closed interval. Within that interval the face topology is unchanged and the
handle has an exact volume and first-moment law. The initial kinds are:

- through-hole radius and transverse center shift (ADR-0021);
- planar face offset along its outward normal, for faces whose neighbors
  remain unchanged surfaces extended or trimmed by the move.

Further kinds, such as blind drilled holes, countersinks, counterbores and
drafted or filleted neighbors, are admitted only after their exact law,
interval derivation, and rebuild validation have been measured on imported
fixtures.

Interval ends are events: the first parameter value at which the moved region
meets other geometry or the edit changes topology. Hosts derive them
conservatively from geometry before any edit. Beyond an event, a handle is
unsupported rather than extrapolated.

Responses have two tiers, which stay distinct in API, documentation and
evidence:

- exact laws: volume, surface area and first moment as polynomials in the
  handle values inside their intervals, evaluated in exact rational arithmetic
  on the host and in float64 under JAX;
- discretized responses: inertia and per-face manufacturability metrics
  evaluated on a frozen reference mesh moved by handle displacement fields.
  Their derivatives are derivatives of that discretization. The mesh is an
  optional layer: its limits restrict neither a handle's interval nor a rebuild.

Exact constraints are local: wall thickness between two declared parallel
faces, and ligaments from a declared hole to a declared wall or another
declared hole. Each holds only after a geometric check confirms that the
pair overlaps in projection and that no other face lies between them. A
part-wide minimum of a manufacturability metric is a discretized response,
not an exact constraint.

Inverse queries minimize a weighted change of handle values subject to
target bands on responses. With exact-law constraints only, a query returns
an exactly rechecked witness, or a proof that no handle values inside the
supplied box satisfy the band, or an explicitly unresolved region, reusing
the bound-and-exclude approach of ADR-0022. Bounds that support a proof are
computed in exact rational arithmetic; floating-point Bernstein coefficients
can violate inclusion
([Garloff, Smith and Jansson](https://reliable-computing.org/interval.02/garl.pdf)).
With discretized constraints, the query is a local gradient-based search with
no certificate; its result must be validated after rebuilding.

Rebuild each edit in OCCT with local Boolean operations: fill and re-cut for
holes, fuse or cut a swept face prism for planar offsets. Validate the rebuild
against the exact laws, as ADR-0021 does: when whole-solid properties disagree,
measurements inside and outside an envelope around the edited geometry decide.
A certificate applies to the represented handle model; only a validated
rebuild connects it to the edited CAD.

Keep host preparation, interval derivation, rebuilding and validation behind
the OCCT boundary. Keep evaluators array-based and JAX-transformable. The
layer stays experimental. The hole-editing API moved to exact laws and a
separate optional mesh as a clean break, and is the handle implementation for
holes. Planar offsets and the inverse query
follow in separate changes, each gated by an imported-fixture census. The
census counts recognized, validated, wrongly accepted and unsupported
candidates against an independent oracle.

## Consequences

Callers can ask minimal-change questions about imported parts, for example
reducing mass within bounds on selected dimensions. They receive either a
rechecked and rebuilt edit or a model-scoped infeasibility proof. Gradients
of discretized metrics remain available to downstream JAX computations.

Coverage is limited to handles whose laws and events are implemented. Most
faces of the measured parts are not yet editable dimensions, and the
fraction that is must be reported per fixture rather than assumed. Topology
changes, interacting features, curved neighbors and general direct modeling
remain outside this decision. Certificates do not extend to discretized
metrics, to geometry outside the declared handles, or to CAD that has not
been rebuilt and validated. No production dependency or new kernel is
introduced.
