# Applications

Experimental applications compose BRepAX geometry evaluation with host-side CAD
preparation. Their individual contracts define differentiation and input scope.

## STEP through-hole editing

`prepare_hole_edit` recognizes cylindrical through-holes in a supplied B-Rep.
It can edit selected radii and centers without reconstructing the stock's modeling history.
The JAX model moves hole walls and adjoining planar boundaries together; the
host `rebuild` method produces a new OCCT shape for further use or STEP export.

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from brepax.experimental.applications.hole_edit import prepare_hole_edit
from brepax.io.step import read_step

jax.config.update("jax_enable_x64", True)
shape = read_step("part.step")
all_holes = prepare_hole_edit(shape)
# Select one hole by its input face ID; other features remain fixed.
face_id = all_holes.holes[0].face_id
edit = prepare_hole_edit(shape, [face_id], deflection=0.0001)
model = edit.model
radii = (model.reference_radii + model.upper_radii) / 2
assert bool(eqx.filter_jit(model.is_valid)(radii))
volume, derivative = jax.jit(jax.value_and_grad(model.volume))(radii)
areas = jax.jit(jax.vmap(model.surface_area))(
    jnp.stack([model.reference_radii, radii])
)
edited_shape = edit.rebuild(radii)
```

Center motion uses an optional second argument, `center_shifts`, with one
world-coordinate displacement vector per selected hole. It is measured from
the original center in input CAD units. The hole axis stays fixed; shifts must
be perpendicular to it. Omitting the argument preserves radius-only behavior.
For the Z-axis holes in the fixture below:

```python
shifts = jnp.array([[0.2, -0.1, 0.0]])
assert bool(eqx.filter_jit(model.is_valid)(radii, shifts))
center = model.center_of_mass(radii, shifts)
center_jacobian = jax.jit(jax.jacrev(model.center_of_mass, argnums=1))(radii, shifts)
edited_shape = edit.rebuild(radii, shifts)
```

For a reproducible repository example, use `tests/fixtures/box_with_holes.step`
as `part.step`. Its radius-4 through-hole can be selected and changed to 4.4;
the other radius-3 hole remains unchanged. The tests also cover rectangular
and round stocks, nonintersecting orthogonal holes, regenerated placements,
STEP roundtrips, and independent CAD finite-difference references.

### Contract and limits

- Input must contain one valid outward-oriented solid. Editable holes need a
  complete empty cylindrical bore, two perpendicular outward-facing planar
  ends, and clearance from other boundaries. Blind, split, oblique, intersecting and near-boundary holes
  are rejected. Unsupported inward cylinders appear in `edit.rejected`; an
  explicit selection of an unsupported face raises `ValueError`. A counterbore
  shoulder is not a valid end for its larger cylinder; the smaller bore through
  the remaining slab can still qualify independently.
- Radii, center shifts, hole metadata, returned points and metrics use the input CAD's units.
  A fixed `CadFrame` normalizes host preparation and numerical evaluation. By
  default, the input bounding box is centered and its largest extent becomes
  40 working units. The `deflection` argument is in **working units**, not the
  original STEP units. No physical unit is inferred from a numeric coordinate.
- Preparation requires JAX x64. Recognition and CAD rebuilding run outside
  `jit`, `vmap` and differentiation. Geometry/metric methods support all three;
  the Boolean validity predicate supports `jit`/`vmap`. Keep the prepared model
  fixed when differentiating with respect to radii or shifts. For direct compilation of
  a bound model method, use `eqx.filter_jit(model.volume)`; bare
  `jax.jit(model.volume)` can fail because the bound Equinox model contains
  unhashable arrays. Alternatively, JIT a function taking the model as a PyTree
  argument. Compositions such as the `value_and_grad` example above also work.
- Radius bounds are `[0.5*r0, r0 + 0.2*clearance]`. Center shifts and radius
  growth share that clearance: `norm(shift) + max(radius-r0, 0)` must not exceed
  `0.2*clearance`, even when shrinking a hole. Axial shifts are invalid beyond
  roundoff (`64 * x64_epsilon * r0`); the residual is projected out.
  `model.is_valid` also checks
  local triangle orientation. Invalid edits return NaN geometry/metrics under
  JAX; `rebuild` raises. Gradients at invalid inputs are undefined.
- Volume and area are rigid-motion invariant; their radius gradients are too.
  Under positive length scaling `s`, values scale by `s**3` and `s**2`, and
  radius gradients by `s**2` and `s`. Centers/vertices transform as points;
  center radius derivatives rotate as vectors and are unchanged by unit scaling.
  Rotate center shifts with the geometry. Scalar shift gradients rotate as
  vectors and scale like radius gradients. The center-of-mass shift Jacobian
  transforms as `R @ J @ R.T` and is unchanged by unit scaling. These Jacobians
  describe transverse motion, with zero axial component; they do not authorize
  editing the hole depth or orientation.
  Fresh meshing may change discretization error, so compare within an accuracy
  budget rather than demanding identical triangulations.
- Gradients describe a frozen mesh with coupled boundary motion. They do not
  differentiate OCCT reconstruction or remeshing. Rebuild and independently
  check CAD before accepting an optimized design; refine `deflection` when
  value or derivative accuracy is insufficient. The local validity predicate
  is not a general self-intersection certificate.
- Face IDs apply only to the prepared input. Reconstruction unifies connected
  same-domain faces and edges to remove fill seams; unselected geometry stays
  fixed, but its face partition can change. Do not assume that Boolean output
  or a STEP roundtrip preserves face IDs.

See [ADR-0021](../architecture/adr/0021-bounded-through-hole-editing.md) for the
scope decision.

### Inverse editing

The regression example also imports a four-hole STEP plate, selects three
noncollinear holes, fits their radii to independently generated CAD volume and
in-plane center-of-mass targets, and validates the rebuilt STEP while preserving
the fourth hole. Its Newton iteration checks the Jacobian rank and edit bounds;
it is a reproducible example, not a general-purpose optimizer API.

Another regression solves a selected hole's radius and two transverse center
coordinates from independent CAD volume and in-plane center-of-mass targets,
then checks the rebuilt STEP and the unchanged neighboring hole. Moving an
isolated hole at fixed radius leaves volume and area unchanged; those two
measurements alone cannot determine a center position.

Choose enough independent target measurements. For four coplanar equal-height
holes, volume and even the full 3D center of mass need not determine all four
radii uniquely. A small mesh optimization residual also does not certify CAD
accuracy: check the rebuilt CAD and refine the discretization when necessary.

::: brepax.experimental.applications.hole_edit
    options:
      members:
        - prepare_hole_edit
        - PreparedHoleEdit
        - HoleEditModel
        - ThroughHole
        - RejectedHole
      show_source: false

## Circular cut crossing a plate edge

`brepax.experimental.contact` evaluates the remaining volume of a rectangular
plate after a perpendicular circular through-cut, including the transition
from an enclosed hole to an open notch. It integrates the analytic boundary
and differentiates that same continuous volume. It does not accept arbitrary
STEP input or rebuild a solid; the imported-hole API above retains its
clearance constraints.

Use a source checkout containing this experimental module for the example
and benchmark below.

```python
import jax
import jax.numpy as jnp

from brepax.experimental.contact import plate_with_hole_volume

jax.config.update("jax_enable_x64", True)
center = jnp.array([1.5, 0.2])
half_extents = jnp.array([3.0, 2.0])
height = jnp.array(2.0)

def volume(radius):
    return plate_with_hole_volume(center, radius, half_extents, height)

# First contact is at radius 1.5; 1.7 opens the right edge.
value, derivative = jax.jit(jax.value_and_grad(volume))(jnp.array(1.7))
```

Use one length unit and local plate coordinates. All parameters support first
derivatives; exact contact can have a smooth volume even when the solid's
boundary pinches and its mesh is not closed. Moving a fully enclosed hole's
center at fixed radius correctly has zero volume derivative. See
[ADR-0022](../architecture/adr/0022-contact-volume-experiment.md) for the
scope, numerical limitations, and comparison requirements.

Run the independent geometry and estimator comparison from a checkout:

```bash
uv run --group benchmark python -m benchmarks.contact_transition.run --cad --output /tmp/contact.json
```

The output retains failed cases, all declared smoothing choices, finite-difference
steps, fresh OCCT and mesh checks, and synchronized execution costs. Ordinary
AD of the same analytic primal is included. These bounded comparisons do not
establish a general advantage over ordinary AD or arbitrary CAD support.

::: brepax.experimental.contact
    options:
      members:
        - circle_rectangle_intersection_area
        - plate_with_hole_volume
      show_source: false

## Opposing planar face contact

`brepax.experimental.planar_contact` evaluates how intersection volume starts
changing when two solids move from a known opposing planar contact. It takes
a convex contact polygon and a translation/rotation direction. Unlike the
circular-cut example, this is a **one-sided directional response**, generally
nonlinear in direction, with no single gradient representing every motion.

```python
import jax
import jax.numpy as jnp

from brepax.experimental.planar_contact import (
    planar_contact_volume_direction,
    prepare_planar_contact_patch,
)

jax.config.update("jax_enable_x64", True)
patch = prepare_planar_contact_patch(
    [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    origin=[0, 0, 0], basis=[[1, 0], [0, 1], [0, 0]],
)
tilts = jnp.array([[1.0, 0, 0], [-1.0, 0, 0]])
response = jax.jit(jax.vmap(
    lambda omega: planar_contact_volume_direction(
        patch, jnp.zeros(3), omega, jnp.zeros(3)
    )
))(tilts)
# Both tilts create overlap: response == [1, 1].
```

The plane normal is the cross product of the basis columns, outward from the
fixed solid. Inputs use one length unit; translation is length per unit of the
dimensionless path parameter and rotation is radians per unit parameter.
Transform both velocities and the pivot with the geometry under proper rigid
motion. Scaling lengths and translation by `s` scales the response by `s**3`.
Changing the pivot requires the compensating translation documented by the API.
Preparation is host-only and requires x64; queries support `jit` and `vmap`
with a fixed polygon size. Differentiating a response query does not produce
the gradient of Boolean volume at contact.

For actual CAD inputs, `prepare_planar_contact_reference` in
`brepax.experimental.planar_contact_cad` takes two solids and explicit face IDs.
It checks finite convex polyhedra with straight-edged, single-wire planar
faces and unique opposing supports. Its required `mode` distinguishes
`"zero_residual"` from `"projected_nominal"`. Both return a **nominal support
polygon**, with the source gaps, normal residuals, and tolerances retained in
`prepared.diagnostics`; neither certifies exact source contact nor repairs
the solids. Even a positive gap below OCCT tolerance changes the derivative.
Curved or holed faces, initial overlap, additional contact patches, and general
nonconvex CAD are outside this selected-face preparation contract. The collection
API below handles the separate case of explicitly supplied convex cells.

Run the grouped estimator comparison and fresh CAD direction-selection example:

```bash
uv run --group benchmark python -m benchmarks.face_contact.run --output /tmp/face-contact.json
```

The same finite motion candidates and action step are evaluated with actual CAD
volumes. Results include calibration choices, five geometry-group splits,
ablations, setup/runtime costs, and unsupported or unresolved cases. A good
infinitesimal direction does not certify a finite edit; tiny CAD wedges can
fall below OCCT resolution. This experiment does not add a generic optimizer,
contact detector, or arbitrary STEP topology-editing API. See
[ADR-0022](../architecture/adr/0022-contact-volume-experiment.md) for the decision
and the [directional derivation](../explanation/stratification_theory.md#directional-response-at-opposing-planar-contact).

::: brepax.experimental.planar_contact
    options:
      members:
        - PlanarContactPatch
        - prepare_planar_contact_patch
        - planar_contact_volume_direction
        - PlanarContactSet
        - prepare_planar_contact_set
        - planar_contact_set_volume_direction
      show_source: false

::: brepax.experimental.planar_contact_cad
    options:
      members:
        - prepare_planar_contact_reference
        - PreparedPlanarContact
        - PlanarContactDiagnostics
      show_source: false

### Multiple contacts from convex cells

`prepare_planar_contact_set_reference` accepts finite lists of convex
polyhedral CAD cells for the fixed and moving bodies. The moving cells share
one rigid motion. It checks every within-body and cross-body pair; same-body
cell interiors must be disjoint, and initial cross-body overlap is rejected.
All positive-area contacts contribute to the packed response. Positive gaps
and edge/point contacts remain in the inventory even though their first-order
volume response is zero. Shared patch edges are allowed; duplicate cells are
not.

For example, a cube approaching the floor and wall of an L-shaped stock has
two unit-area contact patches:

```python
import jax
import jax.numpy as jnp

from brepax._occt.backend import BRepPrimAPI_MakeBox, gp_Pnt
from brepax.experimental.planar_contact import planar_contact_set_volume_direction
from brepax.experimental.planar_contact_collection_cad import (
    prepare_planar_contact_set_reference,
)

jax.config.update("jax_enable_x64", True)
fixed = [
    BRepPrimAPI_MakeBox(gp_Pnt(-1, 0, -1), 2, 1, 1).Shape(),
    BRepPrimAPI_MakeBox(gp_Pnt(-1, 0, 0), 1, 1, 1).Shape(),
]
moving = [BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Shape()]
prepared = prepare_planar_contact_set_reference(
    fixed, moving, mode="zero_residual"
)
response = planar_contact_set_volume_direction(
    prepared.contact_set,
    jnp.array([-1.0, 0, -1]), jnp.zeros(3), jnp.zeros(3),
)  # 2.0; the finite overlap is 2*t - t**2 for small positive t.
```

Preparation requires JAX x64. The packed set supports `jit` and `vmap`, with
optional triangle capacity and masks
when preparing sets directly. Direct numerical preparation only packages the
supplied patches; it cannot establish contact completeness or absence of
duplicates.

The CAD collection uses represented vertex hulls for pair classification and
retains source-plane residuals for its nominal contact polygons. Neither is a
certificate for exact source-CAD contact. Unlike selected-face counterfactual
projection, this collection never promotes a positive represented gap or
interior overlap into touching: gaps stay separated and overlaps are rejected.
`"projected_nominal"` only permits the bounded face-residual preparation after
the represented pair is touching. Source faces must own the exact represented
support vertices and resolve the same support plane and normal; an unresolved
thin patch is rejected rather than replaced by another nearby face. Rotating
and re-extracting adjacent cells can
introduce tiny represented overlaps and cause conservative rejection. Preparing
an explicit design in its working frame and then transforming the contact model
is a distinct route; finite CAD must still be checked after actual placement.

The bounded cutter/socket example selects both a direction and a finite amount,
then executes an actual stock-minus-cutter Boolean edit:

```bash
uv run --group benchmark python -m benchmarks.multi_contact.run \
  --output /tmp/multi-contact.json --step-output /tmp/cutter-edit.step
```

The report compares target-removal error and fixed-step material removal,
including single-patch ablations, sampled estimators, actual CAD differences,
and scanning the same finite candidates. It keeps preparation, compilation,
query and Boolean costs separate. Common/Cut volume balance and valid shapes
check the executed action; they do not imply that a requested target volume
was achieved exactly. First-order selection can miss a better finite action.
The demonstration STEP assigns millimetres to its generated working lengths
and is read back to verify volume and validity; it does not infer input units.

The single-contact benchmark also records three CAD resolution routes: raw
external construction, reconstruction from trusted dimensions in an explicit
fixed `CadFrame`, and transformation of an already-built B-Rep. Reconstruction
can resolve a small design feature without resolving uncertainty in imported
source CAD. A coordinate transform also transforms stored tolerances and is
not a substitute for constructing the design in suitable working units.

::: brepax.experimental.planar_contact_collection_cad
    options:
      members:
        - prepare_planar_contact_set_reference
        - PreparedPlanarContactSet
        - ContactPreparationError
      show_source: false

### Preserving a supplied placement

`brepax.experimental.planar_contact_placement` keeps local geometry and a
common frame instead of baking a transformation into every vertex. With the
local cells from the preceding example:

```python
from brepax.experimental.planar_contact_placement import (
    placed_planar_contact_set_volume_direction,
    prepare_placed_planar_contact_set_reference,
)

placed = prepare_placed_planar_contact_set_reference(
    fixed, moving, rotation=jnp.eye(3), origin=(2, -1, 3),
    length_scale=0.001, mode="zero_residual",
)
rate = placed_planar_contact_set_volume_direction(
    placed.model, jnp.array([-0.001, 0, -0.001]),
    jnp.zeros(3), jnp.array([2.0, -1, 3]),
)  # 2e-9 in world volume units per dimensionless time.
```

The convention is `world = origin + length_scale * rotation @ local`.
World velocities and pivot are pulled back to the local frame. Diagnostics in
`local_reference` retain local units. A genuine shared proper-rigid top-level
OCCT Location can instead be supplied through
`prepare_located_planar_contact_set_reference`; different Location datum chains,
reflections and scaled source Locations are rejected. Nested source geometry
is preserved. Explicit numerical length scaling remains available separately.

This preserves supplied provenance; it cannot reconstruct a lost frame from
flattened STEP vertices. Positive local gaps remain gaps and initial overlap
is rejected. `original_cad_contact_certified` remains false. The response
supports `jit` and `vmap` over motion directions, not differentiation of the
prepared geometry or frame. Finite edits still require actual CAD validation.

::: brepax.experimental.planar_contact_placement
    options:
      members:
        - PlacedPlanarContactSet
        - PreparedPlacedPlanarContactSet
        - place_planar_contact_set
        - placed_planar_contact_set_volume_direction
        - prepare_placed_planar_contact_set_reference
        - prepare_located_planar_contact_set_reference
      show_source: false

### Finite polyhedral translation

`brepax.experimental.polyhedral_translation` prepares the complete finite
overlap-volume profile along a prescribed translation. Inputs are convex-cell
vertex arrays, not CAD solids. Each body's cell interiors must be disjoint,
and the bodies must have no initial interior overlap. Preparation computes
rational event times and cubic coefficients from the represented geometry;
the JAX evaluator uses floating coefficients. It does not fit CAD samples.

```python
from itertools import product

import jax
import numpy as np

from brepax.experimental.polyhedral_translation import (
    polyhedral_translation_slope,
    polyhedral_translation_volume,
    prepare_polyhedral_translation,
)

jax.config.update("jax_enable_x64", True)
box = np.array(list(product((0.0, 1.0), repeat=3)))
path = prepare_polyhedral_translation(
    [box], [box + (1, 1, 0)], (-1, -1, 0), max_step=2.0,
)
value = jax.jit(polyhedral_translation_volume)(path.profile, 0.25)  # 0.0625
left = polyhedral_translation_slope(path.profile, 1.0, side="left")  # 2
right = polyhedral_translation_slope(path.profile, 1.0, side="right")  # -2
```

This edge-contact example has zero initial slope, grows quadratically, and
then loses overlap after the event at step one. Values have cubic length units;
step is dimensionless. `jit`, `vmap` over steps and ordinary step AD inside
open event intervals are supported. At events use explicit one-sided slopes;
geometry and direction differentiation are outside the contract.

The profile's floating thresholds round rational event times upward for
binary64 query classification. A threshold need not equal the exact event:
`path.events` retains its rational time and exact left/right slope limits.
Distinct events without a resolvable floating interval and overflowing or
underflowing coefficients are rejected. Out-of-range queries, and slope requests from a
side outside the prepared interval, return NaN. Preparation is intended for
small convex cells and can be expensive. It does not support rotation or
certify that a separate CAD model matches the represented hulls.

```bash
uv run --group benchmark python -m benchmarks.multi_contact.translation \
  --output /tmp/translation.json --step-output /tmp/translation-edit.step
```

The benchmark compares independent interval/section references and actual CAD,
checks event slopes, and executes a target-volume Cut on a known monotone
translation interval. A separate rotating-prism control shows why the initial
velocity field cannot determine higher-order finite terms. See the
[derivation and path limits](../explanation/stratification_theory.md#finite-translation-and-event-intervals).

::: brepax.experimental.polyhedral_translation
    options:
      members:
        - PolyhedralTranslationProfile
        - PreparedPolyhedralTranslation
        - TranslationEvent
        - prepare_polyhedral_translation
        - polyhedral_translation_volume
        - polyhedral_translation_slope
      show_source: false

### Target positions and reachable volume

`brepax.experimental.polyhedral_targets` uses the retained exact polynomials
to find target-volume positions over the entire prepared translation path.
It handles multiple roots, roots at extrema, and constant solution intervals.
These are host queries; they do not differentiate the root-selection operation.

```python
from itertools import product

import jax
import numpy as np

from brepax.experimental.polyhedral_targets import (
    polyhedral_translation_range,
    polyhedral_translation_targets,
)
from brepax.experimental.polyhedral_translation import prepare_polyhedral_translation

jax.config.update("jax_enable_x64", True)
box = np.array(list(product((0.0, 1.0), repeat=3)))
path = prepare_polyhedral_translation(
    [box], [box + (1, 1, 0)], (-1, -1, 0), max_step=2.0,
)
solutions = polyhedral_translation_targets(
    path, 0.25, step_tolerance=1e-12, volume_tolerance=1e-12,
)
steps = [root.step for root in solutions.roots]  # [0.5, 1.5]
volume_range = polyhedral_translation_range(
    path, step_tolerance=1e-12, volume_tolerance=1e-12,
)
# The maximum volume enclosure is exactly [1, 1].
```

`unreachable` means no exact solution exists on this represented path.
`unresolved` means exact solutions exist but some cannot be returned as
distinct floating actions meeting the requested tolerances or refinement cap.
`exact_reachable` records existence separately. Root records retain rational
isolating bounds and the exact model residual at each proposed float step;
constant solution intervals are returned separately. Inspect these outcomes
before choosing a position, and verify the actual CAD edit independently.

Step tolerance is absolute and dimensionless. Volume tolerance has cubic
length units and must scale with the target when changing units. A floating
target denotes that exact binary64 value; use `fractions.Fraction` for another
exact rational target. Volume-range enclosures are conservative when their
refinement is unresolved. They can be computed once for repeated targets;
the target solver decides existence directly from the polynomials.

::: brepax.experimental.polyhedral_targets
    options:
      members:
        - TranslationTargetRoot
        - TranslationTargetInterval
        - TranslationTargetResult
        - TranslationRangeCandidate
        - TranslationVolumeRange
        - polyhedral_translation_targets
        - polyhedral_translation_range
      show_source: false

### Design sensitivities along a translation path

`brepax.experimental.polyhedral_parameter` prepares an affine family of convex
vertex cells with fixed facet normals and one or two dimensionless design
parameters. Unlike a fixed translation profile, its traced volume evaluation
changes the geometry with those parameters. Translation velocity stays fixed.

The following example changes the initial gap and the stock extrusion height.
At the reference design the target has two positions; the first is a simple
root whose sensitivity can be evaluated inside its event interval.

```python
from itertools import product

import jax
import numpy as np

from brepax.experimental.polyhedral_parameter import (
    AffinePolyhedralCell,
    polyhedral_parameter_position_sensitivity,
    prepare_polyhedral_parameter,
    specialize_polyhedral_parameter,
)
from brepax.experimental.polyhedral_targets import polyhedral_translation_targets

jax.config.update("jax_enable_x64", True)
stock = np.array([
    (x, y, z) for z in (0.0, 1.0)
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
])
stock_changes = np.zeros((2, 6, 3))
stock_changes[1, 3:, 2] = 1.0
tool = np.array(list(product((-0.375, -0.125), (-0.5, 1.5), (-0.5, 1.5))))
tool_changes = np.zeros((2, 8, 3))
tool_changes[0, :, 0] = -1.0
design = prepare_polyhedral_parameter(
    [AffinePolyhedralCell(stock, stock_changes)],
    [AffinePolyhedralCell(tool, tool_changes)],
    (1.0, 0.0, 0.0), max_step=1.5,
)
theta = np.zeros(2)
path = specialize_polyhedral_parameter(design, theta)
solutions = polyhedral_translation_targets(
    path, 15 / 128, step_tolerance=1e-12, volume_tolerance=1e-12,
)
steps = [root.step for root in solutions.roots]  # [23/32, 5/4]
sensitivity = jax.jit(polyhedral_parameter_position_sensitivity)(
    design.model, 23 / 32, theta,
)  # [1, -15/32]: fixed physical target, dimensionless design parameters.
```

An `AffinePolyhedralCell` has vertices `(N, 3)` and parameter displacements
`(P, N, 3)`. Its vertex map is affine in `theta`; vertices, displacement
coefficients and velocity share a length unit. Preparation checks the fixed
facet-normal assumption and derives sufficient strict linear conditions on
`theta`. These preserve cell validity, disjoint interiors and event ordering;
they need not describe the largest valid region. Conflicting event-order
changes at the reference design are rejected even if volume itself is smooth.

`polyhedral_parameter_volume` supports `jit`, `vmap` and first-order JVP/reverse
derivatives in step and design parameters inside resolved open intervals.
Its custom JVP uses the physical derivative polynomials, avoiding cancellation
from differentiating rounded moving normalization anchors. Prepared model
coefficients are not differentiation variables; higher derivatives are outside
the contract. It returns NaN near moving events or parameter boundaries and
when value/derivative cancellation or numerical range is unresolved. This can
include a mathematically smooth stationary point. That is a limitation of the
numerical derivative chart, not a claim that physical volume is undefined.
Host specialization checks exact parameter
conditions and creates a complete fixed-parameter translation profile with
the existing boundary-value, one-sided-slope and target-query contracts.
Specialization evaluates the affine represented family; independently rounded
new vertices or rebuilt CAD can differ from that model.

The position-sensitivity query computes the local volume-level-set derivative;
it does not verify a target residual or select a root. Check the root first.
Zero or unresolved slope, event boundaries and branch changes require separate
treatment. Supply `target_parameter_gradient` when the target itself changes
with the design. For fixed dimensionless parameters, scaling all lengths by
`s` scales volume and its parameter derivatives by `s**3`; convert physical
length sensitivities by the chain rule. Retain independent CAD validation.

```bash
uv run --group benchmark python -m benchmarks.multi_contact.parametric \
  --output /tmp/parametric.json
```

This benchmark compares finite-profile inversion, first-order proposals and
CAD-only search on the same supplied translation paths and ranges. It checks
root inventories and path-unreachable targets against independent prism
sections, evaluates design sensitivities, and validates actual rebuilt CAD
edits. Parameter-volume differences rebuild CAD directly; position differences
use specialized roots checked against independent section solutions and rebuilt
CAD residuals. Failures and unsupported cases remain in the results. Timings
include preparation and attempted CAD operations, but share one process and
JAX caches; they are diagnostics rather than isolated cold-start comparisons.

::: brepax.experimental.polyhedral_parameter
    options:
      members:
        - AffinePolyhedralCell
        - PolyhedralParameterModel
        - ParameterTranslationEvent
        - PreparedPolyhedralParameter
        - prepare_polyhedral_parameter
        - specialize_polyhedral_parameter
        - polyhedral_parameter_volume
        - polyhedral_parameter_events
        - polyhedral_parameter_position_sensitivity
      show_source: false

### Selecting finite edits with a CAD budget

```bash
uv run --group benchmark python -m benchmarks.multi_contact.selection \
  --output /tmp/finite-selection.json
```

This benchmark compares analytical candidate ordering and CAD confirmation
with CAD-only adaptive selection. Its separate amount-refinement panel uses
bounded proposals and only returns observed actions. It promises neither
monotonic volume along arbitrary motion nor target reachability. A negative
error difference relative to the fixed candidate grid is possible when a
refined amount improves on every grid action.

Fresh synthetic geometry groups and five grouped train/test partitions keep
calibration apart from each evaluated shape; repeated targets are not
independent shapes. Reports retain failed or unsupported inputs and targets
exceeding a known total-material bound. Probe attempts and final independent
Common/Cut checks have separate costs. Cached replay timings do not establish
end-to-end speed superiority. The experimental comparison is not a general
optimizer API or evidence for arbitrary STEP inputs.

## Other experimental applications

::: brepax.experimental.applications.mold_direction
    options:
      show_source: false

::: brepax.experimental.applications.cad_frame.CadFrame
    options:
      show_source: false
