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

## Other experimental applications

::: brepax.experimental.applications.mold_direction
    options:
      show_source: false

::: brepax.experimental.applications.cad_frame.CadFrame
    options:
      show_source: false
