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

## Other experimental applications

::: brepax.experimental.applications.mold_direction
    options:
      show_source: false

::: brepax.experimental.applications.cad_frame.CadFrame
    options:
      show_source: false
