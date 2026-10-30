# Stratification Theory

Boolean relations help organize geometric derivatives, but a topology
change does not by itself imply a first-derivative jump. The objective,
local contact geometry, and numerical discretization must be specified.

## Relation Regions and Contact Sets

For disks or spheres with positive radii and center distance $d$:

| Label | Relationship | Strict region |
|---|---|---|
| 0 | Disjoint | $d > r_1+r_2$ |
| 1 | Partial overlap | $\lvert r_1-r_2\rvert < d < r_1+r_2$ |
| 2 | A strictly contained in B | $d+r_1 < r_2$ |
| 3 | B strictly contained in A | $d+r_2 < r_1$ |

These operational labels are not a complete Whitney stratification.
External tangency, internal tangency, and equal-radius coincidence require
separate sets. All four strict regions are full-dimensional; partial overlap
is not uniquely highest-dimensional. A Whitney stratification additionally
requires smooth strata, frontier conditions, and Whitney regularity, none
of which follows from four labels.

For union measure, disjoint and contained formulas reduce to $V_A+V_B$,
$V_B$, and $V_A$, respectively. Partial overlap requires intersection
measure. Exact primitive-volume derivatives therefore help without solving
the harder contact problem.

## Contact Order for Area and Volume

Let $\delta>0$ denote normal penetration from first contact. For a smooth
$m$-dimensional contact set in $n$ dimensions, suppose the local gap in the
$k=n-1-m$ transverse tangent directions is, to leading order,

$$
\delta-\tfrac12 y^\top H y,\qquad H\text{ positive definite}.
$$

Integrating its positive part gives changed measure proportional to
$\delta^{1+k/2}$. This local model assumes nondegenerate relative curvature,
a regular contact set, and no simultaneous endpoint event.

| Geometry and objective | Leading change as $\delta\downarrow0$ | Contact regularity |
|---|---|---|
| Two disks, overlap area | $C\delta^{3/2}$ | First derivative continuous; second derivative unbounded |
| Two spheres, overlap volume | $C\delta^2$ | First derivative continuous; second derivative jumps |
| Parallel finite cylinders, side contact with positive common axial length | $C\delta^{3/2}$ | First derivative continuous |
| Opposing planar faces with positive contact area $A$ | $A\delta$ | One-sided volume derivatives differ |

Here $C>0$ depends on geometry. Zero relative curvature, corners, contact
endpoints, and coincident surfaces need separate analysis; contact dimension
alone is not a universal classification.

For internal tangency with fixed radii $R>r>0$, let $\delta=d-(R-r)$.
The excess union measure beyond the larger primitive satisfies

$$
A_{\mathrm{union}}-\pi R^2
\sim \frac43\sqrt{\frac{2Rr}{R-r}}\,\delta^{3/2},
\qquad
V_{\mathrm{union}}-\frac{4\pi R^3}{3}
\sim \frac{\pi Rr}{R-r}\,\delta^2.
$$

Ordinary internal and external contacts are therefore both $C^1$ for these
area/volume objectives. The earlier assertion of a first-order jump at
unequal-radius internal tangency was incorrect. Equal-radius coincidence
is a distinct degeneracy not covered by this conclusion.

The objective also matters. For two unit spheres with $d=2-\delta$,
overlap volume is $\pi\delta^2/2-\pi\delta^3/12$, whereas union surface area
is $8\pi-2\pi\delta$ for $0<\delta<2$. Volume is $C^1$ at first contact;
surface area has a first-derivative kink. The overlap volume tends to zero
continuously at first contact.

## Directional Response at Opposing Planar Contact

Let finite solids have disjoint interiors and meet on one isolated convex
planar polygon $M$, with positive thickness on both sides. Let $n$ point out
of the fixed solid toward the moving solid. For a rigid path with velocity
$w(x)=v+\omega\times(x-p)$ and dimensionless parameter $t$, the positive
one-sided intersection-volume response is

$$
D^+ V_{\cap}(v,\omega)
=\lim_{t\downarrow0}\frac{V_{\cap}(t)-V_{\cap}(0)}{t}
=\int_M [-n\cdot w(x)]_+\,dA.
$$

Locally, normal displacement creates an overlap layer with thickness
$t[-n\cdot w(x)]_++O(t^2)$. The moving polygon edges affect strips of area
$O(t)$ and thickness $O(t)$, so their volume does not contribute at first
order. This bounded argument requires regular finite edges, fixed positive
thickness, bounded velocities, and no other first-order boundary contribution.

For $M=[-1,1]^2$ in the XY plane with pivot at the origin, the integrand is
$[-v_z+\omega_y x-\omega_x y]_+$. Normal approach with $v_z=-1$ gives 4;
separation gives 0. Both unit tilts $\omega_x=1$ and $\omega_x=-1$ give 1.
Consequently the response is positively homogeneous and subadditive, but
not linear or odd. It cannot be represented by one ordinary gradient/JVP.
The signed left derivative along a direction is $-D^+V_{\cap}(-v,-\omega)$.
Rigid union and stock-minus-tool volume responses have the opposite sign.

The experimental planar evaluator implements this direction-dependent
integral. Its host CAD preparation returns a nominal projected support-polygon
model with diagnostics. A positive source gap, existing volume overlap, an
omitted positive-area contact, or co-oriented coincidence needs different
geometric information; none is certified by a small OCCT residual. See the
[application contract](../reference/applications.md#opposing-planar-face-contact)
and [ADR-0022](../architecture/adr/0022-contact-volume-experiment.md).

One-sided interior/exterior traces in
[Berggren (2023)](https://doi.org/10.23952/cot.2023.27) explain the relevant
approach/separation distinction. Its nonnegative basis perturbations do not
directly state the mixed-sign rigid-tilt formula above, which follows from
the local overlap argument. The cut-geometry AD results of
[Wegert et al. (2025)](https://arxiv.org/abs/2504.09748v3) and
[Wegert et al. (2026)](https://arxiv.org/abs/2609.15084v2) assume stable cut
topology; in particular, the latter's Assumption 3.5 excludes birth of a
full-dimensional overlap from this face contact. These are prior techniques,
not a direct proof of this event or evidence of theoretical novelty here.

### Several simultaneous polyhedral contacts

Let each body be a finite union of full-dimensional convex polyhedral cells,
with disjoint cell interiors within each body. All moving cells follow the
same rigid path, and the two bodies initially have disjoint interiors. Then
the component-pair intersections partition the total intersection up to
measure-zero boundaries, even at finite displacement:

$$
|A\cap g_t(B)|=\sum_{i,j}|A_i\cap g_t(B_j)|.
$$

Consequently, the positive one-sided volume response is the sum of the
planar integrals over **all** positive-area component-pair contacts. Shared
patch edges require no first-order correction. Duplicate or overlapping
same-body cells invalidate this partition; their contributions cannot simply
be added. Two initially disjoint convex bodies have a separating plane, so
nonparallel contact patches require a nonconvex body or a union of cells.

Positive-gap pairs contribute zero for sufficiently small motion. Edge-only
contacts contribute at most $O(t^2)$ and vertex-only contacts at most $O(t^3)$
for a fixed nondegenerate polyhedral configuration. They must still be kept
in the geometry inventory: zero first-order response is not finite-motion
safety. For example, diagonal motion into an edge can create volume $t^2$.
A cube moving into an L-shaped corner can instead give $2t-t^2$: the two
unit-area contacts contribute a first derivative of 2, while the finite
difference quotient has an $O(t)$ error. Bounds are not uniform as feature
thicknesses or angles degenerate.

The summed response remains convex and positively homogeneous in the supplied
rigid direction. It vanishes exactly when the separating normal velocity is
nonnegative at every vertex of every convex contact polygon. These linear
inequalities describe a cone of motions without **first-order** penetration,
not a certificate for a finite edit. Cell ordering and nonoverlapping patch
subdivision preserve the response, in addition to its rigid-motion, unit,
and compensated-pivot laws.

This is a local polyhedral derivation using the measure partition above.
[Kiderlen and Rataj, Theorem 1.1](https://arxiv.org/pdf/1708.09191) establishes
a related positive-part boundary integral for dilation by finite translation
sets. [Schymura, Corollary 4](https://arxiv.org/pdf/1010.2446) bounds changes
of overlap under rigid motions; paths differing by $O(t^2)$ therefore have
the same first-order response in this bounded setting. Neither result is
being cited as a theorem about arbitrary source CAD or as a novelty claim
for this implementation.

### Finite translation and event intervals

For a prescribed translation $B_t=B+t v$, each component-pair intersection
has fixed plane normals and affine offsets:

$$
P(t)=\{x:n_k\cdot x\le b_k+t c_k\}.
$$

An independent triple of planes defines a candidate vertex $x(t)=a+t d$.
Its remaining inequalities give an exact feasibility interval. Between these
interval endpoints the active vertices and faces are fixed; a consistently
oriented facet triangulation expresses volume as a sum of determinants of
three affine vectors. Each determinant has degree at most three. Summing over
the disjoint-interior cell pairs preserves this piecewise-polynomial structure.
This is established parametric-polytope volume computation; see
[Verdoolaege, Section 5.9](https://compsys-tools.ens-lyon.fr/iscc/barvinok.pdf).
It is not a new general Boolean differentiation theorem.

The experimental preparation derives these intervals and coefficients from
the represented vertex hulls with rational arithmetic, without fitting sampled
CAD volumes. The JAX evaluator uses floating coefficients and must retain their
rounding limits. Geometry and translation direction are fixed by host
preparation; differentiation with respect to step inside an open interval
differentiates its polynomial. At an event, use the specified one-sided slope
instead of assuming a unique derivative. The model is continuous in volume,
but its derivatives need not be continuous.

For appropriately translating unit boxes, the first branches are $t$, $t^2$,
and $t^3$ for face, edge, and point contact. A positive gap postpones overlap
until its entry event. Later exit events can make volume decrease, so neither
zero initial response nor a missing sampled bracket establishes that finite
overlap or a target is impossible. Finite models still require an independent
CAD check when applied to source solids.

### Target positions along a finite path

For a target volume $q$, solve $V(t)-q=0$ on every prepared interval. A
finite-body intersection can enter and leave the stock, attain the target
several times, or have a constant-volume interval. Consequently a sign change
between sampled endpoints is sufficient to locate some roots, but its absence
does not establish that no root exists. A target at a local extremum can be
a repeated root without a sign change.

The exact represented polynomial and target permit rational root counting
and isolation, including repeated roots. Keep an isolating interval separate
from a floating candidate that meets the requested step and volume tolerances.
Failure to resolve such a candidate is a numerical outcome, not a proof of
unreachability. Constant solution intervals are distinct from isolated roots.
Range estimates likewise require the interval endpoints and interior stationary
points, not just sampled values. Any reachability conclusion concerns only the
represented geometry and supplied translation interval.

Polynomial inversion of geometric volume is established. For example,
[Kromer and Bothe, Section 2.4.1](https://arxiv.org/pdf/2101.03861) use the cubic
volume within a bracket to position a cutting plane without further geometric
truncation there. Their plane-cut volume is monotone; that uniqueness
assumption does not extend to a bounded cutter entering and leaving a solid.

### Design parameters and moving event boundaries

With fixed plane normals and affine dimension controls $\theta$, consider

$$
P(t,\theta)=\{x:n_k\cdot x\le b_k+t c_k+d_k^\top\theta\}.
$$

Independent plane triples give vertices affine in both $t$ and $\theta$.
A remaining inequality has the form
$a+\beta t+\gamma^\top\theta\le0$. When $\beta\ne0$, its candidate event is

$$
\tau(\theta)=-\frac{a+\gamma^\top\theta}{\beta},\qquad
\partial_\theta\tau=-\frac{\gamma^\top}{\beta}.
$$

A fixed ordering of these events and fixed active faces define a local
parameter region. Within its event intervals, oriented facet determinants
give volume polynomials of total degree at most three in $(t,\theta)$.
Differentiating those polynomials includes the motion of the intersection
vertices. This uses the same parametric-polytope principle as the fixed-path
construction; it is not a new general shape-differentiation theorem.
[Wegert et al., Section 3.2](https://arxiv.org/html/2504.09748v3#S3.SS2) similarly
propagate derivatives through cut vertices and maps while assuming stable cut
topology under small perturbations.

The parameter region must also preserve valid, bounded, positive-volume
cells, their fixed facet normals, and the disjoint-interior conditions used
to sum component volumes. Checking these properties only at the reference
parameter does not justify evaluation at another parameter. Conservative
validity conditions need not describe the largest admissible region.
Floating checks near a moving event require their own uncertainty policy;
an exact host model does not make a rounded event comparison exact.

For numerical evaluation, expanding near a moving event reduces cancellation
in the value. Differentiating that rounded normalization introduces another
chain-rule cancellation, which can obscure a structurally zero design
derivative. The parameter evaluator therefore uses a custom first-order JVP
from the physical derivative polynomials collected before floating packing.
It represents the same volume model and is linear in the step/design tangents;
the host-prepared coefficients are not differentiation variables. Independent
derivative references are necessary to validate this choice. A query whose
value or intended derivative cannot be resolved numerically returns NaN,
even when its exact host model is valid. Some smooth stationary configurations
are consequently outside the numerical chart's accepted queries.

An event can be a redundant change of representation rather than physical
contact. Moreover, events with distinct sensitivities can coincide: the first
of $1+\theta$ and $1-\theta$ is $1-|\theta|$. Rejecting that event chart does
not prove that the underlying volume lacks a derivative. Zero crossing speed,
event-order changes, and simultaneous events require separate treatment.

For a locally isolated target position $t_*(\theta)$ inside an event interval,
where $V(t_*(\theta),\theta)=q(\theta)$ and $\partial_t V\ne0$, the implicit
function theorem gives

$$
\partial_\theta t_*=
\frac{\partial_\theta q-\partial_\theta V}{\partial_t V}.
$$

For a fixed physical target, its derivative is zero. If the target depends on
$\theta$, include its derivative. This local formula does not select between
multiple roots, certify that an
arbitrary query is a target root, or extend through a zero slope or event
boundary. Event-time sensitivity is not an additive correction to a static
Boolean volume derivative; see [the TOI distinction](toi_derivation.md).

### Rotation depends on the full path

The finite benchmark uses $x(t)=p+R(t\omega)(x-p)+t v$. Let $\Omega x=
\omega\mathbin{\times}x$. Replacing $p$ by $p+d$ and $v$ by $v+\Omega d$
preserves the initial velocity field, but changes the finite trajectory by

$$
(I-R(t\omega))d+t\Omega d=-\tfrac12t^2\Omega^2d+O(t^3).
$$

This is a different physical path, not a violation of coordinate covariance.
Consequently the earlier compensated-pivot law is a first-order law only.

For a prism $[0,w]\times[-a,b]\times[0,h]$ against the wall $x\le0$, take
$p=(q,0,0)$, $v=(0,q,0)$ and $\omega=(0,0,1)$, with $q\ge0$. For
$0<t<\pi/2$, $-a<q\tan(t/2)<b$ and
$0<b\tan t-q(\sec t-1)<w$, the overlap is an uncapped triangle and

$$
V(t)=\frac{h[b\sin t-q(1-\cos t)]^2}{2\sin t\cos t}
=h\left(\frac{b^2}2t-\frac{bq}2t^2+
\left(\frac{b^2}6+\frac{q^2}8\right)t^3\right)+O(t^4).
$$

All these paths have the same initial velocity field. When $q=0$, the
quadratic term vanishes and the exact branch is $h b^2\tan(t)/2$.
At $t_*=\arctan(w/b)$, the far side becomes active and the next branch is
$h[wb-w^2\cot(t)/2]$. Value and first derivative agree at this event, while
second derivatives do not. Extrapolating the old branch through the event
is invalid. These bounded prism formulas do not extend the translation API
to generic rotating polyhedra.

## Current BRepAX Implementation

Method C in `brepax.boolean.stratum` uses a custom VJP:

1. The forward value samples a hard Boolean indicator on a finite grid.
   Exact min/max set composition does not make quadrature exact.
2. Sampled SDF signs select a relation label. Small overlap or protrusion
   regions can be missed. There is no explicit exact-contact classification.
3. Disjoint and contained cases use primitive-volume derivatives when
   finite analytical formulas exist. Exactness requires both correct
   classification and correct formulas; unbounded primitives use a grid
   estimate instead.
4. Partial overlap differentiates a sigmoid indicator on the frozen grid,
   with width equal to the geometric mean cell spacing.

The backward rule is a surrogate for a continuous geometric derivative,
generally not the derivative of the hard sampled forward value. Three
simple labels with analytical formulas do not imply accuracy on three
quarters of configurations or solve moving intersection boundaries.

The analytical disk reference also uses distance floors and angle clipping.
Near contact, independent limits or stable formulas must distinguish these
safeguards from the mathematical derivative. See the
[safe-primal discussion](jax_where_gradient_pitfall.md).

## Invariance and Numerical Evidence

Exact area/volume is invariant under common rigid motion. Uniform length
scaling by $s>0$ scales $n$-dimensional measure by $s^n$ and gradients with
respect to lengths by $s^{n-1}$; center gradients rotate with geometry.
Grid orientation, finite windows, smoothing widths, and sampled detection
can break or approximate these laws. Dimensional tolerances and smoothing
parameters must scale with units.

Separate value error, relation detection, derivative error, and optimization
outcome. At nonsmooth events compare one-sided limits; away from events,
sweep finite-difference steps and resolution against an independent oracle.
Relative error alone is misleading when the true derivative tends to zero.

Selected-fixture errors are not universal convergence rates or gradient-sign
guarantees. AD through a frozen mesh likewise does not establish derivatives
of regenerated CAD with moving trims or changing topology. See the
[design thesis](design_thesis.md) and [TOI discussion](toi_derivation.md).
