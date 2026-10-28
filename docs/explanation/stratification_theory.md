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
