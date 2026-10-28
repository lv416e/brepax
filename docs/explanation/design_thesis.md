# Design Thesis

BRepAX investigates differentiable Boolean geometry alongside practical
STEP/B-Rep evaluation. The research question is how local contact geometry
and moving boundaries determine derivatives, and how to compute them without
confusing a numerical surrogate with the intended CAD objective.

## What Is Implemented

| Method | Current computation | Limitation |
|---|---|---|
| A: smoothing | Public 2D union API: smooth SDF composition and sigmoid measure integral | Temperature, finite-domain, and discretization bias; public 3D dispatch is unimplemented |
| B: TOI | Unimplemented | Event-time sensitivity alone does not define a static gradient correction |
| C: relation dispatch | Hard sampled forward value; primitive-volume derivatives or sigmoid STE backward | Sampled relation detection and intersecting-gradient approximation |
| Experimental circular cut | Analytic disk/rectangle measure and boundary-flux first derivatives | Perpendicular extruded cut; no general CAD reconstruction |
| Experimental planar contact | Positive one-sided volume response to a supplied rigid direction, including complete collections of opposing planar contacts | Known patches or explicit convex-cell bodies with disjoint interiors; nonlinear in direction; finite edits require CAD validation |

The [stratification explanation](stratification_theory.md) defines the
current contract and contact regularity. Exact primitive-volume derivatives
in simple relations do not make Method C exact everywhere or solve moving
intersection boundaries.

## Contact Geometry Is the Mathematical Question

Point, line, and face contact can have different orders of area/volume
change. Smooth sphere contact has a continuous first volume derivative,
while positive-area planar contact can have different one-sided derivatives.
Internal versus external contact alone does not determine the order, and
changing the objective from volume to surface area changes the answer.

Four implementation labels are not a proved Whitney stratification.
Contact sets and degeneracies need explicit analysis. Contact dynamics also
does not establish a mathematical isomorphism with static Boolean
differentiation: a trajectory has flow and reset rules that a static volume
does not. The [TOI discussion](toi_derivation.md) separates these contracts
and explains why the previous static correction was invalid.

## Three Distinct Numerical Questions

| Question | What must be distinguished |
|---|---|
| Locally constant objective | An exact zero derivative versus a deliberately nonlocal exploration signal |
| Near-contact volume derivative | Missed contact regions, smoothing, quadrature, and true one-sided behavior |
| Moving CAD boundaries | A frozen-mesh derivative versus derivatives of regenerated geometry, trims, and topology |

Smoothing may help exploration, but guarantees neither a useful gradient
everywhere nor convergence to an unsmoothed optimum. A hybrid schedule
remains a [proposal](hybrid_optimization_strategy.md).

AD through a fixed discretization differentiates that discretization.
Shape calculus motivates accounting for moving boundaries; it does not
remove frozen trim or connectivity assumptions. Bounded through-hole editing
deliberately preserves clearance: its conservative edit budget is not a
demonstrated physical contact event or topology-changing Boolean derivative.

## Evidence and Related Ideas

Density projection, differentiable rendering, hybrid dynamics, and shape
differentiation offer techniques with different objectives and assumptions.
They do not establish universal convergence, a fixed bias direction, or
uniqueness for BRepAX. For example,
[nvdiffrast antialiasing](https://nvlabs.github.io/nvdiffrast/#antialiasing)
uses an approximate local pixel-coverage model; it is not an exact Boolean
volume derivative.

Comparisons need independent geometric values and derivatives, matched
initializations and budgets, and temperature/resolution sweeps. Check
rigid-motion and unit-scaling laws for both values and gradients.
Convergence on a selected fixture does not prove that gradient sign is
always correct or that a frozen-mesh plateau has one universal cause.
The [historical concept proof](concept_proof_evaluation.md) identifies
earlier conclusions that no longer describe the implementation.
