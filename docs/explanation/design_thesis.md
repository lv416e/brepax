# Design Thesis

The mathematical foundation behind BRepAX: why contact dynamics
formulations from differentiable physics apply to CAD Boolean
operations, and how this enables a differentiable B-Rep kernel
without inventing new theory.

## Core Observation

Gradient discontinuities at topological boundaries in CAD Boolean
operations (where edges or faces appear or disappear) are
mathematically isomorphic to contact events in differentiable
physics simulation. Both involve:

- A smooth objective function with **piecewise-defined** gradients
- A **boundary surface** separating regimes (contact surface / stratum boundary)
- A **gradient jump** at the boundary (velocity impulse / area gradient discontinuity)
- Differentiation through a **root-finding problem** (time of impact / boundary crossing)

The same implicit function theorem (IFT) formula applies to both
domains. See the correspondence table in the
[TOI derivation](toi_derivation.md#5-contact-dynamics-correspondence)
for the formal substitution.

## Three Gradient Strategies

BRepAX maps established contact dynamics formulations to three
Boolean gradient strategies:

| Contact formulation | Boolean strategy | Method | Character |
|---|---|---|---|
| Compliant contact (penalty) | Smooth SDF composition | A (smoothing) | Global signal, O(k) bias |
| LCP / convex optimization | Boundary integral correction | B (TOI) | Exact at boundary, root-finding required |
| Position-based dynamics | Stratum-aware analytical dispatch | C (stratum) | Exact in 3/4 strata, O(h) in 1/4 |

This mapping is also isomorphic to differentiable rendering:

| Rendering | BRepAX | Signal character |
|---|---|---|
| SoftRas (2019) | Method A | Global, approximate |
| Mitsuba 3 boundary integral | Method B | Exact at silhouette |
| nvdiffrast edge sampling | Method C | Exact where visible, approximate at edges |

## Topology Optimization Connection

The domain indicator used in Method A and the intersecting-stratum
STE in Method C share a common form:

$$H_\epsilon(\mathbf{x}) = \sigma\bigl(-\text{sdf}(\mathbf{x}) / \epsilon\bigr)$$

This is mathematically identical to the **density projection**
introduced by Guest (2004) in topology optimization, where a
Heaviside projection sharpens intermediate densities toward 0/1.
The connection was discovered during physics PoC development and
provides theoretical grounding: convergence properties, bias
direction (toward volume increase), and the O(h) intrinsic limit
of grid-based integration of discontinuous integrands are all
established results in the topology optimization literature.

## Three Independent Problems

Empirical measurement revealed that "Stratum 1 gradient accuracy"
decomposes into three independent problems with distinct solutions:

| Problem | Phenomenon | Root cause | Solution |
|---|---|---|---|
| Cross-stratum stall | Gradient is zero in disjoint stratum for intersection volume | SDF carries no overlap information outside the intersection | Method A (smoothing extends signal across boundary) |
| Near-tangent STE bias | 25-542% gradient error approaching tangency | Sigmoid kernel width exceeds intersection feature size | Higher resolution or boundary integral (Path C) |
| Frozen-topology plateau | Divergence theorem gradient plateaus at ~25% | Mesh topology does not adapt to parameter changes | Explicit boundary integral for moving intersection curve |

The first problem is addressed by the hybrid A+C strategy
(see [hybrid optimization strategy](hybrid_optimization_strategy.md)).
The second is an intrinsic O(h) limitation of grid-based methods,
acceptable for practical optimization (convergence validated on
sphere-sphere and cylinder-sphere objectives). The third requires
the Hadamard shape derivative boundary integral, deferred to a
future milestone.

## Practical Implication

Despite theoretical limitations near stratum boundaries, gradient-based
optimization converges to correct optima in practice. The STE gradient
is directionally correct (sign is always right) and the bias toward
volume overestimation acts as acceleration rather than obstruction.
This is validated by the 3D gradient accuracy benchmark
(`tests/benchmarks/test_gradient_accuracy_3d.py`).
