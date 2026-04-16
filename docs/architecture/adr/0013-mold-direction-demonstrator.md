# ADR-0013: Mold Direction Optimizer as Application Demonstrator

## Status

Accepted

## Context

With the core Boolean kernel complete (8 primitives, 3 Boolean operations,
stratum-dispatched gradients), BRepAX lacks a concrete application
demonstrator showing what the library enables in practice.  Without at
least one end-to-end application, the library remains a collection of
differentiable primitives with no visible value proposition.

Mold pull direction optimization is a natural first application because:

- It directly uses differentiable SDF evaluation and gradient computation.
- It has a clear physical interpretation in injection-molding design.
- It requires optimization on a manifold (S^2), showcasing JAX composability.
- The problem scope is bounded enough for a lightweight implementation.

## Decision

Add a lightweight mold direction optimizer as an experimental application
under `src/brepax/experimental/applications/mold_direction.py`.

### Scope

**Included:**

- `undercut_volume()`: Surface-weighted softplus undercut metric estimated
  on a grid.  Uses finite-difference normals (avoids NaN from autodiff at
  degenerate SDF points like cylinder axes).
- `optimize_mold_direction()`: Projected gradient descent on S^2 (Euclidean
  step + re-normalization).
- `MoldDirectionResult` dataclass for optimization diagnostics.
- Example notebook `07_mold_direction_demo.py` with L-bracket shape.
- Unit tests verifying gradient finiteness, directional sensitivity,
  convergence, and unit-vector constraint.

**Explicitly excluded (deferred):**

- Formal r-set based undercut definition — requires topology layer.
- Riemannian optimization on S^2 (exponential map retraction).
- Integration with external cost models.
- STEP file input — BRepAX STEP I/O is not yet implemented.

### Design choices

**Surface-weighted softplus metric** rather than volume-based sigmoid:
the sigmoid undercut indicator satisfies `σ(-a) + σ(a) = 1`, causing
centrosymmetric surface contributions to cancel regardless of direction.
Softplus breaks this cancellation (`softplus(a) + softplus(-a) = |a| + C`),
giving meaningful directional signal even for shapes where non-convex
features are small relative to the total surface.

**Finite-difference normals** instead of `jax.grad(sdf)`: SDF
implementations using `jnp.linalg.norm` produce NaN gradients at
degenerate points (cylinder axis, box center).  Central differences avoid
this entirely and are sufficient for undercut direction estimation.  The
normals are constant with respect to the pull direction, so autodiff
through `undercut_volume` w.r.t. direction is unaffected.

**Simple projection on S^2** instead of Riemannian retraction: for a
demonstrator, `d = d / |d|` after each Euclidean gradient step is adequate.
Riemannian optimization (exponential map) can be added later as a
natural integration point with other manifold optimization libraries.

## Consequences

- BRepAX has a concrete "what can you do with this?" answer for launch.
- The experimental API surface grows, but all new code is under
  `experimental/applications/` with clear "subject to change" labeling.
- Future Phase 2 work (full mold direction optimizer, inverse design)
  can extend this skeleton rather than starting from scratch.
- The softplus metric design and finite-difference normal pattern are
  reusable for other SDF-based analysis tools.
