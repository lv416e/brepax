# 3D Extension Roadmap

Informed by concept proof findings: stratum-aware gradient exactness (Gate 1),
method complementarity (ADR-0011), and the hybrid optimization requirement.

## 3D Primitive Implementation Order

| Priority | Primitive | Rationale |
|----------|-----------|-----------|
| 1 | Sphere | Natural 3D extension of disk-disk. Same stratum structure (disjoint/intersecting/contained), same boundary function form (g = d - r1 - r2). Concept proof patterns reuse directly. |
| 2 | Cylinder | Axis-symmetric, industrially critical. SDF is distance to infinite cylinder minus radius. Stratum structure adds axis-parallel complexity. |
| 3 | Plane | Trivial SDF (signed distance to half-space). Enables **Cylinder + Plane = drilling** demo, the 3D extension midpoint milestone. |
| 4 | Cone | Natural extension of Cylinder (linear radius profile). Reuses Cylinder implementation patterns. |
| 5 | Torus | Non-convex SDF. Most complex parametric surface in the set. |
| 6 | Box | Axis-aligned. Edge and corner singularities create complex stratum structure. |

### Midpoint Milestone: Cylinder + Plane Drilling

After primitives 1-3 are complete, demonstrate differentiable Boolean intersection
of a Cylinder through a Plane (drilling operation). This is a representative
industrial CAD operation and serves as the first 3D demo with external visibility.

## Boolean Operations

Union, intersection, and subtract for all primitive pairs. Method (C) stratum-aware
is the primary method; Method (A) smoothing is retained for cross-stratum exploration.

## Method (B) TOI Correction

Deferred to the STEP pipeline milestone (ADR-0009). Implementation will occur
alongside the hybrid optimizer where cross-stratum gradient transfer has a
concrete use case. The mathematical derivation is complete
(docs/explanation/toi_derivation.md).

## Hybrid Optimizer

The 3D extension delivers the API skeleton in `brepax/experimental/optimizers/`:

- `HybridSchedule`: switch criteria (step count, boundary distance, loss plateau)
- `HybridResult`: trajectory, method log, stratum transitions, convergence status
- `hybrid_optimize()`: function signature fixed, implementation raises NotImplementedError

An example notebook demonstrates manual Method (A) to Method (C) switching
on the disk-disk problem. The automatic scheduler is deferred to the STEP
pipeline milestone.

## Completion Criteria

- [ ] 6 primitives with SDF, jit/vmap/grad compatible, property tests
- [ ] Union/intersection/subtract Boolean operations via Method (C)
- [ ] Cylinder + Plane drilling demo notebook
- [ ] Hybrid optimizer API skeleton in experimental/
- [ ] Gradient accuracy benchmark for 3D Boolean matching analytical baseline
- [ ] GPU benchmark for Gate 3b (vmap scaling on CUDA or Metal)
