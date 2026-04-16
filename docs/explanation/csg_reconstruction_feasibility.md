# CSG Reconstruction Feasibility Assessment

## Context

BRepAX Stage B.1 converts STEP faces to analytical primitives (Plane, Cylinder,
Sphere, Cone, Torus). Stage B.2 builds a face adjacency graph. The question is
whether BRepAX can reconstruct a CSG tree from these ingredients, and if so, how.

## BRepAX Starting Point

Unlike most CSG reconstruction research, BRepAX starts from:

- **Exact analytical primitives** (not learned, not approximated)
- **Face adjacency graph** (which faces share which edges)
- **Differentiable Boolean operations** (stratum-aware Method C)

This combination does not exist in any prior work.

## Literature Landscape

### Classical Algorithms

**Shapiro-Vossler BSP Decomposition (1991-1993).** Converts B-Rep to CSG by
extending face surfaces to half-spaces, enumerating candidate regions via
point-membership classification (PMC), and recovering a Boolean expression in
disjunctive normal form (DNF). Mathematically rigorous for manifold solids
bounded by algebraic surfaces. Worst-case exponential (2^n for n half-spaces),
but prunable. This is the most directly applicable classical method because it
starts from exactly what BRepAX has: classified analytical surfaces.

**Feature-Based Decomposition (Vandenbrande-Requicha, Woo).** Recognizes
machining features (holes, pockets, slots) via subgraph pattern matching on the
face adjacency graph. Produces manufacturing-oriented feature trees, not general
CSG. Useful as a heuristic prior for common industrial parts.

**Cell-Based Decomposition (Rossignac-O'Connor).** Extends all surfaces to form
a spatial arrangement, classifies cells as inside/outside. Complete but produces
flat DNF with cell count explosion in practice.

### Learning-Based Approaches

**CSG-Stump (Ren et al., 2021).** Fixed three-layer structure: primitives as
half-spaces, intersections of subsets, union of results. Essentially learnable
DNF. Connection weights are continuous during training, binarized at inference.
Scalable to ~64 primitives from point clouds.

**UCSG-Net (Kaczynska et al., 2020).** Learns fixed-depth CSG trees with
continuous primitive parameters and soft Booleans from occupancy supervision.
Limited tree depth and primitive vocabulary.

**CAPRI-Net (Yu et al., 2022).** Like CSG-Stump but with neural half-spaces
instead of analytical primitives. More expressive but loses interpretability.

**InverseCSG (Du et al., 2018).** Enumerative program synthesis scored by
Chamfer distance. Exact but exponentially slow (~8-12 primitives max).

All neural methods start from point clouds or voxels and must learn primitive
fitting. BRepAX bypasses this entirely.

### Commercial State of the Art

Industry uses rule-based feature recognizers on B-Rep adjacency graphs, not
general CSG reconstruction. The consensus is that general B-Rep to CSG is
ill-posed (many valid trees produce the same B-Rep). Commercial tools constrain
the problem to manufacturing-relevant features.

## The Gap

No existing work combines:

1. Classified B-Rep faces (exact primitives) as input
2. Face adjacency graph for topological pruning
3. Differentiable Boolean operations for gradient-based refinement

This is the BRepAX opportunity.

## Recommended Approach: Differentiable CSG-Stump

### Core Idea

Implement a differentiable CSG-Stump layer where:

- **Primitives are fixed** from STEP face classification (no fitting needed)
- **Boolean connection matrix** is the only learnable/optimizable parameter
- **Face adjacency graph** prunes the connection space (non-adjacent faces
  unlikely to participate in the same intersection term)
- **Shapiro-Vossler PMC** provides warm-start initialization
- **BRepAX stratum-aware Booleans** enable gradient flow through the tree

### Implementation Phases

**Phase 1: Minimum Viable Reconstruction.** Handle "stock minus features"
patterns: a base primitive (Box) with holes/pockets subtracted (Cylinders).
This covers a large fraction of machined parts. Use adjacency graph to identify
connected groups of subtractive features.

Estimated effort: 1-2 weeks. Expected success rate for simple machined parts:
60-70%.

**Phase 2: DNF Reconstruction.** Implement full CSG-Stump with adjacency-based
pruning and PMC warm-start. Flat DNF output (union of intersections of
half-spaces).

Estimated effort: 2-4 weeks. Handles arbitrary analytical-surface parts.

**Phase 3: Tree Compaction.** Convert flat DNF to compact binary CSG tree via
Boolean expression minimization (Quine-McCluskey or greedy factoring).

Estimated effort: 1-2 weeks. Produces human-readable CSG.

### Failure Modes and Fallbacks

- **Non-manifold geometry:** Skip, return face-level primitives only
- **Interacting features:** Fall back to flat DNF without compaction
- **NURBS faces in the mix:** Exclude from CSG tree, treat as fixed boundary

## Alternative: Face-Level Trimmed SDF

If full CSG reconstruction proves too costly, a pragmatic alternative:

- Each face is an infinite surface primitive (already implemented)
- Trim to finite face boundary using edge distance truncation
- Combine trimmed SDFs via smooth min/max
- Gradient flows through face parameters without CSG tree

**Pros:** Simpler, faster to implement (~1 week). Face-level gradient for
optimization. Sufficient for many practical applications (mold direction,
undercut analysis, wall thickness).

**Cons:** No B-Rep semantics preservation. Cannot represent topology changes
(adding/removing features). Trimmed SDF gradient at trim boundaries is
non-trivial.

## Recommendation

**Start with Phase 1 (minimum viable reconstruction)** targeting "stock minus
features" patterns. This delivers practical value quickly and validates the
CSG-Stump + adjacency pruning approach on real parts. If Phase 1 succeeds,
proceed to Phase 2 (full DNF). If Phase 1 reveals fundamental obstacles, pivot
to the face-level trimmed SDF alternative.

The face-level alternative is always available as a fallback, so there is
limited downside risk to attempting Phase 1 first.

## Academic Positioning

A differentiable CSG-Stump operating on exact B-Rep primitives with
adjacency-based pruning would be, to our knowledge, the first system bridging
classical Shapiro-Vossler theory with modern differentiable programming. This
positions BRepAX at the intersection of computational geometry and
differentiable programming — a novel contribution regardless of whether full
tree compaction (Phase 3) is achieved.
