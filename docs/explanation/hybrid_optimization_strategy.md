# Hybrid Optimization Strategy

## The Complementarity of Smoothing and Stratum-Aware Gradients

BRepAX provides two differentiation strategies with complementary strengths:

| Property | Method (A) Smoothing | Method (C) Stratum-Aware |
|----------|---------------------|--------------------------|
| Gradient accuracy (within stratum) | Biased by temperature k | Analytically exact |
| Gradient existence (cross stratum) | Non-zero everywhere | Zero where objective is stratum-invariant |
| Boundary behavior | Smooth but imprecise | Precise but discontinuous |
| Optimization scope | Global (can cross boundaries) | Local (within current stratum) |

Neither method alone is sufficient for all optimization scenarios. The optimal strategy depends on the optimization phase:

1. **Exploration phase**: Use Method (A) with large k to discover which stratum contains the optimum. The smoothing kernel provides global gradient signal at the cost of precision.
2. **Refinement phase**: Switch to Method (C) for exact convergence within the target stratum. Analytical gradients eliminate smoothing bias.

## Analogy with Differentiable Rendering

This complementarity mirrors the evolution of differentiable rendering:

- **SoftRas (2019)**: Probabilistic rasterization provides gradients for fully occluded objects (analogous to Method A providing cross-stratum signal)
- **nvdiffrast (2020)**: Analytic edge derivatives provide exact gradients at visibility boundaries (analogous to Method C providing exact within-stratum gradients)
- **Modern practice**: Coarse-to-fine scheduling combining soft rasterization for initialization with analytic gradients for refinement

BRepAX follows the same trajectory for CAD geometry.

## Empirical Evidence

### Within-stratum (Method C excels)

Optimizing r1 within the intersecting stratum:
- Method (C): converges to target with position error 7e-6 in 6 steps
- Method (A): converges to biased optimum with position error 0.021 in 6 steps

Method (C) is 3000x more precise.

### Cross-stratum (Method A provides signal)

Optimizing c2_x from disjoint (d=3.0) toward intersecting target (d=1.5):
- Method (C): gradient = 0.0 (area independent of center in disjoint stratum)
- Method (A): gradient = 7.5e-4 (smoothing kernel extends beyond boundary)

Method (A) provides weak but nonzero signal; Method (C) is mathematically stuck.

## Design Implications

The hybrid strategy will be implemented as an optimizer-level concern, not a modification to the Boolean operation API. The `union_area()` function remains method-agnostic; the caller selects the method based on optimization phase. A coarse-to-fine scheduler is planned for future implementation.
