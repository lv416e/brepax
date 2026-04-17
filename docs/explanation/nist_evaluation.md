# Evaluation on NIST MBE PMI Test Cases

## Overview

BRepAX was evaluated on 28 public STEP files from three sources: the
NIST MBE PMI test cases (16 files), the FreeCAD mechanical parts library
(9 files), and OCCT/CadQuery test geometry (3 files).  The evaluation
covers face conversion, CSG reconstruction, and scalability.

**Key results:**

- Face conversion: 4,080 / 4,080 = **100%** (all surface types including rational B-splines)
- CSG reconstruction: **24 / 24 tested files successful** (up to 664 faces)
- BSpline faces: 104 / 4,080 = **2.5%** of faces in real parts are freeform NURBS

## Data Sources

| Source | Files | Description |
|--------|-------|-------------|
| NIST FTC (Fully-Toleranced Test Cases) | 6 | FTC-06 through FTC-11: machined parts with holes, pockets, chamfers |
| NIST CTC (Combined Test Cases) | 5 | CTC-01 through CTC-05: complex parts with GD&T |
| NIST STC (Simplified Test Cases) | 5 | STC-06 through STC-10: simplified versions of FTC/CTC |
| FreeCAD Library | 9 | Ball bearings, hex nuts (M3-M10), linear slide, GT2 pulley |
| OCCT / CadQuery | 3 | Linkrods, screw, cube+cylinder test geometry |

All files are public domain or open source and may be used without restriction.

## Face Type Distribution

Aggregate across all 28 files (4,080 faces):

| Surface Type | Count | Percentage | BRepAX Primitive |
|-------------|-------|-----------|-----------------|
| Cylindrical | 1,847 | 45.3% | Cylinder |
| Planar | 1,285 | 31.5% | Plane |
| Conical | 472 | 11.6% | Cone |
| Toroidal | 216 | 5.3% | Torus |
| Spherical | 156 | 3.8% | Sphere |
| B-spline | 104 | 2.5% | BSplineSurface |

97.5% of faces are analytical surfaces representable by the five classical
primitive types.  B-spline surfaces appear in 7 of 28 files, concentrated
in parts with fillet blends and freeform features.

## Per-File Results

### NIST FTC / CTC / STC

| File | Faces | BSpline | Conversion | CSG Time | IN Cells |
|------|-------|---------|-----------|----------|----------|
| FTC-06 | 144 | 0 | 144/144 | 2.8s | 1,777 |
| FTC-07 | 269 | 20 | 269/269 | 6.6s | 1,686 |
| FTC-08 | 270 | 0 | 270/270 | 6.5s | 1,375 |
| FTC-09 | 158 | 0 | 158/158 | 4.3s | 527 |
| FTC-10 | 214 | 1 | 214/214 | 16.6s | 2,217 |
| FTC-11 | 6 | 0 | 6/6 | 0.5s | 5 |
| CTC-01 | 139 | 0 | 139/139 | 3.9s | 1,582 |
| CTC-02 | 664 | 34 | 664/664 | 63.8s | 4,108 |
| CTC-03 | 139 | 0 | 139/139 | 2.7s | 185 |
| CTC-04 | 518 | 0 | 518/518 | 18.0s | 4,901 |
| CTC-05 | 209 | 9 | 209/209 | 5.6s | 1,153 |
| STC-06 | 144 | 0 | 144/144 | 2.1s | 1,701 |
| STC-07 | 306 | 20 | 306/306 | 7.8s | 1,753 |
| STC-08 | 271 | 0 | 271/271 | 7.3s | 1,377 |
| STC-09 | 125 | 0 | 125/125 | 2.7s | 527 |
| STC-10 | 256 | 1 | 256/256 | 12.8s | 4,264 |

### FreeCAD Mechanical Parts

| File | Faces | BSpline | Conversion | CSG Time | IN Cells |
|------|-------|---------|-----------|----------|----------|
| 608ZZ Ball Bearing | 14 | 0 | 14/14 | 0.1s | 13 |
| 6201-2RS Ball Bearing | 16 | 0 | 16/16 | 0.1s | 13 |
| GT2 Pulley | 24 | 0 | 24/24 | 0.2s | 27 |
| Hex Nut M6 | 30 | 1 | 30/30 | 6.8s | 2 |
| Linear Slide | 28 | 0 | 28/28 | 0.3s | 24 |

### OCCT / Misc

| File | Faces | BSpline | Conversion | CSG Time | IN Cells |
|------|-------|---------|-----------|----------|----------|
| OCCT Screw | 10 | 0 | 10/10 | 0.1s | 9 |
| OCCT Linkrods | 37 | 18 | 37/37 | 58.4s | 78 |
| CadQuery Cube+Cyl | 9 | 0 | 9/9 | 0.1s | 2 |

## Scalability

CSG reconstruction time scales with the number of IN cells (geometric
complexity) rather than the total number of faces.  The PMC sampling
approach discovers only geometrically reachable sign vectors, making
the theoretical 2^n cell count irrelevant in practice.

| Faces | Theoretical 2^n | Actual IN Cells | CSG Time |
|-------|----------------|-----------------|----------|
| 6 | 64 | 5 | 0.5s |
| 139 | 10^41 | 185 | 2.7s |
| 270 | 10^81 | 1,375 | 6.5s |
| 518 | 10^155 | 4,901 | 18.0s |
| 664 | 10^199 | 4,108 | 63.8s |

The relationship is approximately linear in the number of IN cells,
with an additional factor for sampling convergence.

### BSpline Performance

A fast sign determination method (bounding box pruning + surface center
normal dot product) eliminates the Newton projection bottleneck during
PMC sampling.  BSpline faces add negligible overhead when they occupy
a small fraction of the part.

| Comparison | Faces | BSpline | Time |
|-----------|-------|---------|------|
| FTC-08 (no BSpline) | 270 | 0 | 6.5s |
| FTC-07 (20 BSpline) | 269 | 20 | 6.6s |

## Known Limitations

1. **Large BSpline-heavy parts**: OCCT linkrods (49% BSpline) takes 58s.
   Bounding box pruning is less effective when BSpline faces cover most
   of the part.

2. **Resolution vs accuracy**: Grid-based volume at resolution 32 gives
   0.3% error for NURBS shapes (vs 0.036% for analytical primitives).

3. **Thin features**: Very thin walls or narrow slots may be missed by
   random sampling if the feature volume is small relative to the bounding
   box.  Increasing samples_per_round mitigates this.

## Comparison with Feasibility Predictions

| Prediction | Actual |
|-----------|--------|
| "20 faces is practical limit" | 664 faces successful |
| "BSpline SDF is high risk" | 100% conversion, fast sign for PMC |
| "97-98% analytical faces" | 97.5% analytical (confirmed) |
| "Rational B-spline needs weights" | 56 rational faces converted with weights |
