# ABC Dataset evaluation tool

Evaluate any STEP file through the full BRepAX benchmark pipeline
(five volume paths + four face-level metrics + OCCT BRepGProp
reference).  This is the reusable building block for M7a
(20+ external models from the ABC Dataset).

## Evaluate a single STEP file

```bash
uv run python -m benchmarks.abc_evaluation.evaluate_step path/to/file.step
```

Add `--json` for machine-readable output (one JSON object to stdout):

```bash
uv run python -m benchmarks.abc_evaluation.evaluate_step path/to/file.step --json
```

## Obtaining ABC Dataset STEP files

The [ABC Dataset](https://deep-geometry.github.io/abc-dataset/)
(Koch et al., CVPR 2019) contains ~1M parametric CAD models from
Onshape in STEP format.  Files are distributed as `.7z` chunks of
10,000 models each.

**License note.** Copyright of each model belongs to its Onshape
creator.  Public documents are usable for research under the
[Onshape Terms of Use](https://www.onshape.com/en/legal/terms-of-use).
Do not bundle raw STEP files in this repository.

To download one chunk and extract a single model:

```bash
# 1. Get the URL list (one-time)
wget https://archive.nyu.edu/rest/bitstreams/89509/retrieve -O step_v00.txt

# 2. Download chunk 0000 (~1-3 GB compressed)
head -2 step_v00.txt | xargs -n 2 sh -c 'wget --no-check-certificate $0 -O $1'

# 3. Extract one model (requires p7zip / 7z)
7z e abc_0000_step_v00.7z 00000050_80d90bfdd2e74f709a8c_step_000.step -oabc_models/

# 4. Evaluate
uv run python -m benchmarks.abc_evaluation.evaluate_step abc_models/00000050_80d90bfdd2e74f709a8c_step_000.step
```

Adjust the model filename to select a different model.  Use the
corresponding `feat_v00` YAML to filter by surface type
(Plane/Cylinder/Sphere/BSpline/etc.) and patch count.

## What it measures

Same pipeline as `benchmarks/integration_report/run_benchmark.py`:

- **Volume accuracy** against OCCT BRepGProp on five paths:
  `divergence_volume`, `mesh_sdf`, `gwn`, `CSG-Stump`,
  `TrimmedCSGStump`.
- **Face-level metric coverage**: `surface_area_per_face`,
  `min_draft_angle_per_face`, `mean_curvature_per_face`,
  `min_wall_thickness_per_face`.
