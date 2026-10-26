# Integration benchmark report

A reproducible measurement of BRepAX's volume paths and face-level
metric coverage on the project's standard STEP fixture set.

The report below is a checked-in measurement snapshot. Building the docs
includes it without rerunning the benchmark, so its results need not describe
the current source revision. Its update history is available in
[Git history](https://github.com/lv416e/brepax/commits/main/benchmarks/integration_report/REPORT.md).
Regenerate it on the revision being evaluated with:

```bash
uv run python -m benchmarks.integration_report.run_benchmark
```

## What this report does not claim

- No "first" or "unique" claim about any BRepAX path.
- No quantitative speed comparison against external systems.
- No claim that BRepAX outperforms any external system on tasks
  those systems were designed for.

The competitor landscape section below is **qualitative feature
coverage only**, not a numerical head-to-head; cells are marked
"hypothesis" where they reflect best-effort reading of public
documentation.

## The report

--8<-- "benchmarks/integration_report/REPORT.md"
