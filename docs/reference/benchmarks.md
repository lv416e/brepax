# Integration benchmark report

A reproducible measurement of BRepAX's volume paths and face-level
metric coverage on the project's standard STEP fixture set.

The report below is the **output of one command**, regenerated from
the same commit this documentation was built from (so the numbers
shown here and the version of BRepAX you are reading docs for are
guaranteed to be in sync):

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
