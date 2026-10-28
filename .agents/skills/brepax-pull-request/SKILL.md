---
name: brepax-pull-request
description: Prepare BRepAX commits, pushes, pull requests, or explicitly authorized releases using the repository's quality gates, PR template, and release-please conventions.
---

# BRepAX pull requests and releases

Read `AGENTS.md` for authorization boundaries and mandatory quality gates.
Implementation work does not by itself authorize a merge or release.

## Prepare a commit or push

- Inspect the branch, working tree, and final diff. Preserve unrelated changes
  and existing worktrees. Changes go through a feature branch and PR, never
  a direct push to `main`.
- Run all four root quality gates before every commit and push. Also lint and
  format changed Python paths outside the standard `src/` and `tests/` checks.
  Do not use CI as a substitute for local checks.
- Use Conventional Commits. The PR title becomes the squash commit subject,
  in the form `feat(scope): description (#number)` after merging.
- For Claude contributions, include
  `Co-Authored-By: Claude <noreply@anthropic.com>` in the relevant commit/squash
  body.

## Prepare the PR

Use `.github/PULL_REQUEST_TEMPLATE.md` as the canonical body structure:
`Summary`, `Verification`, and `Related` when applicable. Scale the content
to the change. Explain concrete resulting behavior and the reason for it;
do not pad the description to a fixed number of bullets.

Record verification that actually ran and relevant checks not run. New unit
and property tests are chosen by `brepax-validation`, not mandatory paired
checkboxes. New non-trivial public APIs need useful docstrings/examples;
approved architectural decisions need an ADR.

Before requesting merge, verify on GitHub that `ci-gate` passed, the branch
has no merge conflicts, and the title and body follow these conventions.
Report unavailable evidence as unavailable. Use squash merge only and do not
leave ceremonial merge comments.

For ordinary PRs, the repository owner's explicit merge authorization permits
an admin merge without the required approval count when required CI has passed
for the current head and substantive review findings are resolved. Immediately
before merging, recheck the head SHA, base, mergeability, checks, and review
threads; pin the expected head SHA. Never bypass failed or pending required
checks. Do not infer merge authorization from implementation approval alone.

## Release-please

A `chore(main): release x.y.z` PR represents the maintainer's release decision.
Never merge it without explicit instruction, even if checks are green.

Conventional Commits on `main` accumulate into a release-please PR. When an
explicit version override is requested, preserve `Release-As: x.y.z` as the
**only** body text of the triggering squash commit; extra prose can prevent
the intended release override. Check the actual squash body before merging.

After the maintainer-authorized release PR is merged, release-please creates
the GitHub release and tag, and the `publish` job in the same workflow publishes
to PyPI. Verify the release and package availability.

If publishing fails, diagnose the failure first. The manual fallback is
`gh workflow run publish.yaml -f tag=vX.Y.Z` for the authorized release tag.
Inspect the current workflow before using it. Keep TestPyPI validation and
production publishing distinct; see `CONTRIBUTING.md` for setup details.
