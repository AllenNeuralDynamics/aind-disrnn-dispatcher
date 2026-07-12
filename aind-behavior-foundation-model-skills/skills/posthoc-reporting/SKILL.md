---
name: posthoc-reporting
description: Post-hoc analysis and reporting conventions for studies — report files with YAML frontmatter and BEGIN/END result markers, JSON _meta blocks, launch_record results.md stubs, Makefile targets, cache .gitignore policy, one-producer-per-artifact. Use when writing/regenerating analysis reports, figures, or JSON summaries from finished W&B groups.
---

# Post-hoc analysis & reporting

This skill is the source of truth for analysis/reporting conventions (the one-line
rule lives in `AGENTS.md` §12). Full artifact contracts: `references/` here. When
in doubt, mirror a normalized study (e.g. `studies/01-gru-scaling-law/`) rather
than inventing layout.

## Principles

1. **Reports are code** — committed, regenerable, diffable; changed only via their
   producer script (manual prose lives outside markers).
2. **One producer per artifact** — every figure/JSON is written by exactly one script.
3. **Caches are `.gitignore`'d** — anything re-derivable from W&B (per-subject pulls,
   `analysis/_cache_*.json`). Curated summary JSON and reports/figures are committed.
4. **Provenance lives inside the artifact** — JSON `_meta`, report frontmatter.
5. **Idempotent regeneration** — `make r<n>` twice produces identical files
   (see the caveats below: `_meta` timestamps and cross-machine figure bytes).

## Pin the W&B groups — never discover them live (hard rule)

A producer MUST read runs from a **declared allowlist** of W&B groups
(`WANDB_GROUPS = [...]` at module level, queried per group, as in
`studies/01-gru-scaling-law/analysis/nxd_scaling.py`). It must NOT query the whole
project and keep whatever matches its grid.

Why: a project-wide query silently absorbs **any** new run that happens to fit the
cell definition, so a report's numbers change when someone launches something
unrelated. This actually happened — `02-gru-scaling-law-ignore/analysis/scaling.py`
globbed the project, and a 200-step validation smoke run moved its published H16
mean from 0.71635 to 0.70348 (SEM 5.3e-5 → 1.3e-2) with no code change. Pinned
producers (studies 01/03/04) ignored the same run and reproduced exactly.

`_meta.wandb_groups` must be a subset of the declared allowlist — never a set
discovered from whatever the query returned, which makes a contaminated pull look
self-consistent.

## What "reproducible" actually means here (verified 2026-07-11)

- **Data (curated JSON/CSV) is exactly reproducible** — a re-run changes nothing but
  `_meta`. This is the thing to check: `git diff` the JSON, ignore `_meta`, expect zero.
- **`_meta.produced_at_pt` (and `dispatcher_git_sha`) change every run**, so
  regeneration always dirties the tree. A CI check cannot be a bare
  `git diff --exit-code` — it must ignore `_meta`.
- **Figure PNGs are bit-identical only on the same machine.** Across machines the
  bytes (and ~6% of pixels — text/AA rendering) differ, because the *plotting* env
  is not pinned: `environment.lock` pins the wrapper commit that produced the
  *metrics*, not matplotlib/freetype. A PNG diff after regenerating on a different
  box is expected and is **not** evidence that results changed — confirm via the JSON.

## Layout (per study)

```
studies/<study>/
  Makefile                 # one PHONY target per report; `all` regenerates everything
  environment.lock         # pins the wrapper commit that PRODUCED the metrics
  CHANGELOG.md
  analysis/
    <producer>.py          # one script == one report block or one JSON producer
    <producer>.json        # carries _meta (via studies/util/_meta.py, pass study_root=)
    wandb_keys.py          # comment-only contract naming the exact W&B summary keys read
    fig_<slug>.png         # style via studies/util/plot_style.py
    reports/
      INDEX.md             # one row per report
      r<n>-<slug>.md
    provenance/            # backfill history, metric caveats, etc.
```

Shared helpers live in `studies/util/` (`_meta.py`, `plot_style.py`) — reuse them,
never copy per-study. Commit derived outputs (JSON/CSV/PNG); gitignore only caches.

## Contracts in brief (full templates: `references/report-contracts.md`)

- **Report** `r<n>-<slug>.md`: required frontmatter (`id`, `slug`, `status`
  draft→live→superseded, `authors`, `wandb_groups`, `inputs`, `reproduce`);
  script-owned `<!-- BEGIN result-N -->` / `<!-- END result-N -->` regions —
  anything inside is overwritten on regen; producers match marker pairs only.
- **JSON**: opens with `_meta` (`produced_by`, `produced_at_pt` in Seattle time,
  `dispatcher_git_sha`, `wandb_groups`).
- **`launch_record_<label>/results.md`**: written once after the group settles —
  W&B group + project URL, Beaker exp id + resources, timestamps, status, headline
  metric, which reports it feeds.

## Multi-agent collaboration rules

- Never hand-edit inside a `live` report's markers — rerun its producer instead.
- One PR = one report or one producer change; don't silently shift another report's numbers.
- New analysis output ⇒ declare it: add to a report's `inputs:`, add `_meta`, add
  cache patterns to `.gitignore`.
- Never squash-merge (AGENTS.md §9); state assumptions in the PR body and cite the
  W&B groups that justify a changed claim.
- Reports use Seattle time and clickable W&B links (AGENTS.md §7).

## References (read on demand)

- `references/report-contracts.md` — full templates: report frontmatter + marker
  example, `_meta` JSON, `results.md` stub, Makefile convention, `.gitignore`
  policy, enforcement checklist.
- `references/wandb-graphql-sandbox.md` — pulling W&B data from the Claude Science
  sandbox where `wandb.Api()` can't run (GraphQL + `requests` recipe, deleteRun
  gotcha).
