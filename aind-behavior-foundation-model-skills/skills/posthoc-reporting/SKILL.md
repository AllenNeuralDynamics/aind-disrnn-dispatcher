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
5. **Idempotent regeneration** — `make r<n>` twice produces identical files.

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
