---
name: posthoc-reporting
description: Post-hoc analysis and reporting conventions for studies — report files with YAML frontmatter and BEGIN/END result markers, JSON _meta blocks, launch_record results.md stubs, Makefile targets, cache .gitignore policy, one-producer-per-artifact. Use when writing/regenerating analysis reports, figures, or JSON summaries from finished W&B groups.
---

# Post-hoc analysis & reporting

Canonical detail: `docs/posthoc-analysis.md` (this is a **target spec**; parts of
`studies/data-scaling-law/` predate it). **If this skill and the doc conflict, the doc
wins.**

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
  analysis/
    <producer>.py          # one script == one report block or one JSON producer
    <producer>.json        # carries _meta
    fig_<slug>.png
    reports/r<n>-<slug>.md
```

## Report file contract (`reports/r<n>-<slug>.md`)

Required YAML frontmatter: `id`, `slug`, `status` (draft → live → superseded),
`authors`, `wandb_groups`, `inputs` (script/data/figure paths),
`reproduce` (e.g. `make -C studies/<study> r7`).

Script-owned regions are bounded by `<!-- BEGIN result-N -->` / `<!-- END result-N -->`
markers; anything inside is overwritten on regen — human prose goes outside. Producer
scripts match on marker pairs only, never on prose or line numbers. Figures referenced
by relative path. Superseded reports stay in the tree and link forward.

## JSON `_meta` contract

Every analysis JSON opens with:

```json
"_meta": {
  "produced_by": "analysis/<script>.py",
  "produced_at_pt": "<ISO ts, America/Los_Angeles>",
  "dispatcher_git_sha": "<sha>",
  "wandb_groups": ["<variant>@<launch_id>", "..."]
}
```

## Reading W&B from the Claude Science sandbox

Producer scripts run on the HPC node (or any authenticated machine) where the
`wandb` SDK works normally — nothing special needed there. **But** when pulling
run/sweep data from the Claude Science Mac sandbox, `wandb.Api()` fails: it tries
to spawn a `wandb-core` helper subprocess (`ServicePollForTokenError`) and write
config under `~/.config/wandb`, both blocked in the sandbox. Working route there:

- Store the key as the `WANDB_API_KEY` credential; hit the GraphQL endpoint
  directly with `requests`, `auth=('api', KEY)` — no SDK.
- `https://api.wandb.ai` must be allowlisted (one-time `request_network_access`).
- Entity/login is `houhan` (`hanhou`); team entity `AIND-disRNN`.

```python
import requests, os
KEY = os.environ["WANDB_API_KEY"]
GQL = "https://api.wandb.ai/graphql"
q = """query($e:String!,$p:String!,$s:String!){
  project(name:$p,entityName:$e){ sweep(sweepName:$s){ id state
    runs(first:50){ edges{ node{ name displayName state summaryMetrics config }}}}}}"""
r = requests.post(GQL, auth=("api", KEY),
    json={"query": q, "variables": {"e":"AIND-disRNN","p":"hpc_test","s":"<sweep_id>"}},
    timeout=60).json()
# per-run full history: query run(name:$run){ history(samples:100000) }
```

Prefer running the producer on the HPC node when possible; use the GraphQL route
only for ad-hoc pulls from the sandbox.

## `launch_record_<label>/results.md` (written once, after the group settles)

30-second stub: W&B group + project URL, Beaker exp id (cluster/priority/resources),
submitted/settled Seattle timestamps, status (success / partial N-of-M / failed),
one-line headline metric, which reports it feeds, non-obvious notes (preemptions,
retries).

## Multi-agent collaboration rules

- Never hand-edit inside a `live` report's markers — rerun its producer instead.
- One PR = one report or one producer change; don't silently shift another report's numbers.
- New analysis output ⇒ declare it: add to a report's `inputs:`, add `_meta`, add
  cache patterns to `.gitignore`.
- Never squash-merge (AGENTS.md §9); state assumptions in the PR body and cite the
  W&B groups that justify a changed claim.
- Reports use Seattle time and clickable W&B links (AGENTS.md §7).
